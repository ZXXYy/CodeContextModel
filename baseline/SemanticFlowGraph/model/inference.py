import os
import json
import torch
import argparse
import logging

import pandas as pd
import xml.etree.ElementTree as ET

from tqdm import tqdm
from SemanticCodebert import SemanticCodebert
from manager import MixedPrecisionManager
from tokenizer import QueryTokenizer, DocTokenizer
from utils_colbert import get_special_tokens, get_config
from utils import get_device

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('indexer')
TOPK=5

class ModelInference:
    def __init__(self, colbert: SemanticCodebert, args, amp=False):
        assert colbert.training is False

        self.special_tokens = get_special_tokens(args.checkpoint)
        self.emb_cmp = args.embeddings_comparison
        self.colbert = colbert
        # self.query_tokenizer = QueryTokenizer(self.colbert.config, args.query_maxlen)
        self.query_tokenizer = DocTokenizer(self.colbert.config, args.doc_maxlen, self.special_tokens)
        self.doc_tokenizer = DocTokenizer(self.colbert.config, args.doc_maxlen, self.special_tokens)

        self.amp_manager = MixedPrecisionManager(amp)

    def query(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                Q = self.colbert.query_q(*args, **kw_args)
                if 'average' in self.emb_cmp:
                    # unsqueeze(0) to be compatible with ColBERT code;
                    # ColBERT needs to get (#words, #dim) per document,
                    # so unsqueeze(0) will make it (1, #dim) which is fine
                    Q = torch.mean(Q, dim=1).unsqueeze(0).squeeze(1)
                    # Q = [torch.mean(q, dim=0).unsqueeze(0) for q in Q]
                return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                D = self.colbert.doc_q(*args, **kw_args)
                if 'average' in self.emb_cmp:
                    # unsqueeze(0) to be compatible with ColBERT code;
                    # ColBERT needs to get (#words, #dim) per document,
                    # so unsqueeze(0) will make it (1, #dim) which is fine
                    # D = [torch.mean(d, dim=0).unsqueeze(0) for d in D]
                    D = torch.stack([torch.mean(d, dim=0).unsqueeze(0) for d in D]).squeeze(1)

                return D.cpu() if to_cpu else D

    def queryFromText(self, queries, bsize=None, to_cpu=False):
        if bsize:
            batches = self.query_tokenizer.tensorize(queries, bsize=bsize)
            batches = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in batches]
            return torch.cat(batches)

        input_ids, attention_mask = self.query_tokenizer.tensorize(queries)
        # add paddings if the length of query is less than 256
        if input_ids.shape[1] < 256:
            input_ids = torch.cat([input_ids, torch.zeros(input_ids.shape[0], 256-input_ids.shape[1], dtype=torch.long, device=input_ids.device)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.shape[0], 256-attention_mask.shape[1], dtype=torch.long, device=attention_mask.device)], dim=1)
        
        Q = self.query(input_ids, attention_mask)
        return Q, attention_mask

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False):
        torch.cuda.empty_cache()
        if bsize:
            batches, reverse_indices = self.doc_tokenizer.tensorize(docs, bsize=bsize)

            batches = [self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)
                       for input_ids, attention_mask in batches]

            if keep_dims:
                D = _stack_3D_tensors(batches)
                return D[reverse_indices]

            D = [d for batch in batches for d in batch]
            return [D[idx] for idx in reverse_indices.tolist()]

        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        # add paddings if the length of query is less than 256
        if input_ids.shape[1] < 256:
            input_ids = torch.cat([input_ids, torch.zeros(input_ids.shape[0], 256-input_ids.shape[1], dtype=torch.long, device=input_ids.device)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.shape[0], 256-attention_mask.shape[1], dtype=torch.long, device=attention_mask.device)], dim=1)

        return self.doc(input_ids, attention_mask, keep_dims=keep_dims)

    def score(self, Q, D, mask=None, lengths=None):
        if lengths is not None:
            assert mask is None, "don't supply both mask and lengths"

            mask = torch.arange(D.size(1), device=self.colbert.dev) + 1
            mask = mask.unsqueeze(0) <= lengths.to(self.colbert.dev).unsqueeze(-1)

        scores = (D @ Q.T)
        # scores = scores if mask is None else scores * mask.unsqueeze(-1)
        # scores = scores.max(1)
        return scores.squeeze(1).cpu()
        # return scores.values.sum(-1).cpu()


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos

    return output

def load_colbert(args, device):
    config = get_config(args.config)
    token_config = args.checkpoint.split('/')[-1].split('_')[4]
    if os.path.exists(args.checkpoint):
        logger.info('Loading model from {0}'.format(args.checkpoint))
        model = SemanticCodebert(config, token_config, dev=device, query_maxlen=args.query_maxlen, doc_maxlen=args.doc_maxlen,
                        dim=args.dim, similarity_metric=args.similarity, mask_punctuation=args.mask_punctuation)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        return model
    else:
        raise RuntimeError('Cannot load model from {0}. Path does not exist.'.format(args.checkpoint))
    
def load_test_cases(args):
    project_name = args.data_dpath.split('/')[-1]
    train_test_index_path = os.path.join(
        os.path.dirname(args.data_dpath), 
        'train_test_index', 
        project_name, 
        'test_index.json'
    )
    reader = json.load(open(train_test_index_path))
    reader = [x.replace('repo_first_3', project_name) for x in reader]

    queries, total_hunks, total_labels = [], [], []
    for test_case in reader:
        query, hunks, labels = "", [], []
        expanded_model_path = os.path.join(test_case, f"{args.step}_step_seeds_expanded_model.xml")
        model_dir = expanded_model_path.split('/')[-2]
        codes_path = os.path.join(test_case, f"my_java_codes.tsv")
        df_code = pd.read_csv(codes_path, sep='\t')
        if not os.path.exists(expanded_model_path):
            continue

        tree = ET.parse(expanded_model_path)
        root = tree.getroot()
        nodes = root.findall(".//vertex")
        for vertex in nodes:
            node_id = '_'.join([model_dir, vertex.get('kind'), vertex.get('ref_id')]) 
            code = df_code[df_code['id'] == node_id]['code'].values[0]
            if vertex.get('seed', '0') == '1':
                query += code + ' [UNUSED_6] '
            elif vertex.get('origin', '0') == '1':
                hunks.append(code + ' [UNUSED_6] ')
                labels.append(1)
            else:
                hunks.append(code + ' [UNUSED_6] ')
                labels.append(0)
            if len(hunks) > 300:
                break

        queries.append([query])
        total_hunks.append(hunks)
        total_labels.append(labels)
        # logger.info(f"{expanded_model_path} - hunks: {len(hunks)}")
    
    return queries, total_hunks, total_labels

def compute_mrr(scores, labels):
    mrr = 0
    topk = min(100, len(labels))
    topk_indices = torch.topk(scores, topk).indices.flatten() 
    # print(f"topk_indices shape: {topk_indices.shape}")
    # print(topk_indices)
    for i, item in enumerate(topk_indices):
        label = labels[topk_indices[i]]
        if label == 1:
            mrr = 1 / (i + 1)
            return {'MRR': mrr}
    return {'MRR': 0}

def compute_metrics(scores, labels):
    total_hit = {}
    for i in range(1, TOPK+1):
        total_hit[f'top{i}_hit'] = 0
    total_hit['mrr'] = 0

    for topk in range(1, TOPK+1):
        temp = topk
        topk = min(topk, len(scores))
        topk_indices = scores.topk(topk).indices
        hit = 0
        for i in range(0, len(topk_indices)):
            label = labels[topk_indices[i]] # get the index of the top3 embeddings
            if label == 1:
                hit += 1
                break
        total_hit[f"top{temp}_hit"] += 1 if hit > 0 else 0

    mrr = compute_mrr(scores, labels)
    total_hit['mrr'] += mrr['MRR']

    return total_hit

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'])
    parser.add_argument('--dim', dest='dim', default=128, type=int)
    parser.add_argument('--query-maxlen', dest='query_maxlen', default=256, type=int)
    parser.add_argument('--doc-maxlen', dest='doc_maxlen', default=256, type=int)
    parser.add_argument('--mask-punctuation', dest='mask_punctuation', default=True, action='store_true')

    parser.add_argument('--embeddings-comparison', choices=['average', 'token'], default='average')
    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        default='../../../data/zxing/model_ColBERT_zxing_hunks_bertoverflow_QARCL_q256_d256_dim128_cosine_hunk')

    parser.add_argument('--data-dpath', dest='data_dpath', default='../../../data/zxing')
    parser.add_argument('--config', choices=['BERTOverflow', 'BERT', 'CodeBERT'], default='BERTOverflow')
    parser.add_argument('--amp', dest='amp', default=True, action='store_true')

    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.granularity = None
    device = get_device(args.gpu)

    eval_hit_rate = {}
    for i in range(1, 6):
        eval_hit_rate[f'top{i}_hit'] = 0
    eval_hit_rate['mrr'] = 0

    colbert = load_colbert(args, device).to(device)
    inference = ModelInference(colbert, args, amp=args.amp)
    colbert.eval()
    logger.info(f"====load model {args.checkpoint} finished, device: {device}====")

    queries, total_hunks, total_labels = load_test_cases(args)
    logger.info(f"====load test cases finished====")

    for i, query in tqdm(enumerate(queries), total=len(queries)):
        passages = total_hunks[i]
        labels = total_labels[i]
        with torch.no_grad():
            Q, _ = inference.queryFromText(query, bsize=None)
            D = inference.docFromText(passages, bsize=None)
            logger.debug(f"Q type: {type(Q)}, D type: {type(D)}")
            logger.debug(f"Q shape: {Q.shape}, D shape: {D.shape}")
            scores = inference.score(Q, D) 
            logger.debug(f"scores shape: {scores.shape}")
            # get topk scores and indices
            metrics = compute_metrics(scores, labels)
            eval_hit_rate = {k: eval_hit_rate[k] + metrics[k] for k in metrics}
    
    eval_hit_rate = {"eval_"+k: v / len(queries) for k, v in eval_hit_rate.items()}
    print(f"Test finished, Test Metrics {eval_hit_rate}")
                




    