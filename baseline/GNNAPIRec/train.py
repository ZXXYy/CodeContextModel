# 1. embeddings 要重新计算过，算出新的embedding index - embedding.py
# 2. 直接用ExpandGraphDataset的数据集 - code_context_model/build_dataset.py
# 3. 预处理数据集ExpandGraphDataset，将原始数据集转换为可以用于GAPI模型训练的数据集 
# 4. train model & test model

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
import random
import time
import argparse
import torch
import logging
import dgl
import wandb
import atexit
import numpy as np
import collections
import torch.nn.functional as F

from tqdm import tqdm
from torch import nn
from itertools import product
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall

from lex import LexParser
from model import GCNRec
from code_context_model.build_dataset import ExpandGraphDataset


logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

TOPK = 5
CASE_NOT_TOPK_HIT = []


ste_results = collections.defaultdict(list)

# handle the exit event
def exit_handler():
    # wandb finish 
    wandb.finish()
atexit.register(exit_handler)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 计算相似度（例如使用余弦相似度）
def cosine_similarity(x1, x2):
    # 确保输入是1D张量，如果输入是2D或更高维度的张量，可以根据实际需求调整
    if x1.dim() > 1:
        x1 = x1.view(-1)
    if x2.dim() > 1:
        x2 = x2.view(-1)
    return F.cosine_similarity(x1.unsqueeze(0), x2.unsqueeze(0))

def pairwise_cosine_similarity(x1, x2):
    similarities = []
    for i in range(x1.size(0)):
        similarity = cosine_similarity(x1[i], x2[i])
        similarities.append(similarity)
    return torch.stack(similarities)
    
def euclidean_distance(x1, x2):
    # 确保输入是1D张量，如果输入是2D或更高维度的张量，可以根据实际需求调整
    if x1.dim() > 1:
        x1 = x1.view(-1)
    if x2.dim() > 1:
        x2 = x2.view(-1)
    # 计算欧式距离
    return torch.dist(x1, x2).item()

def compute_mrr(labels, similarities):
    mrr = 0
    topk = min(100, len(labels))
    topk_indices = torch.topk(similarities, topk).indices.flatten() 
    # print(f"topk_indices shape: {topk_indices.shape}")
    # print(topk_indices)
    for i, item in enumerate(topk_indices):
        label = labels[topk_indices[i]]
        if label == 1:
            mrr = 1 / (i + 1)
            return {'MRR': mrr}
    return {'MRR': 0}

def compute_metrics(model, graph, node_feats, node_labels):
    total_hit = {}
    for i in range(1, TOPK+1):
        total_hit[f'top{i}_hit'] = 0
    total_hit['mrr'] = 0
    
    for topk in range(1, TOPK+1):
        topk_indices, _ = model.get_top_items(graph, node_feats, node_labels, k=topk)
        topk_indices = topk_indices.cpu().numpy()
        # print(f"{topk}: {topk_indices}")
        labels = node_labels[node_labels != -1].cpu().numpy()
        hit = 0
        for i in range(0, len(topk_indices)):
            label = labels[topk_indices[i]] # get the index of the top3 embeddings
            if label == 1:
                hit += 1
                break
        total_hit[f"top{topk}_hit"] += 1 if hit > 0 else 0

    _, ratings = model.get_top_items(graph, node_feats, node_labels, k=topk)
    mrr = compute_mrr(labels, ratings)
    total_hit['mrr'] += mrr['MRR']

    return total_hit
    
def train(train_loader, valid_loader, verbose=True, **kwargs):
    pretrained_emb_path = kwargs.get('pretrained_emb_path', None)
    lr = kwargs.get('lr', 0.01)
    num_epochs = kwargs.get('num_epochs', 50)
    output_dir = kwargs.get('output_dir', 'output')
    neg_sz = kwargs.get('neg_sz', 2)
    debug = kwargs.get('debug', False)


    parser = LexParser(pretrained_emb_path)
    pre_emb = torch.stack([torch.from_numpy(emb) for emb in parser.pre_embedding]).to(device)

    model = GCNRec(len(parser.vocab), pre_emb).to(device)
    num_params = sum([p.numel() for p in model.parameters()])
    logger.info('total model parameters: {}'.format(num_params))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    logger.info("======= Start training =======")
    for epoch in tqdm(range(num_epochs)):
        total_loss, eval_loss = 0.0, 0.0
        model.train()

        train_hit_rate = {}
        for i in range(1, 6):
            train_hit_rate[f'top{i}_hit'] = 0
        train_hit_rate['mrr'] = 0
        # for user, pos_item, neg_item in tqdm(dataset.gen_batch(batch_sz, neg_sz),
        #                                      total=len(dataset.train)//batch_sz):
        for i, batch_graphs in enumerate(train_loader):
            # label = np.concatenate((np.ones(batch_sz), np.zeros(batch_sz*neg_sz)))
            # loss = model(gvar(user), gvar(pos_item), gvar(neg_item), gvar(label))
            batch_graphs = batch_graphs.to(device)
            batch_graphs.ndata['feat'] = batch_graphs.ndata['feat'].to(device)
            batch_graphs.edata['label'] = batch_graphs.edata['label'].to(device)
            batch_graphs.ndata['label'] = batch_graphs.ndata['label'].to(device)
            loss = model(batch_graphs, batch_graphs.ndata['feat'], batch_graphs.ndata['label'], batch_graphs.edata['label'].squeeze(1))

            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # evaluate
        eval_hit_rate, eval_loss = eval(model, valid_loader, verbose=False)
        wandb_log = {
            "Epoch": epoch,
            "Train Loss": total_loss,
            "Eval Loss": eval_loss,
        }
        wandb_log.update(train_hit_rate)
        wandb_log.update(eval_hit_rate)
        if not debug:
            wandb.log(wandb_log)

        logger.info(f"Epoch {epoch}, Train Loss {total_loss}")
        logger.info(f"Epoch {epoch}, Eval  Loss {eval_loss}, Eval Metrics {eval_hit_rate}")
        # save the model
        torch.save(model.state_dict(), f"{output_dir}/model_{epoch}.pth")
        logger.info(f"Model saved at {output_dir}/model_{epoch}.pth")

    logger.info("======= Training finished =======")

def eval(model, data_loader, **kwargs):
    verbose = kwargs.get('verbose', True)
    eval_loss = 0.0
    eval_hit_rate = {}
    for i in range(1, 6):
        eval_hit_rate[f'top{i}_hit'] = 0
    eval_hit_rate['mrr'] = 0
    
    # logger.info("======= Start evaluating =======")
    model.eval()
    with torch.no_grad():
        eval_graph_num_cnt = 0
        for i, batch_graphs in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_graphs.ndata['feat'] = batch_graphs.ndata['feat'].to(device)
            batch_graphs.edata['label'] = batch_graphs.edata['label'].to(device)
            batch_graphs.ndata['label'] = batch_graphs.ndata['label'].to(device)
            loss = model(batch_graphs, batch_graphs.ndata['feat'], batch_graphs.ndata['label'], batch_graphs.edata['label'].squeeze(1))
            eval_graph_num_cnt += len(batch_graphs.batch_num_nodes())
            eval_loss += loss.item()
            # TODO: compute_metrics
            metrics = compute_metrics(model, batch_graphs, batch_graphs.ndata['feat'], batch_graphs.ndata['label'])
            # print(f"Eval Batch {i}: Metrics {metrics}")
            eval_hit_rate = {k: eval_hit_rate[k] + metrics[k] for k in metrics}
        
        eval_hit_rate = {"eval_"+k: v / eval_graph_num_cnt for k, v in eval_hit_rate.items()}
    
    return eval_hit_rate, eval_loss
    
def test(model, test_loader, **kwargs):
    logger.info("======= Start testing =======")
    test_hit_rate, _ = eval(model, test_loader, verbose=False)
    logger.info(f"Test finished, Test Metrics {test_hit_rate}")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # train args
    parser.add_argument('--do_train', action='store_true', help='train the model')
    parser.add_argument('--device', type=int, default=1, help='device id')
    parser.add_argument('--input_dirs', type=str, nargs='+', default='data', help='input directories')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--train_batch_size', type=int, default=1, help='train batch size')
    parser.add_argument('--valid_batch_size', type=int, default=1, help='train batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--neg_sz', type=int, default=2, help='negative sample size.')
    parser.add_argument('--pretrained_emb_path', type=str, default=None, help='pretrained word2vecembedding path.')
    
    # test args
    parser.add_argument('--do_test', action='store_true', help='test the model')
    parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')
    parser.add_argument('--test_model_pth', type=str, default='model_48.pth', help='test model path')

    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    
    if args.debug:
        args.num_epochs = 1 
        args.test_model_pth = 'model_0.pth'
    elif args.do_train:
        args.output_dir = os.path.join(args.output_dir, f"{time.strftime('%m-%d-%H-%M')}")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        wandb.init(project="code-context-model")
        # 配置wandb
        config = wandb.config
        config.learning_rate = args.lr
        config.batch_size = args.train_batch_size
        config.epochs = args.num_epochs
        config.device = args.device
    if args.do_train:
        args.test_model_pth = os.path.join(args.output_dir, args.test_model_pth)   
    

    args_dict_str = '\n'.join([f"{k}: {v}" for k, v in vars(args).items()])
    logger.info(f"Arguments: \n{args_dict_str}")

    global device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 设置随机种子
    set_seed(args.seed)
    # 加载数据集
    # xml_files = read_xml_dataset(args.input_dir)
    # data_builder = ExpandGraphDataset(xml_files=xml_files, embedding_dir=args.embedding_dir, embedding_model='BgeEmbedding', device=device, debug=args.debug)
    # # 切分数据集
    # train_dataset, valid_dataset, test_dataset = split_dataset(data_builder)
    train_dataset = torch.load(os.path.join(args.input_dirs[0], 'train_dataset.pt'))
    valid_dataset = torch.load(os.path.join(args.input_dirs[0], 'valid_dataset.pt'))
    test_dataset = torch.load(os.path.join(args.input_dirs[0], 'test_dataset.pt'))

    for i in range(1, len(args.input_dirs)):
        train_dataset = train_dataset + torch.load(os.path.join(args.input_dirs[i], 'train_dataset.pt'))
        valid_dataset = valid_dataset + torch.load(os.path.join(args.input_dirs[i], 'valid_dataset.pt'))
        test_dataset = test_dataset + torch.load(os.path.join(args.input_dirs[i], 'test_dataset.pt'))

    if args.debug:
        train_dataset = [train_dataset[i] for i in list(range(64))]  
        valid_dataset = [valid_dataset[i] for i in list(range(16))] 
        test_dataset = [test_dataset[i] for i in list(range(16))] 

    # 使用 DataLoader 加载子集
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=dgl.batch)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True, collate_fn=dgl.batch)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=dgl.batch)
    logger.info(f"Load dataset finished, Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    
    if args.do_train:
        train(
            train_loader=train_loader, 
            valid_loader=valid_loader, 
            verbose=False, 
            lr=args.lr,
            num_epochs=args.num_epochs,
            output_dir=args.output_dir,
            debug=args.debug
        )

    if args.do_test:
        logger.info(f"test model path: {args.test_model_pth}")
        old_state_dict = torch.load(args.test_model_pth)
        # 加载重命名后的 state_dict 到新模型
        parser = LexParser(args.pretrained_emb_path)
        pre_emb = torch.stack([torch.from_numpy(emb) for emb in parser.pre_embedding]).to(device)
        model = GCNRec(len(parser.vocab), pre_emb).to(device)
        # model.load_state_dict(old_state_dict, strict=True)
        model.load_state_dict(torch.load(args.test_model_pth))
        test(
            model=model, 
            test_loader=test_loader, 
        )

    # python train.py --input_dir "" --do_train --do_test --output_dir "" --num_epochs 50 --lr 1e-4 --threshold 0.5 --seed 42