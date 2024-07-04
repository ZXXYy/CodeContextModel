import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import random
import time
import argparse
import torch
import logging
import dgl
import wandb
import numpy as np

from tqdm import tqdm
from torch import nn
from itertools import product
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall

from code_context_model.build_dataset import ExpandGraphDataset, split_dataset
from code_context_model.gnn import RGCN

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

wandb.init(project="code-context-model")

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

def euclidean_distance(x1, x2):
    # 确保输入是1D张量，如果输入是2D或更高维度的张量，可以根据实际需求调整
    if x1.dim() > 1:
        x1 = x1.view(-1)
    if x2.dim() > 1:
        x2 = x2.view(-1)
    # 计算欧式距离
    return torch.dist(x1, x2).item()


def train(model, train_loader, valid_loader, verbose=True, **kwargs):
    logger.info("======= Start training =======")
    lr = kwargs.get('lr', 0.01)
    num_epochs = kwargs.get('num_epochs', 50)
    threshold = kwargs.get('threshold', 0.5)
    output_dir = kwargs.get('output_dir', 'output')
    # 定义损失函数和优化器
    # loss_fn = nn.BCELoss()
    # loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001, betas=(0.9, 0.99))
    
    prec_metrics = BinaryPrecision(threshold=threshold).to(device)
    recall_metrics = BinaryRecall(threshold=threshold).to(device)
    f1_metrics = BinaryF1Score(threshold=threshold).to(device)

    def compute_metrics(logits, labels):
        seed_indices = (labels == -1).nonzero()
        seed_embeddings = logits.squeeze(1)[seed_indices]
        non_seed_indices = (labels != -1).nonzero()
        non_seed_embeddings = logits.squeeze(1)[non_seed_indices]

        similarities = []
        logger.debug(f"num seed embeddings: {len(seed_embeddings)}, num non seed embeddings: {len(non_seed_embeddings)}")
        for i in range(0, len(seed_embeddings)):
            temp = torch.tensor([]).to(device)
            for embedding in non_seed_embeddings:
                temp = torch.cat((temp, cosine_similarity(seed_embeddings[i], embedding)))
            logger.debug(f"similarities for seed_embeddings {i}: {temp}")
            similarities = similarities + [temp]
        logger.debug(f"Similarities: {similarities}")
        similarities = torch.sum(torch.stack(similarities), dim=0)
        logger.debug(f"Similarities: {similarities}")
        logger.debug(f"similarities type: {similarities.shape}")
        # find top3 similar embeddings
        topk = 3 if len(non_seed_indices) >= 3 else len(non_seed_indices)
        topk_indices = torch.topk(similarities, topk).indices
        logger.debug(f"Top3 indices: {topk_indices}")
        hit = 0
        for i in range(0, len(topk_indices)):
            idx = non_seed_indices[topk_indices[i]] # get the index of the top3 embeddings
            if labels[idx] == 1:
                hit = 1
        return {'hit': hit}
        # prec = prec_metrics(logits, labels)
        # recall = recall_metrics(logits, labels)
        # f1 = f1_metrics(logits, labels)
        # return {"precision": prec.item(), "recall": recall.item(), "f1": f1.item()}

    def compute_loss(logits, labels):
        loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        seed_indices = (labels == -1).nonzero()
        positive_indices = (labels == 1).nonzero()
        negative_indices = (labels == 0).nonzero()
        embeddings = logits.squeeze(1)
        logger.debug(f"Seed Indices: {len(seed_indices)}, Positive Indices: {len(positive_indices)}, Negative Indices: {len(negative_indices)}")
        
        # 生成所有可能的 (seed, positive) 组合对
        seed_positive_pairs = list(product(seed_indices, positive_indices))
        logger.debug(f"Seed Positive Pairs: {seed_positive_pairs}")
        # 提取组合对的嵌入
        seed_pair_embeddings = torch.stack([embeddings[pair[0]].squeeze(0) for pair in seed_positive_pairs])
        positive_pair_embeddings = torch.stack([embeddings[pair[1]].squeeze(0) for pair in seed_positive_pairs])
        # 定义 margin
        margin = 1.0
        # 计算正样本对之间的欧氏距离
        positive_distances = torch.nn.functional.pairwise_distance(seed_pair_embeddings, positive_pair_embeddings)
        # 构建正样本对标签（全1）
        positive_labels = torch.ones(positive_distances.size(), device=positive_distances.device)
        # 计算正样本对的 Contrastive Loss
        positive_loss = torch.mean(positive_labels * positive_distances.pow(2)) 

        if len(negative_indices) == 0:
            return positive_loss
        # 生成所有可能的 (seed, negative) 组合对
        seed_negative_pairs = list(product(seed_indices, negative_indices))
        # 提取组合对的嵌入
        seed_pair_embeddings = torch.stack([embeddings[pair[0]].squeeze(0) for pair in seed_negative_pairs])
        negative_pair_embeddings = torch.stack([embeddings[pair[1]].squeeze(0) for pair in seed_negative_pairs])
        # 计算负样本对之间的欧氏距离
        negative_distances = torch.nn.functional.pairwise_distance(seed_pair_embeddings, negative_pair_embeddings)
        # 构建负样本对标签（全0）
        negative_labels = torch.zeros(negative_distances.size(), device=negative_distances.device)
        # 计算负样本对的 Contrastive Loss
        negative_loss = torch.mean((1 - negative_labels) * torch.clamp(margin - negative_distances, min=0.0).pow(2))

        # 总的 Contrastive Loss
        loss = positive_loss + negative_loss
        
        return loss # , non_seed_logits, non_seed_labels

    for epoch in tqdm(range(num_epochs)):
        total_loss, eval_loss = 0.0, 0.0
        # train_avg_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        # eval_avg_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        train_hit_rate = {"hit": 0.0}
        eval_hit_rate = {"hit": 0.0}
        model.train()    
        for i, batch_graphs in enumerate(train_loader):
            # 打印形状以调试
            # logger.info(f"Node features shape: {batch_graphs.ndata['feat'].shape}")
            # logger.info(f"Edge labels shape: {batch_graphs.edata['label'].shape}")
            logits = model(batch_graphs, batch_graphs.ndata['feat'].squeeze(1), batch_graphs.edata['label'].squeeze(1))
            # if epoch > 10:
            #     logger.info(f"Logits: {logits}")
            #     logger.info(f"Labels: {batch_graphs.ndata['label']}")
            # loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
            loss = compute_loss(logits, batch_graphs.ndata['label'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            metrics = compute_metrics(logits, batch_graphs.ndata['label']) # FIXME: wrong train metrics if batchsize > 1
            train_hit_rate = {k: train_hit_rate[k] + metrics[k] for k in metrics}
            if verbose:
                logger.info(f"Train Epoch {epoch}-Batch {i}: Loss {loss.item()}, Metrics {metrics}")
        
        # evaluate
        model.eval()
        with torch.no_grad():
            for i, batch_graphs in enumerate(valid_loader):
                logits = model(batch_graphs, batch_graphs.ndata['feat'].squeeze(1), batch_graphs.edata['label'].squeeze(1))
                loss = compute_loss(logits, batch_graphs.ndata['label'])
                eval_loss += loss.item()
                metrics = compute_metrics(logits, batch_graphs.ndata['label'])
                eval_hit_rate = {k: eval_hit_rate[k] + metrics[k] for k in metrics}
                if verbose:
                    logger.info(f"Valid Epoch {epoch}-Batch {i}: Loss {loss.item()}, Metrics {metrics}")
        
        train_hit_rate = {k: v / len(train_loader) for k, v in train_hit_rate.items()}
        eval_hit_rate = {k: v / len(valid_loader) for k, v in eval_hit_rate.items()}
        wandb.log({
            "Epoch": epoch,
            "Train Loss": total_loss,
            "Train Hit Rate": train_hit_rate['hit'],
            "Eval Loss": eval_loss,
            "Eval Hit Rate": eval_hit_rate['hit'],
        })
        logger.info(f"Epoch {epoch}, Train Loss {total_loss}, Train Metrics {train_hit_rate}")
        logger.info(f"Epoch {epoch}, Eval  Loss {eval_loss}, Eval Metrics {eval_hit_rate}")
        # save the model
        torch.save(model.state_dict(), f"{output_dir}/model_{epoch}.pth")
        logger.info(f"Model saved at {output_dir}/model_{epoch}.pth")

    logger.info("======= Training finished =======")

def test(model, test_loader, **kwargs):
    logger.info("======= Start testing =======")
    threshold = kwargs.get('threshold', 0.5)
    prec_metrics = BinaryPrecision(threshold=threshold).to(device)
    recall_metrics = BinaryRecall(threshold=threshold).to(device)
    f1_metrics = BinaryF1Score(threshold=threshold).to(device)

    def compute_metrics(logits, labels):
        prec = prec_metrics(logits, labels)
        recall = recall_metrics(logits, labels)
        f1 = f1_metrics(logits, labels)
        return {
            "precision": prec.cpu().item(),
            "recall": recall.cpu().item(),
            "f1": f1.cpu().item()
        }
    
    model.eval()
    with torch.no_grad():
        test_avg_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for i, batch_graphs in enumerate(test_loader):
            logits = model(batch_graphs, batch_graphs.ndata['feat'].squeeze(1), batch_graphs.edata['label'].squeeze(1))
            non_seed_indices = (batch_graphs.ndata['label'] != -1).nonzero()
            # 根据non_seed_indices选出对应的logits和labels
            non_seed_logits = logits.squeeze(1)[non_seed_indices]
            non_seed_labels = batch_graphs.ndata['label'].float()[non_seed_indices]
            metrics = compute_metrics(non_seed_logits, non_seed_labels)
            # logger.info(f"Test Batch {i}: Metrics {metrics}")
            test_avg_metrics = {k: test_avg_metrics[k] + metrics[k] for k in metrics}
        
        test_avg_metrics = {k: v / len(test_loader) for k, v in test_avg_metrics.items()}
        logger.info(f"Test finished, Test Metrics {test_avg_metrics}")
    


def read_xml_dataset(data_dir):
    result_xmls = []
    dir_names = os.listdir(data_dir)
    
    for dir_name in dir_names:
        dataset_dir = os.path.join(data_dir, dir_name, "seed_expanded")
        if not os.path.exists(dataset_dir):
            continue
        # for file_name in os.listdir(dataset_dir):
        #     if file_name.endswith(".xml"):
        #         result_xmls.append(os.path.join(dataset_dir, file_name))
        file_names = os.listdir(dataset_dir)
        step_1_files = [f for f in file_names if f.startswith("1_step")]
        if len(step_1_files) > 0:
            result_xmls.append(os.path.join(dataset_dir, step_1_files[random.randint(0, len(step_1_files) - 1)]))
    
    logger.info(f"Read total xml files: {len(result_xmls)}")
    return result_xmls
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train args
    parser.add_argument('--do_train', action='store_true', help='train the model')
    parser.add_argument('--device', type=int, default=1, help='device id')
    parser.add_argument('--input_dir', type=str, default='data', help='input directory')
    parser.add_argument('--embedding_dir', type=str, default='data', help='embedding directory')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--train_batch_size', type=int, default=1, help='train batch size')
    parser.add_argument('--valid_batch_size', type=int, default=1, help='train batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for binary classification')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # test args
    parser.add_argument('--do_test', action='store_true', help='test the model')
    parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')
    parser.add_argument('--test_model_pth', type=str, default='model_48.pth', help='test model path')

    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, f"{time.strftime('%m-%d-%H-%M')}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        args.test_model_pth = os.path.join(args.output_dir, args.test_model_pth)    

    args_dict_str = '\n'.join([f"{k}: {v}" for k, v in vars(args).items()])
    logger.info(f"Arguments: \n{args_dict_str}")

    global device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # 配置wandb
    config = wandb.config
    config.learning_rate = args.lr
    config.batch_size = args.train_batch_size
    config.epochs = args.num_epochs
    config.device = args.device

    # 设置随机种子
    set_seed(args.seed)
    # 加载数据集
    xml_files = read_xml_dataset(args.input_dir)
    data_builder = ExpandGraphDataset(xml_files=xml_files, embedding_dir=args.embedding_dir, embedding_model='BgeEmbedding', device=device, debug=args.debug)
    # 切分数据集
    train_dataset, valid_dataset, test_dataset = split_dataset(data_builder)
    # 使用 DataLoader 加载子集
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=dgl.batch)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True, collate_fn=dgl.batch)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, collate_fn=dgl.batch)

    # # 定义模型
    model = RGCN(in_feat=1024, h_feat=1024, out_feat=1, num_rels=8)
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    model.to(device)

    if args.do_train:
        train(
            model=model, 
            train_loader=train_loader, 
            valid_loader=valid_loader, 
            verbose=False, 
            lr=args.lr,
            num_epochs=args.num_epochs,
            threshold=args.threshold,
            output_dir=args.output_dir
        )

    if args.do_test:
        logger.info(f"test model path: {args.test_model_pth}")
        model.load_state_dict(torch.load(args.test_model_pth))
        test(
            model=model, 
            test_loader=test_loader, 
            threshold=args.threshold
        )
    

    # python train.py --input_dir "" --do_train --do_test --output_dir "" --num_epochs 50 --lr 1e-4 --threshold 0.5 --seed 42