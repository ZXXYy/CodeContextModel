import os
os.environ["DGLBACKEND"] = "pytorch"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import random
import time
import argparse
import torch
import logging
import dgl
import wandb
import atexit
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

from tqdm import tqdm
from torch import nn
from itertools import product
from dgl.dataloading import GraphDataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall

from code_context_model.build_dataset import ExpandGraphDataset, split_dataset
from code_context_model.gnn import RGCN

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

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


def compute_metrics(batch_logits, batch_labels, batch_num_nodes, device):
    batch_size = len(batch_num_nodes)
    start_idx = 0
    total_hit = {}
    for i in range(1, 6):
        total_hit[f'top{i}_hit'] = 0
        # total_hit[f'top{i}_precision'] = 0
        # total_hit[f'top{i}_recall'] = 0

    logger.debug(f"Batch num:{batch_num_nodes}")
    for k in range(batch_size):
        labels = batch_labels[start_idx : start_idx + batch_num_nodes[k]]
        logits = batch_logits[start_idx : start_idx + batch_num_nodes[k]]
        seed_indices = (labels == -1).nonzero()
        positive_indices = (labels == 1).nonzero()
        seed_embeddings = logits[seed_indices]
        non_seed_indices = (labels != -1).nonzero()
        non_seed_embeddings = logits[non_seed_indices]

        similarities = []
        logger.debug(f"num seed indices: {len(seed_indices)}, num non seed indices: {len(non_seed_indices)}")
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
        for topk in range(1,6):
            temp = topk
            topk = topk if len(non_seed_indices) >= topk else len(non_seed_indices)
            topk_indices = torch.topk(similarities, topk).indices
            logger.debug(f"Top{topk} indices: {topk_indices}")
            hit = 0
            for i in range(0, len(topk_indices)):
                idx = non_seed_indices[topk_indices[i]] # get the index of the top3 embeddings
                if labels[idx] == 1:
                    hit += 1
                    break
            # total_hit[f"top{temp}_precision"] += (hit+len(seed_indices)) / (topk+len(seed_indices))
            # total_hit[f"top{temp}_recall"] += (hit+len(seed_indices)) / (len(seed_indices)+len(positive_indices))
            total_hit[f"top{temp}_hit"] += 1 if hit > 0 else 0
        start_idx += batch_num_nodes[k]

    return total_hit
    # prec = prec_metrics(logits, labels)
    # recall = recall_metrics(logits, labels)
    # f1 = f1_metrics(logits, labels)
    # return {"precision": prec.item(), "recall": recall.item(), "f1": f1.item()}

def compute_loss(batch_logits, batch_labels, batch_num_nodes):
    batch_size = len(batch_num_nodes)
    start_idx = 0
    total_loss = 0
    logger.debug(f"Batch num: {batch_num_nodes}")

    for i in range(batch_size):
        # loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        labels = batch_labels[start_idx : start_idx + batch_num_nodes[i]]
        logits = batch_logits[start_idx : start_idx + batch_num_nodes[i]]
        seed_indices = (labels == -1).nonzero()
        positive_indices = (labels == 1).nonzero()
        negative_indices = (labels == 0).nonzero()
        embeddings = logits
        logger.debug(f"Seed Indices: {len(seed_indices)}, Positive Indices: {len(positive_indices)}, Negative Indices: {len(negative_indices)}")
        
        # 生成所有可能的 (seed_index, positive_index) 组合对
        seed_positive_pairs = list(product(seed_indices, positive_indices))
        # 提取组合对的嵌入
        seed_pair_embeddings = torch.stack([embeddings[pair[0]].squeeze(0) for pair in seed_positive_pairs])
        positive_pair_embeddings = torch.stack([embeddings[pair[1]].squeeze(0) for pair in seed_positive_pairs])
        # 定义 margin
        neg_margin = 0.0
        pos_margin = 1.0
        # 计算正样本对之间的欧氏距离
        # positive_distances = torch.nn.functional.pairwise_distance(seed_pair_embeddings, positive_pair_embeddings)
        positive_distances = pairwise_cosine_similarity(seed_pair_embeddings, positive_pair_embeddings)
        # 构建正样本对标签（全1）
        positive_labels = torch.ones(positive_distances.size(), device=positive_distances.device)
        # 计算正样本对的 Contrastive Loss
        positive_loss = torch.mean(positive_labels *  torch.clamp(pos_margin-positive_distances, min=0.0).pow(2)) 

        if len(negative_indices) == 0:
            return positive_loss
        # 生成所有可能的 (seed, negative) 组合对
        seed_negative_pairs = list(product(seed_indices, negative_indices))
        # 提取组合对的嵌入
        seed_pair_embeddings = torch.stack([embeddings[pair[0]].squeeze(0) for pair in seed_negative_pairs])
        negative_pair_embeddings = torch.stack([embeddings[pair[1]].squeeze(0) for pair in seed_negative_pairs])
        # 计算负样本对之间的欧氏距离
        # negative_distances = torch.nn.functional.pairwise_distance(seed_pair_embeddings, negative_pair_embeddings)
        negative_distances = pairwise_cosine_similarity(seed_pair_embeddings, negative_pair_embeddings)
        # 构建负样本对标签（全0）
        negative_labels = torch.zeros(negative_distances.size(), device=negative_distances.device)
        # 计算负样本对的 Contrastive Loss
        negative_loss = torch.mean((1 - negative_labels) * torch.clamp(negative_distances-neg_margin, min=0.0).pow(2))

        # 总的 Contrastive Loss
        total_loss += positive_loss + negative_loss
        start_idx += batch_num_nodes[i]

    return total_loss # , non_seed_logits, non_seed_labels

def train(model: RGCN, train_loader, valid_loader, verbose=True, **kwargs):
    lr = kwargs.get('lr', 0.01)
    num_epochs = kwargs.get('num_epochs', 50)
    threshold = kwargs.get('threshold', 0.5)
    output_dir = kwargs.get('output_dir', 'output')
    debug = kwargs.get('debug', False)
    device = kwargs.get('device', torch.device("cpu"))
    rank = kwargs.get('rank', 0)
    if rank ==0 :
        logger.info("======= Start training =======")

    # 定义损失函数和优化器
    # loss_fn = nn.BCELoss()
    # loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001, betas=(0.9, 0.99))
    
    prec_metrics = BinaryPrecision(threshold=threshold).to(device)
    recall_metrics = BinaryRecall(threshold=threshold).to(device)
    f1_metrics = BinaryF1Score(threshold=threshold).to(device)

    for epoch in tqdm(range(num_epochs)):
        train_loader.set_epoch(epoch)
        total_loss, eval_loss = 0.0, 0.0
        train_graph_num_cnt = 0
        # train_avg_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        # eval_avg_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        train_hit_rate = {}
        eval_hit_rate = {}
        for i in range(1, 6):
            train_hit_rate[f'top{i}_hit'] = 0
            # train_hit_rate[f'top{i}_precision'] = 0
            # train_hit_rate[f'top{i}_recall'] = 0
            eval_hit_rate[f'top{i}_hit'] = 0
            # eval_hit_rate[f'top{i}_precision'] = 0
            # eval_hit_rate[f'top{i}_recall'] = 0
        
        model.train()    
        for i, batch_graphs in enumerate(train_loader):

            train_graph_num_cnt += len(batch_graphs.batch_num_nodes())
            # 打印形状以调试
            logger.debug(f"Node features shape: {batch_graphs.edata['label'].shape}")
            # logger.info(f"Node features shape: {batch_graphs.edata['label'].squeeze(1).shape}")

            # logger.info(f"Edge labels shape: {batch_graphs.edata['label'].shape}")
            batch_graphs = batch_graphs.to(device)
            batch_graphs.ndata['feat'] = batch_graphs.ndata['feat'].to(device)
            batch_graphs.edata['label'] = batch_graphs.edata['label'].to(device)
            batch_graphs.ndata['label'] = batch_graphs.ndata['label'].to(device)
            logger.debug("Model Forwarding...")
            logits = model(batch_graphs, batch_graphs.ndata['feat'], batch_graphs.edata['label'].squeeze(1))
            # if epoch > 10:
            #     logger.info(f"Logits: {logits}")
            #     logger.info(f"Labels: {batch_graphs.ndata['label']}")
            # loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
            logger.debug("Loss computing...")
            loss = compute_loss(logits, batch_graphs.ndata['label'], batch_graphs.batch_num_nodes().tolist())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            logger.debug("Metrics computing...")
            metrics = compute_metrics(logits, batch_graphs.ndata['label'], batch_graphs.batch_num_nodes().tolist(), device) # FIXME: wrong train metrics if batchsize > 1
            train_hit_rate = {k: train_hit_rate[k] + metrics[k] for k in metrics}
            if verbose and rank==0:
                logger.info(f"Train Epoch {epoch}-Batch {i+1}/{len(train_loader)}: Loss {loss.item()/len(batch_graphs.batch_num_nodes())}, Metrics {metrics}")
        
        # evaluate
        model.eval()
        with torch.no_grad():
            eval_graph_num_cnt = 0
            for i, batch_graphs in enumerate(valid_loader):
                batch_graphs = batch_graphs.to(device)
                batch_graphs.ndata['feat'] = batch_graphs.ndata['feat'].to(device)
                batch_graphs.edata['label'] = batch_graphs.edata['label'].to(device)
                batch_graphs.ndata['label'] = batch_graphs.ndata['label'].to(device)
                eval_graph_num_cnt += len(batch_graphs.batch_num_nodes())
                logits = model(batch_graphs, batch_graphs.ndata['feat'], batch_graphs.edata['label'].squeeze(1))
                loss = compute_loss(logits, batch_graphs.ndata['label'], batch_graphs.batch_num_nodes().tolist())
                eval_loss += loss.item()
                metrics = compute_metrics(logits, batch_graphs.ndata['label'], batch_graphs.batch_num_nodes().tolist(), device)
                eval_hit_rate = {k: eval_hit_rate[k] + metrics[k] for k in metrics}
                # if verbose:
                #     logger.info(f"Valid Epoch {epoch}-Batch {i}/{len(valid_loader)}: Loss {loss.item()/len(batch_graphs.batch_num_nodes())}, Metrics {metrics}")
        
        train_hit_rate = {k: v / train_graph_num_cnt for k, v in train_hit_rate.items()}
        eval_hit_rate = {k: v / eval_graph_num_cnt for k, v in eval_hit_rate.items()}

        # Merge all loss and hit_rate from different device
        total_loss = torch.tensor(total_loss, device=device)
        eval_loss = torch.tensor(eval_loss, device=device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

        for i in range(1, 6):
            train_hit_rate[f'top{i}_hit'] = torch.tensor(train_hit_rate[f'top{i}_hit'], device=device)
            eval_hit_rate[f'top{i}_hit'] = torch.tensor(eval_hit_rate[f'top{i}_hit'], device=device)
        for i in range(1, 6):
            # use mean to calculate the hit rate
            dist.all_reduce(train_hit_rate[f'top{i}_hit'], op=dist.ReduceOp.SUM)
            dist.all_reduce(eval_hit_rate[f'top{i}_hit'], op=dist.ReduceOp.SUM)
        

        # only rank 0 process record the metrics and save the modelsx
        if rank != 0:
            continue
        # save the model
        torch.save(model.module.state_dict(), f"{output_dir}/model_{epoch}.pth")
        logger.info(f"Model saved at {output_dir}/model_{epoch}.pth")

        # echo total loss and averaged hit rate
        total_loss = total_loss.item()
        eval_loss = eval_loss.item()
        train_hit_rate = {k: v.item() / dist.get_world_size() for k, v in train_hit_rate.items()}
        eval_hit_rate = {k: v.item() / dist.get_world_size() for k, v in eval_hit_rate.items()}
        logger.info(f"Epoch {epoch}, Train Loss {total_loss}, Train Metrics {train_hit_rate}")
        logger.info(f"Epoch {epoch}, Eval  Loss {eval_loss}, Eval Metrics {eval_hit_rate}")

        # don't record metrics in debug mode
        if debug:
            continue
        wandb_log = {
            "Epoch": epoch,
            "Train Loss": total_loss,
            "Eval Loss": eval_loss,
        }
        wandb_log.update(train_hit_rate)
        wandb_log.update(eval_hit_rate)
        wandb.log(wandb_log)


    logger.info("======= Training finished =======")

def test(model, test_loader, **kwargs):
    device = kwargs.get("device", 0)
    logger.info("======= Start testing =======")
    threshold = kwargs.get('threshold', 0.5)

    model.eval()
    with torch.no_grad():
        test_hit_rate = {}
        for i in range(1, 6):
            test_hit_rate[f'top{i}_hit'] = 0
            test_hit_rate[f'top{i}_precision'] = 0
            test_hit_rate[f'top{i}_recall'] = 0
        for i, batch_graphs in enumerate(test_loader):
            batch_graphs = batch_graphs.to(device)
            batch_graphs.ndata['feat'] = batch_graphs.ndata['feat'].to(device)
            batch_graphs.edata['label'] = batch_graphs.edata['label'].to(device)
            batch_graphs.ndata['label'] = batch_graphs.ndata['label'].to(device)
            logits = model(batch_graphs, batch_graphs.ndata['feat'], batch_graphs.edata['label'].squeeze(1))
            # non_seed_indices = (batch_graphs.ndata['label'] != -1).nonzero()
            # # 根据non_seed_indices选出对应的logits和labels
            # non_seed_logits = logits.squeeze(1)[non_seed_indices]
            # non_seed_labels = batch_graphs.ndata['label'].float()[non_seed_indices]
            metrics = compute_metrics(logits, batch_graphs.ndata['label'], batch_graphs.batch_num_nodes().tolist(), device)
            # logger.info(f"Test Batch {i}: Metrics {metrics}")
            test_hit_rate = {k: test_hit_rate[k] + metrics[k] for k in metrics}
        
        test_hit_rate = {k: v / len(test_loader) for k, v in test_hit_rate.items()}
        logger.info(f"Test finished, Test Metrics {test_hit_rate}")
    

def init_process_group(world_size, rank, port):
    """
        world_size: the number of processes
        rank: the process ID, which should be an integer from 0 to world_size - 1
    """
    dist.init_process_group(
        backend="nccl",  # change to 'nccl' for multiple GPUs
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,  
        rank=rank,
    )

def get_dataloaders(dataset_path: str, **kwargs):
    debug = kwargs.get('debug', False)
    train_batch_size = kwargs.get('train_batch_size', 8)
    valid_batch_size = kwargs.get('valid_batch_size', 8)
    # test_batch_size = kwargs.get('test_batch_size', 1)

    train_dataset = torch.load(os.path.join(dataset_path, 'train_dataset.pt'))
    valid_dataset = torch.load(os.path.join(dataset_path, 'valid_dataset.pt'))
    if debug:
        train_dataset = train_dataset[:64]
        valid_dataset = valid_dataset[:64]
        # test_dataset = test_dataset[:16]

    # 使用 DataLoader 加载子集
    # train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=dgl.batch)
    # valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True, collate_fn=dgl.batch)
    # test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, collate_fn=dgl.batch)
    train_loader = GraphDataLoader(train_dataset, use_ddp=True, batch_size=train_batch_size, shuffle=True)
    valid_loader = GraphDataLoader(valid_dataset, use_ddp=True, batch_size=valid_batch_size)
    # test_loader = GraphDataLoader(test_dataset, batch_size=test_batch_size)
    logger.info(f"Load dataset finished, Train: {len(train_dataset)}, Valid: {len(valid_dataset)}") # Test: {len(test_dataset)}")

    return train_loader, valid_loader

def init_model(seed, device):
    torch.manual_seed(seed)
    model = RGCN(in_feat=1024, h_feat=1024, out_feat=1, num_rels=8)
    model = model.to(device)
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    if device.type == "cpu":
        model = DistributedDataParallel(model)
    else:
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )

    return model


def train_parallel(rank, world_size, args):
    # 初始化 W&B
    if not args.debug and rank==0:
        wandb.init(
            project="code-context-model",
            config=args,
            job_type="training",
        )

    init_process_group(world_size, rank, port=args.port)
    device_ids = args.devices.split(',')
    logger.info(f"Process {rank} started, device_ids: {device_ids}")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_ids[rank]}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    train_loader, valid_loader = get_dataloaders(
        args.input_dir, 
        debug=args.debug, 
        train_batch_size=args.train_batch_size, 
        valid_batch_size=args.valid_batch_size, 
        # test_batch_size=args.test_batch_size
    )

    # # 定义模型
    model = init_model(args.seed, device)
    train(
        model=model, 
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        verbose=True, 
        lr=args.lr,
        num_epochs=args.num_epochs,
        threshold=args.threshold,
        output_dir=args.output_dir,
        debug=args.debug,
        device=device,
        rank=rank
    )
    dist.destroy_process_group()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # train args
    parser.add_argument('--do_train', action='store_true', help='train the model')
    parser.add_argument('--devices', type=str, default=1, help='device id')
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
    global args
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, f"{time.strftime('%m-%d-%H-%M')}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.debug:
        args.num_epochs = 1 
        args.test_model_pth = 'model_0.pth'
    
    if args.do_train:
        args.test_model_pth = os.path.join(args.output_dir, args.test_model_pth)   
    

    args_dict_str = '\n'.join([f"{k}: {v}" for k, v in vars(args).items()])
    logger.info(f"Arguments: \n{args_dict_str}")

    # global device
    # device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 设置随机种子
    set_seed(args.seed)
    # set args.port to random value
    args.port = random.randint(10000, 20000)

    if args.do_train:
        device_ids = args.devices.split(',')
        mp.spawn(
            train_parallel, 
                args=(len(device_ids), args), 
                nprocs=len(device_ids)
            )

    if args.do_test:
        test_dataset = torch.load(os.path.join(args.input_dir, 'test_dataset.pt'))
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, collate_fn=dgl.batch)
        model = RGCN(in_feat=1024, h_feat=1024, out_feat=1, num_rels=8)
        model.to(f"cuda:{args.devices.split(',')[0]}")
        logger.info(f"test model path: {args.test_model_pth}")
        model.load_state_dict(torch.load(args.test_model_pth))
        device = torch.device(f"cuda:{args.devices.split(',')[0]}") if torch.cuda.is_available() else torch.device("cpu")
        test(
            model=model, 
            test_loader=test_loader, 
            threshold=args.threshold,
            device=device
        )
    

    # python train.py --input_dir "" --do_train --do_test --output_dir "" --num_epochs 50 --lr 1e-4 --threshold 0.5 --seed 42