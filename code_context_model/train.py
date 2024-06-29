import os
import random
import argparse
import torch
import logging
import dgl
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall

from code_context_model.expand_dataset import ExpandGraphDataset, split_dataset
from code_context_model.gnn import RGCN

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def train(model, train_loader, valid_loader, verbose=True, **kwargs):
    logger.info("======= Start training =======")
    lr = kwargs.get('lr', 0.01)
    num_epochs = kwargs.get('num_epochs', 50)
    threshold = kwargs.get('threshold', 0.5)
    output_dir = kwargs.get('output_dir', 'output')
    # 定义损失函数和优化器
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001, betas=(0.9, 0.99))
    
    prec_metrics = BinaryPrecision(threshold=threshold)
    recall_metrics = BinaryRecall(threshold=threshold)
    f1_metrics = BinaryF1Score(threshold=threshold)

    def compute_metrics(logits, labels):
        prec = prec_metrics(logits, labels)
        recall = recall_metrics(logits, labels)
        f1 = f1_metrics(logits, labels)
        return {"precision": prec.item(), "recall": recall.item(), "f1": f1.item()}

    
    for epoch in range(num_epochs):
        total_loss, eval_loss = 0.0, 0.0
        train_avg_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        eval_avg_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        model.train()    
        for i, batch_graphs in enumerate(train_loader):
            logits = model(batch_graphs, batch_graphs.ndata['feat'], batch_graphs.edata['label'])
            loss = loss_fn(logits, batch_graphs.ndata['label'])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            
            metrics = compute_metrics(logits, batch_graphs.ndata['label']).item()
            train_avg_metrics = {k: train_avg_metrics[k] + metrics[k] for k in metrics}
            if verbose:
                logger.info(f"Train Epoch {epoch}-Batch {i}: Loss {loss.item()}, Metrics {metrics}")
        
        # evaluate
        model.eval()
        with torch.no_grad():
            for i, batch_graphs in enumerate(valid_loader):
                logits = model(batch_graphs, batch_graphs.ndata['feat'], batch_graphs.edata['label'])
                loss = loss_fn(logits, batch_graphs.ndata['label'])
                eval_loss += loss.item()
                metrics = compute_metrics(logits, batch_graphs.ndata['label']).item()
                eval_avg_metrics = {k: eval_avg_metrics[k] + metrics[k] for k in metrics}
                if verbose:
                    logger.info(f"Valid Epoch {epoch}-Batch {i}: Loss {loss.item()}, Metrics {metrics}")
        
        train_avg_metrics = {k: v / len(train_loader) for k, v in train_avg_metrics.items()}
        eval_avg_metrics = {k: v / len(valid_loader) for k, v in eval_avg_metrics.items()}
        logger.info(f"Epoch {epoch}, Train Loss {total_loss}, Train Metrics {train_avg_metrics}")
        logger.info(f"Epoch {epoch}, Eval  Loss {total_loss}, Eval Metrics {eval_avg_metrics}")
        # save the model
        torch.save(model.state_dict(), f"{output_dir}/model_{epoch}.pth")
        logger.info(f"Model saved at {output_dir}/model_{epoch}.pth")

    logger.info("======= Training finished =======")

def test(model, test_loader, **kwargs):
    logger.info("======= Start testing =======")
    threshold = kwargs.get('threshold', 0.5)
    prec_metrics = BinaryPrecision(threshold=threshold)
    recall_metrics = BinaryRecall(threshold=threshold)
    f1_metrics = BinaryF1Score(threshold=threshold)

    def compute_metrics(logits, labels):
        prec = prec_metrics(logits, labels)
        recall = recall_metrics(logits, labels)
        f1 = f1_metrics(logits, labels)
        return {"precision": prec.item(), "recall": recall.item(), "f1": f1.item()}
    
    model.eval()
    with torch.no_grad():
        test_avg_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for i, batch_graphs in enumerate(test_loader):
            logits = model(batch_graphs, batch_graphs.ndata['feat'], batch_graphs.edata['label'])
            metrics = compute_metrics(logits, batch_graphs.ndata['label'])
            logger.info(f"Test Batch {i}: Metrics {metrics}")
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
        for file_name in os.listdir(dataset_dir):
            if file_name.endswith(".xml"):
                result_xmls.append(os.path.join(dataset_dir, file_name))
    
    logger.info(f"Read total xml files: {len(result_xmls)}")
    return result_xmls
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train args
    parser.add_argument('--do_train', action='store_true', help='train the model')
    parser.add_argument('--input_dir', type=str, default='data', help='input directory')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for binary classification')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # test args
    parser.add_argument('--do_test', action='store_true', help='test the model')
    parser.add_argument('--test_model_pth', type=str, default='model.pth', help='test model path')
    args = parser.parse_args()

    args_dict_str = '\n'.join([f"{k}: {v}" for k, v in vars(args).items()])
    logger.info(f"Arguments: \n{args_dict_str}")

    # 设置随机种子
    set_seed(args.seed)
    # 加载数据集

    xml_files = read_xml_dataset(args.input_dir)
    data_builder = ExpandGraphDataset(xml_files=xml_files)
    # 切分数据集
    train_dataset, valid_dataset, test_dataset = split_dataset(data_builder)
    # 使用 DataLoader 加载子集
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=dgl.batch)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, collate_fn=dgl.batch)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=dgl.batch)

    # 定义模型
    model = RGCN(in_feats=1024, h_feat=1024, out_feat=1, num_rels=8)

    if args.do_train:
        train(
            model=model, 
            train_loader=train_loader, 
            valid_loader=valid_loader, 
            verbose=True, 
            lr=args.lr,
            num_epochs=args.num_epochs,
            threshold=args.threshold,
            output_dir=args.output_dir
        )

    if args.do_test:
        model.load_state_dict(torch.load(args.test_model_pth))
        test(
            model=model, 
            test_loader=test_loader, 
            threshold=args.threshold
        )
    

    # python train.py --input_dir "" --do_train --do_test --output_dir "" --num_epochs 50 --lr 1e-4 --threshold 0.5 --seed 42