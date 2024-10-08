import os
import json
import logging
import math
import argparse
import time
import multiprocessing

import gspan_mining.config
import gspan_mining.main

import networkx as nx
import xml.etree.ElementTree as ET
import numpy as np

from tqdm import tqdm
from contextlib import redirect_stdout
from networkx.algorithms import isomorphism


logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)


def build_gspan_graph(input_dir, output_path):
    # 读取code context model
    model_dirs = json.loads(open(f'{input_dir}/train_index.json').read())
    logger.info(f"train sets len: {len(model_dirs)}")
    all_nodes, all_edges = [], []
    for model_dir in model_dirs:
        model_dir = '/data0/xiaoyez/CodeContextModel/data/mylyn/' +model_dir.split('/')[-1]
        model_file = os.path.join(model_dir, 'code_context_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            logger.info(f"model file not found: {model_file}")
            continue
        # 读取code context model,以及doxygen的结果
        tree = ET.parse(model_file)  # 拿到xml树
        graphs = tree.getroot().findall("graph")
        curr_index = 0
        nodes, edges = [], []
        # form subgraphs in the code context model to a big graph in gspan format 
        for graph in graphs:
            vertex_list = graph.find('vertices').findall('vertex')
            vs = []
            # handle vertex
            for vertex in vertex_list:
                stereotype, _id = vertex.get('stereotype'), int(vertex.get('id'))
                # 去除 notfound
                if not stereotype == 'NOTFOUND':
                    vs.append((_id, stereotype))
            for v in sorted(vs, key=lambda x: x[0]):
                vertex_text = f'v {v[0] + curr_index} {v[1]}\n'
                nodes.append(vertex_text)
            
            # handle edges,
            edge_list = graph.find('edges').findall('edge')
            for edge in edge_list:
                start, end, label = int(edge.get('start')), int(edge.get('end')), edge.get('label')
                keys = [x[0] for x in vs]
                if start in keys and end in keys:
                    edge_text = f'e {start + curr_index} {end + curr_index} {label}\n'
                    edges.append(edge_text)

            curr_index += len(vertex_list)
            
        all_nodes.append(nodes)
        all_edges.append(edges)

    # write context model in gspan format to file 
    with open(output_path, 'w') as f:
        for i in range(len(all_nodes)):
            f.write(f't # {i}\n')
            for node in all_nodes[i]:
                f.write(node)
            for edge in all_edges[i]:
                f.write(edge)
        f.write('t # -1')

    return all_nodes, all_edges


def gspan_miner(input_file, output_file, node_num, min_sup = 0.015 ):
    min_support = math.ceil(min_sup * node_num)  # 0.02 * num_of_graphs
    # '-s' '--min_support' 
    # '-d' '--directed' run for directed graphs
    # '-l' lower bound of number of vertices of output subgraph, default 2
    # '-u' int, upper bound of number of vertices of output subgraph, 'default inf'
    # database_file_name = 'no_graph.data'
    args_str = f'-s {min_support} -d True {input_file}'
    args, _ = gspan_mining.config.parser.parse_known_args(args=args_str.split())
    with open(output_file, 'w') as f:
        with redirect_stdout(f):
            gspan_mining.main.main(args) # 这里的 gsan库有问题，需要根据报错，将包源码的 append 方法修改为 _append 即可
    logger.info(f'min_support: {min_support}')

def load_tests(input_dir, step):
    G1s = []
    # 读取code context model
    model_dirs = json.loads(open(f'{input_dir}/test_index.json').read())
    for model_dir in model_dirs:
        # print('---------------', model_dir)
        model_dir = '/data0/xiaoyez/CodeContextModel/data/mylyn/' +model_dir.split('/')[-1]
        seed_expand_file = os.path.join(model_dir, f'{step}_step_seeds_expanded_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(seed_expand_file):
            # logger.info(f"model file not found: {seed_expand_file}")
            continue
        # 读取code context model,以及doxygen的结果，分1-step,2-step,3-step扩展图
        tree = ET.parse(seed_expand_file)  # 拿到xml树
        code_context_model = tree.getroot()
        graphs = code_context_model.findall("graph") # 一个code context model对应一张图
        gs = get_graph(graphs, step)
        if len(gs) > 0:
            G1s = G1s + gs
    return G1s

def get_graph(graphs: list[ET.Element], step: int):
    gs = []
    if len(graphs) == 0:
        return gs
    for graph in graphs:
        vertices = graph.find('vertices')
        vertex_list = vertices.findall('vertex')
        edges = graph.find('edges')
        edge_list = edges.findall('edge')
        g = nx.DiGraph()
        true_node = 0
        # true_edge = 0
        # 转化为图结构
        remove_nodes = []
        for node in vertex_list:
            g.add_node(int(node.get('id')), label=node.get('stereotype'), origin=int(node.get('origin', 0)), seed=int(node.get('seed', 0)),
                       G1=node.get('G1'))
            if node.get('stereotype') == 'NOTFOUND':
                remove_nodes.append(int(node.get('id')))
            else:
                if int(node.get('origin')) == 1:
                    true_node += 1
        for link in edge_list:
            g.add_edge(int(link.get('start')), int(link.get('end')), label=link.get('label'))
            # if int(link.get('origin')) == 1:
            #     true_edge += 1
        if true_node > step:
            for node_id in remove_nodes:
                g.remove_node(node_id)  # 会自动删除边
            gs.append(g)
    return gs


def load_patterns(patterns):
    G2s = []
    with open(patterns) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('t #'):
                g = nx.DiGraph()
            if line.startswith('v'):
                v = line.split(' ')
                g.add_node(int(v[1]), label=v[2])
            if line.startswith('e'):
                e = line.split(' ')
                g.add_edge(int(e[1]), int(e[2]), label=e[3])
            if line.startswith('Support'):
                G2s.append(g)
    return G2s

def node_match(node1, node2):
    return node1['label'] == node2['label']

def edge_match(edge1, edge2):
    return edge1['label'] == edge2['label']

def graph_match_task(G1: nx.DiGraph, G1_index: int,  G2s: list[nx.DiGraph], step: int):
    logger.info(f'handling: {G1_index}-{G1}')
    begin_time = time.time()
    total_match = 0
    confidence = dict()
    flag = False
    for G2 in G2s:
        curr_time = time.time()
        if curr_time - begin_time > 60 * 10:
            flag = True
            break
        GM = isomorphism.DiGraphMatcher(G1, G2, node_match=node_match, edge_match=edge_match)
        # 检查 G1 是否包含 G2 的子图同构
        if GM.subgraph_is_isomorphic():
            # 遍历每个子图同构
            for sub_iter in GM.subgraph_isomorphisms_iter():
                curr_time = time.time()
                if curr_time - begin_time > 60 * 10:
                    flag = True
                    break

                # 获取同构子图的节点id列表
                nodes = list(map(int, list(sub_iter.keys())))
                num_seeds = 0
                for node in nodes:
                    num_seeds += int(G1.nodes.get(node)['seed'])
                # 匹配的子图中不属于 ground truth的节点数不能超过当前的预测步长 step 
                # FIXME: 这样合理吗
                if len(nodes) - num_seeds > step:
                    continue
                total_match += 1
                for node in nodes:
                    confidence[node] = confidence.setdefault(node, 0) + 1
    if flag: # 计算已经匹配出来的
        # print(f'continue {G1_index}')
        return None # 如果存在超时，跳过

    for i in confidence:
        confidence[i] = confidence.get(i) / total_match
    metrics = compute_metrics(confidence, G1, step)
    return metrics


def pattern_matching(test_dir, pattern_dir, step: int):
    G1s = load_tests(test_dir, step)
    G2s = load_patterns(pattern_dir)
    logger.info(f'Load Test Graphs Len: {len(G1s)}')
    logger.info(f'Load Pattern Graphs Len: {len(G2s)}')
    test_metrics = {f"top{i}_hit": 0 for i in range(1, 6)}
    test_metrics.update({'MRR': 0, 'MAP': 0})

    start_time = time.time()
    # parallelly handle G1s 
    with multiprocessing.Pool(processes=32) as pool:
        results = [pool.apply_async(graph_match_task, args=(G1, G1s.index(G1), G2s, step)) for G1 in G1s]
        cnt = 0
        for res in results:
            metrics = res.get()
            if metrics:
                cnt += 1
                test_metrics = {k: test_metrics[k] + metrics[k] for k in metrics}
        test_metrics = {k: v / cnt for k, v in test_metrics.items()}
    end_time = time.time()
    temp = pattern_dir.split('/')[-1]
    with open(f'./pattern_matching_result_{step}_{temp}.json', 'w') as f:
        f.write(json.dumps(test_metrics))
    logger.info(f"Pattern Matching Test finished!\nTime: {end_time - start_time}\nTest Metrics {test_metrics} ")

def compute_hit_rate(G1, confidence):
    total_hit = {f"top{i}_hit": 0 for i in range(1, 6)}

    for topk in range(1,6):
        temp = topk
        topk = topk if len(confidence) >= topk else len(confidence)
        hit = 0
        for i in range(0, topk):
            idx = confidence[i][0] # get the index of the top3 embeddings
            if  G1.nodes.get(idx)['origin'] == 1: # get positive nodes
                hit = 1
                break
        total_hit[f"top{temp}_hit"] += hit

    return total_hit

def compute_mrr(G1, confidence):
    mrr = 0
    for i, item in enumerate(confidence):
        if i > 100:
            return {'MRR': 0}
        if G1.nodes.get(item[0])['origin'] == 1:
            mrr = 1 / (i + 1)
            break
    return {'MRR': mrr}

def compute_map(G1, confidence, step, topk=5):
    MAP = 0
    positive_cnt = 0
    for i, item in enumerate(confidence):
        if i >= topk:
            break
        if G1.nodes.get(item[0])['origin'] == 1:
            positive_cnt += 1
            MAP += positive_cnt / (i + 1)
    MAP = MAP / step
    return {'MAP': MAP}


def compute_metrics(confidence, G1, step):
    confidence = sorted(confidence.items(), key=lambda d: d[1], reverse=True)  # [(3, 1.0), (17, 0.5), (14, 0.5)]
    # filter out seed nodes
    confidence = list(filter(lambda x: G1.nodes.get(x[0])['seed'] == 0, confidence))  
    
    metrics = compute_hit_rate(G1, confidence)
    mrr = compute_mrr(G1, confidence)
    MAP = compute_map(G1, confidence, step)
    metrics.update(mrr)
    metrics.update(MAP)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true', help='train the patterns')
    parser.add_argument('--do_test', action='store_true', help='test the patterns')
    parser.add_argument('--step', type=int, default='1', help='step of the seed expanded model')
    args = parser.parse_args()

    # mine patterns from code context model
    sup_list = [0.005]
    # i = 0.01
    # while i < 0.1:
    #     sup_list.append(i)
    #     i += 0.005

    if args.do_train:
        min_sup = 0.01
        max_sum = 0.1
        for sup in sup_list:
            all_nodes, _ = build_gspan_graph('/data0/xiaoyez/CodeContextModel/data/train_test_index/mylyn', './no_graph.data')
            gspan_miner('./no_graph.data', f'./no-sup-{sup}.data', len(all_nodes), sup)

    if args.do_test:
        for sup in sup_list:
            pattern_matching('/data0/xiaoyez/CodeContextModel/data/train_test_index/mylyn', f'/data0/xiaoyez/CodeContextModel/no-sup-{sup}.data', args.step)