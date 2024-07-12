import os
import argparse
import random
import json 
import time
import logging
import heapq

import networkx as nx
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

class FuzzyElement:
    def __init__(self, name, degree):
        self.name = name
        self.degree = degree
        
    def merge(self, name, degree):
        assert self.name == name
        new_degree = self.degree + degree - self.degree * degree
        self.degree = new_degree
    
    def traditional_merge(self, name, degree):
        assert self.name == name
        if degree < self.degree:
            return
        else:
            self.degree = degree
    
    def __str__(self):
        return '{} [{:1.2f}]'.format(self.name, self.degree)
        
class Suade:    
    def filterPredecessor(self, node, relation):
        DG = self.DG
        predecessors = set()
        for edge in DG.in_edges(node):
            SRC = edge[0]
            TGT = edge[1]
            assert TGT == node
            edge_label = DG[SRC][TGT]['label']
            if edge_label == relation:
                predecessors.add(SRC)

        return predecessors

    def filterSuccessor(self, node, relation):
        DG = self.DG
        successors = set()
        for edge in DG.out_edges(node):
            SRC = edge[0]
            TGT = edge[1]
            assert SRC == node
            edge_label = DG[SRC][TGT]['label']
            if edge_label == relation:
                successors.add(TGT)

        return successors

    def getForwardSet(self, node, relation, transpose):
        if not transpose:
            return self.filterPredecessor(node, relation)
        else:
            return self.filterSuccessor(node, relation)

    def getBackwordSet(self, node, relation, transpose):
        if not transpose:
            return self.filterSuccessor(node, relation)
        else:
            return self.filterPredecessor(node, relation)

    def analyzeRelation(self, I, relation, transpose, alpha = 0.25):
        Z = dict()
        for x in I:
    #         print(x)
            # FILTERED by relation
            S_forward = self.getForwardSet(x, relation, transpose)

            for s in S_forward:
                if s not in I:            
    #                 print(s)
                    S_backward = self.getBackwordSet(s, relation, transpose)

                    INTER_1 = S_forward.intersection(I)
                    INTER_2 = S_backward.intersection(I)
                    degree = (((float((1 + len(INTER_1)) * len(INTER_2))) / (len(S_forward) * len(S_backward)))) ** alpha
    #                 print(degree)

                    fuzzy_obj = None
                    if s in Z:
                        fuzzy_obj = Z[s]
                        fuzzy_obj.traditional_merge(s, degree)
                    else:
                        fuzzy_obj = FuzzyElement(s, degree)
                        Z[s] = fuzzy_obj

        return Z

    def union(self, S, T):
        for node in T:
            NEW_fuzzy_obj = T[node]
            fuzzy_obj = None
            if node in S:
                fuzzy_obj = S[node]
                fuzzy_obj.merge(NEW_fuzzy_obj.name, NEW_fuzzy_obj.degree)
            else:
                S[node] = NEW_fuzzy_obj

    def main(self, alpha = 0.25):
        S = dict()

        for relation in self.relations:
            T = self.analyzeRelation(self.I, relation, False)
            self.union(S, T)
            T = self.analyzeRelation(self.I, relation, True)
            self.union(S, T)

        return S
    
    # DG is G1 SG
    def __init__(self, DG, I, relations):
        self.DG = DG
        self.I = I
        self.relations = relations

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
        for node in vertex_list:
            g.add_node(int(node.get('id')), label=node.get('stereotype'), origin=int(node.get('origin', 0)), seed=int(node.get('seed', 0)),
                       G1=node.get('G1'))
        for link in edge_list:
            g.add_edge(int(link.get('start')), int(link.get('end')), label=link.get('label'))
            
        gs.append(g)
    return gs


def load_tests(input_dir, step):
    G1s = []
    # 读取code context model
    model_dirs = json.loads(open(f'{input_dir}/test_index.json').read())
    for model_dir in model_dirs:
        # print('---------------', model_dir)
        seed_expand_file = os.path.join(model_dir, f'{step}_step_seeds_expanded_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(seed_expand_file):
            continue
        # 读取code context model,以及doxygen的结果，分1-step,2-step,3-step扩展图
        tree = ET.parse(seed_expand_file)  # 拿到xml树
        code_context_model = tree.getroot()
        graphs = code_context_model.findall("graph") # 一个code context model对应一张图
        gs = get_graph(graphs, step)
        if len(gs) > 0:
            G1s = G1s + gs
    return G1s

def load_interest_nodes(graphs) -> list[list[int]]:
    """
    return a list of interest nodes for each graph
    """
    gI = []
    for graph in graphs:
        I = []
        # get graph nodes
        nodes_with_label = graph.nodes(data=True)
        for node, data in nodes_with_label:
            if int(data['seed']) == 1:
                I.append(node)
        gI.append(I)
    return gI

def compute_metrics(G1, confidence):
    confidence = dict(sorted(confidence.items(), key=lambda item: item[1].degree, reverse=True)) # {1: (1, 0.7), 2: (2, 0.2), 3: (3, 0.5)}
    # filter out seed nodes
    confidence = [v for k, v in confidence.items() if G1.nodes[k]['seed'] == 0]
    # for i in range(len(confidence)):
    #     print(confidence[i])
    # print("="*16)

    
    total_hit = {f'top{i}_hit': 0 for i in range(1,6)}
    for topk in range(1,6):
        temp = topk
        topk = topk if len(confidence) >= topk else len(confidence)
        hit = 0
        # if tie in the topk, we randomly pick the first topk node
        topk_values = heapq.nlargest(topk, set([item.degree for item in confidence]))
        # print(f"top {topk} values: {topk_values}")
        topk_nodes = []
        for value in topk_values:
            nodes_with_value = [item for item in confidence if item.degree == value]
            num_select = min(topk-len(topk_nodes), len(nodes_with_value))
            topk_nodes = topk_nodes + random.sample(nodes_with_value, num_select)
            if len(topk_nodes) == topk:
                break
        
        # for i in range(len(topk_nodes)):
        #     print(topk_nodes[i])
        # print("="*16)

        for node in topk_nodes:
            if  G1.nodes.get(node.name)['origin'] == 1: # get positive nodes
                hit = 1
                break
        total_hit[f"top{temp}_hit"] += hit

    return total_hit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data', help='input directory')
    args = parser.parse_args()

    test_hit_rate = {f"top{i}_hit": 0 for i in range(1, 6)}

    G1s = load_tests(args.input_dir, 1)
    gI = load_interest_nodes(G1s) 
    relations = ["declares", "calls", "inherits", "implements"]
    start_time = time.time()

    for i in range(len(G1s)):
        suade = Suade(G1s[i], gI[i], relations)
        degrees = suade.main()

        metrics = compute_metrics(G1s[i], degrees)
        test_hit_rate = {k: test_hit_rate[k] + metrics[k] for k in metrics}
    
    end_time = time.time()
    test_hit_rate = {k: v / len(G1s) for k, v in test_hit_rate.items()}
    logger.info(f"Saude finished!\nTime: {end_time - start_time}\nTest Metrics {test_hit_rate} ")
