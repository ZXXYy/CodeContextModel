import os
import json
import logging
import math
import gspan_mining.config
import gspan_mining.main

import xml.etree.ElementTree as ET

from contextlib import redirect_stdout


logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)


def build_gspan_graph(input_dir, output_path):
    # 读取code context model
    model_dirs = json.loads(open(f'{input_dir}/train_index.json').read())
    logger.info(f"train sets len: {len(model_dirs)}")
    all_nodes = []
    all_edges = []
    for model_dir in model_dirs:
        model_file = os.path.join(model_dir, 'code_context_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        # 读取code context model,以及doxygen的结果
        tree = ET.parse(model_file)  # 拿到xml树
        graphs = tree.getroot().findall("graph")
        # f.write(f't # {graph_index}\n')
        curr_index = 0
        nodes = []
        edges = []
        for graph in graphs:
            vertex_list = graph.find('vertices').findall('vertex')
            vs = []
            for vertex in vertex_list:
                stereotype, _id = vertex.get('stereotype'), int(vertex.get('id'))
                # 去除 notfound
                if not stereotype == 'NOTFOUND':
                    vs.append((_id, stereotype))
            for v in sorted(vs, key=lambda x: x[0]):
                vertex_text = f'v {v[0] + curr_index} {v[1]}\n'
                nodes.append(vertex_text)
            
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
    

def test():
    pass    

if __name__ == '__main__':
    min_sup = 0.015
    all_nodes, _ = build_gspan_graph('/data0/xiaoyez/CodeContextModel/data/train_test_index', './no_graph.data')
    gspan_miner('./no_graph.data', f'./no-sup-{min_sup}.data', len(all_nodes))