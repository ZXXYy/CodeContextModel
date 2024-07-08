import os
import itertools
import logging
import random
import argparse
import shutil

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
from graphviz import Digraph


logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger('SEED')


def display_graph(graph_path):
    tree = ET.parse(graph_path)
    root = tree.getroot()
    dot = Digraph()
    seed_expanded = False
    vertices = root.findall(".//vertex")
    if 'seed' in vertices[0].attrib:
        seed_expanded = True
    # 渲染结点
    for vertex in root.findall(".//vertex"):
        positive_node = vertex.get('origin') == '1'
        color = 'red' if positive_node else 'lightgrey'
        shape = 'diamond' if seed_expanded and vertex.get('seed') == '1' else 'ellipse' 
        dot.node(vertex.get('id'), color=color, shape=shape)
    # 渲染边
    for edge in root.findall(".//edge"):
        label = edge.get('label')
        dot.edge(edge.get('start'), edge.get('end'), label=label, color='black')
    # 保存并展示图
    dot.render(graph_path.split('.')[0], format='png', view=True)

def subgraph2graph(expanded_model_path):
    if not os.path.exists(expanded_model_path):
        logger.error(f"{expanded_model_path} not exists")
        return
    # if os.path.exists(expanded_model_path.replace('new_', 'big_')):
    #     logger.info(f"{expanded_model_path.replace('new_', 'big_')} already exists")
    #     return
    tree = ET.parse(expanded_model_path)
    root = tree.getroot()
    graphs = root.findall(".//graph")
    total_vertices = 0
    for graph in graphs:
        vertices = graph.findall(".//vertex")
        for vertex in vertices:
            id = int(vertex.get('id'))
            vertex.set('id', str(id + total_vertices))
        for edge in graph.findall(".//edge"):
            start = int(edge.get('start'))
            end = int(edge.get('end'))
            edge.set('start', str(start + total_vertices))
            edge.set('end', str(end + total_vertices))
        total_vertices += len(vertices)
    # write to file
    out_path = expanded_model_path.replace('new_', 'big_')
    tree.write(out_path, encoding='utf-8', xml_declaration=False)

def generate_seed(expanded_model_path, step=1):
    if not os.path.exists(expanded_model_path):
        logger.error(f"{expanded_model_path} not exists")
        return None, None
    tree = ET.parse(expanded_model_path)
    root = tree.getroot()
    vertex_ids = []
    # get vertex id
    for vertex in root.findall(".//vertex"):
        if vertex.get('origin') == '1':
            vertex_ids.append(vertex.get('id'))
    
    if len(vertex_ids)-step < 1:
        logger.error(f"{expanded_model_path}: The number of seeds is less than 1")
        return None, None
    # select len(vertex_ids) - step vertex as seed
    seeds = list(itertools.combinations(vertex_ids, len(vertex_ids)-step))
    # get filtered vertex ids
    filtered_vertex_ids = []
    for seed in seeds:
        filtered_vertex_ids.append(list(set(vertex_ids) - set(seed)))
    logger.debug(seeds)
    logger.debug(filtered_vertex_ids)
    return seeds, filtered_vertex_ids

def form_expand_graph(new_nodes, new_edges, old_xml_graph):
    new_root = ET.Element("expanded_model")
    graph = old_xml_graph.find(".//graph")
    max_nodes_id = len(old_xml_graph.findall(".//vertex"))-1
    new_graph = ET.Element("graph", attrib={
        "repo_name": graph.get('repo_name'),
        "repo_path": graph.get('repo_path')
    })
    # create graph meta info
    new_root.append(new_graph)
    vertices = ET.Element("vertices", attrib={'total': str(len(new_nodes)), 'max_id': str(max_nodes_id)})
    edges = ET.Element("edges", attrib={'total': str(len(new_edges))})
    new_graph.append(vertices)
    new_graph.append(edges)

    # write nodes and edges
    for node in new_nodes:
        vertices.append(node)
    for edge in new_edges:
        edges.append(edge)

    new_tree = ET.ElementTree(new_root)
    return new_tree

def generate_expanded_graph_from_seed(expanded_model_path, seeds, outdir, steps=1):
    if not os.path.exists(expanded_model_path):
        logger.error(f"{expanded_model_path} not exists")
        return
    tree = ET.parse(expanded_model_path)
    root = tree.getroot()
    logger.debug(seeds)
    new_edges = []
    new_nodes = []
    # add seeds to new graph
    for seed in seeds:
        vertex = root.find(f".//vertex[@id='{seed}']")
        if vertex not in new_nodes:
            vertex.set('seed', '1')
            new_nodes.append(vertex)
            logger.debug(vertex.get('id'))

    for _ in range(steps):
        nodes = new_nodes.copy()
        # 针对每一个nodes，找到其相邻的边和结点
        for seed in new_nodes:
            seed_id = seed.get('id')
            logger.debug(f"seed: {seed_id}")
            for edge in root.findall(".//edge"):
                source = edge.get('start') 
                target = edge.get('end')
                if not (source == seed_id or target == seed_id):
                    continue
                logger.debug(f"edge: {edge.get('start')} -> {edge.get('end')})")
                # add the edge
                if edge not in new_edges:
                    new_edges.append(edge)
                # add the neighbor vertex
                vertex_id = target if source == seed_id else source
                vertex = root.find(f".//vertex[@id='{vertex_id}']")
                if vertex not in nodes:
                    logger.debug(f"vertex: {vertex.get('id')}")
                    vertex.set('seed', '0')
                    nodes.append(vertex)
            logger.debug(f"===========")
        # 基于新的nodes继续扩展
        new_nodes = nodes

    # 两个选中的结点之间可能还存在边，需要添加
    vertex_ids = [vertex.get('id') for vertex in new_nodes]
    for edge in root.findall(".//edge"):
        source = edge.get('start') 
        target = edge.get('end')
        if source in vertex_ids and target in vertex_ids and edge not in new_edges:
            new_edges.append(edge)
    
    # write new graph to file
    logger.debug(f"new_nodes: {len(new_nodes)}, new_edges: {len(new_edges)}")
    new_tree = form_expand_graph(new_nodes, new_edges, root)
    new_tree.write(f"{outdir}/{steps}_step_seeds_expanded_model.xml", encoding='utf-8', xml_declaration=True)

def collaspe_variables(expanded_model_path, code_path, model_dir, outdir, step):
    if not os.path.exists(expanded_model_path):
        logger.error(f"{expanded_model_path} not exists")
        return
    if not os.path.exists(code_path):
        logger.warning(f"{code_path} not exists")
        return
    df_code = pd.read_csv(code_path, sep='\t')
    tree = ET.parse(expanded_model_path)
    root = tree.getroot()
    # 1. 找到所有的class结点
    new_variable_id = int(root.find(".//vertices").get('max_id')) + 1
    vertices_class = root.findall(".//vertex[@kind='class']")
    logger.debug(f"Number of class: {len(vertices_class)}")
    for i, vertex in enumerate(vertices_class):
        class_id = vertex.get('id')
        # 2. 对于每一个class结点，找到其所有的variable结点
        class_edges = root.findall(f".//edge[@start='{class_id}']")
        variable_codes = ""
        variable_ids = []
        variable_origin = False
        logger.debug(f"Number of class edges: {len(class_edges)}")
        for edge in class_edges:
            end_id = edge.get('end')
            variable = root.find(f".//vertex[@id='{end_id}'][@kind='variable']")
            if variable is None:
                continue
            logger.debug(f"class declared variables: {variable}")
            variable_id = f"{model_dir}_{variable.get('kind')}_{variable.get('ref_id')}"
            variable_code = df_code[df_code['id'] == variable_id]['code'].values[0]
            logger.debug(f'{code_path}:\n {variable_id}\n{variable_code}')
            logger.debug('='*10)

            # 2.1 如果len(variable_code) < 100, 则将variable结点合并
            if len(variable_code) < 100:
                # 2.1.1 找到这些结点对应的代码片段，合并
                variable_codes += variable_code + "\n"
                variable_ids.append(variable.get('id'))
                variable_origin = variable_origin or variable.get('origin') == '1'
                pass
            # 2.2 如果len(variable_code) >= 100, variable结点保存
            else:
                pass
        if len(variable_ids) == 0:
            continue
        new_variable = ET.Element("vertex", attrib={
            "id": str(new_variable_id),
            "kind": "variable",
            "origin": "1" if variable_origin else "0",
            "ref_id": f"{step}_step_collapsed_variable_{i}",
        })
        vertices = root.find('.//vertices')
        vertices.append(new_variable)
        code_id = f"{model_dir}_{new_variable.get('kind')}_{new_variable.get('ref_id')}", 
        if df_code[df_code['id'] == code_id].empty:
            df_code = pd.concat([df_code, pd.DataFrame({
                'id': code_id, 
                'code': variable_codes}, index=[0])],ignore_index=True)
        # 3. 获取删除结点的所有边，将这些边的start和end指向新的variable结点
        for variable_id in variable_ids:
            variable = root.find(f".//vertex[@id='{variable_id}']")
            parent_variable = root.find(f".//vertices")
            parent_variable.remove(variable)
            # 删除边
            def update_edges(edges, variable_id):
                parent_edges = root.find(f".//edges")
                for edge in edges:
                    start_id = edge.get('start')
                    end_id = edge.get('end')
                    logger.debug(f"edge: {start_id} -> {end_id}")
                    def update_edge(start, end):
                         # 如果边已经存在，则不添加
                        if root.find(f".//edge[@start='{start}'][@end='{end}']") is None:
                            edge.set('start', str(start))
                            edge.set('end', str(end))
                            logger.debug(f"new edge: {start} -> {end}")
                            parent_edges.append(edge)
                        parent_edges.remove(edge)
                    if start_id == end_id:
                        update_edge(new_variable_id,new_variable_id)
                    elif start_id == variable_id:
                        update_edge(new_variable_id,end_id)
                    elif end_id == variable_id:
                        update_edge(start_id,new_variable_id)
            start_edges = root.findall(f".//edge[@start='{variable_id}']")
            end_edges = root.findall(f".//edge[@end='{variable_id}']")
            update_edges(start_edges, variable_id)
            update_edges(end_edges, variable_id)
        new_variable_id += 1
    if not os.path.exists(f"{outdir}/{step}_step_collaspe"):
        os.makedirs(f"{outdir}/{step}_step_collaspe")
    # 5. 写入文件
    tree.write(f"{outdir}/collapse_{expanded_model_path.split('/')[-1]}", encoding='utf-8', xml_declaration=True)
    df_code.to_csv(f"{outdir}/my_java_codes_collapse.tsv", sep='\t', index=False)
            

def generate_big_graphs(input_path, step=1):
    """
    将所有的子图合并成一个大图 new_1_step_expanded_model.xml -> big_1_step_expanded_model.xml
    """
    context_models = os.listdir(input_path)
    for context_model in tqdm(context_models):
        logger.debug(f"Processing {context_model}")
        subgraph2graph(f"{input_path}/{context_model}/new_{step}_step_expanded_model.xml")

def generate_seed_expanded_graphs(input_path, step=1):
    """
    根据seed生成扩展图，保存到seed_expanded文件夹下
    big_x_step_expanded_model.xml -> x_step_seeds_expanded_model.xml
    """
    context_models = os.listdir(input_path)
    for context_model in tqdm(context_models):
        logger.debug(f"Processing {context_model}")
        seeds, filtered_vertex_ids = generate_seed(f"{input_path}/{context_model}/big_{step}_step_expanded_model.xml", step)
        if seeds is None:
            continue
        expanded_model_path = f"{input_path}/{context_model}/big_{step}_step_expanded_model.xml"
        index = random.randint(0, len(seeds) - 1) 
        seed = seeds[index]
        logger.debug(f"Seed: {seed}")
        generate_expanded_graph_from_seed(expanded_model_path, seed, outdir=f"{input_path}/{context_model}", steps=step)

        # for i, seed in enumerate(seeds):
        #     logger.debug(f"Seed: {seed}")

def generate_variable_collapsed_graphs(input_path, step=1):
    """
    合并变量结点
    x_step_seeds_expanded_model.xml -> collaspe_x_step_seeds_expanded_model.xml
    my_java_codes.tsv -> my_java_codes_collapse.tsv
    """
    context_models = os.listdir(input_path)
    for context_model in tqdm(context_models):
        logger.debug(f"Processing {context_model}")
        expanded_model_dir = f"{input_path}/{context_model}"
        model_dir = context_model
        outdir = f"{input_path}/{context_model}/"
        code_path = f"{input_path}/{context_model}/my_java_codes.tsv"
        if os.path.exists(f"{input_path}/{context_model}/my_java_codes_collapse.tsv"):
            code_path = f"{input_path}/{context_model}/my_java_codes_collapse.tsv"

        expanded_model_path = os.path.join(expanded_model_dir, f"{step}_step_seeds_expanded_model.xml")
        if not os.path.exists(expanded_model_path):
            logger.error(f"{expanded_model_path} not exists")
            continue
        collaspe_variables(expanded_model_path, code_path=code_path, model_dir=model_dir, outdir=outdir, step=step)

        # for expanded_model_path in os.listdir(expanded_model_dir):
        #     expanded_model_path = os.path.join(expanded_model_dir, expanded_model_path)

def get_statistics(input_path):
    context_models = os.listdir(input_path)
    statistics = {
        'positive_variables': [],
        'variables': [],
        'positive_functions': [],
        'functions': [],
        'positive_classes': [],
        'classes': [],
        'variable_percentage': [],
    }
    variables_code = []
    for context_model in tqdm(context_models):
        logger.debug(f"Processing {context_model}")
        if not os.path.exists(f"{input_path}/{context_model}/my_java_codes.tsv"):
            continue
        df_code = pd.read_csv(f"{input_path}/{context_model}/my_java_codes.tsv", sep='\t')
        for i, row in df_code.iterrows():
            if 'variable' in row['id'] :
                variables_code.append(len(row['code']))

        if not os.path.exists(f"{input_path}/{context_model}/big_1_step_expanded_model.xml"):
            continue
        tree = ET.parse(f"{input_path}/{context_model}/big_1_step_expanded_model.xml")
        root = tree.getroot()
        vertices = root.findall(".//vertex")
        num_nodes = len(vertices)
        variables, positive_variables = 0, 0
        functions, positive_functions = 0, 0
        classes, positive_classes = 0, 0

        for vertex in vertices:
            if vertex.get('kind') == 'variable':
                variables += 1
                positive_variables += 1 if vertex.get('origin') == '1' else 0
            elif vertex.get('kind') == 'function':
                functions += 1
                positive_functions += 1 if vertex.get('origin') == '1' else 0
            elif vertex.get('kind') == 'class':
                classes += 1
                positive_classes += 1 if vertex.get('origin') == '1' else 0
        statistics['positive_variables'].append(positive_variables)
        statistics['variables'].append(variables)
        statistics['positive_functions'].append(positive_functions)
        statistics['functions'].append(functions)
        statistics['positive_classes'].append(positive_classes)
        statistics['classes'].append(classes)
        if num_nodes == 0:
            continue
        statistics['variable_percentage'].append(variables/num_nodes)

        
        print(f"{context_model}: classes={positive_classes}/{classes} functions={positive_functions}/{functions} variables={positive_variables}/{variables} ")
    print(f"positive variables = 0: {statistics['positive_variables'].count(0)}")
    # count the number of variable_percentage > 0.5
    num_variable_percentage = len([i for i in statistics['variable_percentage'] if i > 0.2])
    print(f"variable_percentage > 0.2: {num_variable_percentage}")
    variables_code_lens = [x for x in variables_code if x <= 1024]
    print(f"variables code length <= 1024: {len(variables_code_lens)}/{len(variables_code)}")
    variables_code_lens = [x for x in variables_code if x <= 200]
    print(f"variables code length <= 200: {len(variables_code_lens)}/{len(variables_code)}")
    variables_code_lens = [x for x in variables_code if x <= 100]
    print(f"variables code length <= 100: {len(variables_code_lens)}/{len(variables_code)}")
    # plot the statistics num
    count_values = [statistics['positive_variables'].count(i) for i in range(0, max(statistics['positive_variables'])+1)]

    # plt.bar(range(0, max(statistics['positive_variables'])+1), count_values)
    # plt.hist(statistics['variable_percentage'], bins=50, alpha=0.5, label='variables')
    # plt.hist(variables_code, bins=50, alpha=0.5, label='variables')

    plt.show()

def clearup(input_path):
    context_models = os.listdir(input_path)
    for context_model in tqdm(context_models):
        logger.debug(f"Processing {context_model}")            
        for step in range(1, 4):
            # remove big expanded graphs
            expanded_model_dir = f"{input_path}/{context_model}/big_{step}_step_expanded_model.xml"
            if os.path.exists(expanded_model_dir):
                os.remove(expanded_model_dir)
            # remove seed expanded graphs
            expanded_model_dir = f"{input_path}/{context_model}/{step}_step_seed_expanded"
            if os.path.exists(expanded_model_dir):
                shutil.rmtree(expanded_model_dir)
            collapse_model_dir = f"{input_path}/{context_model}/{step}_step_collaspe"
            if os.path.exists(collapse_model_dir):
                shutil.rmtree(collapse_model_dir)
            seed_expanded_file = f"{input_path}/{context_model}/{step}_step_seeds_expanded_model.xml"
            if os.path.exists(seed_expanded_file):
                os.remove(seed_expanded_file)
            collapse_file = f"{input_path}/{context_model}/collapse_{step}_step_seeds_expanded_model.xml"
            if os.path.exists(collapse_file):
                os.remove(collapse_file)
        # remove collasped graphs
        if os.path.exists(f"{input_path}/{context_model}/my_java_codes_collapse.tsv"):
            os.remove(f"{input_path}/{context_model}/my_java_codes_collapse.tsv")
        if os.path.exists(f"{input_path}/{context_model}/seed_expanded"):
            shutil.rmtree(f"{input_path}/{context_model}/seed_expanded")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data', help='input directory')
    parser.add_argument('--step', type=int, default=1, help='expand step')
    parser.add_argument('--action', type=str, default='generate', help='action to perform: generate, statistics, display, clear')
    args = parser.parse_args()

    if args.action == 'generate':
        logger.info(f"=====start to generate big graphs from {args.input_dir}=====")
        generate_big_graphs(args.input_dir, args.step)
        logger.info(f"=====Big graphs generated successfully=====")
        
        logger.info(f"=====start to generate seed expanded graphs from {args.input_dir}=====")
        generate_seed_expanded_graphs(args.input_dir, args.step)
        logger.info(f"=====Seed expanded graphs generated successfully=====")

        logger.info(f"=====start to generate collapsed variable graphs from {args.input_dir}=====")
        generate_variable_collapsed_graphs(args.input_dir, args.step)
        logger.info(f"=====Collasped variable graphs generated successfully=====")
    elif args.action == 'statistics':
        # 使用下述代码获取数据集中变量/函数/类的统计信息
        # get_statistics('/Users/zhengxiaoye/Desktop/codeContextGNN/CodeContextModel/data')
        get_statistics(args.input_dir)
    elif args.action == 'display':
        # 使用下述代码展示图     
        # display_graph('/Users/zhengxiaoye/Desktop/codeContextGNN/CodeContextModel/data/mylyn/1729/seed_expanded/1_step_seeds_0_expanded_model.xml')
        display_graph(args.input_dir)
    elif args.action == 'clear':
        clearup(args.input_dir)
    
   