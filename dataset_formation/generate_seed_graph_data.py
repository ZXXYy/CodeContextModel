
import xml.etree.ElementTree as ET
import itertools
from graphviz import Digraph
import logging
from tqdm import tqdm
import click
import os

logging.basicConfig(level=logging.DEBUG, format='[%(filename)s:%(lineno)d] - %(message)s')
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
    new_graph = ET.Element("graph", attrib={
        "repo_name": graph.get('repo_name'),
        "repo_path": graph.get('repo_path')
    })
    # create graph meta info
    new_root.append(new_graph)
    vertices = ET.Element("vertices", attrib={'total': str(len(new_nodes))})
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

def generate_expanded_graph_from_seed(expanded_model_path, seeds, idx, steps=1):
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
    prefix_output = os.path.join(os.path.dirname(expanded_model_path), "seed_expanded")
    if not os.path.exists(prefix_output):
        os.makedirs(prefix_output)
    # dir not empty, remove all files
    for file in os.listdir(prefix_output):
        os.remove(os.path.join(prefix_output, file))
    logger.debug(f"new_nodes: {len(new_nodes)}, new_edges: {len(new_edges)}")
    new_tree = form_expand_graph(new_nodes, new_edges, root)
    new_tree.write(f"{prefix_output}/{steps}_step_seeds_{idx}_expanded_model.xml", encoding='utf-8', xml_declaration=True)

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--step', default=1, help='The number of steps to expand the seed graph')
def main(input_path, step):
    context_models = os.listdir(input_path)
    for context_model in tqdm(context_models):
        if context_model != '42':
            continue
        logger.debug(f"Processing {context_model}")
        seeds, filtered_vertex_ids = generate_seed(f"{input_path}/{context_model}/big_{step}_step_expanded_model.xml", step)
        if seeds is None:
            continue
        for i, seed in enumerate(seeds):
            logger.debug(f"Seed: {seed}")
            generate_expanded_graph_from_seed(f"{input_path}/{context_model}/big_{step}_step_expanded_model.xml", seed, idx=i, steps=step)
            break
            # display_graph(f"tmp/1_step_seeds_{i}_expanded_model.xml")

if __name__ == '__main__':
    main()

    # 使用下述代码转化子图为一张完整的图
    # input_path = '/Users/zhengxiaoye/Desktop/codeContextGNN/CodeContextModel/data/mylyn'
    # context_models = os.listdir(input_path)
    # for context_model in tqdm(context_models):
    #     logger.debug(f"Processing {context_model}")
    #     for step in range(1, 4):
    #         subgraph2graph(f"{input_path}/{context_model}/new_{step}_step_expanded_model.xml")

    # 使用下述代码展示图     
    # display_graph('/Users/zhengxiaoye/Desktop/codeContextGNN/CodeContextModel/data/mylyn/42/seed_expanded/2_step_seeds_0_expanded_model.xml')