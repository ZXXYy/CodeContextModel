
import xml.etree.ElementTree as ET
import itertools
from graphviz import Digraph
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('SEED')


def display_graph(graph_path):
    tree = ET.parse(graph_path)
    root = tree.getroot()
    dot = Digraph()
    # 渲染结点
    for vertex in root.findall(".//vertex"):
        positive_node = vertex.get('origin') == '1'
        color = 'red' if positive_node else 'lightgrey'
        dot.node(vertex.get('id'), color=color)
    # 渲染边
    for edge in root.findall(".//edge"):
        label = edge.get('label')
        dot.edge(edge.get('start'), edge.get('end'), label=label, color='black')
    # 保存并展示图
    dot.render("tmp/"+graph_path.split('.')[0], format='png', view=True)

def generate_seed(cxt_model_path, step=1):
    tree = ET.parse(cxt_model_path)
    root = tree.getroot()
    # get vertex id
    vertex_ids = [vertex.get('id') for vertex in root.findall(".//vertex")]
    # select len(vertex_ids) - step vertex as seed
    seeds = list(itertools.combinations(vertex_ids, len(vertex_ids)-step))
    # get filtered vertex ids
    filtered_vertex_ids = []
    for seed in seeds:
        filtered_vertex_ids.append(list(set(vertex_ids) - set(seed)))
    
    print(seeds)
    print(filtered_vertex_ids)
    return seeds, filtered_vertex_ids

def generate_expanded_graph_from_seed(expanded_model_path, seeds, idx, step=1):
    tree = ET.parse(expanded_model_path)
    root = tree.getroot()
    logger.debug(seeds)
    new_edges = []
    new_nodes = []
    # add seeds to new graph
    for seed in seeds:
        vertex = root.find(f".//vertex[@id='{seed}']")
        if vertex not in new_nodes:
            new_nodes.append(vertex)
            logger.debug(vertex.get('id'))

    for step in range(step):
        nodes = new_nodes.copy()
        for seed in new_nodes:
            seed_id = seed.get('id')
            logger.debug(f"seed: {seed_id}")
            for edge in root.findall(".//edge"):
                source = edge.get('start') 
                target = edge.get('end')
                if source == seed_id or target == seed_id:
                    logger.debug(f"edge: {edge.get('start')} -> {edge.get('end')})")
                    if edge not in new_edges:
                        new_edges.append(edge)
                    vertex_id = target if source == seed_id else source
                    # get the vertex
                    vertex = root.find(f".//vertex[@id='{vertex_id}']")
                    if vertex not in nodes:
                        logger.debug(f"vertex: {vertex.get('id')}")
                        nodes.append(vertex)
            logger.debug(f"===========")
        new_nodes = nodes

    # 两个选中的结点之间可能还存在边，需要添加
    vertex_ids = [vertex.get('id') for vertex in new_nodes]
    for edge in root.findall(".//edge"):
        source = edge.get('start') 
        target = edge.get('end')
        if source in vertex_ids and target in vertex_ids and edge not in new_edges:
            new_edges.append(edge)
    
    # write new graph to file
    new_root = ET.Element("expanded_model", attrib={'seednum': str(idx)})
    graph = root.find(".//graph")
    new_graph = ET.Element("graph", attrib={
        "repo_name": graph.get('repo_name'),
        "repo_path": graph.get('repo_path')
    })
    new_root.append(new_graph)
    vertices = ET.Element("vertices", attrib={'total': str(len(new_nodes))})
    edges = ET.Element("edges", attrib={'total': str(len(new_edges))})
    print(f"nodes: {len(new_nodes)}, edges: {len(new_edges)}")
    new_graph.append(vertices)
    new_graph.append(edges)
    for node in new_nodes:
        vertices.append(node)
    for edge in new_edges:
        edges.append(edge)
    new_tree = ET.ElementTree(new_root)
    new_tree.write(f"tmp/{step}_step_seeds_{idx}_expanded_model.xml", encoding='utf-8', xml_declaration=True)


seeds, filtered_vertex_ids = generate_seed("/Users/zhengxiaoye/Desktop/codeContextGNN/CodeContextModel/data/repo_first_3/60/code_context_model.xml")
generate_expanded_graph_from_seed("/Users/zhengxiaoye/Desktop/codeContextGNN/CodeContextModel/data/repo_first_3/60/new_1_step_expanded_model.xml", seeds[0], 1, step=2)
display_graph('/Users/zhengxiaoye/Desktop/codeContextGNN/CodeContextModel/data/repo_first_3/60/new_1_step_expanded_model.xml')
display_graph('/Users/zhengxiaoye/Desktop/codeContextGNN/CodeContextModel/0_step_seeds_1_expanded_model.xml')

