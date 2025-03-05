import os
import sys

import xml.etree.ElementTree as ET

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.xmltree_parser import XMLTreeParser

mylyn_dir = "/data0/xiaoyez/CodeContextModel/data/mylyn"
cross_file_cases = []

def contain_edge_type(parser, edge_type):
    edges = parser.get_edges()
    for edge in edges:
        if edge.get('label') == edge_type and edge.get('start') != edge.get('end'):
            return True
    return False

def contain_vertex_type(parser, vertex_type):
    vertices = parser.get_vertices()
    for vertex in vertices:
        if vertex.get('kind') == vertex_type:
            return True
    return False

for ccm_dir in os.listdir(mylyn_dir):
    ccm_path = os.path.join(mylyn_dir, ccm_dir, "code_context_model.xml")
    if not os.path.exists(ccm_path):
        continue
    parser = XMLTreeParser(ccm_path)
    graphs = parser.get_graphs()
    if len(graphs) == 2:
        cross_file_cases.append(ccm_dir)
        vertices_count = parser.get_vertices_count()
        if vertices_count[0] <= 3 and vertices_count[1] <= 3:
            if contain_edge_type(parser, 'calls') and contain_vertex_type(parser, 'variable'):
                print(ccm_path)

print(len(cross_file_cases))