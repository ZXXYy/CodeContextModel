import os
import sys
import json
import argparse
import torch
import time

import xml.etree.ElementTree as ET

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.xmltree_parser import XMLTreeParser
from code_context_model.build_dataset import ExpandGraphDataset
from initial_cmm.initial_ccm import inference_dataset

mylyn_dir = "/data0/xiaoyez/CodeContextModel/data/mylyn"
test_dir = "/data0/xiaoyez/CodeContextModel/data/train_test_index/mylyn"
CROSS_FILE_CASES_DIR = "/data2/xiaoyez/CodeContextModel/dataset_cross_file_cases_step1"
EMBEDDING_DIR = "/data0/xiaoyez/CodeContextModel/embedding_bge"
EMBEDDING_MODEL = "BgeEmbedding"

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

def build_dataset(xml_files, cross_file: str):
    if os.path.exists(os.path.join(CROSS_FILE_CASES_DIR, f"{cross_file}_seed_dataset.pt")):
        return os.path.join(CROSS_FILE_CASES_DIR, f"{cross_file}_seed_dataset.pt")
    test_dataset = ExpandGraphDataset(
        xml_files=xml_files, 
        embedding_dir=EMBEDDING_DIR, 
        embedding_model=EMBEDDING_MODEL, 
        debug=False
    )
    outfile = f"{cross_file}_seed_dataset.pt" if cross_file else "dataset.pt"
    if not os.path.exists(CROSS_FILE_CASES_DIR):
        os.makedirs(CROSS_FILE_CASES_DIR)
    torch.save(test_dataset,  os.path.join(CROSS_FILE_CASES_DIR, outfile))
    print(f"save dataset to {os.path.join(CROSS_FILE_CASES_DIR, outfile)}")
    return os.path.join(CROSS_FILE_CASES_DIR, outfile)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cross_file_cases = []
    non_cross_file_cases = []

    total_cases = 0
    test_cases = json.load(open(os.path.join(test_dir, "test_index.json")))
    test_cases = [case.replace("/data0/xiaoyez/CodeContextModel/data/repo_first_3/", "/data0/xiaoyez/CodeContextModel/data/mylyn/") for case in test_cases]
    for test_case in test_cases:
        ccm_path = os.path.join(test_case, "code_context_model.xml")
        expanded_ccm_path = os.path.join(test_case, "1_step_seeds_expanded_model.xml")
        parser = XMLTreeParser(ccm_path)
        graphs = parser.get_graphs()
        if len(graphs) >= 2:
            cross_file_cases.append(expanded_ccm_path)
        else:
            non_cross_file_cases.append(expanded_ccm_path)

    print(f"cross file cases: {len(cross_file_cases)}")
    print(f"non cross file cases: {len(non_cross_file_cases)}")

    cross_file_dataset_path = build_dataset(cross_file_cases, "cross_file")
    non_cross_file_dataset_path = build_dataset(non_cross_file_cases, "non_cross_file")
    print(f"========== inference cross file cases ==========")
    inference_dataset(cross_file_dataset_path, args)
    print(f"========== inference non cross file cases ==========")
    inference_dataset(non_cross_file_dataset_path, args)