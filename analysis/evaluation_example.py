import os
import sys
import json
import torch
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.xmltree_parser import XMLTreeParser
from code_context_model.build_dataset import ExpandGraphDataset
from initial_cmm.initial_ccm import inference_dataset

EVALUATION_EXAMPLE_DIR = "/data2/xiaoyez/CodeContextModel/dataset_evaluation_example"
EMBEDDING_DIR = "/data0/xiaoyez/CodeContextModel/embedding_bge"
EMBEDDING_MODEL = "BgeEmbedding"
test_dir = "/data0/xiaoyez/CodeContextModel/data/train_test_index/mylyn"
test_cases = json.load(open(os.path.join(test_dir, "test_index.json")))
test_cases = [case.replace("/data0/xiaoyez/CodeContextModel/data/repo_first_3/", "/data0/xiaoyez/CodeContextModel/data/mylyn/") for case in test_cases]

def build_dataset(xml_files: list, test_case: str):
    if os.path.exists(os.path.join(EVALUATION_EXAMPLE_DIR, f"{test_case}_seed_dataset.pt")):
        return os.path.join(EVALUATION_EXAMPLE_DIR, f"{test_case}_seed_dataset.pt")
    test_dataset = ExpandGraphDataset(
        xml_files=xml_files, 
        embedding_dir=EMBEDDING_DIR, 
        embedding_model=EMBEDDING_MODEL, 
        debug=False
    )
    outfile = f"{test_case}_seed_dataset.pt"
    if not os.path.exists(EVALUATION_EXAMPLE_DIR):
        os.makedirs(EVALUATION_EXAMPLE_DIR)
    torch.save(test_dataset,  os.path.join(EVALUATION_EXAMPLE_DIR, outfile))
    print(f"save dataset to {os.path.join(EVALUATION_EXAMPLE_DIR, outfile)}")
    return os.path.join(EVALUATION_EXAMPLE_DIR, outfile)

def get_evaluation_example_candidates():
    evaluation_example_candidates = []
    for test_case in test_cases:
        ccm_path = os.path.join(test_case, "code_context_model.xml")
        expanded_ccm_path = os.path.join(test_case, "1_step_seeds_expanded_model.xml")
        if not os.path.exists(expanded_ccm_path):
            continue
        parser = XMLTreeParser(ccm_path)
        expanded_parser = XMLTreeParser(expanded_ccm_path)
        graphs = parser.get_graphs()
        if len(graphs) >= 2:
            vertices = parser.get_vertices()
            expanded_vertices = expanded_parser.get_vertices()
            if len(vertices) <= 5 and len(expanded_vertices) <= 20:
                evaluation_example_candidates.append(test_case)
                print(test_case)
    return evaluation_example_candidates

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    evaluation_example_candidates = get_evaluation_example_candidates()
    evaluation_example = "/data0/xiaoyez/CodeContextModel/data/mylyn/1174/1_step_seeds_expanded_model.xml"
    example_file_dataset_path = build_dataset([evaluation_example], "1174")
    inference_dataset(example_file_dataset_path, args)