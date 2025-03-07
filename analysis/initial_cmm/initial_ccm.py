# 1. randomly select the initial ccm for each case with number of vertices k
# 2. expand the initial ccm to expanded ccm 
# 3. label the expanded ccm with label_type: seed, context, non-context
# 4. build dataset, infer the label, and calculate the metrics 

import os
import sys
import json
import torch
import dgl
import argparse
import logging
import numpy as np
import xml.etree.ElementTree as ET

from typing import List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seed_strategies import CountBasedStrategy, OrderBasedStrategy, ExperienceBasedStrategy
from utils.xmltree_parser import XMLTreeParser
from dataset_formation.generate_seed_graph_data import generate_expanded_graph_from_seed
from code_context_model.build_dataset import ExpandGraphDataset
from code_context_model.gnn import RGCN
from code_context_model.train import test
from results_visualize import visualize_count_based_results, visualize_order_based_results

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

CCM_EXPANDED_GRAPH_FILE = "big_1_step_expanded_model.xml"
INITIAL_CCM_DIR = "/data2/xiaoyez/CodeContextModel/initial_ccm"
EMBEDDING_DIR = "/data0/xiaoyez/CodeContextModel/embedding_bge"
EMBEDDING_MODEL = "BgeEmbedding"
INFERENCE_MODEL_PATH = "/data0/xiaoyez/CodeContextModel/model_output/07-14-00-19/model_48.pth"

test_dir = "/data0/xiaoyez/CodeContextModel/data/train_test_index/mylyn"

# new_1_step_expanded_model.xml 是基于code_context_model.xml生成的一步扩展图
# big_1_step_expanded_model.xml 是基于new_1_step_expanded_model.xml生成的所有子图合并后的大图
# 所以 这里的seed从big_1_step_expanded_model.xml中选取，写入到count_based_seed_expanded_model.xml

def generate_initial_seed(test_case, seed_strategy, num_seed=1) -> List[tuple]:
    if seed_strategy == "count_based":
        seed_strategy = CountBasedStrategy(num_seed)
        graph = XMLTreeParser(os.path.join(test_case, CCM_EXPANDED_GRAPH_FILE))
        initial_seed = seed_strategy.generate_seed(graph)
        return initial_seed
    elif seed_strategy == "order_based":
        seed_strategy = OrderBasedStrategy(num_seed)
        graph = XMLTreeParser(os.path.join(test_case, CCM_EXPANDED_GRAPH_FILE))
        initial_seeds = seed_strategy.generate_seed(graph)
        return initial_seeds
    elif seed_strategy == "experience_based":
        seed_strategy = ExperienceBasedStrategy(test_case.split("/")[-1])
        graph = XMLTreeParser(os.path.join(test_case, CCM_EXPANDED_GRAPH_FILE))
        initial_seed = seed_strategy.generate_seed(graph)
        return initial_seed
    else:
        raise ValueError(f"Invalid seed strategy: {seed_strategy}")
    return initial_seed

def generate_expanded_ccm_from_seed(initial_seed, seed_strategy, test_case, expanded_ccm_id):
    # expanded_ccm_id 在count_based中是num_seed, 在order_based中是initial_seed的index
    graph = XMLTreeParser(os.path.join(test_case, CCM_EXPANDED_GRAPH_FILE))
    test_case_id = test_case.split("/")[-1]
    outdir = os.path.join(INITIAL_CCM_DIR, seed_strategy, "raw_data", test_case_id)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if os.path.exists(os.path.join(outdir, f"{expanded_ccm_id}_seed_expanded_model.xml")):
        return os.path.join(outdir, f"{expanded_ccm_id}_seed_expanded_model.xml")
    # print(os.path.join(outdir, f"{num_seed}_seed_expanded_model.xml"))
    generate_expanded_graph_from_seed(
        graph.root, 
        initial_seed, 
        outdir=outdir, 
        outpath=os.path.join(outdir, f"{expanded_ccm_id}_seed_expanded_model.xml")
    )
    return os.path.join(outdir, f"{expanded_ccm_id}_seed_expanded_model.xml")
    

def build_dataset(xml_files, seed_strategy, expanded_ccm_id):
    if os.path.exists(os.path.join(INITIAL_CCM_DIR, seed_strategy, "dataset", f"{expanded_ccm_id}_seed_dataset.pt")):
        return os.path.join(INITIAL_CCM_DIR, seed_strategy, 'dataset', f"{expanded_ccm_id}_seed_dataset.pt")
    test_dataset = ExpandGraphDataset(
        xml_files=xml_files, 
        embedding_dir=EMBEDDING_DIR, 
        embedding_model=EMBEDDING_MODEL, 
        debug=False
    )
    outfile = f"{expanded_ccm_id}_seed_dataset.pt" if expanded_ccm_id else "dataset.pt"
    if not os.path.exists(os.path.join(INITIAL_CCM_DIR, seed_strategy, "dataset")):
        os.makedirs(os.path.join(INITIAL_CCM_DIR, seed_strategy, "dataset"))
    torch.save(test_dataset,  os.path.join(INITIAL_CCM_DIR, seed_strategy, "dataset", outfile))
    print(f"save dataset to {os.path.join(INITIAL_CCM_DIR, seed_strategy, 'dataset', outfile)}")
    return os.path.join(INITIAL_CCM_DIR, seed_strategy, 'dataset', outfile)

def inference_dataset(dataset_path, args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(dataset_path)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dgl.batch)
    logger.info(f"Load dataset finished, Test: {len(dataset)}")

    logger.info(f"test model path: {INFERENCE_MODEL_PATH}")
    old_state_dict = torch.load(INFERENCE_MODEL_PATH)
    mapping = {
        'conv1': 'conv_layers.0',
        'conv2': 'conv_layers.1',
        'conv3': 'conv_layers.2'
    }
    new_state_dict = {}
    for old_key, value in old_state_dict.items():
        for old_prefix, new_prefix in mapping.items():
            if old_key.startswith(old_prefix):
                new_key = old_key.replace(old_prefix, new_prefix)
                new_state_dict[new_key] = value

    # 加载重命名后的 state_dict 到新模型
    num_dims = 1024 # bge embedding dims
    model = RGCN(in_feat=num_dims, h_feat=num_dims, gnn_layers=3, out_feat=1, num_rels=8)
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)
    # model.load_state_dict(torch.load(args.test_model_pth))
    test_hit_rate = test(
        model=model, 
        test_loader=test_loader, 
        device=device,
        threshold=0.5
    )
    return test_hit_rate
        
def write_result(test_hit_rates, seed_strategy):
    with open(os.path.join(INITIAL_CCM_DIR, seed_strategy, f"result.json"), "w") as f:
        json.dump(test_hit_rates, f)

def run_count_based_initial_ccm(test_cases, args):
    if not os.path.exists(os.path.join(INITIAL_CCM_DIR, "count_based", "result.json")):
        test_hit_rates = defaultdict(list)
        for num_seed in range(1, 16):
            expanded_ccms = []
            for test_case in tqdm(test_cases):
                initial_seed = generate_initial_seed(test_case, "count_based", num_seed)
                if initial_seed is None:
                    continue
                expanded_ccm = generate_expanded_ccm_from_seed(initial_seed[0], "count_based", test_case, num_seed)
                expanded_ccms.append(expanded_ccm)
            dataset_path = build_dataset(expanded_ccms, "count_based", num_seed)
            test_hit_rate = inference_dataset(dataset_path, args)
            test_hit_rates[num_seed] = test_hit_rate
        write_result(test_hit_rates, "count_based")    
         
    results = json.load(open(os.path.join(INITIAL_CCM_DIR, "count_based", "result.json")))
    visualize_count_based_results(results, os.path.join(INITIAL_CCM_DIR, "count_based", "visualization"))

def run_order_based_initial_ccm(test_cases, args):
    if not os.path.exists(os.path.join(INITIAL_CCM_DIR, "order_based", "result.json")):
        test_hit_rates = defaultdict(list)
        for test_case in tqdm(test_cases):
            test_case_id = test_case.split("/")[-1]
            initial_seeds = generate_initial_seed(test_case, "order_based", 5)
            if initial_seeds is None:
                continue
            for index, initial_seed in enumerate(initial_seeds):
                expanded_ccm = generate_expanded_ccm_from_seed(initial_seed, "order_based", test_case, index)
                dataset_path = build_dataset([expanded_ccm], "order_based", f"{test_case_id}_{index}")
                test_hit_rate = inference_dataset(dataset_path, args)
                test_hit_rates[test_case_id].append(test_hit_rate)
        write_result(test_hit_rates, "order_based")
    
    results = json.load(open(os.path.join(INITIAL_CCM_DIR, "order_based", "result.json")))
    visualize_order_based_results(results, os.path.join(INITIAL_CCM_DIR, "order_based", "visualization"))

def run_experience_based_initial_ccm(test_cases, args):
    if not os.path.exists(os.path.join(INITIAL_CCM_DIR, "experience_based", "result.json")):
        test_hit_rates = defaultdict(list)
        have_experience_based_seed_count, have_experience_based_cases = 0, []
        seeds_num_list = []
        expanded_ccms = []
        for test_case in tqdm(test_cases):
            test_case_id = test_case.split("/")[-1]
            initial_seed = generate_initial_seed(test_case, "experience_based")
            if initial_seed is None:
                continue            
            have_experience_based_seed_count += 1
            expanded_ccm = generate_expanded_ccm_from_seed(initial_seed[0], "experience_based", test_case, "experience_based")
            expanded_ccms.append(expanded_ccm)
            have_experience_based_cases.append(test_case_id)
            seeds_num_list.append(len(initial_seed[0]))
        dataset_path = build_dataset(expanded_ccms, "experience_based", "experience_based")
        test_hit_rate = inference_dataset(dataset_path, args)
        test_hit_rates["experience_based"] = test_hit_rate

        step_1_expanded_ccms = [os.path.join(
            '/data0/xiaoyez/CodeContextModel/data/mylyn',
            case_id, 
            "1_step_seeds_expanded_model.xml"
        ) for case_id in have_experience_based_cases]
        step_1_dataset_path = build_dataset(step_1_expanded_ccms, "experience_based", "non_experience_based_step_1")
        step_1_test_hit_rate = inference_dataset(step_1_dataset_path, args)
        test_hit_rates["non_experience_based_step_1"] = step_1_test_hit_rate

        write_result(test_hit_rates, "experience_based")
        print(f"have {have_experience_based_seed_count}/{len(test_cases)} experience based seeds")
        print(f"seeds num list: {np.quantile(seeds_num_list, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])}")
    # results = json.load(open(os.path.join(INITIAL_CCM_DIR, "experience_based", "result.json")))

def parse_args():
    parser = argparse.ArgumentParser(description='Generate initial CCM')
    parser.add_argument('--seed_strategy', type=str, default="step_based", help='seed strategy')
    parser.add_argument('--device', type=str, default="0", help='device')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    test_cases = json.load(open(os.path.join(test_dir, "test_index.json")))
    test_cases = [case.replace("/data0/xiaoyez/CodeContextModel/data/repo_first_3/", "/data0/xiaoyez/CodeContextModel/data/mylyn/") for case in test_cases]
    if args.seed_strategy == "count_based":
        run_count_based_initial_ccm(test_cases, args)
    elif args.seed_strategy == "order_based":
        run_order_based_initial_ccm(test_cases, args)
    elif args.seed_strategy == "experience_based":
        run_experience_based_initial_ccm(test_cases, args)
    else:
        raise ValueError(f"Invalid seed strategy: {args.seed_strategy}")
    # results = infer_expanded_ccms(expanded_ccms, mylyn_dir, test_dir)
    # calculate_metrics(results)