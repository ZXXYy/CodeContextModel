import os
import json
import logging

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from collections import Counter

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger('Eembedding')


# plot the distribution of the number of nodes
def plot_distribution(data, outfig):
    count = Counter(data)
    count = dict(sorted(count.items()))
    # get cumulative distribution
    total = sum(count.values())
    cumulative_count = {key: sum(v for k, v in count.items() if k <= key)*1.0/total for key in count.keys()}
    cumulative_count = dict(sorted(cumulative_count.items()))

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    # plot cdf and count in same figure
    ax.bar(count.keys(), count.values(), width=0.8)
    ax.set_xlabel('Number of Code Elements in Code Context Model')
    ax.set_ylabel('Count')
    ax.tick_params(labelsize=25)
    ax.yaxis.label.set_size(25)
    ax.xaxis.label.set_size(25)

    ax2 = ax.twinx()
    ax2.plot(cumulative_count.keys(), cumulative_count.values(), color='navy')
    ax2.set_ylabel('Percentage')
    ax2.yaxis.label.set_size(25)
    ax2.tick_params(labelsize=25)

    ax.grid(True, linestyle='--', which="major")
    plt.savefig(f'{outfig}.png')

def get_nodes(codes_path):
    tree = ET.parse(codes_path)
    root = tree.getroot()
    nodes = root.findall(".//vertex")
    return nodes

def get_code_context_model_statistics(project_dir):
    id_dirs = sorted(os.listdir(project_dir))
    num_nodes = []
    for id_dir in tqdm(id_dirs):
        logger.debug(f"{project_dir}, {id_dir}")
        codes_path = os.path.join(project_dir, id_dir, "code_context_model.xml")
        nodes = get_nodes(codes_path)
        if len(nodes) == 0:
            continue  
        num_nodes.append(len(nodes))
        logger.debug(f"{codes_path} nodes: {len(nodes)}")

    logger.info(f"{len(num_nodes)}")
    logger.info(f"Average number of nodes: {sum(num_nodes) / len(num_nodes)}")
    logger.info(f"Median number of nodes: {np.median(num_nodes)}")
    logger.info(f"Max number of nodes: {max(num_nodes)}")
    logger.info(f"Min number of nodes: {min(num_nodes)}")
    if not os.path.exists('analysis/figs'):
        os.makedirs('analysis/figs')
    outpath = os.path.join('analysis/figs', f"stats_{project_dir.split('/')[-1]}")
    plot_distribution(num_nodes, outpath)

def get_cross_file_count(codes_path):
    file_paths = set()
    tree = ET.parse(codes_path)
    root = tree.getroot()
    graphs = root.findall(".//graph")
    for graph in graphs:
        file_path = graph.attrib['repo_path']
        file_paths.add(file_path)
    
    if len(file_paths) == 2:
        logger.info(f"{codes_path} files: {len(file_paths)}")
    logger.debug(f"{codes_path} files: {len(file_paths)}")
    return file_paths

def get_cross_file_statistics(project_dir):
    id_dirs = sorted(os.listdir(project_dir))
    num_files = []
    for id_dir in tqdm(id_dirs):
        codes_path = os.path.join(project_dir, id_dir, "code_context_model.xml")
        file_paths = get_cross_file_count(codes_path)
        if len(file_paths) == 0:
            continue
        num_files.append(len(file_paths))
        
    outpath = os.path.join('analysis/figs', f"cross_file_{project_dir.split('/')[-1]}")
    plot_distribution(num_files, outpath)

def filter_cross_file_from_test(test_index_file):
    test_index = json.load(open(test_index_file))
    cross_file_test = []
    
    for project_id_dir in test_index:
        project_id_dir = project_id_dir.replace("repo_first_3", "mylyn")
        codes_path = os.path.join(project_id_dir, "code_context_model.xml")
        file_paths = get_cross_file_count(codes_path)
        if len(file_paths) < 2:
            continue
        cross_file_test.append(project_id_dir)
    
    logger.info(f"Total cross file test: {len(cross_file_test)}")
    outpath = test_index_file.replace("test_index.json", "cross_file_test_index.json")
    json.dump(cross_file_test, open(outpath, 'w'))

def filter_few_nodes_from_test(test_index_file):
    test_index = json.load(open(test_index_file))
    initial_task_test = []
    
    for project_id_dir in test_index:
        project_id_dir = project_id_dir.replace("repo_first_3", "mylyn")
        codes_path = os.path.join(project_id_dir, "code_context_model.xml")
        nodes = get_nodes(codes_path)
        if len(nodes) > 4:
            continue
        initial_task_test.append(project_id_dir)
    
    logger.info(f"Total initial task test: {len(initial_task_test)}")
    outpath = test_index_file.replace("test_index.json", "initial_task_test_index.json")
    json.dump(initial_task_test, open(outpath, 'w'))


if __name__ == "__main__":
    # project_dirs = [
    #     '/data0/xiaoyez/CodeContextModel/data/mylyn',
    #     '/data0/xiaoyez/CodeContextModel/data/PDE',
    #     '/data0/xiaoyez/CodeContextModel/data/Platform',
    # ]
    # for project_dir in project_dirs:
    #     get_code_context_model_statistics(project_dir)
    #     get_cross_file_statistics(project_dir)

    test_index_file = '/data0/xiaoyez/CodeContextModel/data/train_test_index/mylyn/test_index.json'
    filter_cross_file_from_test(test_index_file)
    filter_few_nodes_from_test(test_index_file)

