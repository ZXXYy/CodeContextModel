
import os
import re
import argparse
import random
import logging
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm
from gensim.models import Word2Vec
from lex import LexParser

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger('Eembedding')

MAX_SEQ_LEN = 10

def get_nodes_text(expand_graph_path: str) -> pd.DataFrame:
    tree = ET.parse(expand_graph_path)
    root = tree.getroot()
    nodes = root.findall(".//vertex")
    nodes_id = []
    node_id2type = {}
    model_dir = expand_graph_path.split('/')[-2]
    codes_path = os.path.dirname(expand_graph_path) + "/my_java_codes.tsv"
    # read tsv file
    df_code = pd.read_csv(codes_path, sep='\t')
    for vertex in nodes:
        node_id = '_'.join([model_dir, vertex.get('kind'), vertex.get('ref_id')]) 
        nodes_id.append(node_id)
        node_id2type[node_id] = vertex.get('kind')
    df_code = df_code[df_code['id'].isin(nodes_id)]

    # get variable / method / class name
    def get_node_name(node_id):
        code = df_code[df_code['id'] == node_id]['code'].values[0].split('\n')[0]
        name = ""
        if node_id2type[node_id] == 'variable':
            name = 'VARIABLE/' + code.split('=')[0].split()[-1]
        elif node_id2type[node_id] == 'function':
            name =  'FUNCTION/' + code.split('(')[0].split()[-1]
        elif node_id2type[node_id] == 'class':
            name =  'CLASS/' + code.split('class')[1].split()[0]
        
        return name
    
    df_code['code'] = df_code['id'].apply(get_node_name)
    return df_code

def get_expanded_model_path(input_dir, id_dir):
    for step in [3, 2, 1]:
        expand_graph_path = os.path.join(input_dir, id_dir, f"{step}_step_seeds_expanded_model.xml")
        if os.path.exists(expand_graph_path):
            return expand_graph_path
    return None

def get_word_corpus(input_dir, debug=False):
    id_dirs = sorted(os.listdir(input_dir))
    id_dirs = id_dirs[:10] if debug else id_dirs
    corpus = []
    logger.info("Start reading codes...")
    
    for id_dir in tqdm(id_dirs):
        logger.debug(f"{input_dir}, {id_dir}")
        # use 3-step seed expansion model first for all possible corpus        
        expand_graph_path = get_expanded_model_path(input_dir, id_dir)
        if expand_graph_path is None:
            logger.info(f"No expanded file in { os.path.join(input_dir, id_dir)}")
            continue
        codes = get_nodes_text(expand_graph_path)["code"].tolist()
        corpus.extend(codes)
    print(corpus[:10])
    print(len(corpus))

    return corpus

def build_word_embedding(corpus, min_count=1):
    parser = LexParser(corpus)
    return parser

def embedding_inference(input_dir, output_dir, parser, debug=False):
    id_dirs = sorted(os.listdir(input_dir))
    id_dirs = id_dirs[:10] if debug else id_dirs
    corpus = []
    logger.info("Start reading codes...")
    
    for id_dir in tqdm(id_dirs):
        logger.debug(f"{input_dir}, {id_dir}")
        # use 3-step seed expansion model for all possible corpus
        expand_graph_path = get_expanded_model_path(input_dir, id_dir)
        if expand_graph_path is None:
            logger.info(f"No file in {expand_graph_path}")
            continue

        # get node embedding
        df_code = get_nodes_text(expand_graph_path)
        codes = df_code["code"]
        embeddings = [parser.get_embedding(code) for code in codes]
        df_code.loc[:, 'embedding'] = embeddings   
        df_code = df_code.drop(columns=['code']) 
        # print(df_code)

        # write embedding to file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_fn = os.path.join(output_dir, f'{id_dir}_{parser.model_name}_embedding.pkl')
        df_code.to_pickle(output_fn)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data', help='input directory')
    parser.add_argument('--output_dir', type=str, default='data', help='output directory')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    pretrain_model_path = '/data0/xiaoyez/CodeContextModel/word2vec.pretrain'
    if not os.path.exists(pretrain_model_path):
        corpus = get_word_corpus(args.input_dir, args.debug)
        parser = build_word_embedding(corpus)
    else:
        parser = LexParser(None, pretrain_model_path=pretrain_model_path)

    embedding_inference(args.input_dir, args.output_dir, parser, args.debug)