import torch
from transformers import AutoModel, AutoTokenizer
import xml.etree.ElementTree as ET
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

def get_nodes_text(expand_graph_path: str) -> pd.DataFrame:
    tree = ET.parse(expand_graph_path)
    root = tree.getroot()
    nodes = root.findall(".//vertex")
    nodes_id = []
    model_dir = expand_graph_path.split('/')[-3]
    codes_path = os.path.dirname(os.path.dirname(expand_graph_path)) + "/my_java_codes.tsv"
    # read tsv file
    df_code = pd.read_csv(codes_path, sep='\t')
    for vertex in nodes:
        node_id = '_'.join([model_dir, vertex.get('kind'), vertex.get('ref_id')]) 
        # node_text = df_code[df_code['id'] == node_id]['code'].values[0]
        # print(len(node_text))
        nodes_id.append(node_id)
    return df_code[df_code['id'].isin(nodes_id)]

class TextEmbedding():
    def __init__(self):
        self.tokenizer = None
        self.model = None
    def get_embedding(self, input_texts: str) -> torch.Tensor:
        pass


class BgeEmbedding(TextEmbedding):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        self.model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5', revision="refs/pr/13")
    
    def __str__(self):
        return "BgeEmbedding"

    def get_embedding(self, input_texts: str) -> torch.Tensor:  
        batch_dict = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = outputs[0][:, 0] # FIXME: bge只需要取第一个token的embedding，不同的模型不一样
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        print(embeddings.shape)
        print("Sentence embeddings:", embeddings)
        return embeddings

def get_tokenizer_data(input_texts):
    model = BgeEmbedding()
    batch_dict = model.tokenizer(input_texts, padding=True, return_tensors='pt')
    input_ids = batch_dict['input_ids']
    # [  240,   281,    33, 15898,    23,    23,  2028]
    # [240, 281,  33, 512,  23,  23, 512]
    token_counts = (input_ids != model.tokenizer.pad_token_id).sum(dim=1)
    return token_counts.tolist()

def get_tokenizer_statistics(input_dir):
    total_token_counts = []
    cnt = 0
    for id_dir in tqdm(os.listdir(input_dir)):
        seed_dir = input_dir + id_dir + "/seed_expanded/"
        if not os.path.exists(seed_dir):
            continue
        file = os.listdir(seed_dir)[random.randint(0, len(os.listdir(seed_dir)) - 1)]
        if file.endswith(".xml"):
            print(f"processing {seed_dir + file}")
            nodes_text = get_nodes_text(seed_dir + file)['code'].tolist()
            token_counts = get_tokenizer_data(nodes_text)
            total_token_counts.extend( [x for x in token_counts if x <= 2000])
        cnt += 1
        if cnt == 1000:
            break
    print(total_token_counts)
    bins = 50  # 可以根据需要调整 bin 的数量
    hist, bin_edges = np.histogram(total_token_counts, bins=bins)
    plt.figure(figsize=(12, 6))
    plt.hist(total_token_counts, bins=bin_edges, color='b', edgecolor='black')
    plt.savefig("token_counts.png")
    plt.show()

def embedding_inference(input_dir):
    model = BgeEmbedding()
    id_dirs = sorted(os.listdir(input_dir))
    for id_dir in id_dirs:
        print(input_dir, id_dir)
        codes_path = os.path.join(input_dir, id_dir, "my_java_codes.tsv")
        print(f"processing {codes_path}")
        df_code = pd.read_csv(codes_path, sep='\t')
        nodes_text = df_code['code'].tolist()
        embeddings = model.get_embedding(nodes_text)
        df_code['embedding'] = embeddings   
        df_code = df_code.drop(columns=['code']) 
        df_code.to_pickle(os.path.join(input_dir, id_dir, f'{model}_embedding.pkl'))
        break
            


if __name__ == '__main__':
    # nodes_text = get_nodes_text("data/mylyn/60/seed_expanded/1_step_seeds_0_expanded_model.xml")
    # model = BgeEmbedding()
    # model.get_embedding(nodes_text)
    input_dir = "data/mylyn/"
    embedding_inference(input_dir)
    
