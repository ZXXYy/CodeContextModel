import torch
from transformers import AutoModel, AutoTokenizer
import xml.etree.ElementTree as ET
import os
import pandas as pd

def get_nodes_text(expand_graph_path: str) -> str:
    tree = ET.parse(expand_graph_path)
    root = tree.getroot()
    nodes = root.findall(".//vertex")
    nodes_text = []
    model_dir = expand_graph_path.split('/')[-3]
    codes_path = os.path.dirname(os.path.dirname(expand_graph_path)) + "/my_java_codes.tsv"
    # read tsv file
    df_code = pd.read_csv(codes_path, sep='\t')
    for vertex in nodes:
        node_id = '_'.join([model_dir, vertex.get('kind'), vertex.get('ref_id')]) 
        node_text = df_code[df_code['id'] == node_id]['code'].values[0]
        print(len(node_text))
        nodes_text.append(node_text)
    return nodes_text

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

    def get_embedding(self, input_texts: str) -> torch.Tensor:  
        batch_dict = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = outputs[0][:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        print(embeddings.shape)
        print("Sentence embeddings:", embeddings)


if __name__ == '__main__':
    nodes_text = get_nodes_text("data/mylyn/60/seed_expanded/1_step_seeds_0_expanded_model.xml")
    model = BgeEmbedding()
    model.get_embedding(nodes_text)