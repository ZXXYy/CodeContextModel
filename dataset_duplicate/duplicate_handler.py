import argparse
import json
import os
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from os import path

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from unionfind import unionfind


def calculate_bleu_nltk(reference_code, candidate_code):
    # 保证 reference 是长的那个
    if len(reference_code) < len(candidate_code):
        reference_code, candidate_code = candidate_code, reference_code

    reference_tokens = word_tokenize(reference_code)
    candidate_tokens = word_tokenize(candidate_code)

    smoothing_function = SmoothingFunction().method1
    score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing_function)
    return score


def build_similar_code_clusters(code_snippets, similarity_threshold=0.8):
    print("<--- start Build similar code clusters...")
    n = len(code_snippets)
    uf = unionfind(n)

    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            bleu_score = calculate_bleu_nltk(code_snippets[i], code_snippets[j])
            if bleu_score >= similarity_threshold:
                print("find one similar code cluster", i, j, bleu_score)
                count += 1
                uf.unite(i, j)
    print("number of code snippets: ", n)
    print("number of similar code snippets: ", count)
    print("---> end build similar code clusters...")
    return uf


class DuplicateHandler:
    def __init__(self, projects, dataset_path):
        self.duplicate_model_pairs = None
        self.analyzer = None
        self.idx_to_index = None
        self.similar_code_pairs = None
        self.all_index = None
        self.data = None
        self.projects: list[str] = projects
        self.dataset_files: dict[str, str] = dict()
        self.index_files: dict[str, tuple[str, str]] = dict()
        for project in projects:
            self.dataset_files[project] = path.join(dataset_path, project)
            self.index_files[project] = (path.join(dataset_path, "train_test_index", project,
                                                   'train_index.json'),
                                         path.join(dataset_path, "train_test_index", project,
                                                   'test_index.json'))

    def load_all_index(self) -> dict[str, tuple[list[str], list[str]]]:
        print("<--- start load all index")
        all_indexes = dict()
        for project, v in self.index_files.items():
            print("handle train index project: ", project)
            train = list()
            with open(v[0], 'r') as f:
                train = [i[i.rindex("/") + 1:] for i in json.load(f)]
            print("handle test index project: ", project)
            with open(v[1], 'r') as f:
                test = [i[i.rindex("/") + 1:] for i in json.load(f)]
            all_indexes[project] = (train, test)
        print("<--- end load all index")
        return all_indexes

    def load_all_dataset(self) -> dict[str, dict[str, list[tuple[str, str]]]]:
        """加载所有的model数据，{project: {model_id: [(idx, code), ...]}"""
        print("<--- start load all dataset")
        all_models = dict()
        for project, v in self.dataset_files.items():
            print("handle project: ", project)
            dirs = os.listdir(str(v))
            t = dict()
            for d in dirs:
                id_code = list()
                codes_path = path.join(str(v), d, 'my_java_codes.tsv')
                java_codes = pd.read_csv(codes_path, delimiter='\t')
                model_file = path.join(str(v), d, "code_context_model.xml")
                tree = ET.parse(model_file)
                code_context_model = tree.getroot()
                graphs = code_context_model.findall("graph")
                for graph in graphs:
                    vertices = graph.find('vertices')
                    vertex_list = vertices.findall('vertex')
                    for vertex in vertex_list:
                        idx = d + '_' + vertex.get('kind') + '_' + vertex.get('ref_id')
                        code = java_codes[java_codes['id'] == idx]['code'].iloc[0]
                        id_code.append((idx, str(code)))
                t[d] = id_code
            all_models[project] = t
        print("---> end load all dataset")
        return all_models

    def build_similar_code_pairs(self):
        print("<--- start build similar code_pairs")
        ufs = dict[str, unionfind]
        index_to_idx = dict()
        for project, models in self.data.items():
            print("handle project: ", project)
            codes = list()
            for model in models.values():
                for v in model:
                    index_to_idx[v[0]] = len(codes)
                    codes.append(v[1])
            uf = build_similar_code_clusters(codes)
            ufs[project] = uf
        print("---> end build similar code_pairs")
        return ufs, index_to_idx

    def handle(self):
        # get data
        self.data = self.load_all_dataset()
        self.all_index = self.load_all_index()

        # build_similar_code_pairs
        self.similar_code_pairs, self.idx_to_index = self.build_similar_code_pairs()

        # create CodeSimilarityAnalyzer
        self.analyzer = CodeSimilarityAnalyzer()
        self.duplicate_model_pairs = self.analyzer.find_duplicates(data=self.data, all_index=self.all_index,
                                                                   similar_code_pairs=self.similar_code_pairs,
                                                                   idx_to_index=self.idx_to_index)

    def print_result(self):
        print(self.similar_code_pairs)
        print(self.duplicate_model_pairs)


class CodeSimilarityAnalyzer:
    def __init__(self, jaccard_threshold=0.7):
        self.jaccard_threshold = jaccard_threshold

    def calculate_jaccard(self, similar_code_pairs: unionfind, test: set, train: set) -> float:
        """计算Jaccard相似度"""
        if not test or not train:  # 处理空集
            return 0.0
        intersection = 0
        for te in test:
            for tr in train:
                intersection += 1 if similar_code_pairs.issame(te, tr) else 0
        union = len(test.union(train))
        return intersection / union if union > 0 else 0.0

    def find_duplicates(self, data, all_index, similar_code_pairs, idx_to_index) -> dict[
        str, list[tuple[int, int, float]]]:
        """找出数据集中的重复数据对"""
        print("<--- start find duplicated test models based on train models")
        duplicates = dict()

        for project, v in all_index.items():  # model_id
            dup = []
            for test_index in v[1]:  # 遍历测试集
                for train_index in v[0]:  # 匹配训练集
                    test = data[project][test_index]  # [(idx, code),...]
                    train = data[project][train_index]
                    test_set = set()
                    for snippet in test:
                        test_set.add(idx_to_index[snippet[0]])
                    train_set = set()
                    for snippet in train:
                        train_set.add(idx_to_index[snippet[0]])
                    jaccard = self.calculate_jaccard(similar_code_pairs, test_set, train_set)
                    if jaccard >= self.jaccard_threshold:
                        dup.append((v[1], v[0], jaccard))
            duplicates[project] = dup
            print(f"project {project} find {len(dup)} duplicate test-train pairs")
            print(f"project {project} find {len(set([i[0] for i in dup]))} duplicated test models")
        print("<--- end find duplicated test models")
        return duplicates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script takes two strings and one array.")
    parser.add_argument('--path', type=str, help='dataset path', default="/data0/xiaoyez/CodeContextModel/data/")
    parser.add_argument('--output', type=str, help='output file', default="output.txt")
    parser.add_argument('--projects', type=str, help='projects to handle', default="mylyn,PDE,Platform")
    args = parser.parse_args()
    projects = args.projects.split(',')

    with open(args.output, 'a', buffering=1) as f:
        # 在 with 块中重定向输出
        with redirect_stdout(f):
            print(f"dataset path: {args.path}")
            print(f"output file: {args.output}")
            print(f"projects to handle: {projects}")
            handler = DuplicateHandler(projects, args.path)
            handler.handle()
            handler.print_result()
