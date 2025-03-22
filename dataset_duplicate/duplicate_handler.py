import argparse
import json
import os
import re
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
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


# def build_similar_code_clusters(code_snippets, similarity_threshold=0.8):
#     print("<--- start Build similar code clusters...")
#     n = len(code_snippets)
#     uf = unionfind(n)
#
#     total = 0
#     count = 0
#     for i in range(n):
#         for j in range(i + 1, n):
#             total += 1
#             bleu_score = calculate_bleu_nltk(code_snippets[i], code_snippets[j])
#             if bleu_score >= similarity_threshold:
#                 print("find one similar code cluster", i, j, bleu_score)
#                 count += 1
#                 uf.unite(i, j)
#     print("number of compared code snippets: ", total)
#     print("number of similar code snippets: ", count)
#     print("---> end build similar code clusters...")
#     return uf

def build_similar_code_clusters(code_snippets, project="mylyn", similarity_threshold=0.8):
    if True:
        print("<--- start Build similar code clusters...")
        n = len(code_snippets)
        uf = unionfind(n)
        # 读取文件
        with open(f"similar_codes_{project}.txt", "r") as f:
            for line in f:
                # 使用正则表达式匹配两个连续的数字
                match = re.search(r"(\d+)\s+(\d+)", line)
                if match:
                    first_num, second_num = match.groups()
                    print(int(first_num), int(second_num))
                    uf.unite(int(first_num), int(second_num))
        print("---> end build similar code clusters...")
        return uf
    print("<--- start Build similar code clusters...")
    n = len(code_snippets)
    uf = unionfind(n)
    lock = threading.Lock()  # 用于保护 uf 操作的线程安全
    lock1 = threading.Lock()  # 用于保护 uf 操作的线程安全

    cache = dict()

    # 定义线程任务
    def process_pair(i, j):
        if i < 22290:
            return
        # bleu_score = calculate_bleu_nltk(code_snippets[i], code_snippets[j])
        reference_code = code_snippets[i]
        candidate_code = code_snippets[j]
        if len(reference_code) < len(candidate_code):
            reference_code, candidate_code = candidate_code, reference_code
            i, j = j, i

        def get_or_cache(index, code_tokens):
            if index not in cache:
                with lock1:
                    if index not in cache:  # 防止重复写入
                        cache[index] = word_tokenize(code_tokens)
            return cache[index]

        # 使用 get_or_cache 函数简化代码
        reference_tokens = get_or_cache(i, reference_code)
        candidate_tokens = get_or_cache(j, candidate_code)

        smoothing_function = SmoothingFunction().method1
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing_function)

        if bleu_score >= similarity_threshold:
            with lock:  # 确保对 uf 的操作线程安全
                uf.unite(i, j)
                print("find one similar code cluster", i, j, bleu_score)

    # 创建所有组合的索引对
    all_pairs = list(combinations(range(n), 2))
    total = len(all_pairs)

    # 使用线程池执行任务
    with ThreadPoolExecutor() as executor:
        executor.map(lambda pair: process_pair(pair[0], pair[1]), all_pairs)

    print("number of compared code snippets: ", total)
    print("---> end build similar code clusters...")
    return uf


class DuplicateHandler:
    def __init__(self, projects, dataset_path):
        self.duplicate_inner_model_pairs = None
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
                if d == "timer" or d.startswith("model"):
                    continue
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
        ufs = dict()
        index_to_idx = dict()
        for project, models in self.data.items():
            print("handle project: ", project)
            codes = list()
            i_to_i = dict()
            for model in models.values():
                for v in model:
                    i_to_i[v[0]] = len(codes)
                    codes.append(v[1])
            uf = build_similar_code_clusters(codes, project)
            ufs[project] = uf
            index_to_idx[project] = i_to_i
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

        self.duplicate_inner_model_pairs = self.analyzer.find_test_inner_duplicates(data=self.data,
                                                                                    all_index=self.all_index,
                                                                                    similar_code_pairs=self.similar_code_pairs,
                                                                                    idx_to_index=self.idx_to_index)

    def print_result(self):
        print("<--- start print_result")
        print("-------similar_code_pairs-------")
        print(self.similar_code_pairs)
        print("-------idx_to_index-------")
        print(self.idx_to_index)
        print("-------duplicate_model_pairs-------")
        print(self.duplicate_model_pairs)
        print("-------duplicate_inner_model_pairs-------")
        print(self.duplicate_inner_model_pairs)
        print("---> end print_result")


class CodeSimilarityAnalyzer:
    def __init__(self, jaccard_threshold=0.7):
        self.jaccard_threshold = jaccard_threshold

    def calculate_jaccard(self, similar_code_pairs: unionfind, test: set, train: set) -> float:
        """计算Jaccard相似度"""
        if not test or not train:  # 处理空集
            return 0.0
        # 使用集合去重，确保 intersection 计算正确
        intersection = len({te for te in test for tr in train if similar_code_pairs.issame(te, tr)})

        union = len(test) + len(train) - intersection
        return intersection / union if union > 0 else 0.0
        # intersection = 0
        # for te in test:
        #     for tr in train:
        #         intersection += 1 if similar_code_pairs.issame(te, tr) else 0
        # union = len(test) + len(train) - intersection
        # return intersection / union if union > 0 else 0.0

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
                        test_set.add(idx_to_index[project][snippet[0]])
                    train_set = set()
                    for snippet in train:
                        train_set.add(idx_to_index[project][snippet[0]])
                    jaccard = self.calculate_jaccard(similar_code_pairs[project], test_set, train_set)
                    if jaccard >= self.jaccard_threshold:
                        dup.append((test_index, train_index, jaccard))
            duplicates[project] = dup
            print(f"project {project} has {len(v[1])} test models")
            print(f"project {project} has {len(v[0])} train models")
            print(f"project {project} find {len(dup)} duplicate test-train pairs")
            print(f"project {project} find {len(set([i[0] for i in dup]))} duplicated test models")
        print("<--- end find duplicated test models")
        return duplicates

    def find_test_inner_duplicates(self, data, all_index, similar_code_pairs, idx_to_index) -> dict[
        str, list[tuple[int, int, float]]]:
        """找出测试数据集中的自重复数据对"""
        print("<--- start find inner duplicated test models")
        duplicates = dict()

        for project, v in all_index.items():  # model_id
            dup = []
            for i in range(len(v[1])):
                for j in range(i + 1, len(v[1])):
                    test_index = v[1][i]
                    test_index_another = v[1][j]
                    test = data[project][test_index]  # [(idx, code),...]
                    test_another = data[project][test_index_another]
                    test_set = set()
                    for snippet in test:
                        test_set.add(idx_to_index[project][snippet[0]])
                    test_another_set = set()
                    for snippet in test_another:
                        test_another_set.add(idx_to_index[project][snippet[0]])
                    jaccard = self.calculate_jaccard(similar_code_pairs[project], test_set, test_another_set)
                    if jaccard >= self.jaccard_threshold:
                        dup.append((test_index, test_index_another, jaccard))
            duplicates[project] = dup
            print(f"project {project} has {len(v[1])} test models")
            print(f"project {project} find {len(dup)} duplicate test-test pairs")
            print(f"project {project} find {len(set([i[0] for i in dup]))} duplicated test models")
        print("<--- end find inner duplicated test models")
        return duplicates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script takes path strings and projects array.")
    parser.add_argument('--path', type=str, help='dataset path', default="/data0/xiaoyez/CodeContextModel/data/")
    parser.add_argument('--projects', type=str, help='projects to handle', default="mylyn,PDE,Platform")
    args = parser.parse_args()
    projects = args.projects.split(',')

    print(f"dataset path: {args.path}")
    print(f"projects to handle: {projects}")
    handler = DuplicateHandler(projects, args.path)
    handler.handle()
    handler.print_result()
