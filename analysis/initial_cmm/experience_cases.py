import pandas as pd
import json
import os
import sys
import torch
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_context_model.build_dataset import ExpandGraphDataset, split_dataset

AUTHOR_INFO_FILE = "/data0/xiaoyez/CodeContextModel/analysis/crawl_metadata/metadata/bug_report_metadata/Mylyn_ccm_metadata.csv"
EXPEIRENCE_CASES_FILE = "/data0/xiaoyez/CodeContextModel/analysis/initial_cmm/data/initial_ccm/experience_based/experience_based_cases.json"
TRAIN_TEST_CASES_FILE = "/data0/xiaoyez/CodeContextModel/data/train_test_index/mylyn/train_index.json"
TEST_TEST_CASES_FILE = "/data0/xiaoyez/CodeContextModel/data/train_test_index/mylyn/test_index.json"
NEW_TRAIN_TEST_CASES_FILE = "/data2/xiaoyez/CodeContextModel/initial_ccm/experience_based/new_train_index.json"
EMBEDDING_DIR = "/data0/xiaoyez/CodeContextModel/embedding_bge"
EMBEDDING_MODEL = "BgeEmbedding"
OUTPUT_DIR = "/data2/xiaoyez/CodeContextModel/initial_ccm/experience_based/dataset"
AUTHOR_CASES_FILE = "/data2/xiaoyez/CodeContextModel/initial_ccm/experience_based/author_cases.json"

def load_author_info(author_info_file):
    author_info = pd.read_csv(author_info_file)
    return author_info

def load_experience_cases(experience_cases_file):
    experience_cases = json.load(open(experience_cases_file))
    return experience_cases

def load_train_test_cases(train_test_cases_file):
    train_test_cases = json.load(open(train_test_cases_file))
    return train_test_cases

def main():
    df_ccm_metadata = load_author_info(AUTHOR_INFO_FILE)
    experience_cases = load_experience_cases(EXPEIRENCE_CASES_FILE)
    authors = set()
    cases = []
    for case in experience_cases:
        author_info = df_ccm_metadata[df_ccm_metadata["ccm_id"] == int(case)]["assignee"].values[0]
        authors.add(author_info)
        author_info_cases = df_ccm_metadata[df_ccm_metadata["assignee"] == author_info]
        cases.extend(list(author_info_cases["ccm_id"]))
    print(f"cases: {len(set(cases))}")

    author_cases = defaultdict(list)
    for author in authors:
        author_info_cases = df_ccm_metadata[df_ccm_metadata["assignee"] == author]
        for idx, case in author_info_cases.iterrows():
            author_cases[author].append({
                "ccm_id": case["ccm_id"],
                "bug_report_id": case["bug_report_id"]
            })
    json.dump(author_cases, open(AUTHOR_CASES_FILE, "w"))

    # cases = list(set(cases))
    # train_cases = load_train_test_cases(TRAIN_TEST_CASES_FILE)
    # test_cases = load_train_test_cases(TEST_TEST_CASES_FILE)
    # new_train_cases = []
    # print(f"train_test_cases: {len(train_cases)}")
    # for case in train_cases:
    #     case = int(case.split("/")[-1]) 
    #     if case in cases:
    #         continue
    #     new_train_cases.append(f"/data0/xiaoyez/CodeContextModel/data/mylyn/{case}/1_step_seeds_expanded_model.xml")

    # with open(NEW_TRAIN_TEST_CASES_FILE, "w") as f:
    #     json.dump(new_train_cases, f)
    # print(f"new_train_cases: {len(new_train_cases)}")
    # train_data_builder = ExpandGraphDataset(
    #     xml_files=new_train_cases, 
    #     embedding_dir=EMBEDDING_DIR, 
    #     embedding_model=EMBEDDING_MODEL, 
    #     debug=False
    # )

    # train_dataset, valid_dataset = split_dataset(train_data_builder)

    # print(f"train dataset: {len(train_dataset)}")
    # print(f"valid dataset: {len(valid_dataset)}")

    # if not os.path.exists(OUTPUT_DIR):
    #     os.makedirs(OUTPUT_DIR)
    # # write the dataset to disk
    # torch.save(train_dataset, os.path.join(OUTPUT_DIR, 'train_dataset.pt'))
    # torch.save(valid_dataset, os.path.join(OUTPUT_DIR, 'valid_dataset.pt'))


if __name__ == "__main__":
    main()
    