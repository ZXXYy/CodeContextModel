import re
import os
import sys
import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.xmltree_parser import XMLTreeParser
# /data2/shunliu/pythonfile/code_context_model_prediction/params_validation/repo_vs_commit_order/IQR_code_timestamp/05/Mylyn
PROJECT_NAME = "Mylyn"
CODE_CONTEXT_MODEL_ID_DIR = os.path.join(
    "/data2/shunliu/pythonfile/code_context_model_prediction", 
    "params_validation",
    "repo_vs_commit_order",
    # "periods",
    "IQR_code_timestamp",
    "05",
    PROJECT_NAME
)
# /data2/shunliu/pythonfile/code_context_model_prediction/params_validation/working_periods/code_elements/05/Mylyn/47.xml
WORKING_PERIOD_DIR = os.path.join(
    "/data2/shunliu/pythonfile/code_context_model_prediction", 
    "params_validation",
    "working_periods",
    "code_elements",
    "05",
    PROJECT_NAME
)
# /data0/xiaoyez/CodeContextModel/data/mylyn
CODE_CONTEXT_MODEL_FINAL_DIR = os.path.join(
    "/data0/xiaoyez/CodeContextModel/data",
    PROJECT_NAME if PROJECT_NAME != "Mylyn" else "mylyn"
)
METADATA_DIR = "/data2/xiaoyez/CodeContextModel/bug_report_metadata"

def get_assignee(bug_id):
    metadata_file = os.path.join(METADATA_DIR, "processed", f"{bug_id}.json")
    metadata = json.load(open(metadata_file))
    return metadata["assignee"]

def get_task_type(bug_id):
    metadata_file = os.path.join(METADATA_DIR, "processed", f"{bug_id}.json")
    metadata = json.load(open(metadata_file))
    return metadata["importance"]

if __name__ == "__main__":
    items = []
    for ccm in tqdm(os.listdir(CODE_CONTEXT_MODEL_ID_DIR)):
        if ccm.endswith(".xml"):
            xml_path = os.path.join(CODE_CONTEXT_MODEL_ID_DIR, ccm)
            xml_parser = XMLTreeParser(xml_path)
            bug_id = xml_parser.get_bug_id()
            working_period_id = xml_parser.get_working_period_id()
            start_time = xml_parser.get_start_time()
            end_time = xml_parser.get_end_time()
            assignee = get_assignee(bug_id)
            # replace with x votes
            task_type = re.sub(r'with \d+ votes?', '', get_task_type(bug_id)).strip()
            task_type = re.sub(r'P\d?', '', task_type).strip()
            items.append({
                "bug_report_id": bug_id,
                "working_period_id": working_period_id,
                "ccm_id": ccm.split(".")[0],
                "assignee": assignee,
                "task_type": task_type,
                "start_time": start_time,
                "end_time": end_time
            })
    df = pd.DataFrame(items)
    df = df[df["ccm_id"].isin(os.listdir(CODE_CONTEXT_MODEL_FINAL_DIR))]
    print(df["task_type"].value_counts())
    print(df["assignee"].value_counts())
    print(len(df))
    print(df.head())
    df.to_csv(os.path.join(METADATA_DIR, f"{PROJECT_NAME}_ccm_metadata.csv"), index=False)
    










