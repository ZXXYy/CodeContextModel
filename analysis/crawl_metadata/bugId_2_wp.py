import os
import sys
import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.xmltree_parser import XMLTreeParser

WORKING_PERIOD_DIR = os.path.join(
    "/data2/shunliu/pythonfile/code_context_model_prediction", 
    "params_validation",
    "working_periods",
    # "periods",
    "code_elements",
    "05",
    # "Mylyn"
)
METADATA_DIR = "/data2/xiaoyez/CodeContextModel/bug_report_metadata"

if __name__ == "__main__":
    bug_id_2_wp = defaultdict(list)
    # for working_period in tqdm(os.listdir(WORKING_PERIOD_DIR)):
    #     if working_period.endswith(".xml"):
    #         xml_path = os.path.join(WORKING_PERIOD_DIR, working_period)
    #         xml_parser = XMLTreeParser(xml_path)
    #         bug_id = xml_parser.get_bug_id()
    #         bug_id_2_wp[bug_id].append(working_period)
    df = pd.read_csv(os.path.join(WORKING_PERIOD_DIR, "working_periods_events.tsv"), sep=",")
    df = df[df["project"] == "Mylyn"]
    for index, row in df.iterrows():
        bug_id = row["bug_id"]
        wp = row["period_index"]
        bug_id_2_wp[bug_id].append(wp)
    
    output_file = os.path.join(METADATA_DIR, "bugId_2_wp.json")
    json.dump(bug_id_2_wp, open(output_file, "w"))










