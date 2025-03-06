import os
import json
import logging
import requests

from tqdm import tqdm
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger('crawl_metadata')

METADATA_DIR = "/data2/xiaoyez/CodeContextModel/bug_report_metadata"

def get_bug_report_ids(bug_report_dir: str):
    bug_report_ids = os.listdir(bug_report_dir)
    return bug_report_ids

# crawl the bug report from the bug report id
def get_bug_repot(bug_report_id: str):
    if not os.path.exists(os.path.join(METADATA_DIR, "raw")):
        os.makedirs(os.path.join(METADATA_DIR, "raw"))
    output_file = os.path.join(METADATA_DIR, "raw", f"{bug_report_id}.html")
    if os.path.exists(output_file):
        return
    try:
        url = f"https://bugs.eclipse.org/bugs/show_bug.cgi?id={bug_report_id}"
        logger.info(url) 
        response = requests.get(url)
        with open(output_file, "w") as f:
            f.write(response.text)
    except Exception as e:
        logger.error(f"Error crawling bug report {bug_report_id}: {e}")

def handle_bug_report(bug_report_id: str):
    raw_file = os.path.join(METADATA_DIR, "raw", f"{bug_report_id}.html")
    if not os.path.exists(raw_file):
        return
    with open(raw_file, "r") as f:
        content = f.read()
    soup = BeautifulSoup(content, "html.parser")

    bug_metadata = {}
    bug_metadata["id"] = bug_report_id
    bug_metadata["summary"] = soup.find("span", id="short_desc_nonedit_display").text.strip()
    th_element = soup.find("th", id="field_label_assigned_to")
    bug_metadata["assignee"] = th_element.find_parent("tr").find("td").text.strip()
    vote_element = soup.find("span", id="votes_container")
    importance_strs = vote_element.find_parent("td").text.split("\n")
    importance_strs = [s.strip() for s in importance_strs if s.strip()]
    bug_metadata["importance"] = " ".join(importance_strs[:-1])
    comment_elements = soup.find_all("div", class_="bz_comment")
    comments = []
    for comment_element in comment_elements:
        comment_person = comment_element.find("span", class_="bz_comment_user").text.strip()
        comment_content = comment_element.find("pre", class_="bz_comment_text").text.strip()
        comments.append({
            "person": comment_person,
            "content": comment_content
        })
    bug_metadata["comments"] = comments
    
    logger.debug(bug_metadata["summary"])
    logger.debug(bug_metadata["assignee"])
    logger.debug(bug_metadata["importance"])
    logger.debug(bug_metadata["comments"])
    
    if not os.path.exists(os.path.join(METADATA_DIR, "processed")):
        os.makedirs(os.path.join(METADATA_DIR, "processed"))
    output_file = os.path.join(METADATA_DIR, "processed", f"{bug_report_id}.json")
    if os.path.exists(output_file):
        return
    with open(output_file, "w") as f:
        json.dump(bug_metadata, f)
    return bug_metadata

if __name__ == "__main__":
    bug_report_dir = "/data2/shunliu/pythonfile/code_context_model_prediction/2023_dataset/mylyn_zip/Mylyn"
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    bug_report_ids = get_bug_report_ids(bug_report_dir)
    for bug_report_id in tqdm(bug_report_ids):
        get_bug_repot(bug_report_id)

    for bug_report_id in bug_report_ids:
        handle_bug_report(bug_report_id)