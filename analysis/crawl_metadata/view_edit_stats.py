import os
import sys
import logging
import pandas as pd
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from standard_element import solve_one
from utils.xmltree_parser import XMLTreeParser


logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger('view_edit_stats')

PROJECT_NAME = "Platform"
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

def get_view_edit_stats(ccm_id: str, wp_id: str):
    wp = os.path.join(WORKING_PERIOD_DIR, f"{wp_id}.xml")
    ccm = os.path.join(CODE_CONTEXT_MODEL_FINAL_DIR, str(ccm_id), "code_context_model.xml")
    wp_parser = XMLTreeParser(wp)
    ccm_parser = XMLTreeParser(ccm)

    vertices = ccm_parser.get_vertices()
    vertices_labels = [vertex.get("label") for vertex in vertices]
    events = wp_parser.get_events()
    stats = []
    for vertex_label in vertices_labels:
        event_start_date = None
        view_times, edit_times = [], []
        view_count, edit_count = 0, 0
        for event_idx, event in enumerate(events):
            event_structure_handle = event.get("event_structure_handle", "")
            event_structure_handle = solve_one(event_structure_handle)[1]
            if event_structure_handle != vertex_label:
                continue
            event_kind = event.get("event_kind", "")
            if event_idx == len(events) - 1:
                end_time = datetime.strptime(wp_parser.get_event_end_time(), "%Y-%m-%d %H:%M:%S")
                start_time = datetime.strptime(event.get("event_start_date"), "%Y-%m-%d %H:%M:%S")
                event_time = (end_time - start_time).total_seconds()
            else:
                end_time = datetime.strptime(events[event_idx + 1].get("event_start_date"), "%Y-%m-%d %H:%M:%S")
                start_time = datetime.strptime(event.get("event_start_date"), "%Y-%m-%d %H:%M:%S")
                event_time = (end_time - start_time).total_seconds()
            if event_kind == "selection":
                view_times.append(event_time)
                view_count += 1
            elif event_kind == "edit":
                edit_times.append(event_time)
                edit_count += 1
            logger.debug(
                f"{vertex_label}:\n"
                f"  #views: {view_count}\n"
                f"  #edits: {edit_count}\n" 
            f"  avg_view: {sum(view_times) / view_count if view_count > 0 else 0:.2f}\n"
            f"  avg_edit: {sum(edit_times) / edit_count if edit_count > 0 else 0:.2f}"
        )
        # if vertex_label == solve_one(events[len(events) - 1].get("event_structure_handle", ""))[1]:
        #     if edit_count == 0 and view_count == 1:
        #         continue
        stats.append({
            "ccm_id": ccm_id,
            "wp_id": wp_id,
            "vertex_label": vertex_label,
            "view_count": view_count,
            "edit_count": edit_count,
            "sum_view": sum(view_times),
            "sum_edit": sum(edit_times)
        })
    return stats
if __name__ == "__main__":
    ccm_metadata = pd.read_csv(os.path.join(METADATA_DIR, f"{PROJECT_NAME}_ccm_metadata.csv"))
    total_stats = []
    for index, row in tqdm(ccm_metadata.iterrows(), total=len(ccm_metadata)):
        ccm_id = row["ccm_id"]
        wp_id = row["working_period_id"]
        stats = get_view_edit_stats(ccm_id, wp_id)
        total_stats.extend(stats)

    total_stats_df = pd.DataFrame(total_stats)
    total_stats_df.to_csv(os.path.join(METADATA_DIR, f"{PROJECT_NAME}_view_edit_stats.csv"), index=False)
    
    VIEW_THRESHOLD = 3
    VIEW_COUNT_THRESHOLD = 1
    print(total_stats_df.describe())
    non_zero_edit_count = len(total_stats_df[total_stats_df['edit_count'] > 0])
    print(f"non-zero edit count: {non_zero_edit_count}")
    view_only_df = total_stats_df.loc[total_stats_df['edit_count'] == 0]
    print(f"view only stats:\n {view_only_df.describe()}")
    # zero_edit_view_gt_count = len(total_stats_df.loc[(total_stats_df['edit_count'] == 0) & (total_stats_df['view_count'] > VIEW_COUNT_THRESHOLD)])
    # print(f"zero edit but view count > {VIEW_COUNT_THRESHOLD}: {zero_edit_view_gt_count}")
    # df_zero_edit_view_lt = total_stats_df.loc[(total_stats_df['edit_count'] == 0) & (total_stats_df['view_count'] <= VIEW_COUNT_THRESHOLD)]
    # zero_edit_view_lt_avg_view_gt_threshold_count = len(df_zero_edit_view_lt.loc[df_zero_edit_view_lt['sum_view'] > VIEW_THRESHOLD])
    # print(f"zero edit but view count <= {VIEW_COUNT_THRESHOLD} and sum view > {VIEW_THRESHOLD}s: {zero_edit_view_lt_avg_view_gt_threshold_count}")
    # # print(df_zero_edit_view_lt_3.describe())
    # non_misnavigation_count = non_zero_edit_count + zero_edit_view_gt_count + zero_edit_view_lt_avg_view_gt_threshold_count
    # print(f"total: {non_misnavigation_count} / {len(total_stats_df)} {non_misnavigation_count / len(total_stats_df):.2%}")
    # df_misnavigation = df_zero_edit_view_lt.loc[df_zero_edit_view_lt['sum_view'] < VIEW_THRESHOLD]
    # df_misnavigation.to_csv(os.path.join(METADATA_DIR, f"{PROJECT_NAME}_misnavigation.csv"), index=False)
    # print(len(df_misnavigation["ccm_id"].unique()))
