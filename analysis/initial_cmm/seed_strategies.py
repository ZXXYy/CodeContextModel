import random
import os
import sys
import logging
import itertools
import pandas as pd
import xml.etree.ElementTree as ET

from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from utils.xmltree_parser import XMLTreeParser
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.crawl_metadata.standard_element import solve_one

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger('seed_strategies')

PROJECT_NAME = "Mylyn"
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

class SeedStrategy(ABC):
    """抽象基类，定义种子选择策略的接口"""
    @abstractmethod
    def generate_seed(self, graph: XMLTreeParser, step: int = 1) -> Optional[List[tuple]]:
        """生成种子节点集合"""
        pass

class CountBasedStrategy(SeedStrategy):
    """按照指定数量随机选择种子的策略"""
    def __init__(self, seed_count: int = None, percentage: float = None, max_seed_count: int = None):
        """
        Parameters:
            seed_count: 指定要选择的种子节点数量。
            如果为None，则使用len(vertex_ids)-step作为数量（原来的行为）
        """
        self.seed_count = seed_count
        self.max_seed_count = max_seed_count
        self.percentage = percentage

    def generate_seed(self, graph: XMLTreeParser, step: int = 1) -> Optional[List[tuple]]:
        vertex_ids = []
        # get vertex in code context model
        for vertex in graph.get_vertices():
            if graph.is_origin_vertex(vertex):
                vertex_ids.append(graph.get_vertex_id(vertex))
        if self.percentage is not None:
            seed_size = int(len(vertex_ids) * self.percentage)
            logger.info(f"seed_size: {seed_size}, percentage: {self.percentage}")
        else:
            seed_size = self.seed_count if self.seed_count is not None else len(vertex_ids) - step
        logger.debug(f"seed_size: {seed_size}, seed_count: {self.seed_count}, max_seed_count: {self.max_seed_count}")
        logger.debug(f"len(vertex_ids): {len(vertex_ids)}")
        if self.seed_count is not None and len(vertex_ids) - self.seed_count < 1:   # 确保至少留下一个种子节点用于预测
            return None
        if self.max_seed_count is not None and len(vertex_ids) - self.max_seed_count < 1:
            return None
        if seed_size < 1:
            return None
        # select seed_size vertices as seed
        # seeds = list(itertools.combinations(vertex_ids, seed_size))
        # index = random.randint(0, len(seeds) - 1) 
        # seed = seeds[index]
        seed = tuple(random.sample(vertex_ids, seed_size))

        return [seed]
    
class OrderBasedStrategy(SeedStrategy):
    """按照指定顺序选择种子的策略"""
    def __init__(self, seed_count: int = 5):
        self.seed_count = seed_count

    def get_chronology_nodes(self, test_case: str) -> dict[str, datetime]:
        df_ccm_metadata = pd.read_csv(os.path.join(METADATA_DIR, f"{PROJECT_NAME}_ccm_metadata.csv"))
        test_case_id = int(test_case.split("/")[-1])
        logger.debug(f"{df_ccm_metadata[df_ccm_metadata['ccm_id'] == test_case_id]}")
        wp_id = df_ccm_metadata[df_ccm_metadata["ccm_id"] == test_case_id]["working_period_id"].values[0]
        wp_path = os.path.join(WORKING_PERIOD_DIR, f"{wp_id}.xml")
        wp = XMLTreeParser(wp_path)
        wp_events = wp.get_events()
        chronology_nodes = {}
        # get chronology nodes
        for event in wp_events:
            event_id = event.get("event_structure_handle")
            event_structure_handle = solve_one(event_id)[1]
            event_time = datetime.strptime(event.get("event_start_date"), "%Y-%m-%d %H:%M:%S")
            if event_structure_handle not in chronology_nodes:
                chronology_nodes[event_structure_handle] = event_time
            else:
                if event_time > chronology_nodes[event_structure_handle]:
                    chronology_nodes[event_structure_handle] = event_time
        chronology_nodes = sorted(chronology_nodes.items(), key=lambda x: x[1])
        return chronology_nodes

    def generate_seed(self, graph: XMLTreeParser, test_case: str) -> Optional[List[tuple]]:
        chronology_nodes = self.get_chronology_nodes(test_case)
        logger.debug(f"len(chronology_nodes): {len(chronology_nodes)}")
        origin_nodes = {}
        # get vertex in code context model
        total_origin_vertex_cnt = 0
        for vertex in graph.get_vertices():
            if graph.is_origin_vertex(vertex):
                total_origin_vertex_cnt += 1
                flag = False
                # get vertex accessed time
                for chronology_node in chronology_nodes:
                    if vertex.get("label") == chronology_node[0]:
                        origin_nodes[vertex.get("id")] = chronology_node[1]
                        flag = True
                        break
                if not flag:
                    logger.info(f"vertex {vertex.get('id')} not in chronology_nodes")
                
        # sort origin_nodes by accessed time
        origin_nodes = sorted(origin_nodes.items(), key=lambda x: x[1])
        selected_seeds = [origin_node[0] for origin_node in origin_nodes[:-1]]
        # logger.info(f"selected_seeds: {len(selected_seeds)}, total_vertex_cnt: {total_origin_vertex_cnt}")
        return [tuple(selected_seeds)]
    
class ExperienceBasedStrategy(SeedStrategy):
    """基于程序员经验的种子选择策略"""
    METADATA_DIR = "/data2/xiaoyez/CodeContextModel/bug_report_metadata"
    PROJECT_NAME = "Mylyn"
    CODE_CONTEXT_MODEL_FINAL_DIR = os.path.join(
        "/data0/xiaoyez/CodeContextModel/data",
        PROJECT_NAME if PROJECT_NAME != "Mylyn" else "mylyn"
    )
    def __init__(self, ccm_id: str, delta_days: int = 7):
        self.ccm_id = int(ccm_id) # 根据当前的cmm_id，检索相同assignee之前的cmm_id
        self.delta_days = delta_days
        self.experience_elements = self.load_experience_elements()


    def load_experience_elements(self):
        experience_elements = []
        ccm_metadata = pd.read_csv(os.path.join(self.METADATA_DIR, f"{self.PROJECT_NAME}_ccm_metadata.csv"))
        assignee = ccm_metadata[ccm_metadata["ccm_id"] == self.ccm_id]["assignee"].values[0]
        start_time = ccm_metadata[ccm_metadata["ccm_id"] == self.ccm_id]["start_time"].values[0]
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

        # 检索相同assignee之前的cmm_id
        ccm_metadata = ccm_metadata[ccm_metadata["assignee"] == assignee]
        ccm_metadata["end_time"] = pd.to_datetime(ccm_metadata["end_time"])
        ccm_metadata = ccm_metadata[ccm_metadata["end_time"] < start_time]
        ccm_metadata = ccm_metadata[ccm_metadata["end_time"] > start_time - timedelta(days=self.delta_days)]
        logger.debug(f"{assignee} has {len(ccm_metadata)} previous ccm")
        for index, row in ccm_metadata.iterrows():
            code_context_model = os.path.join(self.CODE_CONTEXT_MODEL_FINAL_DIR, str(row["ccm_id"]), "code_context_model.xml")
            code_context_model = XMLTreeParser(code_context_model)
            vertices = code_context_model.get_vertices()
            experience_elements.extend([vertex.get("label") for vertex in vertices])
        logger.debug(f"#experience elements: {len(experience_elements)}")
        return experience_elements
    
    def generate_seed(self, graph: ET.Element, step: int = 1) -> Tuple[Optional[List[tuple]], Optional[List[List[str]]]]:
        vertices = []
        origin_vertex_cnt = 0
        for vertex in graph.get_vertices():
            if graph.is_origin_vertex(vertex):
                origin_vertex_cnt += 1
                vertices.append(vertex)
        vertices = [vertex for vertex in vertices if vertex.get("label") in self.experience_elements]
        if len(vertices) == 0:
            return None
        if origin_vertex_cnt - len(vertices) < 1:
            return None
        experience_seed = tuple(graph.get_vertex_id(vertex) for vertex in vertices)
        logger.debug(f"#experience seed: {len(experience_seed)}")
        return [experience_seed]