import random
import os
import logging
import itertools
import pandas as pd
import xml.etree.ElementTree as ET

from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from utils.xmltree_parser import XMLTreeParser

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger('seed_strategies')

class SeedStrategy(ABC):
    """抽象基类，定义种子选择策略的接口"""
    @abstractmethod
    def generate_seed(self, graph: XMLTreeParser, step: int = 1) -> Optional[List[tuple]]:
        """生成种子节点集合"""
        pass

class CountBasedStrategy(SeedStrategy):
    """按照指定数量随机选择种子的策略"""
    def __init__(self, seed_count: int = None, max_seed_count: int = None):
        """
        Parameters:
            seed_count: 指定要选择的种子节点数量。
            如果为None，则使用len(vertex_ids)-step作为数量（原来的行为）
        """
        self.seed_count = seed_count
        self.max_seed_count = max_seed_count

    def generate_seed(self, graph: XMLTreeParser, step: int = 1) -> Optional[List[tuple]]:
        vertex_ids = []
        # get vertex in code context model
        for vertex in graph.get_vertices():
            if graph.is_origin_vertex(vertex):
                vertex_ids.append(graph.get_vertex_id(vertex))
            
        # 确定种子数量
        seed_size = self.seed_count if self.seed_count is not None else len(vertex_ids) - step
        if self.seed_count is not None and len(vertex_ids) - self.seed_count < 1:   # 确保至少留下一个种子节点用于预测
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

    def generate_seed(self, graph: XMLTreeParser, step: int = 1) -> Optional[List[tuple]]:
        vertex_ids = []
        # get vertex in code context model
        for vertex in graph.get_vertices():
            if graph.is_origin_vertex(vertex):
                vertex_ids.append(graph.get_vertex_id(vertex))
            
        # 确定种子数量
        seed_size = self.seed_count if self.seed_count is not None else len(vertex_ids) - step
        if len(vertex_ids) - seed_size < 1:   # 确保至少留下一个种子节点用于预测
            return None
        
        # select seed_size vertices as seed
        seeds = list(itertools.combinations(vertex_ids, seed_size))
        num_selections = min(self.seed_count, len(seeds))
        selected_seeds = random.sample(seeds, num_selections)
        return selected_seeds
    
class ExperienceBasedStrategy(SeedStrategy):
    """基于程序员经验的种子选择策略"""
    METADATA_DIR = "/data2/xiaoyez/CodeContextModel/bug_report_metadata"
    PROJECT_NAME = "Mylyn"
    CODE_CONTEXT_MODEL_FINAL_DIR = os.path.join(
        "/data0/xiaoyez/CodeContextModel/data",
        PROJECT_NAME if PROJECT_NAME != "Mylyn" else "mylyn"
    )
    def __init__(self, ccm_id: str):
        self.ccm_id = int(ccm_id) # 根据当前的cmm_id，检索相同assignee之前的cmm_id
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
        ccm_metadata = ccm_metadata[ccm_metadata["end_time"] > start_time - timedelta(days=30)]
        logger.info(f"{assignee} has {len(ccm_metadata)} previous ccm")
        for index, row in ccm_metadata.iterrows():
            code_context_model = os.path.join(self.CODE_CONTEXT_MODEL_FINAL_DIR, str(row["ccm_id"]), "code_context_model.xml")
            code_context_model = XMLTreeParser(code_context_model)
            vertices = code_context_model.get_vertices()
            experience_elements.extend([vertex.get("label") for vertex in vertices])
        logger.info(f"#experience elements: {len(experience_elements)}")
        return experience_elements
    
    def generate_seed(self, graph: ET.Element, step: int = 1) -> Tuple[Optional[List[tuple]], Optional[List[List[str]]]]:
        vertices = []
        for vertex in graph.get_vertices():
            if graph.is_origin_vertex(vertex):
                vertices.append(vertex)
        vertices = [vertex for vertex in vertices if vertex.get("label") in self.experience_elements]
        if len(vertices) == 0:
            return None
        experience_seed = tuple(graph.get_vertex_id(vertex) for vertex in vertices)
        logger.info(f"#experience seed: {len(experience_seed)}")
        return [experience_seed]