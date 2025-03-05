import random
import os
import itertools
import xml.etree.ElementTree as ET

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from utils.xmltree_parser import XMLTreeParser

class SeedStrategy(ABC):
    """抽象基类，定义种子选择策略的接口"""
    @abstractmethod
    def generate_seed(self, graph: XMLTreeParser, step: int = 1) -> Optional[List[tuple]]:
        """生成种子节点集合"""
        pass

class CountBasedStrategy(SeedStrategy):
    """按照指定数量随机选择种子的策略"""
    def __init__(self, seed_count: int = None):
        """
        Parameters:
            seed_count: 指定要选择的种子节点数量。
            如果为None，则使用len(vertex_ids)-step作为数量（原来的行为）
        """
        self.seed_count = seed_count

    def generate_seed(self, graph: XMLTreeParser, step: int = 1) -> List[tuple]:
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
        # seeds = list(itertools.combinations(vertex_ids, seed_size))
        # index = random.randint(0, len(seeds) - 1) 
        # seed = seeds[index]
        seed = tuple(random.sample(vertex_ids, seed_size))

        return seed
    
    
class ExperienceBasedStrategy(SeedStrategy):
    """基于程序员经验的种子选择策略"""
    def __init__(self, experience_rules: dict):
        self.rules = experience_rules  # 可以包含各种经验规则

    def generate_seed(self, graph: ET.Element, step: int = 1) -> Tuple[Optional[List[tuple]], Optional[List[List[str]]]]:
        # 实现基于经验规则的种子选择逻辑
        # 这里可以根据具体的经验规则来实现
        pass