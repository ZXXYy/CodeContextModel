import xml.etree.ElementTree as ET

class XMLTreeParser:
    def __init__(self, xml):
        if isinstance(xml, str):
            self.tree = ET.parse(xml)
            self.root = self.tree.getroot()
        elif isinstance(xml, ET.ElementTree):
            self.tree = xml
            self.root = self.tree.getroot()
        else:
            raise ValueError(f"Invalid input type: {type(xml)}")

    def get_root(self):
        return self.root

    def get_graphs(self):
        return self.root.findall('graph')
    
    def get_edges(self):
        edges = []
        for graph in self.get_graphs():
            edges.extend(graph.find('edges').findall('edge'))
        return edges

    def get_vertices(self):
        vertices = []
        for graph in self.get_graphs():
            vertices.extend(graph.find('vertices').findall('vertex'))
        return vertices
    
    def get_vertices_count(self):
        graphs = self.get_graphs()
        vertices_count = []
        for graph in graphs:
            vertices = graph.find('vertices')
            if vertices is not None:
                vertices_count.append(int(vertices.get('total', 0)))
        return vertices_count
    
    def get_events(self):
        return self.root.findall('event')

    def get_event_end_time(self):
        return self.root.find('timestamp').get('last')

    def get_vertex_id(self, vertex):
        return vertex.get('id')
    
    def get_bug_id(self):
        return self.root.find('bug_id').text
    
    def get_working_period_id(self):
        return self.root.find('id').text
    
    def get_start_time(self):
        return self.root.find('timestamps').find('first').text
    
    def get_end_time(self):
        return self.root.find('timestamps').find('last').text
    
    def is_origin_vertex(self, vertex):
        return vertex.get('origin') == '1'
    
    def display_graph(self):
        pass

    # 递归检查整个树中的None值
    def check_element_recursively(self, element, path=""):
        current_path = f"{path}/{element.tag}"
        # 检查当前元素的属性
        for key, value in element.attrib.items():
            assert value is not None, f"Found None value at {current_path}, attribute: {key}, id: {element.get('id', 'unknown')}"
            
        # 递归检查所有子元素
        for child in element:
            self.check_element_recursively(child, current_path)