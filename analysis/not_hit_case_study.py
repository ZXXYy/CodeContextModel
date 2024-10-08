

import os
import json
import subprocess
import xml.etree.ElementTree as ET

dataset_fn = "/data0/xiaoyez/CodeContextModel/data/train_test_index/mylyn/test_index.json"
original_dataset_dir = "/data0/xiaoyez/CodeContextModel/data/mylyn"

case_indices = [4981, 3717, 4782, 4585, 5411, 5467, 5478, 5514, 4598, 5770]
# case_indices = [5270]

# with open(dataset_fn, 'r', encoding='utf-8') as file:
#     test_datas = json.load(file)  # 使用 json.load() 读取 JSON 数据

num_vertex = []

for ind in case_indices:
    print(ind)

    # file_id = test_datas[ind].split('/')[-1]

    file_dir = os.path.join(original_dataset_dir, f"{ind}")
    big_one_step_seed_file_fn = os.path.join(file_dir, f"big_1_step_expanded_model.xml")
    one_step_seed_file_fn = os.path.join(file_dir, f"1_step_seeds_expanded_model.xml")
    # get vertex num
    tree = ET.parse(one_step_seed_file_fn)
    root = tree.getroot()
    # 获取顶点
    graph_element = root.find('graph')
    # 获取顶点
    vertices = graph_element.find('vertices').findall('vertex')
    num_vertex.append(len(vertices))

#     try:
#         script_command = f"""\
# python dataset_formation/generate_seed_graph_data.py \
# --input_dir {one_step_seed_file_fn} \
# --action display\
# """
#         script_command = script_command.split(" ")
#         print(script_command)
#         big_script_command = f"""\
# python dataset_formation/generate_seed_graph_data.py \
# --input_dir {big_one_step_seed_file_fn} \
# --action display\
# """
#         big_script_command = big_script_command.split(" ")
#         print(big_script_command)

#         subprocess.run(
#             script_command, capture_output=True, text=True
#         )
#         subprocess.run(
#             big_script_command, capture_output=True, text=True
#         )

#     except Exception as e:
#         print(e)
    

print(num_vertex)
print(f"min: {min(num_vertex)}, max: {max(num_vertex)}")
print(f"average: {sum(num_vertex) / len(num_vertex)}")
print(f"median: {sorted(num_vertex)[len(num_vertex) // 2]}")