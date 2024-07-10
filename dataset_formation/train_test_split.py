import os
import json
import random
import argparse

import numpy as np
import xml.etree.ElementTree as ET



def split_dataset_by_ratio(input_dir: str, output_dir: str, train_ratio: int = 0.8):
    """
    返回比例范围内的model，根据first_time排序

    :param input_dir: context models directory
    :param train_ratio: split train ratio
    :param output_dir: train/test index file output directory
    """
    model_dirs = os.listdir(input_dir)
    all_models = []
    for model_dir in model_dirs:
        model_file = os.path.join(input_dir, model_dir, 'code_context_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        tree = ET.parse(model_file)  # 拿到xml树
        # 获取XML文档的根元素
        code_context_model = tree.getroot()
        first_time = code_context_model.get('first_time')
        all_models.append([os.path.join(input_dir, model_dir), first_time])
    all_models = sorted(all_models, key=lambda x: x[1])
    all_models = np.array(all_models)
    train_size = int(0.8 * len(all_models))
    train_models = all_models[:train_size, 0].tolist()
    test_models = all_models[train_size:, 0].tolist()
    # write the training and testing sets to files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with  open(f'{output_dir}/train_index.json', 'w') as f:
        json.dump(train_models, f)
    with  open(f'{output_dir}/test_index.json', 'w') as f:
        json.dump(test_models, f)
    # return m[int(len(m) * start_ratio):int(len(m) * end_ratio), 0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data', help='input directory')
    parser.add_argument('--output_dir', type=str, default='data', help='output directory')
    args = parser.parse_args()
    # split_datasets(args.input_dir, args.output_dir)
    split_dataset_by_ratio(args.input_dir, args.output_dir)
