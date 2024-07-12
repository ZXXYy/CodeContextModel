import os
import json

import xml.etree.ElementTree as ET
import pandas as pd

from tqdm import tqdm

def assign_stereotype(dataset_dir: str, stereotype_path: str, step: int):
    stereotypes = pd.read_csv(stereotype_path, sep=' ')
    print(stereotypes)
    # 读取code context model
    # train_model_dirs = json.loads(open(f'{dataset_dir}/train_index.json').read())
    test_model_dirs = json.loads(open(f'{dataset_dir}/test_index.json').read())
    model_dirs = test_model_dirs
    
    # index = model_dir_list.index('459')
    all_stereotypes = {'FIELD'}
    for model_dir in tqdm(model_dirs):
        model_file = os.path.join(model_dir, f'new_{step}_step_expanded_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        tree = ET.parse(model_file)  # 拿到xml树
        # 获取XML文档的根元素
        code_context_model = tree.getroot()
        sub_stereotypes = stereotypes[stereotypes['model'] == int(model_dir.split('/')[-1])]
        # if len(sub_stereotypes) == 0:
        #     sub_stereotypes = stereotypes
        # print(len(sub_stereotypes))
        graphs = code_context_model.findall("graph")
        for graph in graphs:
            repo = graph.get('repo_name')
            sub_sub_stereotypes = sub_stereotypes[sub_stereotypes['project'] == repo]
            # print(len(sub_sub_stereotypes))
            vertices = graph.find('vertices')
            vertex_list = vertices.findall('vertex')
            for vertex in vertex_list:
                kind, label = vertex.get('kind'), vertex.get('label')
                label = label.replace("final ", '').replace(' ', '').replace('...', '').replace(
                    '@SuppressWarnings("unchecked")', '')
                # print(label)
                if kind == 'variable':
                    vertex.set('stereotype', 'FIELD')
                else:
                    s = sub_sub_stereotypes[sub_sub_stereotypes['label'].str.contains(label, regex=False)]
                    if len(s) != 0:
                        # print(s['stereotype'])
                        vertex.set('stereotype', s['stereotype'].values[0])
                        all_stereotypes.add(s['stereotype'].values[0])
                    else:
                        if kind == 'function':
                            func = label[label.rfind('.') + 1:]
                            label = label[:label.rfind('.')]
                            cla = label[label.rfind('.') + 1:]
                            s = sub_sub_stereotypes[
                                sub_sub_stereotypes['label'].str.contains(f'{cla}.{func}', regex=False)]
                            if len(s) != 0:
                                # print(s['stereotype'].values[0])
                                vertex.set('stereotype', s['stereotype'].values[0])
                                all_stereotypes.add(s['stereotype'].values[0])
                            else:
                                label = label[:label.rfind('.')]
                                pack = label[label.rfind('.') + 1:]
                                s = sub_sub_stereotypes[
                                    sub_sub_stereotypes['label'].str.contains(f'{pack}.{cla}.{func}', regex=False)]
                                if len(s) != 0:
                                    # print(s['stereotype'].values[0])
                                    vertex.set('stereotype', s['stereotype'].values[0])
                                    all_stereotypes.add(s['stereotype'].values[0])
                                else:
                                    print(f'NOTFOUND-method-{vertex.get("label")}-{pack}.{cla}.{func}')
                                    vertex.set('stereotype', 'NOTFOUND')
                                    all_stereotypes.add('NOTFOUND')
                        else:
                            cla = label[label.rfind('.') + 1:]
                            s = sub_sub_stereotypes[sub_sub_stereotypes['label'].str.contains(cla, regex=False)]
                            if len(s) != 0:
                                # print(s['stereotype'].values[0])
                                vertex.set('stereotype', s['stereotype'].values[0])
                                all_stereotypes.add(s['stereotype'].values[0])
                            else:
                                label = label[:label.rfind('.')]
                                pack = label[label.rfind('.') + 1:]
                                s = sub_sub_stereotypes[
                                    sub_sub_stereotypes['label'].str.contains(f'{pack}.{cla}', regex=False)]
                                if len(s) != 0:
                                    # print(s['stereotype'].values[0])
                                    vertex.set('stereotype', s['stereotype'].values[0])
                                    all_stereotypes.add(s['stereotype'].values[0])
                                else:
                                    print(f'NOTFOUND-class-{vertex.get("label")}-{pack}.{cla}')
                                    vertex.set('stereotype', 'NOTFOUND')
                                    all_stereotypes.add('NOTFOUND')
        if os.path.exists(os.path.join(model_dir, f'{step}_step_expanded_model_with_stereotype.xml')):
            os.remove(os.path.join(model_dir, f'{step}_step_expanded_model_with_stereotype.xml'))
        tree.write(os.path.join(model_dir, f'new_{step}_step_expanded_model.xml'))
        # tree.write(join(model_path, 'code_context_model.xml'))
        print('stereotype {} code context model over~~~~~~~~~~~~'.format(model_file))
    print(all_stereotypes, len(all_stereotypes))


if __name__ == '__main__':
    # merge_stereotypes()
    # main_func('my_mylyn', step=1)
    # main_func('my_mylyn', step=2)
    assign_stereotype(
        dataset_dir='/data0/xiaoyez/CodeContextModel/data/train_test_index', 
        stereotype_path='/data0/xiaoyez/CodeContextModel/baseline/pattern_matching/stereotypes.tsv',
        step=3
    )
