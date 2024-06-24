import xml.etree.ElementTree as ET
from datetime import timedelta
from dateutil import parser, tz
from collections import defaultdict
import os
import click
from tqdm import tqdm

WORKING_PERIOD = 2.5  # 2 hours
# 定义tzinfos来处理未知时区
TZINFOS = {
    'CET': tz.gettz('CET'),      # Central European Time
    'CDT': tz.gettz('CST6CDT'),  # Central Daylight Time (USA)
    'CEST': tz.gettz('CET'),     # Central European Summer Time
    'COT': tz.gettz('America/Bogota'),  # Colombia Time
    'PET': tz.gettz('America/Lima'),    # Peru Time
    'IST': tz.gettz('Asia/Kolkata'),    # India Standard Time
    'CST': tz.gettz('CST6CDT'),  # Central Standard Time (USA)
    'BST': tz.gettz('Europe/London'),   # British Summer Time
    'GMT': tz.gettz('GMT'),      # Greenwich Mean Time
    'PDT': tz.gettz('PST8PDT'),  # Pacific Daylight Time (USA)
    'PST': tz.gettz('PST8PST'),  # Pacific Standard Time (USA)
    'EET': tz.gettz('EET'),      # Eastern European Time
    'EDT': tz.gettz('EST5EDT'),  # Eastern Daylight Time (USA)
    'MDT': tz.gettz('MST7MDT'),  # Mountain Daylight Time (USA)
    'EST': tz.gettz('EST5EST'),  # Eastern Standard Time (USA)
    'MST': tz.gettz('MST7MST'),  # Mountain Standard Time (USA)
    'BRT': tz.gettz('America/Sao_Paulo'),  # Brasilia Time
    'EEST': tz.gettz('EET'),     # Eastern European Summer Time
    'BRST': tz.gettz('America/Sao_Paulo'),  # Brasilia Summer Time
    'JST': tz.gettz('Asia/Tokyo'),  # Japan Standard Time
    'YAKT': tz.gettz('Asia/Yakutsk'),  # Yakutsk Time
    'ICT': tz.gettz('Asia/Bangkok')  # Indochina Time (Vietnam, Thailand, etc.)
}
# Pares StartDate
def parse_start_date(elem):
    return parser.parse(elem.attrib['StartDate'], tzinfos=TZINFOS)

# split interaction traces by working period 
def group_events_by_interval(events, interval_hours=WORKING_PERIOD):
    grouped_events = defaultdict(list)
    start_time = parse_start_date(events[0])
    interval = timedelta(hours=interval_hours)
    current_group = 0

    for event in events:
        event_time = parse_start_date(event)
        if event_time >= start_time + interval:
            start_time = event_time
            current_group += 1
        # 只添加StructureKind为java的元素
        if event.attrib.get('StructureKind') == 'java':
            grouped_events[current_group].append(event)

    return grouped_events

def save_grouped_events(grouped_events, root, output_file):
    for group, events in grouped_events.items():
        group_root = ET.Element("WorkingPeriod", Id=root.attrib['Id'], Version=root.attrib['Version'])
        # 将组内的事件添加到新的根元素下
        for event in events:
            group_root.append(event)
        # 创建ElementTree对象
        group_tree = ET.ElementTree(group_root)
        # 保存为独立的文件
        filename = f"{output_file}_group_{group}.xml" # data/working_periods/195691_83176_zip_group_0.xml
        group_tree.write(filename, encoding='unicode', xml_declaration=True)

def handle_raw_xml(file_path:str, output_dir:str):
    # 读取和解析XML文件
    tree = ET.parse(file_path)
    root = tree.getroot()
    # 查找所有Kind为selection或edit的元素
    selection_edit_elements = root.findall(".//InteractionEvent[@Kind='selection']") + \
                              root.findall(".//InteractionEvent[@Kind='edit']")
    sorted_elements = sorted(selection_edit_elements, key=parse_start_date)

    if len(sorted_elements) == 0:
        return
    # 按2小时间隔分组
    grouped_events = group_events_by_interval(sorted_elements)
    # 保存分组后的事件
    output_file = file_path.split('/')[-2]
    save_grouped_events(grouped_events, root, f"{output_dir}/{output_file}") # data/working_periods/195691_83176_zip


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
def main(input_path):
    dirs = os.listdir(input_path)
    xml_files = []
    for directory in dirs:
        for root, dirs, files in os.walk(f"{input_path}/{directory}"):
            for file in files:
                if file.endswith('.xml'):
                    xml_files.append(os.path.join(root, file))

    if not os.path.exists('data/working_periods'):
        os.makedirs('data/working_periods')

    for xml_file in tqdm(xml_files):
        handle_raw_xml(xml_file, 'data/working_periods')


if __name__ == '__main__':
    main()