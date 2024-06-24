import xml.etree.ElementTree as ET
from handle_raw import get_repos_from_xml,TZINFOS
from dateutil import parser, tz
from datetime import datetime
import logging
import click
import os
from tqdm import tqdm
import pandas as pd
from git import Repo

logging.basicConfig(level=logging.DEBUG)

def clone_repos(repo_path='data/repos'):
    repo_urls = [
        "https://github.com/eclipse-mylyn/org.eclipse.mylyn.git",
        "https://github.com/eclipse-mylyn/org.eclipse.mylyn.docs.git",
        "https://github.com/eclipse-mylyn/mylyn-website.git",
        "https://github.com/eclipse-mylyn/org.eclipse.mylyn.context.git",
        "https://github.com/eclipse-mylyn/org.eclipse.mylyn.reviews.git",
        "https://github.com/eclipse-mylyn/org.eclipse.mylyn.commons.git",
        "https://github.com/eclipse-mylyn/org.eclipse.mylyn.builds.git",
        "https://github.com/eclipse-mylyn/org.eclipse.mylyn.tasks.git",
        "https://github.com/eclipse-mylyn/org.eclipse.mylyn.versions.git",
        "https://github.com/eclipse-mylyn/org.eclipse.mylyn.docs.intent.main.git",
        "https://github.com/eclipse-mylyn/org.eclipse.mylyn.context.mft.git",
        "https://github.com/eclipse-mylyn/org.eclipse.mylyn.docs.vex.git",
        "https://github.com/eclipse-mylyn/org.eclipse.mylyn.incubator.git",
        "https://github.com/eclipse-mylyn/org.eclipse.mylyn.reviews.ui.git",
    ]
    for repo_url in repo_urls:
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        to_path = os.path.join(repo_path, repo_name)
        if os.path.exist(to_path):
            logging.info(f"Repo {repo_name} already exists")
            continue
        Repo.clone_from(repo_url, to_path)

def get_repos(xml_file, repos_path):
    xml_repos = get_repos_from_xml(xml_file)
    repos = set()
   
    if len(xml_repos) == 0:
        logging.warning("No Mylyn repo found in the xml file")
        return
    # print(xml_repos)
    # find repo paths
    repo_paths = os.listdir(repos_path)
    for repo in xml_repos:
        for repo_path in repo_paths:
            if repo_path in repo:
                repos.add(os.path.join(repos_path, repo_path))
                break
    # print(repos)
    return repos

def get_commits(repo, start_time, end_time):
    repo = Repo(repo)
    target_commits = {
        'before_start_commit': [],
        'between_start_end_commits': [],
    }
    try:
        # 获取所有提交
        commits = reversed(list(repo.iter_commits())) # 从最老的提交开始
        # 找到start_date之前的最后一次提交
        for commit in commits:
            commit_date = datetime.fromtimestamp(commit.committed_date, tz=tz.tzutc())
            if commit_date <= start_time:
                target_commits['before_start_commit'] = [commit]
            elif commit_date > start_time and commit_date < end_time:
                # target_commits['between_start_end_commits'].append(commit)  
                continue
            else:
                break 
        # print(target_commits)
        return target_commits
    except Exception as e:
        logging.error(f"Error in getting commits: {e}")
        return []

def associate_commit(xml_file, repos_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    results = []
    # 获取xml_file中的第一个startDate, working period中的时间已经排好序
    start_date = parser.parse(root.findall(".//InteractionEvent")[0].get('StartDate'), tzinfos=TZINFOS)
    # 获取xml_file中的所有end Date
    end_dates = [event.get('EndDate') for event in root.findall(".//InteractionEvent")]
    end_date = parser.parse(sorted(end_dates)[-1], tzinfos=TZINFOS)
    repos = get_repos(xml_file, repos_path)
    if repos is None:
        return []
    for repo in repos:
        target_commits = get_commits(repo, start_date, end_date)
        results.append({
            'work_period': xml_file,
            'repo': repo,
            'commits': target_commits
        })
    return results
    # print(start_date, end_date)
    

@click.command()
@click.argument('repos_path', type=click.Path(exists=True))
@click.argument('working_periods_path', type=click.Path(exists=True))
def main(repos_path, working_periods_path):
    xml_files = os.listdir(working_periods_path)
    results = []
    # xml_file = '/Users/zhengxiaoye/Desktop/codeContextGNN/CodeContextModel/data/working_periods/116487_83035_zip_group_0.xml'
    for xml_file in tqdm(xml_files[:1000]):
        commits = associate_commit(os.path.join(working_periods_path, xml_file), repos_path)
        results.extend(commits)
    pd.DataFrame(results).to_xml('data/associated_commits.xml', index=False)
    
    # associate_commit(xml_file, repos_path)

if __name__ == '__main__':
    # clone_repos()
    main()
