import os
import math

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

def visualize_count_based_results(results: dict, output_dir: str):
    """
    可视化CountBasedStrategy的结果，将MRR和Hit Rate分开展示
    
    Parameters:
        results: 包含不同seed数量的评估指标结果的字典
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取数据
    seed_counts = sorted([float(k) for k in results.keys()])
    metrics = {
        'hits': {
            'top1_hit': [],
            'top3_hit': [],
            'top5_hit': [],
        },
        'ranking': {
            'mrr': []
        }
    }
    hit2label = {
        'top1_hit': 'R@1',
        'top3_hit': 'R@3',
        'top5_hit': 'R@5',
    }

    for seed_count in seed_counts:
        for metric_group in metrics.values():
            for metric in metric_group:
                metric_group[metric].append(results[str(seed_count)][metric])
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 设置颜色和标记样式
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    # Convert seed counts to percentages
    seed_counts_pct = [x * 100 for x in seed_counts]
    
    # Update x-axis ticks and labels
    ax1.set_xticks(seed_counts_pct)
    ax1.set_xticklabels([f'{int(x)}%' for x in seed_counts_pct])
    ax2.set_xticks(seed_counts_pct)
    ax2.set_xticklabels([f'{int(x)}%' for x in seed_counts_pct])
    
    # 绘制Hit Rate指标
    for i, (metric, values) in enumerate(metrics['hits'].items()):
        ax1.plot(seed_counts_pct, values, 
                label=hit2label[metric],
                marker=markers[i],
                color=colors[i],
                linewidth=2,
                markersize=8)
    
    # 绘制MRR指标
    ax2.plot(seed_counts_pct, metrics['ranking']['mrr'],
            label='MRR',
            marker=markers[0],
            color=colors[0],
            linewidth=2,
            markersize=8)
    
    # 设置第一个子图属性 (Hit Rates)
    ax1.set_xlabel('Percentage of Code Elements in Code Context Model', fontsize=14, labelpad=15)
    ax1.set_ylabel('Topk Recall', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12, loc='lower left')
    ax1.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label size
    
    # 设置第二个子图属性 (MRR)
    ax2.set_xlabel('Percentage of Code Elements in Code Context Model', fontsize=14, labelpad=15)
    ax2.set_ylabel('MRR', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=12, loc='lower left')
    ax2.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label size
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "count_based_strategy_results.png"), dpi=300)

def visualize_order_based_results(results: dict, output_dir: str):
    """
    可视化OrderBasedStrategy的结果
    
    Parameters:
        results: 包含不同seed数量的评估指标结果的字典
    """
    
    # 创建保存可视化结果的目录
    os.makedirs(output_dir, exist_ok=True)
    # 使用entropy来衡量不同seed选择对成功率topk的影响，entropy越小，说明结果越确定
    # 使用variance来衡量不同seed选择对mrr的影响，variance越小，说明结果越稳定
    # ====1. 成功率以及entropy分析 for topk====
    success_rates = []
    entropies = []
    for test_id, predictions in results.items():
        metrics = ['top1_hit', 'top3_hit', 'top5_hit']
        # 计算每个指标的成功率以及entropy
        rates = {}
        entropy_values = {}
        for metric in metrics:
            values = [pred[metric] for pred in predictions]
            p = sum(values) / len(values)
            entropy = 0 if p == 0 or p == 1 else -p * np.log2(p) - (1-p) * np.log2(1-p)
            rates[f"{metric}_success_rate"] = p
            entropy_values[f"{metric}_entropy"] = entropy

        
        rates['test_id'] = test_id
        entropy_values['test_id'] = test_id
        success_rates.append(rates)
        entropies.append(entropy_values)

    success_df = pd.DataFrame(success_rates)
    entropy_df = pd.DataFrame(entropies)

    # ====2. 方差分析 for mrr ====
    variances = []
    for test_id, predictions in results.items():
        metrics = ['mrr']
        # 计算每个指标的方差
        variance = {}
        for metric in metrics:
            values = [pred[metric] for pred in predictions]
            variance[f"{metric}_variance"] = np.var(values)
        variance['test_id'] = test_id
        variances.append(variance)
    variances_df = pd.DataFrame(variances)

    # ====3. 可视化====
    # 1. 成功率分布直方图
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(['top1_hit_success_rate', 'top3_hit_success_rate', 'top5_hit_success_rate']):
        plt.subplot(1, 3, i+1)
        sns.violinplot(success_df[metric])
        plt.title(f'Distribution of Success Rate - {metric}')
        plt.xlabel('Success Rate')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_rate_distribution.png"), dpi=300)
    print(success_df.describe())
    # 2. 熵分布直方图
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(['top1_hit_entropy', 'top3_hit_entropy', 'top5_hit_entropy']):
        plt.subplot(1, 3, i+1)
        sns.violinplot(entropy_df[metric])
        plt.title(f'Distribution of Entropy - {metric}')
        plt.xlabel('Entropy')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "entropy_distribution.png"), dpi=300)
    print(entropy_df.describe())

    # 3. 方差分布直方图
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(['mrr_variance']):
        plt.subplot(1, 1, i+1)
        sns.violinplot(variances_df[metric])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "variance_distribution.png"), dpi=300)
    print(variances_df.describe())

def visualize_experience_based_results(results: dict, output_dir: str):
    """
    可视化基于经验和非基于经验的结果对比
    
    Parameters:
        results: 包含不同时间窗口的评估指标结果的字典
    """
    os.makedirs(output_dir, exist_ok=True)

    # 提取时间窗口
    delta_days = [int(key.split('_')[-1]) for key in results.keys()]
    
    # 准备数据
    exp_metrics = {
        'mrr': [],
        'top1_hit': [],
        'top3_hit': [],
        'top5_hit': []
    }
    non_exp_metrics = {
        'mrr': [],
        'top1_hit': [],
        'top3_hit': [],
        'top5_hit': []
    }

    # 收集数据
    for day_key in sorted(results.keys()):
        for metric in exp_metrics.keys():
            exp_metrics[metric].append(results[day_key]['experience_based'][metric])
            non_exp_metrics[metric].append(results[day_key]['non_experience_based'][metric])

    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 绘制MRR对比
    ax1.plot(delta_days, exp_metrics['mrr'], 'o-', label='Experience-based', color='#2ca02c')
    ax1.plot(delta_days, non_exp_metrics['mrr'], 's-', label='Non-experience-based', color='#1f77b4')
    ax1.set_xlabel('Time Window (days)', fontsize=12)
    ax1.set_ylabel('MRR', fontsize=12)
    ax1.set_title('MRR Comparison between Experience and Non-Experience', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1)

    # 绘制Recall@K对比
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, k in enumerate([1, 3, 5]):
        metric = f'top{k}_hit'
        ax2.plot(delta_days, exp_metrics[metric], f'{markers[i]}-', 
                label=f'R@{k} based on experience', color=colors[i], linestyle='--')
        ax2.plot(delta_days, non_exp_metrics[metric], f'{markers[i]}-',
                label=f'R@{k} based on non-experience', color=colors[i])

    ax2.set_xlabel('Time Window (days)', fontsize=12)
    ax2.set_ylabel('Recall@K', fontsize=12)
    ax2.set_title('Recall@K Comparison between Experience and Non-Experience', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "experience_based_results.png"), dpi=300, bbox_inches='tight')
    plt.close()
   