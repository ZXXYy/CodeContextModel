import os
import math

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

def visualize_count_based_results(results: dict, output_dir: str):
    """
    可视化CountBasedStrategy的结果
    
    Parameters:
        results: 包含不同seed数量的评估指标结果的字典
    """
    
    # 创建保存可视化结果的目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取数据
    seed_counts = sorted([int(k) for k in results.keys()])
    metrics = {
        'top1_hit': [],
        # 'top2_hit': [],
        'top3_hit': [],
        # 'top4_hit': [],
        'top5_hit': [],
        'mrr': []
    }
    
    for seed_count in seed_counts:
        for metric in metrics:
            metrics[metric].append(results[str(seed_count)][metric])
    
    # 绘制图形
    plt.figure(figsize=(12, 8))
    
    # 设置颜色和标记样式
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    # 绘制所有指标
    for i, (metric, values) in enumerate(metrics.items()):
        plt.plot(seed_counts, values, 
                    label=metric.replace('_', ' ').upper() if metric in ['mrr', 'map'] else f"{metric.replace('_', ' ').title()}", 
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    linewidth=2,
                    markersize=8)
    
    # 设置图表属性
    # plt.title('Performance Metrics by Seed Count', fontsize=16)
    plt.xlabel('Number of Seed Nodes', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='lower right')
    
    # 设置x轴刻度为整数
    plt.xticks(seed_counts)
    
    # 设置y轴范围
    plt.ylim(0.4, 1.0)
    
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
