import os
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