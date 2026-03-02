import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 设置全局字体为 Times New Roman (ECCV 标准)
# ==========================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def plot_performance_gap():
    # 1. 数据准备
    labels = ['AT', 'R&P', 'Bit-RD', 'Diff', 'JPEG', 'FD', 'RS', 'HGD', 'AVG']

    # BSR-TI-DIM (Baseline)
    bsr_data = [30.9, 96.9, 95.5, 42.5, 97.3, 96.8, 69.4, 93.3, 77.8]

    # BGPF-SI-TI-DIM (Ours)
    bgpf_data = [36.8, 98.5, 98.5, 61.9, 98.9, 99.1, 83.4, 97.4, 84.3]

    x = np.arange(len(labels))  # 标签位置
    width = 0.35  # 柱子宽度

    # 2. 创建画布
    # figsize=(6, 4) 是一个适合单栏插入的大小
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=150)

    # 3. 绘制柱状图
    # 颜色选择：
    # BSR: #5B7388 (Slate Blue - 沉稳的冷色调，代表基准)
    # BGPF: #D62728 (Tab:Red - 醒目的红色，代表你们的方法)
    rects1 = ax.bar(x - width / 2, bsr_data, width, label='BSR-TI-DIM', color='blue', alpha=0.9, edgecolor='white')
    rects2 = ax.bar(x + width / 2, bgpf_data, width, label='BGPF-SI-TI-DIM (Ours)', color='#D62728', alpha=0.9,
                    edgecolor='white')

    # 4. 设置轴标签和标题
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    # ax.set_xlabel('Defense Models', fontsize=12) # X轴标签可以省略，因为刻度已经很清楚


    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, rotation=0)  # 如果标签太挤，可以改成 rotation=45
    ax.set_ylim(0, 110)  # 留出空间给数字标注

    # 5. 添加图例 (放在合适的位置，通常是上方或角落)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=11)

    # 6. 自动添加数值标注 (Highlight Improvements)
    def autolabel(rects_base, rects_ours):
        """在柱子上方显示数值，并在红色柱子上显示提升幅度"""
        for base, ours in zip(rects_base, rects_ours):
            height_base = base.get_height()
            height_ours = ours.get_height()
            gap = height_ours - height_base

            # 在蓝色柱子上只标数值 (可选，为了不乱可以不标，或者只标数字)
            # ax.annotate(f'{height_base:.1f}',
            #             xy=(base.get_x() + base.get_width() / 2, height_base),
            #             xytext=(0, 3), textcoords="offset points",
            #             ha='center', va='bottom', fontsize=8, color='#5B7388')

            # 在红色柱子上标数值 + 提升幅度
            # 如果提升很大 (>5%)，加粗显示提升值
            font_weight = 'bold' if gap > 5 else 'normal'

            # 显示 ASR 数值
            ax.annotate(f'{height_ours:.1f}',
                        xy=(ours.get_x() + ours.get_width() / 2, height_ours),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

            # 显示提升箭头 (放在柱子中间或者上方更高处)
            # 这里选择在两个柱子中间上方显示 +X.X%
            center_x = (base.get_x() + ours.get_x() + base.get_width() + ours.get_width()) / 2
            max_h = max(height_base, height_ours)

            if gap > 0:
                ax.annotate(f'+{gap:.1f}%',
                            xy=(center_x, max_h),
                            xytext=(0, 15), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, color='#D62728', fontweight='bold',
                            arrowprops=dict(arrowstyle='-', color='#D62728', lw=0.5, shrinkA=0, shrinkB=0))

    autolabel(rects1, rects2)

    # 7. 美化细节
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
    ax.set_axisbelow(True)  # 让网格线在柱子后面

    # 去除顶部和右侧的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 加粗底部和左侧边框
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    # 8. 保存
    plt.tight_layout()
    plt.savefig('fig1_b_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig1_b_performance.eps', format='eps', bbox_inches='tight')
    print("Saved as fig1_b_performance.png and .eps")
    plt.show()


if __name__ == "__main__":
    plot_performance_gap()