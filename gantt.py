import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
#图片保存到gantt文件夹
folder_name = "gantt"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
def draw_complete_activity_network():
    fig, ax = plt.subplots(figsize=(18, 6))

    node_modes = {
        0: [(0, 0, 0, 0)],
        1: [(6, 3, 4, 9), (5, 9, 10, 0)],
        2: [(9, 3, 4, 0), (7, 5, 8, 2), (6, 7, 8, 0)],
        3: [(9, 4, 4, 8), (4, 10, 11, 0)], 
        4: [(2, 2, 4, 8), (2, 5, 6, 0)],
        5: [(5, 3, 4, 10), (5, 7, 8, 0)],
        6: [(2, 2, 3, 6), (1, 6, 7, 0)],
        7: [(0, 0, 0, 0)],
        "j": [r"($R_{jm}, \bar{d}_{jm}, \bar{d}_{jm}+u_{jm}, C_{jm}$)"]
    }

    nodes_config = {
        0: (0, 3, '#ffffff', '#000000'), 
        1: (3, 4, '#dee6f7', '#5b9bd5'), 
        2: (3, 2, '#fbe5d6', '#ed7d31'), 
        3: (6, 5, '#fff2cc', '#ffc000'), 
        4: (6, 3, '#e2f0d9', '#70ad47'), 
        5: (9, 4, '#daefef', '#4472c4'), 
        6: (9, 2, '#fce4d6', '#c00000'), 
        7: (12, 3, '#ffffff', '#000000'), 
        "j": (15, 2, '#ffffff', '#000000'), 
    }
    
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 6), (3, 5), (4, 5), (5, 7), (6, 7)]
    node_size = 0.8

    # --- 3. 绘制节点、模式标注与箭头 ---
    for node, (x, y, fc, ec) in nodes_config.items():
        rect = patches.FancyBboxPatch(
            (x - node_size/2, y - node_size/2), node_size, node_size,
            boxstyle="round,pad=0.1,rounding_size=0.2",
            facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=4
        )
        ax.add_patch(rect)
        
        display_id = f"${node}$" if node == "j" else str(node)
        ax.text(x, y, display_id, ha='center', va='center', fontsize=16, zorder=5)

        if node in node_modes:
            modes = node_modes[node]
            if node == "j":
                mode_text = modes[0]
            else:
                mode_text = "\n".join([f"({', '.join(map(str, m))})" for m in modes])
            
            ax.text(x, y + 0.65, mode_text, ha='center', va='bottom', 
                    fontsize=10, linespacing=1.2, color='#333333', zorder=5)

    for u, v in edges:
        x_s, y_s = nodes_config[u][0], nodes_config[u][1]
        x_e, y_e = nodes_config[v][0], nodes_config[v][1]
        ax.annotate("", 
                    xy=(x_e - 0.55, y_e), 
                    xytext=(x_s + 0.55, y_s),
                    arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1, 
                                    mutation_scale=15, connectionstyle="arc3,rad=0"), 
                    zorder=2)

    # --- 4. 自适应裁剪逻辑 ---
    # 动态获取所有节点的坐标，自动设置范围
    all_x = [pos[0] for pos in nodes_config.values()]
    all_y = [pos[1] for pos in nodes_config.values()]
    
    # 增加一点外边距（Margin）以容纳上方的文字
    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 2) # 上方给模式描述留更多空间
    
    ax.grid(True, linestyle=':', alpha=0.3, zorder=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # 关键修改：自动调整子图参数，使内容填满画布
    plt.tight_layout()

    
    save_path = os.path.join(folder_name, "activity_network_layout_j.jpg")
    
    # 保存并裁剪
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"文件夹已同步，图片保存至: {save_path}")
 
    
     

draw_complete_activity_network()


# --- 2. 模式组合定义 ---
modes_normal = [1, 2, 1, 2, 1, 1] 
modes_robust = [1, 3, 2, 1, 1, 1] 

# --- 3. 数据定义 ---
activity_normal_tau0 = [(1, 6, 0, 3, 1, '#DEE6F7', '#5B9BD5'), 
                    (1, 7, 3, 8, 2, '#FBE5D6', '#ED7D31'), 
                    (1, 9, 8, 12, 3, '#FFF2CC', '#FFC000'),
                    (8, 9, 3, 8, 4, '#E2F0D9', '#70AD47'), 
                    (1, 5, 12, 15, 5, '#DAEFEF', '#4472C4'), 
                    (1, 2, 15, 17, 6, '#FCE4D6', '#C00000')]
activity_normal_tau2 = [(1, 6, 0, 4, 1, '#DEE6F7', '#5B9BD5', [3, 4]), 
                    (1, 7, 4, 12, 2, '#FBE5D6', '#ED7D31', [9, 12]), 
                    (1, 9, 12, 16, 3, '#FFF2CC', '#FFC000'), 
                    (8, 9, 4, 9, 4, '#E2F0D9', '#70AD47'), 
                    (1, 5, 16, 19, 5, '#DAEFEF', '#4472C4'), 
                    (1, 2, 19, 21, 6, '#FCE4D6', '#C00000')]
activity_robust_tau0 = [(1, 6, 0, 3, 1, '#DEE6F7', '#5B9BD5'), (1, 6, 3, 10, 2, '#FBE5D6', '#ED7D31'), (7, 10, 3, 13, 3, '#FFF2CC', '#FFC000'), (1, 2, 10, 12, 4, '#E2F0D9', '#70AD47'), (1, 5, 13, 16, 5, '#DAEFEF', '#4472C4'), (1, 2, 16, 18, 6, '#FCE4D6', '#C00000')]
activity_robust_tau2 = [(1, 6, 0, 4, 1, '#DEE6F7', '#5B9BD5', [3, 4]), (1, 6, 4, 11, 2, '#FBE5D6', '#ED7D31'), (7, 10, 4, 15, 3, '#FFF2CC', '#FFC000', [14, 15]), (1, 2, 11, 13, 4, '#E2F0D9', '#70AD47'), (1, 5, 15, 18, 5, '#DAEFEF', '#4472C4'), (1, 2, 18, 20, 6, '#FCE4D6', '#C00000')]

def draw_wide_gantt(merged_task):
    # 自动识别变量名
    activity_name = "unknown"
    for name, val in globals().items():
        if val is merged_task:
            activity_name = name
            break
    
    current_modes = modes_robust if "robust" in activity_name else modes_normal

    UNIFORM_FONT_SIZE = 11
    SHRINK = 0.05 
    num_rows = 10
    GLOBAL_MAX_TIME = 21 
    
    current_max_time = max([t[3] for t in merged_task])

    fig, ax = plt.subplots(figsize=(16, 4))

    for task in merged_task:
        r_start, r_end, t_start, t_end, job_idx, fc, ec = task[:7]
        highlight = task[7] if len(task) > 7 else None
        
        mode_val = current_modes[job_idx - 1]
        display_label = f" Activity {job_idx}\nMode {mode_val}"

        y_top = num_rows - r_start + 1.45
        y_bottom = num_rows - r_end + 0.55
        draw_y = y_bottom + SHRINK
        draw_height = (y_top - y_bottom) - 2 * SHRINK
        
        # 1. 绘制主色块
        rect = patches.FancyBboxPatch(
            (t_start + SHRINK, draw_y), (t_end - t_start) - 2 * SHRINK, draw_height,
            boxstyle=f"round,pad=0,rounding_size=0.15",
            facecolor=fc, edgecolor=ec, linewidth=2, zorder=2
        )
        ax.add_patch(rect)

        # 2. 绘制加深区域并标注“延期”
        if highlight:
            h_s, h_e = highlight
            h_rect = patches.FancyBboxPatch(
                (h_s + SHRINK, draw_y), (h_e - h_s) - 2 * SHRINK, draw_height,
                boxstyle=f"round,pad=0,rounding_size=0.15",
                facecolor=ec, edgecolor='none', alpha=0.35, zorder=3
            )
            ax.add_patch(h_rect)
            
            # --- 新增：在加深区域标注“延期” ---
            # 如果你使用的是中文环境，请确保系统中装有中文字体，
            # 否则这里建议标注为 "Delay"
            ax.text(h_s + (h_e - h_s)/2, y_bottom + (y_top - y_bottom)/2, 
                    "Delay", ha='center', va='center', 
                    fontsize=UNIFORM_FONT_SIZE,  zorder=7)
        
        # 3. 绘制主标签
        ax.text(t_start + (t_end - t_start)/2, y_bottom + (y_top - y_bottom)/2, 
                display_label, ha='center', va='center', 
                fontsize=UNIFORM_FONT_SIZE, color='black',
                linespacing=1.2, zorder=6)

    # 绘制背景网格
    for x in range(0, GLOBAL_MAX_TIME + 1):
        ax.plot([x, x], [0.5, num_rows + 0.5], color='gray', linestyle=':', lw=1, alpha=0.4, zorder=5)
    for y in range(1, num_rows + 2):
        ax.plot([0, GLOBAL_MAX_TIME], [y - 0.5, y - 0.5], color='gray', linestyle=':', lw=1, alpha=0.4, zorder=5)

    # 坐标轴格式
    ax.set_xlim(-0.1, GLOBAL_MAX_TIME + 0.1)
    ax.set_ylim(0.4, num_rows + 0.6)
    ax.set_aspect('auto') 
    ax.set_xticks(range(1, current_max_time + 1)) 

    ax.set_yticks(range(1, num_rows + 1))
    ax.set_yticklabels([str(i) for i in range(num_rows, 0, -1)])
    # --- 添加轴标签 ---
    # labelpad 用于控制标签距离轴线的距离
    ax.set_xlabel("Time", fontsize=12,   labelpad=5)
    ax.set_ylabel("Resource ID", fontsize=12,   labelpad=5)

    # 如果你希望标签显示在轴的最末端（类似坐标轴箭头处），可以使用 ax.text
    # 或者直接使用常规的轴中心标注（如下所示）
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=2) 

    plt.tight_layout() 
    save_path = os.path.join(folder_name, f"{activity_name}_gantt_chart.jpg")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"文件夹已同步，图片保存至: {save_path}")
     

# --- 执行绘制 ---
draw_wide_gantt(activity_normal_tau0)
draw_wide_gantt(activity_normal_tau2)
draw_wide_gantt(activity_robust_tau0)
draw_wide_gantt(activity_robust_tau2)