#!/usr/bin/env python3
"""
generate_mode_meta.py

从 MRCPSP 的 .mm 文件生成活动模式元数据 CSV 文件。
包含每个活动每个模式的偏离时间(u_abs)和模式选择成本(cost)。

生成规则说明（确保trade-off，无支配关系）：

1. barT: 直接从mm文件读取的基准持续时间(duration)

2. u_abs: 偏离时间，与资源需求**负相关**
   - 资源越多 → 偏离稍小
   - 资源越少 → 偏离稍大
   - 对于 duration=0 的虚拟活动(supersource/sink)，u_abs=0

3. maxT: 最大持续时间 = barT + u_abs

4. cost: 模式选择成本，与持续时间**负相关**（时间短则成本高）
   - 持续时间最长的模式 cost=0（基准/经济模式）
   - 持续时间越短，cost越高（花钱买时间）
   - 这样确保每个模式都有存在意义：快但贵 vs 慢但便宜

5. Rreq: 直接从mm文件读取的可再生资源需求（取所有可再生资源的总和）

用法：
    # 命令行调用
    python generate_mode_meta.py <input.mm> [output.csv]
    
    # Python 模块调用
    from generate_mode_meta import generate_mode_meta_from_mm
    csv_path, deviations, cost = generate_mode_meta_from_mm("input.mm")

选项：
    --seed=<int>        设置随机种子以获得可重复的结果
    --cost-factor=<int> 设置成本因子（默认为1）
"""

import sys
import os
import csv
import random
import re


def parse_mm_file(filepath):
    """
    解析 .mm 文件，提取活动模式信息。
    
    返回:
        jobs: dict, {jobnr: [(mode, duration, [resource_reqs]), ...]}
        num_renewable: int, 可再生资源数量
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    
    jobs = {}
    num_renewable = 0
    num_nonrenewable = 0
    
    # 解析资源数量
    for line in lines:
        if '- renewable' in line and 'non' not in line.lower():
            match = re.search(r':\s*(\d+)', line)
            if match:
                num_renewable = int(match.group(1))
        elif '- nonrenewable' in line:
            match = re.search(r':\s*(\d+)', line)
            if match:
                num_nonrenewable = int(match.group(1))
    
    # 总资源列数 = 可再生资源 + 非再生资源
    total_resource_cols = num_renewable + num_nonrenewable
    
    # 找到 REQUESTS/DURATIONS 部分
    in_requests_section = False
    current_job = None
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        if 'REQUESTS/DURATIONS:' in line_stripped:
            in_requests_section = True
            continue
        
        if in_requests_section:
            # 跳过标题行和分隔线
            if line_stripped.startswith('jobnr') or line_stripped.startswith('-'):
                continue
            
            # 到达下一个部分
            if line_stripped.startswith('*'):
                break
            
            if not line_stripped:
                continue
            
            # 解析数据行
            parts = line_stripped.split()
            if not parts:
                continue
            
            # 尝试解析为数字
            try:
                nums = [int(p) for p in parts]
            except ValueError:
                continue
            
            # 判断是新活动行还是续行
            # 新活动行格式: jobnr mode duration R1 R2 ... N1 N2 ...
            #   列数 = 3 + total_resource_cols
            # 续行格式:     mode duration R1 R2 ... N1 N2 ...
            #   列数 = 2 + total_resource_cols
            
            expected_new_job_cols = 3 + total_resource_cols
            expected_continuation_cols = 2 + total_resource_cols
            
            if len(nums) == expected_new_job_cols:
                # 新活动行
                current_job = nums[0]
                mode = nums[1]
                duration = nums[2]
                # 只取可再生资源
                resource_reqs = nums[3:3 + num_renewable]
                
                if current_job not in jobs:
                    jobs[current_job] = []
                jobs[current_job].append((mode, duration, resource_reqs))
                
            elif len(nums) == expected_continuation_cols and current_job is not None:
                # 续行（同一活动的其他模式）
                mode = nums[0]
                duration = nums[1]
                # 只取可再生资源
                resource_reqs = nums[2:2 + num_renewable]
                jobs[current_job].append((mode, duration, resource_reqs))
    
    return jobs, num_renewable


def generate_mode_meta(jobs, num_renewable, seed=None, cost_factor=1):
    """
    生成模式元数据（确保trade-off，无支配关系）。
    
    核心逻辑：
    - 持续时间短 → 成本高（花钱买时间）
    - 持续时间长 → 成本低(0)（基准/经济模式）
    - 资源需求高 → 偏离小
    - 资源需求低 → 偏离大
    
    参数:
        jobs: dict, 从 parse_mm_file 返回的活动数据
        num_renewable: int, 可再生资源数量
        seed: int, 随机种子（可选）
        cost_factor: int, 成本因子，用于计算模式选择成本（默认为1）
    
    返回:
        meta_data: list of dict, 每行包含 jobnr, mode, barT, maxT, u_abs, cost, Rreq
    """
    if seed is not None:
        random.seed(seed)
    
    meta_data = []
    
    for jobnr in sorted(jobs.keys()):
        modes = jobs[jobnr]
        
        # 获取该活动所有模式的资源需求和持续时间
        resource_list = []
        duration_list = []
        for mode, duration, resource_reqs in modes:
            # 取所有可再生资源的总和
            req = sum(resource_reqs) if resource_reqs else 0
            resource_list.append(req)
            duration_list.append(duration)
        
        min_req = min(resource_list) if resource_list else 0
        max_req = max(resource_list) if resource_list else 0
        req_range = max_req - min_req if max_req > min_req else 1
        
        min_duration = min(duration_list) if duration_list else 0
        max_duration = max(duration_list) if duration_list else 0
        
        for mode, duration, resource_reqs in modes:
            barT = duration
            # 取所有可再生资源的总和
            Rreq = sum(resource_reqs) if resource_reqs else 0
            
            # 生成偏离时间 u_abs（与资源需求负相关）
            if barT == 0:
                # 虚拟活动（supersource/sink），无偏离
                u_abs = 0
            else:
                # 归一化资源需求到 [0, 1]，资源越高值越大
                if req_range > 0:
                    normalized_req = (Rreq - min_req) / req_range
                else:
                    normalized_req = 0.5
                
                # 基础偏离范围（温和）
                base_deviation = max(1, barT // 4)
                
                # 偏离系数：资源需求高时较小，资源需求低时较大
                deviation_factor = 1.0 - 0.5 * normalized_req
                
                # 计算该模式的偏离范围
                min_dev = 1
                max_dev = max(1, int(base_deviation * deviation_factor) + 1)
                
                # 随机生成偏离时间
                u_abs = random.randint(min_dev, max_dev)
            
            maxT = barT + u_abs
            
            # 计算模式选择成本（与持续时间负相关，确保trade-off）
            if barT == 0:
                # 虚拟活动无成本
                cost = 0
            else:
                # 持续时间最长的模式 cost=0（基准/经济模式）
                # 持续时间越短，cost越高（花钱买时间）
                cost = (max_duration - barT) * cost_factor
            
            meta_data.append({
                'jobnr': jobnr,
                'mode': mode,
                'barT': barT,
                'maxT': maxT,
                'u_abs': u_abs,
                'cost': cost,
                'Rreq': Rreq
            })
    
    return meta_data


def write_csv(meta_data, output_path):
    """
    将模式元数据写入 CSV 文件。
    """
    fieldnames = ['jobnr', 'mode', 'barT', 'maxT', 'u_abs', 'cost', 'Rreq']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(meta_data)


def meta_to_deviations_and_cost(meta_data):
    """
    将模式元数据转换为 deviations 和 cost 数据结构。
    
    参数:
        meta_data: list of dict, 模式元数据
    
    返回:
        deviations: dict, {jobnr: [u_abs_mode1, u_abs_mode2, ...]}
        cost: list of list, [[cost_job0_mode1, ...], [cost_job1_mode1, ...], ...]
    """
    # 按 jobnr 分组
    jobs_data = {}
    for row in meta_data:
        jobnr = row['jobnr']
        if jobnr not in jobs_data:
            jobs_data[jobnr] = []
        jobs_data[jobnr].append(row)
    
    # 构建 deviations: {jobnr: [u_abs for each mode]}
    deviations = {}
    for jobnr in sorted(jobs_data.keys()):
        modes = jobs_data[jobnr]
        deviations[jobnr] = [row['u_abs'] for row in modes]
    
    # 构建 cost: [[cost for each mode] for each job]
    # 注意：cost 是 list of list，索引从 0 开始
    cost = []
    for jobnr in sorted(jobs_data.keys()):
        modes = jobs_data[jobnr]
        cost.append([row['cost'] for row in modes])
    
    return deviations, cost


def generate_mode_meta_from_mm(mm_path, output_path=None, seed=None, cost_factor=1):
    """
    从 .mm 文件生成模式元数据 CSV 文件，并返回 deviations 和 cost 数据结构。
    
    这是供其他 Python 模块调用的接口函数。
    
    参数:
        mm_path: str, 输入的 .mm 文件路径
        output_path: str, 输出的 CSV 文件路径（可选，默认自动生成）
        seed: int, 随机种子（可选）
        cost_factor: int, 成本因子（默认为1）
    
    返回:
        csv_path: str, 生成的 CSV 文件路径
        deviations: dict, {jobnr: [u_abs_mode1, u_abs_mode2, ...]}
        cost: list of list, [[cost_job0_mode1, ...], [cost_job1_mode1, ...], ...]
    """
    # 默认输出文件名：与 mm 文件同目录，文件名为 <basename>_mode_meta.csv
    if output_path is None:
        dir_name = os.path.dirname(mm_path)
        base_name = os.path.splitext(os.path.basename(mm_path))[0]
        output_path = os.path.join(dir_name, f"{base_name}_mode_meta.csv") if dir_name else f"{base_name}_mode_meta.csv"
    
    # 解析 mm 文件
    jobs, num_renewable = parse_mm_file(mm_path)
    
    if not jobs:
        raise ValueError(f"未能从文件中解析出活动数据: {mm_path}")
    
    # 生成模式元数据
    meta_data = generate_mode_meta(jobs, num_renewable, seed=seed, cost_factor=cost_factor)
    
    # 写入 CSV
    write_csv(meta_data, output_path)
    
    # 转换为 deviations 和 cost 数据结构
    deviations, cost = meta_to_deviations_and_cost(meta_data)
    
    return output_path, deviations, cost


def main():
    if len(sys.argv) < 2:
        print("用法: python generate_mode_meta.py <input.mm> [output.csv] [--seed=<int>] [--cost-factor=<int>]")
        print("\n选项:")
        print("  --seed=<int>        设置随机种子以获得可重复的结果")
        print("  --cost-factor=<int> 设置成本因子（默认为1）")
        print("\n生成逻辑（确保trade-off）:")
        print("  - 持续时间短 → 成本高（花钱买时间）")
        print("  - 持续时间长 → 成本低(0)（基准/经济模式）")
        print("  - 资源需求高 → 偏离小")
        print("  - 资源需求低 → 偏离大")
        print("\nPython 模块调用:")
        print("  from generate_mode_meta import generate_mode_meta_from_mm")
        print("  csv_path, deviations, cost = generate_mode_meta_from_mm('input.mm')")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # 解析可选参数
    output_file = None
    seed = None
    cost_factor = 1
    
    for arg in sys.argv[2:]:
        if arg.startswith('--seed='):
            seed = int(arg.split('=')[1])
        elif arg.startswith('--cost-factor='):
            cost_factor = int(arg.split('=')[1])
        elif not arg.startswith('--'):
            output_file = arg
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        sys.exit(1)
    
    # 生成 CSV
    print(f"正在解析: {input_file}")
    output_path, deviations, cost = generate_mode_meta_from_mm(input_file, output_file, seed=seed, cost_factor=cost_factor)
    print(f"已生成: {output_path}")
    
    # 打印摘要
    jobs, _ = parse_mm_file(input_file)
    total_modes = sum(len(modes) for modes in jobs.values())
    print(f"\n生成摘要:")
    print(f"  总活动数: {len(jobs)}")
    print(f"  总模式数: {total_modes}")
    if seed is not None:
        print(f"  随机种子: {seed}")
    print(f"  成本因子: {cost_factor}")
    
    print(f"\n数据结构预览:")
    print(f"  deviations: {deviations}")
    print(f"  cost: {cost}")


if __name__ == '__main__':
    main()