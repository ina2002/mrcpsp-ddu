# MRCPSP-DDU 统一求解框架设计文档

## 1. 框架概述

本框架为**带决策依赖不确定性的多模式资源受限项目调度问题 (MRCPSP-DDU)** 提供了一个统一的求解入口。框架整合了两种不同的求解算法，用户可以通过命令行参数灵活选择。

### 1.1 核心问题

MRCPSP-DDU 问题的目标是在资源约束和工期不确定性下，为每个任务选择执行模式并确定执行顺序，以最小化综合成本（模式成本 + 工期惩罚）。

### 1.2 两种求解算法

| 算法 | 文件 | 核心思想 | 适用场景 |
|------|------|----------|----------|
| **CCG** | `ccg.py` | Column-and-Constraint Generation，通过迭代生成极点和约束来逼近最优解 | 理论研究、中小规模问题 |
| **Benders** | `benders.py` | Benders 分解，将问题分解为主问题（决策）和子问题（评估）迭代求解 | 工程应用、大规模问题 |

两种算法求解的是**同一个问题**，但采用了不同的数学分解策略。

## 2. 框架架构

```
┌─────────────────────────────────────────────────────────────┐
│                        run.py                                │
│                    (统一入口脚本)                             │
│  - 解析命令行参数                                             │
│  - 调用相应算法                                               │
│  - 格式化输出结果                                             │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│      CCG 算法分支        │     │    Benders 算法分支      │
│                         │     │                         │
│  translate.py           │     │  mrcpsp.py              │
│  (解析 .mm → JSON)      │     │  (解析 .mm → Instance)  │
│         │               │     │         │               │
│         ▼               │     │         ▼               │
│  ccg.py                 │     │  benders.py             │
│  (Variant2Solver)       │     │  (Benders 类)           │
└─────────────────────────┘     └─────────────────────────┘
```

## 3. 文件职责

| 文件 | 职责 |
|------|------|
| `run.py` | **统一入口**：解析参数、调用算法、输出结果 |
| `translate.py` | **文件解析器**：将 PSPLIB `.mm` 文件转换为 JSON 格式（供 CCG 使用） |
| `mrcpsp.py` | **数据模型**：定义 `Instance`、`Job` 类，直接解析 `.mm` 文件（供 Benders 使用） |
| `ccg.py` | **CCG 算法**：实现 `Variant2Solver` 类，使用 Column-and-Constraint Generation 方法 |
| `benders.py` | **Benders 算法**：实现 `Benders` 类，使用 Benders 分解方法 |

## 4. 使用方法

### 4.1 基本用法

```bash
# 使用 CCG 算法
python run.py --input <mm文件> --algorithm ccg --gamma <Gamma值>

# 使用 Benders 算法
python run.py --input <mm文件> --algorithm benders --gamma <Gamma值>
```

### 4.2 完整参数说明

```
必需参数:
  --input, -i           PSPLIB .mm 文件路径
  --algorithm, -a       求解算法: 'ccg' 或 'benders'

通用参数:
  --gamma, -g           鲁棒性参数 Gamma（默认: 2）
  --verbose, -v         打印详细求解日志
  --output, -o          输出结果到 JSON 文件
  --e-over              工期惩罚系数（默认: 1.0）

Benders 特定参数:
  --time-limit, -t      时间限制，单位秒（默认: 60）
  --uncertainty-level   不确定性水平（默认: 0.7）

CCG 特定参数:
  --max-iter            最大迭代次数（默认: 50）
  --mode-meta           模式元数据 CSV 文件路径
```

### 4.3 使用示例

```bash
# 示例 1: 使用 Benders 算法，Gamma=2，时间限制 60 秒
python run.py -i mrcpsp_toy_example.mm -a benders -g 2 -t 60

# 示例 2: 使用 CCG 算法，带模式元数据
python run.py -i mrcpsp_toy_example.mm -a ccg -g 2 --mode-meta mrcpsp_toy_mode_meta.csv

# 示例 3: 详细输出并保存结果到 JSON
python run.py -i mrcpsp_toy_example.mm -a benders -g 2 -v -o result.json
```

## 5. 输出格式

### 5.1 控制台输出

```
============================================================
求解结果
============================================================
算法:           BENDERS
实例:           mrcpsp_toy_example
Gamma:          2
目标值:         21.0000
目标下界:       21.0000
运行时间:       1.27 秒
迭代次数:       30
平均迭代时间:   0.0418 秒

模式选择:       {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
网络结构:       [(0, 1), (0, 2), (0, 3), ...]
============================================================
```

### 5.2 JSON 输出（使用 `--output` 参数）

```json
{
  "algorithm": "benders",
  "instance": "mrcpsp_toy_example",
  "gamma": 2,
  "objective_value": 21.0,
  "objective_bound": 21.0,
  "runtime": 1.27,
  "iterations": 30,
  "modes": {"0": 0, "1": 0, "2": 0, ...},
  "network": [[0, 1], [0, 2], [0, 3], ...]
}
```

## 6. 扩展指南

### 6.1 添加新算法

如需添加新的求解算法，请按以下步骤操作：

1. 创建新的算法文件（如 `new_algorithm.py`）
2. 在 `run.py` 中添加导入和调用逻辑
3. 在 `argparse` 中添加 `--algorithm` 的新选项
4. 实现 `solve_with_new_algorithm()` 函数

### 6.2 自定义成本函数

可以通过修改 `run.py` 中的 `cost` 参数来自定义模式成本：

```python
# 自定义成本矩阵
cost = [[0], [10, 20], [5, 15, 25], ...]  # 每个任务每个模式的成本
```

## 7. 依赖项

- Python 3.8+
- Gurobi (gurobipy)
- 标准库: json, argparse, pathlib, typing

## 8. 注意事项

1. **Gurobi 许可证**：两种算法都依赖 Gurobi 求解器。CCG 算法生成的模型规模较大，可能需要完整的商业许可证。

2. **Gamma 参数**：Gamma 控制鲁棒性水平，值越大表示考虑更多的不确定性场景，但求解难度也会增加。

3. **时间限制**：对于大规模问题，建议适当增加 `--time-limit` 参数值。
