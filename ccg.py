from __future__ import annotations

import random
rnd = random.Random(2026)
u_noise = 0.1

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Any, Optional
import math

import gurobipy as gp
from gurobipy import GRB


# =========================
# Data structures
# =========================

@dataclass(frozen=True)
class Mode:
    bar_d: float
    u: float                     # ABSOLUTE deviation upper bound (same unit as bar_d)
    r: Dict[int, float]          # renewable resources only, k=1..K


@dataclass
class DDUInstance:
    N_real: int
    K: int
    R: Dict[int, float]          # renewable capacities, k=1..K
    modes: Dict[int, Dict[int, Mode]]     # modes[i][m]
    A_prec: Set[Tuple[int, int]]          # forced precedence arcs (0-based nodes)
    Gamma: float
    e_overhead: float
    b_price: Dict[int, float]
    M_big: float                 # time big-M in recourse constraints
    c_fixed: Optional[Dict[Tuple[int, int], float]] = None  # (i,m)->cost override (toy)
    use_flow: bool = True


# =========================
# Helpers
# =========================

def complete_arc_set(n_nodes: int, sink: int) -> List[Tuple[int, int]]:
    """
    Candidate arc set P following the paper:
    - i != j
    - exclude arcs into source: (j,0) not in P
    - exclude arcs out of sink: (sink,j) not in P
    """
    P: List[Tuple[int, int]] = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            if j == 0:
                continue
            if i == sink:
                continue
            P.append((i, j))
    return P


def compute_c_im(inst: DDUInstance, i: int, m: int) -> float:
    """Example: c_im = sum_k r_imk * bar_d_im * b_k (resource-hour price).
    Toy interface: if inst.c_fixed provides (i,m), override cost.
    """
    if inst.c_fixed is not None and (i, m) in inst.c_fixed:
        return float(inst.c_fixed[(i, m)])
    mode = inst.modes[i][m]
    return sum(mode.r.get(k, 0.0) * mode.bar_d * inst.b_price[k] for k in range(1, inst.K + 1))


def activity_Dmax(inst: DDUInstance, i: int) -> float:
    """Upper bound of activity i duration among all modes: max_m (bar_d_im + u_im)."""
    return max(inst.modes[i][m].bar_d + inst.modes[i][m].u for m in inst.modes[i])


def build_instance_from_psplib_json(
    data: Dict[str, Any],
    Gamma: float,
    e_overhead: float,
    b_price: Optional[Dict[int, float]] = None,
    mode_meta_csv: Optional[str] = None,
    u_max: float = 0.3,
    u_min: float = 0.0,
    M_big: Optional[float] = None,
    use_flow: bool = True,
) -> DDUInstance:
    """
    JSON -> DDUInstance

    编号映射（PSPLIB 标准）：
    - jobnr: 1 是 dummy source, (n+2) 是 dummy sink
    - 模型节点：0..N+1，其中 0 对应 jobnr=1，N+1 对应 jobnr=n+2
    - 现实活动：1..N，对应 jobnr=2..n+1

    生成 u_im 的方案（工期越长越不确定）：
    - 先生成相对偏差比例 u_rel ∈ [u_min, u_max]
    - 再转为绝对偏差：u_abs = bar_d * u_rel
    """
    jobs_incl = int(data["jobs_incl_dummy"])          # n+2
    N_real = jobs_incl - 2                            # n
    K = int(data["n_renew"])

    if u_max < u_min:
        raise ValueError(f"u_max({u_max}) 必须 >= u_min({u_min})")

    # ---------- renewable capacities R_k ----------
    caps = (
        data.get("R_renew")
        or data.get("renewable_capacities")
        or data.get("renewable_capacity")
        or data.get("R")
        or data.get("cap_renew")
    )
    if caps is None:
        raise ValueError("JSON 中未找到可再生资源容量字段（例如 R_renew / renewable_capacities / R）。请检查 translate.py 输出键名。")

    if isinstance(caps, dict):
        R = {int(k): float(v) for k, v in caps.items()}
    else:
        caps = list(caps)
        if len(caps) < K:
            raise ValueError("renewable capacities 长度不足 K")
        R = {k: float(caps[k-1]) for k in range(1, K+1)}

    if b_price is None:
        b_price = {k: 1.0 for k in range(1, K + 1)}
    else:
        for k in range(1, K + 1):
            if k not in b_price:
                raise ValueError("b_price 必须覆盖 1..K")

    jobs = data["jobs"]

    # ---------- (可选) toy 接口：读取每个 (i,m) 的 u_abs 与 cost ----------
    meta_u: Dict[Tuple[int, int], float] = {}
    meta_c: Dict[Tuple[int, int], float] = {}
    if mode_meta_csv is not None:
        import csv
        with open(mode_meta_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("jobnr") is None or row.get("mode") is None:
                    continue
                jobnr = int(row["jobnr"])
                mm = int(row["mode"])
                # 只覆盖现实活动：jobnr=2..N_real+1
                if jobnr < 2 or jobnr > (N_real + 1):
                    continue
                i = jobnr - 1
                # u_abs：优先读 u_abs；否则用 maxT-barT
                u_abs = None
                if row.get("u_abs") not in (None, ""):
                    u_abs = float(row["u_abs"])
                elif row.get("maxT") not in (None, "") and row.get("barT") not in (None, ""):
                    u_abs = float(row["maxT"]) - float(row["barT"])
                if u_abs is not None:
                    meta_u[(i, mm)] = float(u_abs)
                # cost：优先读 cost；否则读 Money
                cost_val = None
                if row.get("cost") not in (None, ""):
                    cost_val = float(row["cost"])
                elif row.get("Money") not in (None, ""):
                    cost_val = float(row["Money"])
                if cost_val is not None:
                    meta_c[(i, mm)] = float(cost_val)


    # precedence arcs: (jobnr_i, jobnr_j) -> (node_i, node_j) where node = jobnr-1
    A_prec: Set[Tuple[int, int]] = set((int(i) - 1, int(j) - 1) for i, j in data["E_prec"])

    # ---------- 收集所有现实活动所有模式 bar_d 做归一化 ----------
    durations_all: List[float] = []
    for i in range(1, N_real + 1):
        jobnr = i + 1
        info = jobs.get(str(jobnr))
        if info is None:
            raise ValueError(f"JSON 缺少 job {jobnr} 的信息（对应现实活动 i={i}）。")
        durs = info.get("durations", {})
        for dur in durs.values():
            dur = float(dur)
            if dur > 1e-12:
                durations_all.append(dur)

    if not durations_all:
        raise ValueError("未收集到任何正 duration，无法生成 u_im。")

    d_min = min(durations_all)
    d_max = max(durations_all)

    def norm(d: float) -> float:
        if d_max - d_min <= 1e-12:
            return 0.0
        return (d - d_min) / (d_max - d_min)

    # ---------- 构建 modes ----------
    modes: Dict[int, Dict[int, Mode]] = {}
    for i in range(1, N_real + 1):
        jobnr = i + 1
        info = jobs.get(str(jobnr))
        nmodes = int(info.get("modes", 1))
        modes[i] = {}

        durs = info.get("durations", {})
        reqs = info.get("req", {})

        for m in range(1, nmodes + 1):
            if str(m) not in durs or str(m) not in reqs:
                continue
            dur = float(durs[str(m)])
            if dur <= 1e-12:
                raise ValueError(f"现实活动 i={i} (jobnr={jobnr}) 的 mode {m} duration=0，不允许。")

            req = reqs[str(m)]
            r_dict = {k: float(req[k - 1]) for k in range(1, K + 1)}            # 若提供 toy CSV，则用 meta_u[(i,m)] 覆盖；否则按原逻辑生成
            if (i, m) in meta_u:
                u_abs = float(meta_u[(i, m)])
            else:
                d_norm = norm(dur)
                base = u_min + (u_max - u_min) * d_norm
                u_rel = base + rnd.uniform(-u_noise, u_noise)
                u_rel = max(u_min, min(u_max, u_rel))
                u_abs = dur * u_rel
            modes[i][m] = Mode(bar_d=dur, u=u_abs, r=r_dict)

        if len(modes[i]) == 0:
            raise ValueError(f"现实活动 i={i} (jobnr={jobnr}) 没有解析到任何可用 mode。")

    # ---------- 计算 time big-M（论文：sum_i max_m (bar+u)） ----------
    if M_big is None:
        tmp = DDUInstance(
            N_real=N_real, K=K, R=R, modes=modes, A_prec=A_prec,
            Gamma=Gamma, e_overhead=e_overhead, b_price=b_price, M_big=1.0, c_fixed=(meta_c if mode_meta_csv is not None else None), use_flow=use_flow
        )
        M_big = sum(activity_Dmax(tmp, i) for i in range(1, N_real + 1))

    return DDUInstance(
        N_real=N_real,
        K=K,
        R=R,
        modes=modes,
        A_prec=A_prec,
        Gamma=Gamma,
        e_overhead=e_overhead,
        b_price=b_price,
        M_big=float(M_big),
        c_fixed=(meta_c if mode_meta_csv is not None else None),
        use_flow=use_flow
    )


# =========================
# Solver: Simplified Variant 2 (MP2 with recourse replication)
# =========================

class Variant2Solver:
    """
    简化 Variant 2（只生成极点）：
    - Master: MP2，对每个极点 pi^k 复制 OU(x,pi^k) + recourse(s^k) 约束
    - SP2: 在 Π(Ay) 上生成极点 pi*
    - 启用资源流 f_{i,j,k}（与论文的一阶段约束一致）
    """

    def __init__(
        self,
        inst: DDUInstance,
        tol: float = 1e-6,
        max_iter: int = 50,
        verbose: bool = True,
        alpha_max: float = 1e4,
    ):
        self.inst = inst
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.alpha_max = float(alpha_max)

        self.nodes = list(range(0, inst.N_real + 2))  # 0..N+1
        self.real = list(range(1, inst.N_real + 1))   # 1..N
        self.sink = inst.N_real + 1

        # Candidate arc set P (paper-compliant)
        self.P = complete_arc_set(len(self.nodes), self.sink)

        # Precompute adjacency lists on P
        self.P_out: Dict[int, List[int]] = {i: [] for i in self.nodes}
        self.P_in: Dict[int, List[int]] = {i: [] for i in self.nodes}
        for (i, j) in self.P:
            self.P_out[i].append(j)
            self.P_in[j].append(i)

        # 极点集合（路径流）：pi^k_{ij} ∈ {0,1}
        self.pis: List[Dict[Tuple[int, int], int]] = []

    # -------------------------
    # Master (MP2)
    # -------------------------

    def build_master(self) -> gp.Model:
        inst = self.inst
        m = gp.Model("MP2_variant2_recourse_flow_absDDU")
        m.Params.OutputFlag = 1 if self.verbose else 0

        # 1) 一阶段变量：x, y, theta, f, eta
        x: Dict[Tuple[int, int], gp.Var] = {}
        for i in self.real:
            for mm in inst.modes[i]:
                x[i, mm] = m.addVar(vtype=GRB.BINARY, name=f"x[{i},{mm}]")

        y: Dict[Tuple[int, int], gp.Var] = {}
        for (i, j) in self.P:
            y[i, j] = m.addVar(vtype=GRB.BINARY, name=f"y[{i},{j}]")

        n = len(self.nodes)
        Mtheta = n - 1

        theta = m.addVars(
            self.nodes,
            lb=0.0,
            ub=n - 1,
            vtype=GRB.CONTINUOUS,
            name="theta"
        )
        m.addConstr(theta[0] == 0.0, name="theta0")

        # 资源流变量（连续）
        f: Dict[Tuple[int, int, int], gp.Var] = {}
        if inst.use_flow:
            for (i, j) in self.P:
                for k in range(1, inst.K + 1):
                    f[i, j, k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"f[{i},{j},{k}]")

        eta = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="eta")

        # 2) 一阶段基本约束
        # (1) 每个现实活动选择一种模式
        for i in self.real:
            m.addConstr(gp.quicksum(x[i, mm] for mm in inst.modes[i]) == 1, name=f"mode_select[{i}]")

        # (2) 强制紧前关系 y_{ij}=1 for (i,j) in A
        for (i, j) in inst.A_prec:
            if (i, j) not in y:
                raise ValueError(f"forced precedence arc {(i,j)} not in candidate P. Ensure A ⊆ P.")
            m.addConstr(y[i, j] == 1, name=f"prec[{i},{j}]")

        # (3) 无环（theta 约束，只对 (i,j) in P）
        for (i, j) in self.P:
            m.addConstr(
                theta[j] - theta[i] >= 1 - Mtheta * (1 - y[i, j]),
                name=f"acyclic[{i},{j}]"
            )

        # 3) 资源流约束（严格按论文：cap_mode + netflow + flowout_rho + strengthening UBs）
        if inst.use_flow:
            def rho_expr(i: int, k: int) -> gp.LinExpr:
                if i in self.real:
                    return gp.quicksum(x[i, mm] * inst.modes[i][mm].r.get(k, 0.0) for mm in inst.modes[i])
                return gp.LinExpr(0.0)

            # (cap_mode) rho_{ik}(x) <= R_k
            for i in self.real:
                for k in range(1, inst.K + 1):
                    m.addConstr(rho_expr(i, k) <= inst.R[k], name=f"cap_mode[{i},{k}]")

            # Precompute rU for real-real arcs
            rU: Dict[Tuple[int, int, int], float] = {}
            for i in self.real:
                for j in self.real:
                    if i == j:
                        continue
                    for k in range(1, inst.K + 1):
                        max_i = max(inst.modes[i][mm].r.get(k, 0.0) for mm in inst.modes[i])
                        max_j = max(inst.modes[j][nn].r.get(k, 0.0) for nn in inst.modes[j])
                        rU[i, j, k] = max(max_i, max_j)

            # (flow_ub_rr) real->real strengthening (m,n enumeration)
            for i in self.real:
                for j in self.real:
                    if i == j:
                        continue
                    if (i, j) not in y:
                        continue
                    for k in range(1, inst.K + 1):
                        for mm in inst.modes[i]:
                            r_imk = inst.modes[i][mm].r.get(k, 0.0)
                            for nn in inst.modes[j]:
                                r_jnk = inst.modes[j][nn].r.get(k, 0.0)
                                rL = min(r_imk, r_jnk)
                                m.addConstr(
                                    f[i, j, k]
                                    <= rL * y[i, j]
                                    + (2 - x[i, mm] - x[j, nn]) * (rU[i, j, k] - rL),
                                    name=f"flow_ub_rr[{i},{j},{k},{mm},{nn}]"
                                )

            # (flow_ub_0r) source->real: f_{0jk} <= rho_{jk}(x) * y_{0j}
            for j in self.real:
                if (0, j) not in y:
                    continue
                for k in range(1, inst.K + 1):
                    m.addConstr(
                        f[0, j, k] <= rho_expr(j, k) * y[0, j],
                        name=f"flow_ub_0r[{j},{k}]"
                    )

            # (flow_ub_rs) real->sink: f_{i,sink,k} <= rho_{ik}(x) * y_{i,sink}
            for i in self.real:
                if (i, self.sink) not in y:
                    continue
                for k in range(1, inst.K + 1):
                    m.addConstr(
                        f[i, self.sink, k] <= rho_expr(i, k) * y[i, self.sink],
                        name=f"flow_ub_rs[{i},{k}]"
                    )

            # (netflow_piecewise) out - in = {R,0,-R}
            for i in self.nodes:
                for k in range(1, inst.K + 1):
                    outflow = gp.quicksum(f[i, j, k] for j in self.P_out[i])
                    inflow = gp.quicksum(f[j, i, k] for j in self.P_in[i])

                    if i == 0:
                        rhs = inst.R[k]
                    elif i == self.sink:
                        rhs = -inst.R[k]
                    elif i in self.real:
                        rhs = 0.0
                    else:
                        rhs = 0.0
                    m.addConstr(outflow - inflow == rhs, name=f"netflow[{i},{k}]")

            # (flowout_rho) sum_j f_{i,j,k} = rho_{i,k}(x), i in real
            for i in self.real:
                for k in range(1, inst.K + 1):
                    outflow = gp.quicksum(f[i, j, k] for j in self.P_out[i])
                    m.addConstr(outflow == rho_expr(i, k), name=f"flowout_rho[{i},{k}]")

        # 4) 目标：min C1(x) + e_overhead * eta
        C1 = gp.quicksum(
            compute_c_im(inst, i, mm) * x[i, mm]
            for i in self.real
            for mm in inst.modes[i]
        )
        m.setObjective(C1 + inst.e_overhead * eta, GRB.MINIMIZE)

        # 5) 对每个已加入极点 pi^k 添加 OU-block + recourse(s^k) block
        for k_idx, pi in enumerate(self.pis):
            self._add_OU_and_recourse_block(m, x, y, eta, pi, k_idx)

        m.update()

        # 保存引用
        m._x = x
        m._y = y
        m._eta = eta
        m._theta = theta
        m._f = f
        return m

    def _add_OU_and_recourse_block(
        self,
        m: gp.Model,
        x: Dict[Tuple[int, int], gp.Var],
        y: Dict[Tuple[int, int], gp.Var],
        eta: gp.Var,
        pi: Dict[Tuple[int, int], int],
        k_idx: int
    ) -> None:
        """
        给定极点 pi^k：
        1) OU(x,pi^k) 块：生成最坏持续时间 d^k（绝对偏差预算集 D(x)）
        2) recourse 块：原变量 s^k 及差分约束，并 eta >= s_sink^k
        """
        inst = self.inst

        # q_i = sum_j pi_{ij}（现实活动）
        q = {i: 0 for i in self.real}
        for (i, j), val in pi.items():
            if val == 1 and i in q:
                q[i] += 1

        # --------- OU variables ---------
        d = {i: m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"d[{k_idx},{i}]") for i in self.real}
        alphaL = {i: m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"aL[{k_idx},{i}]") for i in self.real}
        alphaU = {i: m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"aU[{k_idx},{i}]") for i in self.real}
        alpha = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"a[{k_idx}]")

        Dmax = {i:activity_Dmax(inst, i) for i in self.real}
        AlphaMax = self.alpha_max

        # linearization vars for products with x
        zD, zAL, zAU, zA = {}, {}, {}, {}
        for i in self.real:
            for mm in inst.modes[i]:
                zD[i, mm]  = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"zD[{k_idx},{i},{mm}]")
                zAL[i, mm] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"zAL[{k_idx},{i},{mm}]")
                zAU[i, mm] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"zAU[{k_idx},{i},{mm}]")
                zA[i, mm]  = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"zA[{k_idx},{i},{mm}]")

                # zD = d * x
                m.addConstr(zD[i, mm] <= Dmax[i] * x[i, mm], name=f"zD_ub1[{k_idx},{i},{mm}]")
                m.addConstr(zD[i, mm] <= d[i],            name=f"zD_ub2[{k_idx},{i},{mm}]")
                m.addConstr(zD[i, mm] >= d[i] - Dmax[i] * (1 - x[i, mm]), name=f"zD_lb[{k_idx},{i},{mm}]")

                # zAL = alphaL * x
                m.addConstr(zAL[i, mm] <= AlphaMax * x[i, mm], name=f"zAL_ub1[{k_idx},{i},{mm}]")
                m.addConstr(zAL[i, mm] <= alphaL[i],          name=f"zAL_ub2[{k_idx},{i},{mm}]")
                m.addConstr(zAL[i, mm] >= alphaL[i] - AlphaMax * (1 - x[i, mm]), name=f"zAL_lb[{k_idx},{i},{mm}]")

                # zAU = alphaU * x
                m.addConstr(zAU[i, mm] <= AlphaMax * x[i, mm], name=f"zAU_ub1[{k_idx},{i},{mm}]")
                m.addConstr(zAU[i, mm] <= alphaU[i],          name=f"zAU_ub2[{k_idx},{i},{mm}]")
                m.addConstr(zAU[i, mm] >= alphaU[i] - AlphaMax * (1 - x[i, mm]), name=f"zAU_lb[{k_idx},{i},{mm}]")

                # zA = alpha * x
                m.addConstr(zA[i, mm] <= AlphaMax * x[i, mm], name=f"zA_ub1[{k_idx},{i},{mm}]")
                m.addConstr(zA[i, mm] <= alpha,              name=f"zA_ub2[{k_idx},{i},{mm}]")
                m.addConstr(zA[i, mm] >= alpha - AlphaMax * (1 - x[i, mm]), name=f"zA_lb[{k_idx},{i},{mm}]")

        # primal bounds: d_i in [L(x), U(x)] with ABS u
        for i in self.real:
            Lx = gp.quicksum(inst.modes[i][mm].bar_d * x[i, mm] for mm in inst.modes[i])
            Ux = gp.quicksum((inst.modes[i][mm].bar_d + inst.modes[i][mm].u) * x[i, mm] for mm in inst.modes[i])
            m.addConstr(d[i] >= Lx, name=f"d_lb[{k_idx},{i}]")
            m.addConstr(d[i] <= Ux, name=f"d_ub[{k_idx},{i}]")

        # ABS budget:
        # sum_{i,m in M_i+} (d_i - bar_im)/u_im * x_im <= Gamma
        # <=> sum zD/u <= Gamma + sum (bar/u)*x
        lhs = gp.LinExpr(0.0)
        rhs_extra = gp.LinExpr(0.0)
        for i in self.real:
            for mm in inst.modes[i]:
                u = inst.modes[i][mm].u
                if u <= 1e-12:
                    continue
                bar = inst.modes[i][mm].bar_d
                lhs += zD[i, mm] / u
                rhs_extra += (bar / u) * x[i, mm]
        m.addConstr(lhs <= inst.Gamma + rhs_extra, name=f"budget_abs[{k_idx}]")

        # dual constraints: -aL_i + aU_i + sum_{m in M_i+} (alpha/u_im)*x_im = q_i
        for i in self.real:
            ax_alpha = gp.LinExpr(0.0)
            for mm in inst.modes[i]:
                u = inst.modes[i][mm].u
                if u <= 1e-12:
                    continue
                ax_alpha += zA[i, mm] / u
            m.addConstr(-alphaL[i] + alphaU[i] + ax_alpha == float(q[i]), name=f"dual_feas[{k_idx},{i}]")

        # strong duality
        primal_obj = gp.quicksum(float(q[i]) * d[i] for i in self.real)

        dual_obj = gp.LinExpr(0.0)
        # lower bounds
        for i in self.real:
            for mm in inst.modes[i]:
                dual_obj += -inst.modes[i][mm].bar_d * zAL[i, mm]
        # upper bounds
        for i in self.real:
            for mm in inst.modes[i]:
                dual_obj += (inst.modes[i][mm].bar_d + inst.modes[i][mm].u) * zAU[i, mm]
        # budget part: Gamma*alpha + sum_{u>0} (bar/u) * (alpha*x) = Gamma*alpha + sum (bar/u)*zA
        dual_obj += inst.Gamma * alpha
        for i in self.real:
            for mm in inst.modes[i]:
                u = inst.modes[i][mm].u
                if u <= 1e-12:
                    continue
                dual_obj += (inst.modes[i][mm].bar_d / u) * zA[i, mm]

        m.addConstr(primal_obj == dual_obj, name=f"strong_duality[{k_idx}]")

        # -----------------------
        # Recourse block: s^k variables and precedence constraints
        # -----------------------
        s = {v: m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"s[{k_idx},{v}]") for v in self.nodes}
        m.addConstr(s[0] == 0.0, name=f"s0[{k_idx}]")

        # time precedence only for arcs in P (paper)
        for (i, j) in self.P:
            dur_i = 0.0
            if i in self.real:
                dur_i = d[i]
            m.addConstr(
                s[j] - s[i] >= dur_i - inst.M_big * (1 - y[i, j]),
                name=f"time_prec[{k_idx},{i},{j}]"
            )

        # eta >= makespan of this block
        m.addConstr(eta >= s[self.sink], name=f"eta_ge_makespan[{k_idx}]")

    # -------------------------
    # Solve master
    # -------------------------

    def solve_master(self):
        mp = self.build_master()
        mp.optimize()
        if mp.Status != GRB.OPTIMAL:
            raise RuntimeError(f"主问题未最优，status={mp.Status}")

        x_sol = {(i, mm): int(round(v.X)) for (i, mm), v in mp._x.items()}
        y_sol = {(i, j): int(round(v.X)) for (i, j), v in mp._y.items()}
        eta_sol = float(mp._eta.X)
        obj = float(mp.ObjVal)
        return x_sol, y_sol, eta_sol, obj

    # -------------------------
    # SP2: generate extreme point pi* in Π(Ay)
    # -------------------------

    def solve_SP2(self, x_sol, y_sol):
        """
        SP2：在候选弧集 P 上分离 pi ∈ Π 的极点（0-1 简单路径流）。
        采用绝对偏差预算集：d_i ∈ [bar_i, bar_i+u_i], sum (d_i-bar_i)/u_i <= Gamma（u_i>0）。
        """
        inst = self.inst
        sp = gp.Model("SP2_Pi_absDDU")
        sp.Params.OutputFlag = 1 if self.verbose else 0

        # 1) rho_{ij}：弧集 P 上的 0-1 路径变量（对应极点 pi）
        rho = {arc: sp.addVar(vtype=GRB.BINARY, name=f"rho[{arc[0]},{arc[1]}]") for arc in self.P}

        # 禁止直接 0->sink 的“空路径”（增强分离）
        if (0, self.sink) in rho:
            sp.addConstr(rho[0, self.sink] == 0, name="no_direct_0_sink")

        # 2) s-t 单位流守恒 + 入出度<=1（保证是简单路径极点）
        for v in self.nodes:
            out = gp.quicksum(rho[v, j] for j in self.P_out[v])
            inn = gp.quicksum(rho[i, v] for i in self.P_in[v])

            if v == 0:
                sp.addConstr(out == 1, name="src_out")
                sp.addConstr(inn == 0, name="src_in")
            elif v == self.sink:
                sp.addConstr(out == 0, name="sink_out")
                sp.addConstr(inn == 1, name="sink_in")
            else:
                sp.addConstr(out == inn, name=f"flow_bal[{v}]")
                sp.addConstr(out <= 1,  name=f"deg_out[{v}]")
                sp.addConstr(inn <= 1,  name=f"deg_in[{v}]")

        # 3) z_i：活动 i 是否在路径上（出度=1 即在路径上）
        z = {i: sp.addVar(vtype=GRB.BINARY, name=f"z[{i}]") for i in self.real}
        for i in self.real:
            sp.addConstr(z[i] == gp.quicksum(rho[i, j] for j in self.P_out[i]),
                         name=f"zdef[{i}]")

        # 防退化：路径必须经过至少一个现实活动
        sp.addConstr(gp.quicksum(z[i] for i in self.real) >= 1, name="use_at_least_one_real")

        # 4) 在给定 x_sol 下确定每个 i 的 (bar_i, u_i) —— u_i 为 ABS 上界
        bar, u = {}, {}
        for i in self.real:
            chosen = None
            for mm in inst.modes[i]:
                if x_sol.get((i, mm), 0) == 1:
                    chosen = mm
                    break
            if chosen is None:
                chosen = next(iter(inst.modes[i].keys()))
            bar[i] = inst.modes[i][chosen].bar_d
            u[i] = inst.modes[i][chosen].u

        # 5) 绝对不确定持续时间：d_i ∈ [bar_i, bar_i+u_i]，sum_{u_i>0} (d_i-bar_i)/u_i <= Gamma
        d = {i: sp.addVar(vtype=GRB.CONTINUOUS, lb=bar[i], ub=bar[i] + u[i], name=f"d[{i}]") for i in self.real}
        lhs = gp.LinExpr(0.0)
        for i in self.real:
            if u[i] > 1e-12:
                lhs += (d[i] - bar[i]) / u[i]
        sp.addConstr(lhs <= inst.Gamma, name="budget_abs")

        # 6) w_i = d_i * z_i（线性化）
        w = {i: sp.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"w[{i}]") for i in self.real}
        for i in self.real:
            Dmax = bar[i] + u[i]
            sp.addConstr(w[i] <= d[i], name=f"wub1[{i}]")
            sp.addConstr(w[i] <= Dmax * z[i], name=f"wub2[{i}]")
            sp.addConstr(w[i] >= d[i] - Dmax * (1 - z[i]), name=f"wlb[{i}]")

        # 7) 目标：max  Σ_{i in path} d_i  -  M Σ_{(i,j) in P} (1-y_ij)*rho_ij
        penalty = gp.quicksum(inst.M_big * (1 - y_sol[i, j]) * rho[i, j] for (i, j) in self.P)
        sp.setObjective(gp.quicksum(w[i] for i in self.real) - penalty, GRB.MAXIMIZE)

        sp.optimize()
        if sp.Status != GRB.OPTIMAL:
            raise RuntimeError(f"SP2 未最优，status={sp.Status}")

        val = float(sp.ObjVal)
        pi_star = {arc: int(round(rho[arc].X)) for arc in self.P}

        # ---- store worst-case durations d* (and xi) for plotting/debug ----
        d_star = {i: float(d[i].X) for i in self.real}
        xi_star = {}
        budget_used = 0.0
        for i in self.real:
            if u[i] > 1e-12:
                xi = (d_star[i] - bar[i]) / u[i]
                xi_star[i] = float(xi)
                budget_used += float(xi)
            else:
                xi_star[i] = 0.0

        # recover path order from rho
        succ = {}
        for (i, j), rv in pi_star.items():
            if rv == 1:
                succ[i] = j
        path_nodes = [0]
        cur = 0
        seen = set([0])
        while cur in succ:
            nxt = succ[cur]
            if nxt in seen:
                break
            path_nodes.append(nxt)
            seen.add(nxt)
            cur = nxt
            if cur == self.sink:
                break

        self.last_worst_case = {
            "d": d_star,
            "xi": xi_star,
            "budget_used": float(budget_used),
            "path_nodes": path_nodes,
            "pi": pi_star,
        }
        return val, pi_star

    # -------------------------
    # Main loop
    # -------------------------

    def run(self):
        LB = -math.inf
        UB = math.inf
        best = None


        hist = []  # (iter, LB, UB) history for convergence plot
        for it in range(1, self.max_iter + 1):
            x_sol, y_sol, eta_sol, obj = self.solve_master()
            LB = obj

            val, pi_star = self.solve_SP2(x_sol, y_sol)
            # -------------------------
            # DEBUG 1/2: SP2 penalty term and violating arcs
            # penalty = M_big * sum_{(i,j)} (1 - y_ij) * rho_ij
            # -------------------------
            viol_arcs = [arc for arc, rv in pi_star.items() if rv >= 0.5 and y_sol.get(arc, 0) == 0]
            viol_count = len(viol_arcs)
            viol_sum = sum((1 - y_sol.get(arc, 0)) * (1.0 if rv >= 0.5 else 0.0) for arc, rv in pi_star.items())
            penalty_val = self.inst.M_big * viol_sum
            raw_path_len = val + penalty_val  # since Obj = sum(w) - penalty
            if self.verbose:
                if viol_count == 0:
                    print(f"[Iter {it}] SP2 penalty = 0 (all rho arcs satisfy y=1). raw_path_len={raw_path_len:.6f}")
                else:
                    show = viol_arcs[:20]
                    more = "" if viol_count <= 20 else f" ... (+{viol_count-20} more)"
                    print(
                        f"[Iter {it}] SP2 penalty = {penalty_val:.6f} = M_big({self.inst.M_big:.6f}) * {viol_count}. "
                        f"raw_path_len={raw_path_len:.6f}"
                    )
                    print(f"          violating arcs (rho=1,y=0): {show}{more}")


            # UB = min UB, C1(x) + e * worst_makespan
            C1_val = obj - self.inst.e_overhead * eta_sol
            UB = min(UB, C1_val + self.inst.e_overhead * val)

            if self.verbose:
                print(
                    f"\n[Iter {it}] LB={LB:.6f}, UB={UB:.6f}, gap={UB-LB:.6e}, "
                    f"eta={eta_sol:.6f}, SP2={val:.6f}"
                )

            hist.append((it, LB, UB))

            # keep the latest incumbent solution for downstream plotting/export
            self.best_solution = {"x": x_sol, "y": y_sol, "eta": eta_sol, "LB": LB, "UB": UB}
            best = (x_sol, y_sol, eta_sol, LB, UB)

            # convergence: UB-LB gap
            if UB - LB <= self.tol:
                if self.verbose:
                    print("收敛：UB-LB 已小于 tol。")
                break

            # add new extreme point
            self.pis.append(pi_star)


        # store history for external use
        self.history = hist

        # --- Visualization: UB/LB convergence curve ---
        if len(hist) >= 2:
            import matplotlib.pyplot as plt

            # Set the font to Times New Roman
            plt.rcParams['font.family'] = 'Times New Roman'

            # Example data (replace 'hist' with your actual data)
            iters = [t[0] for t in hist]
            LBs = [t[1] for t in hist]
            UBs = [t[2] for t in hist]

            plt.figure(figsize=(7, 4))
            plt.plot(iters, LBs, marker="o", linewidth=2, label="LB")
            plt.plot(iters, UBs, marker="s", linewidth=2, label="UB")
            plt.fill_between(iters, LBs, UBs, alpha=0.2)

            # Set the x-axis to only show integer ticks
            plt.xticks(range(min(iters), max(iters) + 1))  # This sets the x-axis ticks to integers

            plt.xlabel("Iteration")
            plt.ylabel("Objective value") 
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            #{input_mm}_convergence.jpg

            plt.savefig("ccg_convergence.jpg", dpi=300)
            

            # Show interactively (Windows will pop up a window)
            plt.show()

        return best



# =========================
# Scheduling utilities (for Gantt)
# =========================
def _earliest_start_times_from_y(nodes: List[int], y_sol: Dict[Tuple[int,int], int], dur: Dict[int, float]) -> Dict[int, float]:
    """Compute earliest start times from activated arcs y (assumed acyclic).
    Constraints: s_j >= s_i + dur[i] for all arcs with y[i,j]=1.
    """
    # build adjacency / indegree
    adj: Dict[int, List[int]] = {v: [] for v in nodes}
    indeg: Dict[int, int] = {v: 0 for v in nodes}
    for (i, j), v in y_sol.items():
        if v >= 0.5 and i in adj and j in indeg:
            adj[i].append(j)
            indeg[j] += 1

    # Kahn topological order
    q = [v for v in nodes if indeg[v] == 0]
    topo = []
    while q:
        v = q.pop(0)
        topo.append(v)
        for w in adj[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(topo) != len(nodes):
        # cycle fallback: keep natural order
        topo = list(nodes)

    # DP for earliest start
    s = {v: 0.0 for v in nodes}
    # compute predecessors on the fly
    pred = {v: [] for v in nodes}
    for (i, j), v in y_sol.items():
        if v >= 0.5 and i in pred and j in pred:
            pred[j].append(i)

    for v in topo:
        if pred[v]:
            s[v] = max(s[i] + float(dur.get(i, 0.0)) for i in pred[v])
    return s    
    def _chosen_modes(self, x_sol: Dict[Tuple[int,int], int]) -> Dict[int, int]:
        chosen = {}
        for i in self.real:
            mm = None
            for m in self.inst.modes[i].keys():
                if x_sol.get((i, m), 0) == 1:
                    mm = m
                    break
            if mm is None:
                mm = next(iter(self.inst.modes[i].keys()))
            chosen[i] = mm
        return chosen

    def compute_worstcase_schedule(self) -> Dict[int, float]:
        """Return earliest start times under the last worst-case durations d*."""
        if not hasattr(self, "best_solution") or self.best_solution is None:
            raise RuntimeError("best_solution 不存在：请先运行 solver.run()。")
        if not hasattr(self, "last_worst_case") or self.last_worst_case is None:
            raise RuntimeError("last_worst_case 不存在：请确保 SP2 至少求解过一次。")

        x_sol = self.best_solution["x"]
        y_sol = self.best_solution["y"]

        chosen = self._chosen_modes(x_sol)
        dur = {0: 0.0, self.sink: 0.0}
        for i in self.real:
            mm = chosen[i]
            dur[i] = float(self.last_worst_case["d"].get(i, self.inst.modes[i][mm].bar_d))

        return _earliest_start_times_from_y(self.nodes, y_sol, dur)

    def save_worstcase_gantt(self, out_jpg: str = "worstcase_gantt.jpg", title: Optional[str] = None) -> str:
        """Save a worst-case Gantt chart (activity-based) to jpg."""
        if not hasattr(self, "best_solution") or self.best_solution is None:
            raise RuntimeError("best_solution 不存在：请先运行 solver.run()。")
        if not hasattr(self, "last_worst_case") or self.last_worst_case is None:
            raise RuntimeError("last_worst_case 不存在：请确保 SP2 至少求解过一次。")

        import matplotlib.pyplot as plt

        x_sol = self.best_solution["x"]
        y_sol = self.best_solution["y"]
        chosen = self._chosen_modes(x_sol)

        # durations
        dur = {}
        for i in self.real:
            mm = chosen[i]
            dur[i] = float(self.last_worst_case["d"].get(i, self.inst.modes[i][mm].bar_d))

        # start times
        s = self.compute_worstcase_schedule()

        # plot
        fig = plt.figure(figsize=(10, 0.5 * len(self.real) + 2))
        ax = fig.add_subplot(111)

        ys = list(sorted(self.real))
        for idx, i in enumerate(ys):
            start = float(s.get(i, 0.0))
            length = float(dur[i])
            ax.barh(idx, length, left=start, height=0.6)
            ax.text(start + 0.02 * max(1.0, length), idx, f"{i}(m{chosen[i]})", va="center", fontsize=9)

        ax.set_yticks(list(range(len(ys))))
        ax.set_yticklabels([str(i) for i in ys])
        ax.set_xlabel("Time")
        ax.set_ylabel("Activity")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        cmax = float(s.get(self.sink, max(s.get(i, 0.0) + dur[i] for i in self.real)))
        if title is None:
            title = f"Worst-case Gantt (Gamma={self.inst.Gamma}, Cmax={cmax:.2f})"
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_jpg, dpi=300)
        plt.close(fig)
        return out_jpg


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="translate.py 输出的 PSPLIB JSON 文件路径")
    parser.add_argument("--Gamma", type=float, required=True)
    parser.add_argument("--e_overhead", type=float, default=1.0)
    parser.add_argument("--u_max", type=float, default=0.3, help="relative u_max for generating u_abs = bar_d * u_rel")
    parser.add_argument("--u_min", type=float, default=0.0)
    parser.add_argument("--mode_meta_csv", type=str, default=None, help="(可选) toy接口 CSV：指定每个 mode 的 u_abs 与 cost")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--alpha_max", type=float, default=1e4)
    parser.add_argument("--use_flow", action="store_true", help="启用资源流 f_{i,j,k}")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    inst = build_instance_from_psplib_json(
        data=data,
        Gamma=args.Gamma,
        e_overhead=args.e_overhead,
        b_price=None,
        u_max=args.u_max,
        u_min=args.u_min,
        mode_meta_csv=args.mode_meta_csv,
        M_big=None,
        use_flow=True if args.use_flow else True  # 默认启用
    )

    solver = Variant2Solver(inst, tol=args.tol, max_iter=args.max_iter, verbose=args.verbose, alpha_max=args.alpha_max)
    sol = solver.run()

    x_sol, y_sol, eta_sol, LB, UB = sol
    print("\n=== Done ===")
    print(f"eta = {eta_sol}")
    print(f"LB  = {LB}")
    print(f"UB  = {UB}")

# =========================
# Worst-case Gantt helper (monkeypatch to Variant2Solver)
# =========================
def _v5_earliest_start_times(nodes, y_sol, dur):
    # Build adjacency and indegree
    adj = {v: [] for v in nodes}
    indeg = {v: 0 for v in nodes}
    for (i, j), v in y_sol.items():
        if v >= 0.5 and i in adj and j in indeg:
            adj[i].append(j)
            indeg[j] += 1
    # Kahn
    q = [v for v in nodes if indeg[v] == 0]
    topo = []
    while q:
        v = q.pop(0)
        topo.append(v)
        for w in adj[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(topo) != len(nodes):
        topo = list(nodes)
    pred = {v: [] for v in nodes}
    for (i, j), v in y_sol.items():
        if v >= 0.5 and i in pred and j in pred:
            pred[j].append(i)
    s = {v: 0.0 for v in nodes}
    for v in topo:
        if pred[v]:
            s[v] = max(s[i] + float(dur.get(i, 0.0)) for i in pred[v])
    return s

def _v5_chosen_modes(inst, real_nodes, x_sol):
    chosen = {}
    for i in real_nodes:
        mm = None
        for m in inst.modes[i].keys():
            if x_sol.get((i, m), 0) == 1:
                mm = m
                break
        if mm is None:
            mm = next(iter(inst.modes[i].keys()))
        chosen[i] = mm
    return chosen

def _v5_compute_worstcase_schedule(self):
    if not hasattr(self, "best_solution") or self.best_solution is None:
        raise RuntimeError("best_solution 不存在：请先运行 solver.run()。")
    if not hasattr(self, "last_worst_case") or self.last_worst_case is None:
        raise RuntimeError("last_worst_case 不存在：请确保 SP2 至少求解过一次。")
    x_sol = self.best_solution["x"]
    y_sol = self.best_solution["y"]
    chosen = _v5_chosen_modes(self.inst, self.real, x_sol)
    dur = {0: 0.0, self.sink: 0.0}
    for i in self.real:
        mm = chosen[i]
        dur[i] = float(self.last_worst_case["d"].get(i, self.inst.modes[i][mm].bar_d))
    return _v5_earliest_start_times(self.nodes, y_sol, dur)

def _v5_save_worstcase_gantt(self, out_jpg="worstcase_gantt.jpg", title=None):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import math
    
    if not hasattr(self, "best_solution") or self.best_solution is None:
        raise RuntimeError("best_solution 不存在：请先运行 solver.run()。")
    if not hasattr(self, "last_worst_case") or self.last_worst_case is None:
        raise RuntimeError("last_worst_case 不存在：请确保 SP2 至少求解过一次。")
    
    x_sol = self.best_solution["x"]
    chosen = _v5_chosen_modes(self.inst, self.real, x_sol)
    s = self.compute_worstcase_schedule()
    
    dur = {}
    for i in self.real:
        mm = chosen[i]
        dur[i] = float(self.last_worst_case["d"].get(i, self.inst.modes[i][mm].bar_d))
    
    # 动态调整高度
    fig = plt.figure(figsize=(12, 0.55 * len(self.real) + 2))
    ax = fig.add_subplot(111)
    
    ys = list(sorted(self.real))
    for idx, i in enumerate(ys):
        start = float(s.get(i, 0.0))
        length = float(dur[i])
        ax.barh(idx, length, left=start, height=0.6, color='#5DADE2', edgecolor='#2E4053')
        ax.text(start + 0.1, idx, f"{i}(m{chosen[i]})", va="center", fontsize=9, fontweight='bold')
    
    # --- 强制以 1 为单位的核心设置 ---
    # 计算最大结束时间，用于确定坐标轴范围
    cmax = max(float(s.get(i, 0.0) + dur[i]) for i in self.real) if self.real else 0.0
    
    # 设置主刻度间隔为 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    
    # 设置 x 轴范围，从 0 到向上取整的 cmax + 1
    ax.set_xlim(0, math.ceil(cmax) + 1)
    # -------------------------------

    ax.set_yticks(list(range(len(ys))))
    ax.set_yticklabels([str(i) for i in ys])
    ax.set_xlabel("Time (Unit = 1)")
    ax.set_ylabel("Activity")
    
    # 让网格线对齐每一个整数单位，增加可读性
    ax.grid(True, axis="x", linestyle=":", alpha=0.7)
    
    if title is None:
        title = f"Worst-case Gantt (Gamma={self.inst.Gamma}, Cmax={cmax:.2f})"
    
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_jpg, dpi=300)
    plt.show()
    plt.close(fig)
    
    return out_jpg

# monkeypatch
try:
    Variant2Solver.compute_worstcase_schedule = _v5_compute_worstcase_schedule
    Variant2Solver.save_worstcase_gantt = _v5_save_worstcase_gantt
except Exception:
    pass
