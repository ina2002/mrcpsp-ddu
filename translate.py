# -*- coding: utf-8 -*-
# translate.py
# ----------------
# 将 PSPLIB 的 .mm / .bas（多模式 RCPSP）文本实例转译为 Python 可读的 JSON。
#
# 支持解析：
# - jobs / horizon / #renewable / #nonrenewable
# - precedence relations（successors）
# - requests/durations（每个 job 的每个 mode：duration + 资源需求）
# - resource availabilities
#
# 用法：
#     python translate.py --input mf1_.mm --output mf1_.json
#
# 说明：
# - PSPLIB 文件常见标题块由 "********" 分隔。此解析器采用“标题行状态机”，
#   比单纯按 block 计数更稳健。

#全局结构
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


_SECTION_NONE = "NONE"
_SECTION_PRECEDENCE = "PRECEDENCE"
_SECTION_REQUESTS = "REQUESTS"
_SECTION_AVAIL = "AVAIL"


def _ints(line: str) -> List[int]:
    return list(map(int, re.findall(r"-?\d+", line)))


def parse_psplib_mm(filepath: str) -> Dict[str, Any]:
    """
    解析 PSPLIB .mm / .bas 格式文件，返回字典（可直接 json.dump）。

    返回字段（主要）：
    - jobs_incl_dummy, jobs_real, horizon, n_renew, n_nonrenew
    - jobs[jobnr]["modes"], ["successors"], ["durations"][mode], ["req"][mode]
    - E_prec: precedence arc list [[i,j],...]
    - R: resource availabilities (renewable + nonrenewable)
    """
    path = Path(filepath)
    raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    data: Dict[str, Any] = {
        "file": str(path),
        "projects": None,
        "jobs_incl_dummy": None,
        "jobs_real": None,
        "horizon": None,
        "n_renew": None,
        "n_nonrenew": None,
        "resources": {},
        "R": None,
        "jobs": {},
        "E_prec": [],
    }

    section = _SECTION_NONE
    current_job: Optional[int] = None

    def has_key(s: str, key: str) -> bool:
        return key.lower() in s.lower()

    for line in raw:
        s = line.strip()
        if not s:
            continue

        # global scalars
        if has_key(s, "projects"):
            ints = _ints(s)
            if ints:
                data["projects"] = ints[0]
            continue

        if has_key(s, "jobs (incl. supersource/sink"):
            ints = _ints(s)
            if ints:
                data["jobs_incl_dummy"] = ints[0]
                data["jobs_real"] = ints[0] - 2
            continue

        if has_key(s, "horizon"):
            ints = _ints(s)
            if ints:
                data["horizon"] = ints[0]
            continue

        if has_key(s, "- renewable"):
            ints = _ints(s)
            if ints:
                data["n_renew"] = ints[0]
            continue

        if has_key(s, "- nonrenewable"):
            ints = _ints(s)
            if ints:
                data["n_nonrenew"] = ints[0]
            continue

        # section switches
        if has_key(s, "PRECEDENCE RELATIONS"):
            section = _SECTION_PRECEDENCE
            continue

        if has_key(s, "REQUESTS/DURATIONS"):
            section = _SECTION_REQUESTS
            current_job = None
            continue

        if has_key(s, "RESOURCEAVAILABILITIES"):
            section = _SECTION_AVAIL
            continue

        ints = _ints(s)
        if not ints:
            continue

        if section == _SECTION_PRECEDENCE:
            # jobnr. #modes #successors successors...
            jobnr = ints[0]
            nmodes = ints[1]
            n_succ = ints[2]
            succ = ints[3:3 + n_succ] if n_succ > 0 else []

            key = str(jobnr)
            data["jobs"].setdefault(key, {})
            data["jobs"][key]["modes"] = nmodes
            data["jobs"][key]["successors"] = succ

            for j in succ:
                data["E_prec"].append([jobnr, j])

        elif section == _SECTION_REQUESTS:
            n_renew = data["n_renew"]
            n_non = data["n_nonrenew"]
            if n_renew is None or n_non is None:
                raise ValueError("n_renew / n_nonrenew 未解析成功，请检查文件格式。")
            n_res = int(n_renew) + int(n_non)

            if len(ints) == 3 + n_res:
                current_job = ints[0]
                mode = ints[1]
                dur = ints[2]
                req = ints[3:3 + n_res]
            elif len(ints) == 2 + n_res and current_job is not None:
                mode = ints[0]
                dur = ints[1]
                req = ints[2:2 + n_res]
            else:
                continue

            key = str(current_job)
            data["jobs"].setdefault(key, {})
            data["jobs"][key].setdefault("durations", {})
            data["jobs"][key].setdefault("req", {})
            data["jobs"][key]["durations"][str(mode)] = dur
            data["jobs"][key]["req"][str(mode)] = req

        elif section == _SECTION_AVAIL:
            n_renew = data["n_renew"]
            n_non = data["n_nonrenew"]
            if n_renew is None or n_non is None:
                raise ValueError("n_renew / n_nonrenew 未解析成功，请检查文件格式。")
            n_res = int(n_renew) + int(n_non)

            if len(ints) >= n_res:
                data["R"] = ints[:n_res]

    # sanity checks
    if data["jobs_incl_dummy"] is None:
        raise ValueError("未解析到 jobs (incl. supersource/sink)。")
    if data["R"] is None:
        raise ValueError("未解析到 RESOURCEAVAILABILITIES。")
    if data["horizon"] is None:
        raise ValueError("未解析到 horizon。")

    n_renew = int(data["n_renew"] or 0)
    n_non = int(data["n_nonrenew"] or 0)
    data["resources"] = {
        "renewable": [f"R{k}" for k in range(1, n_renew + 1)],
        "nonrenewable": [f"N{k}" for k in range(1, n_non + 1)],
    }
    return data


def save_json(data: Dict[str, Any], outpath: str) -> None:
    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="PSPLIB .mm/.bas 文件路径")
    ap.add_argument("--output", required=True, help="输出 JSON 路径")
    args = ap.parse_args()

    data = parse_psplib_mm(args.input)
    save_json(data, args.output)
    print(f"[OK] parsed -> {args.output}")


if __name__ == "__main__":
    main()
