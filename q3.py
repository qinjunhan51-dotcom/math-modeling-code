# -*- coding: utf-8 -*-
r"""
本版本情景参数调整：
1. S2 与综合场景 E1：客户16时间窗提前为 [08:50, 09:40]；
2. 综合场景 E5：客户Y坐标改为 (-12.0, 8.0)，保持区外普通订单属性。
3. 综合场景 E5：客户Y时间窗由 [17:00, 19:00] 改为 [17:00, 20:30]，用于降低新增区外订单的迟到时间。

q3.py

问题三：动态事件下的实时车辆调度策略
------------------------------------------------------------
本代码面向建模手给出的“2 个单事件场景 + 1 个综合事件场景”：

S1：12:00，客户25追加订单，新增 180 kg、0.45 m³。
S2：09:30，客户16时间窗提前为 [08:50, 09:40]。
S3：综合事件，按时间线依次处理：
    E1 09:30 客户16时间窗提前；
    E2 10:30 客户48取消订单；
    E3 11:00 新增绿色区急单客户X，坐标(3.5, 6.8)，重量210kg，体积0.52m³；
    E4 14:00 客户25地址变更为(9.2, 10.8)；
    E5 15:30 新增区外普通订单客户Y，坐标(-12.0, 8.0)，重量350kg，体积0.88m³。

建模口径：
1. 继承问题二绿色配送区政策：8:00-16:00燃油车不得进入绿色区或经过穿区弧段；
2. 成本、时间窗、容量、能耗、碳排放和闭合路径口径均继承 q2.py；
3. 事件发生后进行状态截断，冻结已完成和冻结窗口内的任务；
4. 对冻结窗口外剩余任务进行有限候选路线 + 关键插入位置的快速重调度；
5. 输出每个场景/阶段的调整前后成本、距离、等待、迟到、碳排放、换车客户数、路径变化和响应时间。

推荐运行：
    D:\A_Question\.venv\Scripts\python.exe D:\A_Question\q3.py

若想指定目录：
    D:\A_Question\.venv\Scripts\python.exe D:\A_Question\q3.py ^
        --data-dir D:\A_Question ^
        --solver D:\A_Question\q2.py ^
        --base-output-dir D:\A_Question\问题二输出 ^
        --outdir D:\A_Question\问题三输出

说明：
    优先读取 问题二输出 中已经跑好的问题二路线作为初始方案；
    若读取失败，则自动调用 q2 求解器用 base-seed/base-iterations 重新生成初始方案。
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 一、导入 q2 求解器
# =============================================================================

def import_solver(solver_path: Path):
    if not solver_path.exists():
        raise FileNotFoundError(f"没有找到 q2 求解器文件：{solver_path}")

    spec = importlib.util.spec_from_file_location("q2_solver_for_q3", str(solver_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入求解器：{solver_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules["q2_solver_for_q3"] = mod
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# 二、通用工具
# =============================================================================



def clean_q3_output_dir(outdir: Path):
    """
    清理问题三输出目录中的旧结果，避免不同版本 q3 的场景文件夹混在一起。
    只删除本程序/旧版 q3 生成的已知结果项，不删除外部原始数据和 q1/q2 结果。
    """
    if not outdir.exists():
        return

    names_to_remove = [
        "baseline_static_solution",
        "阶段0_问题二初始方案",
        "S1_已有客户新增订单",
        "S1_订单取消",
        "S2_客户时间窗提前",
        "S2_时间窗提前",
        "S3_综合多事件联动",
        "S3_绿色区新增订单",
        "问题三_动态事件结果汇总.xlsx",
        "问题三_全部重调度修复日志.xlsx",
        "问题三_输出说明.txt",
    ]

    for name in names_to_remove:
        p = outdir / name
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            try:
                p.unlink()
            except Exception:
                pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(x, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(round(float(x)))
    except Exception:
        return default


def clock_to_min(clock: str) -> int:
    """
    将 'HH:MM' 转为相对 8:00 的分钟数。
    """
    h, m = str(clock).split(":")
    return (int(h) - 8) * 60 + int(m)


def min_to_clock(t: float) -> str:
    """
    将相对 8:00 的分钟数转为 HH:MM。
    """
    t = int(round(float(t)))
    h = 8 + t // 60
    m = t % 60
    return f"{h:02d}:{m:02d}"


def clone_solution(solution: List[dict]) -> List[dict]:
    return copy.deepcopy(solution)


def clear_solution_cache(solution: List[dict]) -> List[dict]:
    for r in solution:
        r.pop("cached_eval", None)
    return solution


def find_existing_file(folder: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        p = folder / name
        if p.exists():
            return p
    return None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(path)
    return pd.read_excel(path)


def choose_col(df: pd.DataFrame, candidates: List[str], required: bool = False) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in cols}

    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        c_low = str(c).strip().lower()
        if c_low in lower_map:
            return lower_map[c_low]

    for cand in candidates:
        cand_low = str(cand).strip().lower()
        for col in cols:
            col_low = str(col).strip().lower()
            if cand_low in col_low or col_low in cand_low:
                return col

    if required:
        raise KeyError(f"未找到字段 {candidates}，当前字段：{cols}")
    return None


# =============================================================================
# 三、从问题二输出读取基准方案
# =============================================================================

def parse_route_string(route_str: Any) -> List[str]:
    """
    解析类似 0-63_5-12-0 的闭合路径表达。
    """
    s = str(route_str).strip()
    s = s.replace("→", "-").replace("->", "-").replace("—", "-").replace("－", "-")
    parts = [x.strip() for x in s.split("-") if x.strip() != ""]
    stops = [x for x in parts if x != "0" and x.lower() != "depot"]
    return stops


def load_solution_from_output(solver, instance: dict, output_dir: Path) -> Optional[List[dict]]:
    """
    从已经跑好的 q2 输出文件夹读取路线汇总，避免重新跑 ALNS。
    要求路线汇总中至少能识别：
        - 车型列 vehicle_type / 车型
        - 路径列 route / 路径 / route_str
    """
    if output_dir is None or not output_dir.exists():
        return None

    route_file = find_existing_file(output_dir, [
        "问题二_路线汇总.xlsx", "问题二_路线汇总.csv",
        "路线汇总.xlsx", "路线汇总.csv",
    ])
    if route_file is None:
        return None

    df = read_table(route_file)
    route_col = choose_col(df, ["route", "路径", "路径节点", "route_str", "node_sequence", "路线"])
    veh_col = choose_col(df, ["vehicle_type", "车型", "车辆类型"])

    if route_col is None or veh_col is None:
        return None

    solution = []
    missing_nodes = []
    for _, row in df.iterrows():
        vt = str(row[veh_col]).strip()
        stops = parse_route_string(row[route_col])

        clean_stops = []
        for n in stops:
            n = str(n)
            if n in instance["node_dict"]:
                clean_stops.append(n)
            else:
                missing_nodes.append(n)

        if not clean_stops:
            continue

        route = {"vehicle_type": vt, "stops": clean_stops, "locked_len": 0}
        try:
            solver.rebuild_route_cache(route, instance["node_dict"])
        except Exception:
            pass
        solution.append(route)

    if not solution:
        return None

    solver.refresh_solution_cache(solution, instance)
    cost, _, feasible = solver.solution_cost(solution, instance)
    cover = solver.coverage_report(solution, instance)

    print(f"  已从既有输出读取基准方案：{route_file}")
    print(f"  读取路线数: {len(solution)} | 成本: {cost:.2f} | 可行: {feasible} | 缺失节点: {cover.get('missing_node_count')}")
    if missing_nodes:
        print(f"  警告：有 {len(missing_nodes)} 个节点在当前实例中未识别，已跳过。示例：{missing_nodes[:5]}")

    if not feasible or cover.get("missing_node_count", 999999) != 0:
        print("  读取的基准方案不是完全可行方案，将回退到重新求解。")
        return None

    return solution


def build_base_solution_by_solver(
    solver,
    instance: dict,
    seed: int,
    iterations: int,
    skip_local_search: bool = False,
) -> List[dict]:
    """
    若无法读取已有问题二输出，则重新构造一份基准方案。
    """
    init_solution = solver.construct_initial_solution(instance)
    solver.refresh_solution_cache(init_solution, instance)

    best_solution, _ = solver.alns_solve(instance, init_solution, iterations=iterations, seed=seed)
    solver.refresh_solution_cache(best_solution, instance)

    if not skip_local_search:
        try:
            local_solution = solver.local_search_vns(best_solution, instance, max_passes=1)
            solver.refresh_solution_cache(local_solution, instance)
            local_cost, _, local_feasible = solver.solution_cost(local_solution, instance)
            best_cost, _, _ = solver.solution_cost(best_solution, instance)
            if local_feasible and local_cost <= best_cost + 1e-9 and solver.solution_covers_all_nodes(local_solution, instance):
                best_solution = local_solution
        except Exception:
            pass

    try:
        reduced = solver.merge_routes_postprocess(best_solution, instance, max_passes=1)
        solver.refresh_solution_cache(reduced, instance)
        reduced_cost, _, reduced_feasible = solver.solution_cost(reduced, instance)
        best_cost, _, _ = solver.solution_cost(best_solution, instance)
        if reduced_feasible and reduced_cost <= best_cost + 1e-9 and solver.solution_covers_all_nodes(reduced, instance):
            best_solution = reduced
    except Exception:
        pass

    solver.refresh_solution_cache(best_solution, instance)
    return best_solution


# =============================================================================
# 四、距离矩阵与新增客户处理
# =============================================================================

def get_coord_dict(instance: dict) -> Dict[int, Tuple[float, float]]:
    coords = instance.get("coordinates", pd.DataFrame()).copy()
    out = {}

    if not coords.empty:
        id_col = choose_col(coords, ["ID", "客户编号", "id"], required=False)
        x_col = choose_col(coords, ["X (km)", "x", "X"], required=False)
        y_col = choose_col(coords, ["Y (km)", "y", "Y"], required=False)

        if id_col and x_col and y_col:
            for _, r in coords.iterrows():
                try:
                    cid = int(r[id_col])
                    out[cid] = (float(r[x_col]), float(r[y_col]))
                except Exception:
                    pass

    depot = instance.get("depot", None)
    if depot is not None:
        out[0] = (float(depot["x"]), float(depot["y"]))
    else:
        out.setdefault(0, (0.0, 0.0))

    return out


def estimate_road_factor(instance: dict) -> float:
    """
    用已有距离矩阵和坐标估计道路距离/欧氏距离比例。
    """
    coords = get_coord_dict(instance)
    dist = instance["distance_numpy"]
    id_to_pos = instance["distance_id_to_pos"]

    ratios = []
    ids = [int(x) for x in id_to_pos.keys() if str(x).lstrip("-").isdigit()]
    ids = ids[:80]

    for i in range(len(ids)):
        for j in range(i + 1, min(len(ids), i + 10)):
            a, b = ids[i], ids[j]
            if a not in coords or b not in coords:
                continue
            xa, ya = coords[a]
            xb, yb = coords[b]
            eu = math.hypot(xa - xb, ya - yb)
            if eu <= 1e-9:
                continue
            try:
                da = dist[id_to_pos[str(a)], id_to_pos[str(b)]]
                if da > 0:
                    ratios.append(float(da) / eu)
            except Exception:
                pass

    if not ratios:
        return 1.2
    return float(np.median(ratios))


def ensure_distance_id(instance: dict, customer_id: int, x: float, y: float):
    """
    为新增客户或地址变更客户补充/更新距离矩阵。
    如果 customer_id 已存在，则更新该行列；若不存在，则扩展 distance_numpy。
    """
    cid = int(customer_id)
    key = str(cid)
    coords = get_coord_dict(instance)
    road_factor = estimate_road_factor(instance)

    id_to_pos = instance["distance_id_to_pos"]
    dist = instance["distance_numpy"]

    if key not in id_to_pos:
        old_n = dist.shape[0]
        new_dist = np.zeros((old_n + 1, old_n + 1), dtype=float)
        new_dist[:old_n, :old_n] = dist
        id_to_pos[key] = old_n
        dist = new_dist
        instance["distance_numpy"] = dist

    # 更新坐标表
    if "coordinates" in instance and isinstance(instance["coordinates"], pd.DataFrame):
        coords_df = instance["coordinates"].copy()
        id_col = choose_col(coords_df, ["ID", "客户编号", "id"], required=False)
        x_col = choose_col(coords_df, ["X (km)", "x", "X"], required=False)
        y_col = choose_col(coords_df, ["Y (km)", "y", "Y"], required=False)
        if id_col and x_col and y_col:
            mask = pd.to_numeric(coords_df[id_col], errors="coerce").astype("Int64") == cid
            if mask.any():
                coords_df.loc[mask, x_col] = x
                coords_df.loc[mask, y_col] = y
            else:
                new_row = {id_col: cid, x_col: x, y_col: y}
                if "类型" in coords_df.columns:
                    new_row["类型"] = "客户"
                coords_df = pd.concat([coords_df, pd.DataFrame([new_row])], ignore_index=True)
            instance["coordinates"] = coords_df

    coords = get_coord_dict(instance)
    coords[cid] = (x, y)

    pos_c = id_to_pos[key]
    for other_key, pos_o in list(id_to_pos.items()):
        try:
            oid = int(other_key)
        except Exception:
            continue
        if oid not in coords:
            continue
        xo, yo = coords[oid]
        d = road_factor * math.hypot(x - xo, y - yo)
        instance["distance_numpy"][pos_c, pos_o] = d
        instance["distance_numpy"][pos_o, pos_c] = d

    instance["distance_numpy"][pos_c, pos_c] = 0.0


def add_new_customer_node(
    solver,
    instance: dict,
    customer_id: int,
    x: float,
    y: float,
    weight: float,
    volume: float,
    earliest: int,
    latest: int,
    service_time: int = 20,
    node_prefix: str = "NEW",
) -> str:
    """
    增加一个新客户节点，并用欧氏距离*道路系数近似补充距离矩阵。
    """
    ensure_distance_id(instance, customer_id, x, y)

    existing = set(instance["service_nodes"]["node_id"].astype(str).tolist())
    node_id = f"{node_prefix}_{customer_id}"
    k = 1
    while node_id in existing or node_id in instance["node_dict"]:
        k += 1
        node_id = f"{node_prefix}_{customer_id}_{k}"

    node = solver.Node(
        node_id=node_id,
        original_customer_id=int(customer_id),
        sub_id=999000 + k,
        weight=float(weight),
        volume=float(volume),
        earliest=int(earliest),
        latest=int(latest),
        service_time=int(service_time),
        x=float(x),
        y=float(y),
        is_virtual=True,
    )
    instance["node_dict"][node_id] = node

    new_row = {
        "node_id": node_id,
        "original_customer_id": int(customer_id),
        "sub_id": 999000 + k,
        "weight": float(weight),
        "volume": float(volume),
        "earliest": int(earliest),
        "latest": int(latest),
        "service_time": int(service_time),
        "x": float(x),
        "y": float(y),
        "is_virtual": True,
        "order_count": 1,
    }

    try:
        tmp = pd.DataFrame([new_row])
        tmp = solver.add_vehicle_compatibility_columns(tmp)
        new_row = tmp.iloc[0].to_dict()
    except Exception:
        pass

    instance["service_nodes"] = pd.concat([instance["service_nodes"], pd.DataFrame([new_row])], ignore_index=True)
    rebuild_green_metadata_if_possible(solver, instance)
    return node_id


# =============================================================================
# 五、绿色区元数据更新
# =============================================================================

def rebuild_green_metadata_if_possible(solver, instance: dict):
    try:
        depot = instance.get("depot", {"node_id": "0", "x": 0.0, "y": 0.0})
        green_nodes, cross_green_arcs, green_arc_df = solver.build_green_zone_metadata(
            instance["service_nodes"], depot
        )
        instance["green_nodes"] = green_nodes
        instance["cross_green_arcs"] = cross_green_arcs
        instance["green_arc_df"] = green_arc_df

        if "node_id" in instance["service_nodes"].columns:
            instance["service_nodes"]["in_green_zone"] = (
                instance["service_nodes"]["node_id"].astype(str).map(green_nodes).fillna(False)
            )

        if "green_zone_summary" not in instance:
            instance["green_zone_summary"] = {}
        instance["green_zone_summary"]["green_service_node_count"] = int(
            instance["service_nodes"].get("in_green_zone", pd.Series(dtype=bool)).sum()
        )
        instance["green_zone_summary"]["cross_green_arc_count"] = int(
            green_arc_df["cross_green_zone"].sum()
        ) if isinstance(green_arc_df, pd.DataFrame) and not green_arc_df.empty else 0
    except Exception as e:
        print(f"  [警告] 绿色区元数据更新失败：{e}")


# =============================================================================
# 六、状态截断、冻结窗口、扰动指标
# =============================================================================

def build_node_plan_map(solver, solution: List[dict], instance: dict) -> Dict[str, dict]:
    node_map = {}
    for ridx, route in enumerate(solution):
        ev = solver.evaluate_route(route, instance)
        for pos, nid in enumerate(route.get("stops", [])):
            info = {
                "route_index": ridx,
                "vehicle_type": route["vehicle_type"],
                "position": pos,
                "arrival_time": None,
                "service_start": None,
                "service_end": None,
                "late_min": None,
                "wait_min": None,
            }
            if pos < len(ev.details):
                det = ev.details[pos]
                info.update({
                    "arrival_time": safe_float(det.get("arrival_time")),
                    "service_start": safe_float(det.get("service_start")),
                    "service_end": safe_float(det.get("service_end")),
                    "late_min": safe_float(det.get("late_min")),
                    "wait_min": safe_float(det.get("wait_min")),
                })
            node_map[str(nid)] = info
    return node_map


def compute_locked_nodes(
    solver,
    solution: List[dict],
    instance: dict,
    event_time: int,
    freeze_min: int,
) -> Tuple[List[int], set, set]:
    """
    返回：
    1. 每条路线锁定前缀长度；
    2. 已完成节点；
    3. 冻结节点。
    """
    locked_prefix_lengths = []
    completed_nodes = set()
    frozen_nodes = set()
    lock_until = event_time + freeze_min

    for route in solution:
        ev = solver.evaluate_route(route, instance)
        locked_len = 0
        for pos, nid in enumerate(route.get("stops", [])):
            service_end = 10**9
            if pos < len(ev.details):
                service_end = safe_float(ev.details[pos].get("service_end"), 10**9)

            if service_end <= event_time + 1e-9:
                completed_nodes.add(str(nid))
                frozen_nodes.add(str(nid))
                locked_len = pos + 1
            elif service_end <= lock_until + 1e-9:
                frozen_nodes.add(str(nid))
                locked_len = pos + 1
            else:
                break
        locked_prefix_lengths.append(locked_len)

    return locked_prefix_lengths, completed_nodes, frozen_nodes


def build_locked_partial_solution(
    solver,
    solution: List[dict],
    instance: dict,
    locked_prefix_lengths: List[int],
) -> Tuple[List[dict], set]:
    partial = []
    locked_nodes = set()

    for ridx, route in enumerate(solution):
        locked_len = locked_prefix_lengths[ridx] if ridx < len(locked_prefix_lengths) else 0
        locked_stops = [str(x) for x in route.get("stops", [])[:locked_len] if str(x) in instance["node_dict"]]
        if locked_stops:
            r = {
                "vehicle_type": route["vehicle_type"],
                "stops": locked_stops,
                "locked_len": len(locked_stops),
            }
            try:
                solver.rebuild_route_cache(r, instance["node_dict"])
            except Exception:
                pass
            partial.append(r)
            locked_nodes.update(locked_stops)

    return partial, locked_nodes


def expected_service_nodes(instance: dict) -> set:
    return set(instance["service_nodes"]["node_id"].astype(str).tolist())


def compute_disturbance_metrics(
    solver,
    old_solution: List[dict],
    new_solution: List[dict],
    old_instance: dict,
    new_instance: dict,
) -> dict:
    old_map = build_node_plan_map(solver, old_solution, old_instance)
    new_map = build_node_plan_map(solver, new_solution, new_instance)

    old_nodes = set(old_map.keys())
    new_nodes = set(new_map.keys())
    common = old_nodes & new_nodes

    changed_vehicle = 0
    changed_route = 0
    changed_position = 0
    time_shift_sum = 0.0
    time_shift_count = 0
    affected_routes = set()

    for nid in common:
        old = old_map[nid]
        new = new_map[nid]
        if old.get("vehicle_type") != new.get("vehicle_type"):
            changed_vehicle += 1
        if old.get("route_index") != new.get("route_index"):
            changed_route += 1
            affected_routes.add(old.get("route_index"))
            affected_routes.add(new.get("route_index"))
        if old.get("position") != new.get("position"):
            changed_position += 1
        if old.get("arrival_time") is not None and new.get("arrival_time") is not None:
            time_shift_sum += abs(safe_float(old["arrival_time"]) - safe_float(new["arrival_time"]))
            time_shift_count += 1

    return {
        "removed_node_count": len(old_nodes - new_nodes),
        "added_node_count": len(new_nodes - old_nodes),
        "changed_vehicle_node_count": changed_vehicle,
        "changed_route_node_count": changed_route,
        "changed_position_node_count": changed_position,
        "affected_route_count": len([x for x in affected_routes if x is not None]),
        "mean_arrival_time_shift_min": time_shift_sum / time_shift_count if time_shift_count else 0.0,
    }


# =============================================================================
# 七、事件映射
# =============================================================================

def find_customer_nodes(instance: dict, customer_id: int) -> List[str]:
    out = []
    for nid, nd in instance["node_dict"].items():
        try:
            if int(nd.original_customer_id) == int(customer_id):
                out.append(str(nid))
        except Exception:
            pass
    return out


def choose_unlocked_node_for_customer(
    solver,
    solution: List[dict],
    instance: dict,
    customer_id: int,
    completed_nodes: set,
    frozen_nodes: set,
) -> Optional[str]:
    nodes = find_customer_nodes(instance, customer_id)
    if not nodes:
        return None

    candidates = [n for n in nodes if n not in completed_nodes and n not in frozen_nodes]
    if not candidates:
        candidates = [n for n in nodes if n not in completed_nodes]
    if not candidates:
        return None

    # 优先选择原计划到达时间靠后的节点，避免修改已经很接近执行的节点
    plan = build_node_plan_map(solver, solution, instance)
    candidates = sorted(candidates, key=lambda n: safe_float(plan.get(n, {}).get("arrival_time"), 10**9), reverse=True)
    return candidates[0]


def update_node_demand(instance: dict, node_id: str, add_weight: float, add_volume: float, service_time: Optional[int] = None):
    node_id = str(node_id)
    if node_id not in instance["node_dict"]:
        return

    nd = instance["node_dict"][node_id]
    nd.weight += float(add_weight)
    nd.volume += float(add_volume)
    if service_time is not None:
        nd.service_time = int(service_time)

    mask = instance["service_nodes"]["node_id"].astype(str) == node_id
    instance["service_nodes"].loc[mask, "weight"] = nd.weight
    instance["service_nodes"].loc[mask, "volume"] = nd.volume
    if service_time is not None:
        instance["service_nodes"].loc[mask, "service_time"] = int(service_time)

    try:
        tmp = instance["service_nodes"].loc[mask].copy()
        tmp = instance["service_nodes"].loc[mask].copy()
    except Exception:
        pass


def update_customer_time_window(instance: dict, customer_id: int, earliest: int, latest: int, nodes_limit: Optional[List[str]] = None):
    nodes = nodes_limit if nodes_limit is not None else find_customer_nodes(instance, customer_id)
    for nid in nodes:
        if nid not in instance["node_dict"]:
            continue
        nd = instance["node_dict"][nid]
        nd.earliest = int(earliest)
        nd.latest = int(latest)

    mask = instance["service_nodes"]["node_id"].astype(str).isin(nodes)
    instance["service_nodes"].loc[mask, "earliest"] = int(earliest)
    instance["service_nodes"].loc[mask, "latest"] = int(latest)


def remove_customer_unlocked_nodes(
    instance: dict,
    solution: List[dict],
    customer_id: int,
    completed_nodes: set,
    frozen_nodes: set,
) -> List[str]:
    nodes = find_customer_nodes(instance, customer_id)
    remove_nodes = [n for n in nodes if n not in completed_nodes and n not in frozen_nodes]

    if not remove_nodes:
        return []

    remove_set = set(remove_nodes)
    instance["service_nodes"] = instance["service_nodes"][~instance["service_nodes"]["node_id"].astype(str).isin(remove_set)].copy()
    for n in remove_nodes:
        instance["node_dict"].pop(n, None)

    for r in solution:
        r["stops"] = [n for n in r.get("stops", []) if str(n) not in remove_set]
        r.pop("cached_eval", None)

    solution[:] = [r for r in solution if len(r.get("stops", [])) > 0]
    return remove_nodes


def update_customer_address(
    solver,
    instance: dict,
    customer_id: int,
    x: float,
    y: float,
):
    cid = int(customer_id)
    ensure_distance_id(instance, cid, x, y)

    nodes = find_customer_nodes(instance, cid)
    for nid in nodes:
        nd = instance["node_dict"][nid]
        nd.x = float(x)
        nd.y = float(y)

    mask = instance["service_nodes"]["node_id"].astype(str).isin(nodes)
    instance["service_nodes"].loc[mask, "x"] = float(x)
    instance["service_nodes"].loc[mask, "y"] = float(y)

    rebuild_green_metadata_if_possible(solver, instance)
    return nodes


def apply_event(
    solver,
    instance: dict,
    solution: List[dict],
    event: dict,
    completed_nodes: set,
    frozen_nodes: set,
) -> Tuple[dict, List[dict], dict]:
    """
    应用单个事件，返回更新后的 instance/solution 和事件日志。
    """
    instance = copy.deepcopy(instance)
    solution = clone_solution(solution)

    notes = {
        "event_id": event.get("event_id", ""),
        "event_type": event.get("type", ""),
        "target_customer": event.get("customer_id", ""),
        "applied": True,
        "added_nodes": [],
        "removed_nodes": [],
        "updated_nodes": [],
        "message": "",
    }

    etype = event.get("type")

    if etype == "add_to_existing_customer":
        cid = int(event["customer_id"])
        nid = choose_unlocked_node_for_customer(solver, solution, instance, cid, completed_nodes, frozen_nodes)
        if nid is None:
            notes["applied"] = False
            notes["message"] = f"客户{cid}在事件时刻无可修改的未服务节点，事件未应用。"
            return instance, solution, notes

        update_node_demand(
            instance, nid,
            add_weight=float(event["add_weight"]),
            add_volume=float(event["add_volume"]),
            service_time=event.get("service_time", None),
        )
        notes["updated_nodes"] = [nid]
        notes["message"] = f"客户{cid}追加订单已并入服务节点 {nid}。"

    elif etype == "time_window":
        cid = int(event["customer_id"])
        nodes = find_customer_nodes(instance, cid)
        target_nodes = [n for n in nodes if n not in completed_nodes]
        if not target_nodes:
            notes["applied"] = False
            notes["message"] = f"客户{cid}在事件时刻已完成服务，时间窗调整未应用。"
            return instance, solution, notes

        update_customer_time_window(
            instance,
            cid,
            earliest=int(event["new_earliest"]),
            latest=int(event["new_latest"]),
            nodes_limit=target_nodes,
        )
        notes["updated_nodes"] = target_nodes
        notes["message"] = f"客户{cid}时间窗已更新为 [{min_to_clock(event['new_earliest'])}, {min_to_clock(event['new_latest'])}]。"

    elif etype == "cancel_customer":
        cid = int(event["customer_id"])
        removed = remove_customer_unlocked_nodes(instance, solution, cid, completed_nodes, frozen_nodes)
        notes["removed_nodes"] = removed
        if removed:
            notes["message"] = f"客户{cid}未冻结订单节点已取消：{removed}"
        else:
            notes["applied"] = False
            notes["message"] = f"客户{cid}无可取消的未完成/未冻结节点。"

    elif etype == "new_customer":
        cid = int(event["customer_id"])
        nid = add_new_customer_node(
            solver,
            instance,
            customer_id=cid,
            x=float(event["x"]),
            y=float(event["y"]),
            weight=float(event["weight"]),
            volume=float(event["volume"]),
            earliest=int(event["earliest"]),
            latest=int(event["latest"]),
            service_time=int(event.get("service_time", 20)),
            node_prefix=event.get("node_prefix", "NEW"),
        )
        notes["added_nodes"] = [nid]
        notes["message"] = f"新增客户{cid}已添加为节点 {nid}，距离矩阵按道路系数近似补充。"

    elif etype == "address_change":
        cid = int(event["customer_id"])
        nodes = update_customer_address(
            solver,
            instance,
            cid,
            x=float(event["new_x"]),
            y=float(event["new_y"]),
        )
        notes["updated_nodes"] = nodes
        notes["message"] = f"客户{cid}地址已更新，距离矩阵行列按道路系数近似更新。"

    else:
        notes["applied"] = False
        notes["message"] = f"未知事件类型：{etype}"

    for r in solution:
        try:
            solver.rebuild_route_cache(r, instance["node_dict"])
        except Exception:
            pass
        r.pop("cached_eval", None)

    rebuild_green_metadata_if_possible(solver, instance)
    return instance, solution, notes


# =============================================================================
# 八、快速重调度：有限候选路线 + 关键插入位置
# =============================================================================

def route_insert_positions(route: dict) -> List[int]:
    locked = int(route.get("locked_len", 0))
    return list(range(locked, len(route.get("stops", [])) + 1))


def choose_candidate_routes(
    solution: List[dict],
    node_id: str,
    baseline_node_map: Dict[str, dict],
    max_candidate_routes: int,
) -> List[int]:
    all_idx = list(range(len(solution)))
    old = baseline_node_map.get(str(node_id))
    ordered = []

    if old is not None:
        old_r = safe_int(old.get("route_index"), -1)
        if 0 <= old_r < len(solution):
            ordered.append(old_r)

        old_v = old.get("vehicle_type")
        for i in all_idx:
            if i not in ordered and solution[i].get("vehicle_type") == old_v:
                ordered.append(i)

    for i in all_idx:
        if i not in ordered:
            ordered.append(i)

    return ordered[:max_candidate_routes]


def choose_insert_positions(
    route: dict,
    node_id: str,
    ridx: int,
    baseline_node_map: Dict[str, dict],
    max_insert_positions: int,
) -> List[int]:
    positions = route_insert_positions(route)
    if len(positions) <= max_insert_positions:
        return positions

    locked = int(route.get("locked_len", 0))
    endp = len(route.get("stops", []))
    cand = {locked, endp}

    old = baseline_node_map.get(str(node_id))
    if old is not None and safe_int(old.get("route_index"), -1) == ridx:
        old_pos = safe_int(old.get("position"), locked)
        for p in [old_pos - 1, old_pos, old_pos + 1]:
            if p in positions:
                cand.add(p)

    arr = np.linspace(positions[0], positions[-1], max_insert_positions).round().astype(int).tolist()
    for p in arr:
        if p in positions:
            cand.add(p)

    return sorted(cand)[:max_insert_positions]


def make_new_route_for_node(solver, instance: dict, solution: List[dict], node_id: str) -> Optional[dict]:
    node = instance["node_dict"][str(node_id)]
    vehicles = instance["vehicle_types_list"]
    used = solver.solution_vehicle_counts(solution)

    candidates = [
        v for v in vehicles
        if node.weight <= v.capacity_weight + 1e-9 and node.volume <= v.capacity_volume + 1e-9
    ]

    # 绿色区节点优先新能源车
    green_nodes = instance.get("green_nodes", {})
    if bool(green_nodes.get(str(node_id), False)):
        candidates = sorted(candidates, key=lambda v: (0 if v.fuel_type == "electric" else 1, v.capacity_weight))
    else:
        candidates = sorted(candidates, key=lambda v: (v.capacity_weight, v.capacity_volume))

    for v in candidates:
        if used.get(v.vehicle_type, 0) >= instance["vehicle_stock"].get(v.vehicle_type, 0):
            continue
        r = {"vehicle_type": v.vehicle_type, "stops": [str(node_id)], "locked_len": 0}
        try:
            solver.rebuild_route_cache(r, instance["node_dict"])
        except Exception:
            pass
        ev = solver.evaluate_route(r, instance)
        if ev.feasible:
            return r
    return None


def insert_node(route: dict, node_id: str, pos: int, solver, instance: dict) -> dict:
    r = copy.deepcopy(route)
    r["stops"] = list(r.get("stops", []))
    r["stops"].insert(pos, str(node_id))
    r.pop("cached_eval", None)
    try:
        solver.rebuild_route_cache(r, instance["node_dict"])
    except Exception:
        pass
    return r


def repair_remaining_tasks(
    solver,
    instance: dict,
    partial_solution: List[dict],
    remaining_nodes: List[str],
    baseline_node_map: Dict[str, dict],
    max_candidate_routes: int,
    max_insert_positions: int,
    disturbance_weight: float,
    time_shift_weight: float,
) -> Tuple[List[dict], List[dict]]:
    solution = clone_solution(partial_solution)
    logs = []

    def node_key(nid):
        nd = instance["node_dict"][str(nid)]
        green = 1 if instance.get("green_nodes", {}).get(str(nid), False) else 0
        old = 0 if str(nid) in baseline_node_map else 1
        cap = max(nd.weight / 3000.0, nd.volume / 15.0)
        return (-green, old, nd.latest, -cap, str(nid))

    remaining_nodes = sorted([str(n) for n in remaining_nodes], key=node_key)

    for order, nid in enumerate(remaining_nodes, start=1):
        if nid not in instance["node_dict"]:
            continue

        base_cost, _, _ = solver.solution_cost(solution, instance)
        best_sol = None
        best_score = float("inf")
        best_delta = None
        best_action = ""

        route_indices = choose_candidate_routes(solution, nid, baseline_node_map, max_candidate_routes)

        for ridx in route_indices:
            route = solution[ridx]
            positions = choose_insert_positions(route, nid, ridx, baseline_node_map, max_insert_positions)

            for pos in positions:
                cand_route = insert_node(route, nid, pos, solver, instance)
                cand_sol = clone_solution(solution)
                cand_sol[ridx] = cand_route
                solver.refresh_solution_cache(cand_sol, instance)
                cost, _, feasible = solver.solution_cost(cand_sol, instance)
                if not feasible:
                    continue

                delta = cost - base_cost
                penalty = 0.0

                old = baseline_node_map.get(str(nid))
                if old is not None:
                    if old.get("vehicle_type") != cand_route.get("vehicle_type"):
                        penalty += disturbance_weight

                    old_arr = old.get("arrival_time")
                    if old_arr is not None and time_shift_weight > 0:
                        ev = solver.evaluate_route(cand_route, instance)
                        try:
                            new_pos = cand_route["stops"].index(str(nid))
                            if new_pos < len(ev.details):
                                new_arr = safe_float(ev.details[new_pos].get("arrival_time"))
                                penalty += abs(new_arr - safe_float(old_arr)) * time_shift_weight
                        except Exception:
                            pass

                score = delta + penalty
                if score < best_score:
                    best_score = score
                    best_delta = delta
                    best_sol = cand_sol
                    best_action = f"insert_route_{ridx}_pos_{pos}"

        if best_sol is None:
            new_route = make_new_route_for_node(solver, instance, solution, nid)
            if new_route is not None:
                cand_sol = clone_solution(solution) + [new_route]
                solver.refresh_solution_cache(cand_sol, instance)
                cost, _, feasible = solver.solution_cost(cand_sol, instance)
                if feasible:
                    best_sol = cand_sol
                    best_delta = cost - base_cost
                    best_action = "open_new_route"

        if best_sol is None:
            logs.append({
                "repair_order": order,
                "node_id": nid,
                "status": "failed",
                "action": "",
                "delta_cost": "",
            })
            raise RuntimeError(f"节点 {nid} 无法插入或新开车辆。")

        solution = best_sol
        solver.refresh_solution_cache(solution, instance)
        logs.append({
            "repair_order": order,
            "node_id": nid,
            "status": "success",
            "action": best_action,
            "delta_cost": best_delta,
        })

    return solution, logs


def light_relocate_improve(
    solver,
    instance: dict,
    solution: List[dict],
    max_trials: int,
) -> List[dict]:
    """
    轻量 relocate 改进，只动非锁定节点。
    """
    if max_trials <= 0:
        return solution

    sol = clone_solution(solution)
    solver.refresh_solution_cache(sol, instance)
    best_cost, _, feasible = solver.solution_cost(sol, instance)
    if not feasible:
        return sol

    trials = 0
    improved = True

    while improved and trials < max_trials:
        improved = False

        for i, ri in enumerate(list(sol)):
            locked_i = int(ri.get("locked_len", 0))
            for pi in range(locked_i, len(ri.get("stops", []))):
                if trials >= max_trials:
                    break
                nid = ri["stops"][pi]

                for j, rj in enumerate(list(sol)):
                    if trials >= max_trials:
                        break
                    locked_j = int(rj.get("locked_len", 0))
                    candidate_positions = [locked_j, len(rj.get("stops", []))]
                    candidate_positions = sorted(set([p for p in candidate_positions if 0 <= p <= len(rj.get("stops", []))]))

                    for pj in candidate_positions:
                        trials += 1
                        if i == j and (pj == pi or pj == pi + 1):
                            continue

                        cand = clone_solution(sol)
                        moved = cand[i]["stops"].pop(pi)
                        insert_pos = pj
                        if i == j and pj > pi:
                            insert_pos = pj - 1
                        cand[j]["stops"].insert(insert_pos, moved)
                        cand = [r for r in cand if len(r.get("stops", [])) > 0]

                        for r in cand:
                            r.pop("cached_eval", None)
                            try:
                                solver.rebuild_route_cache(r, instance["node_dict"])
                            except Exception:
                                pass
                        solver.refresh_solution_cache(cand, instance)

                        cost, _, feas = solver.solution_cost(cand, instance)
                        if feas and cost + 1e-9 < best_cost:
                            sol = cand
                            best_cost = cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return sol


# =============================================================================
# 九、动态重调度流程
# =============================================================================

def summarize_solution_basic(solver, solution: List[dict], instance: dict) -> dict:
    solver.refresh_solution_cache(solution, instance)
    cost, evals, feasible = solver.solution_cost(solution, instance)
    cover = solver.coverage_report(solution, instance)

    fixed = sum(ev.fixed_cost for ev in evals)
    wait_cost = sum(ev.wait_cost for ev in evals)
    late_cost = sum(ev.late_cost for ev in evals)
    energy = sum(ev.energy_cost for ev in evals)
    carbon = sum(ev.carbon_cost for ev in evals)
    distance = sum(ev.total_distance for ev in evals)
    travel = sum(ev.total_travel_minutes for ev in evals)
    wait_min = sum(sum(ev.waits) for ev in evals)
    late_min = sum(sum(ev.lates) for ev in evals)

    return {
        "total_cost": float(cost),
        "feasible": bool(feasible),
        "fixed_cost": float(fixed),
        "wait_cost": float(wait_cost),
        "late_cost": float(late_cost),
        "energy_cost": float(energy),
        "carbon_cost": float(carbon),
        "distance_km": float(distance),
        "travel_minutes": float(travel),
        "wait_minutes": float(wait_min),
        "late_minutes": float(late_min),
        "used_vehicle_count": int(cover.get("used_vehicle_count", len(solution))),
        "missing_node_count": int(cover.get("missing_node_count", 0)),
        "actual_weight": float(cover.get("actual_weight", 0.0)),
        "expected_weight": float(cover.get("expected_weight", 0.0)),
        "actual_volume": float(cover.get("actual_volume", 0.0)),
        "expected_volume": float(cover.get("expected_volume", 0.0)),
    }



def solution_complete_feasible(solver, solution: List[dict], instance: dict) -> Tuple[bool, dict]:
    """
    判断当前方案是否满足：
    1. 路径评估可行；
    2. 服务节点覆盖完整；
    3. 缺失节点数为 0。
    """
    solver.refresh_solution_cache(solution, instance)
    cost, _, feasible = solver.solution_cost(solution, instance)
    cover = solver.coverage_report(solution, instance)
    ok = bool(feasible and cover.get("missing_node_count", 999999) == 0)
    return ok, {"total_cost": cost, **cover}


def get_node_late_sum(solver, solution: List[dict], instance: dict, node_ids: List[str]) -> float:
    """
    统计指定节点在当前方案中的迟到分钟数。
    对时间窗提前事件，若目标节点仍然迟到，则不能直接接受原方案。
    """
    targets = {str(x) for x in node_ids}
    late_sum = 0.0
    for route in solution:
        ev = solver.evaluate_route(route, instance)
        for pos, nid in enumerate(route.get("stops", [])):
            if str(nid) in targets and pos < len(ev.details):
                late_sum += safe_float(ev.details[pos].get("late_min"), 0.0)
    return late_sum


def affected_route_indices_by_nodes(solution: List[dict], node_ids: List[str]) -> List[int]:
    targets = {str(x) for x in node_ids}
    out = []
    for i, route in enumerate(solution):
        if any(str(n) in targets for n in route.get("stops", [])):
            out.append(i)
    return out


def build_targeted_partial_solution(
    solver,
    solution: List[dict],
    instance: dict,
    affected_route_indices: List[int],
    locked_prefix_lengths: List[int],
    additional_remaining_nodes: Optional[List[str]] = None,
) -> Tuple[List[dict], set, List[str]]:
    """
    构造更符合实际调度的局部修复初始解。

    思路：
    1. 不再把所有未服务节点全部拆出来重插；
    2. 非受影响路线保持完整，并将 locked_len 设为整条路线长度，表示原则上不改变；
    3. 受影响路线只保留冻结前缀，将冻结前缀之后的节点作为待修复节点；
    4. 新增订单节点作为额外待插入节点。
    """
    affected_set = set(int(x) for x in affected_route_indices)
    partial = []
    fixed_nodes = set()
    remaining = []

    for ridx, route in enumerate(solution):
        stops = [str(x) for x in route.get("stops", []) if str(x) in instance["node_dict"]]
        if not stops:
            continue

        if ridx in affected_set:
            locked_len = locked_prefix_lengths[ridx] if ridx < len(locked_prefix_lengths) else 0
            locked_len = max(0, min(int(locked_len), len(stops)))
            prefix = stops[:locked_len]
            suffix = stops[locked_len:]

            if prefix:
                r = {
                    "vehicle_type": route["vehicle_type"],
                    "stops": prefix,
                    "locked_len": len(prefix),
                }
                try:
                    solver.rebuild_route_cache(r, instance["node_dict"])
                except Exception:
                    pass
                partial.append(r)
                fixed_nodes.update(prefix)

            remaining.extend(suffix)
        else:
            # 非受影响路线保持完整，减少扰动；只允许必要时在路线末端追加。
            r = {
                "vehicle_type": route["vehicle_type"],
                "stops": stops,
                "locked_len": len(stops),
            }
            try:
                solver.rebuild_route_cache(r, instance["node_dict"])
            except Exception:
                pass
            partial.append(r)
            fixed_nodes.update(stops)

    if additional_remaining_nodes:
        for n in additional_remaining_nodes:
            n = str(n)
            if n in instance["node_dict"] and n not in fixed_nodes and n not in remaining:
                remaining.append(n)

    # 去重并过滤不存在节点
    seen = set()
    remaining_clean = []
    for n in remaining:
        n = str(n)
        if n in instance["node_dict"] and n not in seen and n not in fixed_nodes:
            remaining_clean.append(n)
            seen.add(n)

    return partial, fixed_nodes, remaining_clean


def should_accept_event_solution_directly(
    solver,
    event: dict,
    event_solution: List[dict],
    event_instance: dict,
    event_notes: dict,
) -> Tuple[bool, str]:
    """
    判断事件应用后是否可以直接接受，不进入大规模重调度。

    更符合实际的逻辑：
    - 追加订单、地址变更、订单取消：如果当前方案仍完整可行，则直接接受；
    - 时间窗提前：如果当前方案完整可行且目标客户不迟到，则直接接受；
    - 新增客户：因为新节点尚未插入，不能直接接受。
    """
    ok, _ = solution_complete_feasible(solver, event_solution, event_instance)
    if not ok:
        return False, "事件应用后当前方案不可行或覆盖不完整，需要局部修复。"

    etype = event.get("type", "")
    if etype == "new_customer":
        return False, "新增客户尚未插入任何路线，需要局部插入。"

    if etype == "time_window":
        target_nodes = event_notes.get("updated_nodes", [])
        late_sum = get_node_late_sum(solver, event_solution, event_instance, target_nodes)
        if late_sum <= 1e-9:
            return True, "时间窗调整后目标节点无迟到，原方案可直接接受。"
        return False, f"时间窗调整后目标节点仍迟到 {late_sum:.2f} 分钟，需要局部修复。"

    return True, "事件应用后当前方案仍完整可行，直接接受，避免不必要的大范围重排。"


def get_event_target_nodes(event_notes: dict) -> List[str]:
    nodes = []
    for key in ["updated_nodes", "added_nodes", "removed_nodes"]:
        for n in event_notes.get(key, []) or []:
            nodes.append(str(n))
    out = []
    seen = set()
    for n in nodes:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def targeted_repair_after_event(
    solver,
    event: dict,
    event_instance: dict,
    event_solution: List[dict],
    event_notes: dict,
    baseline_map: Dict[str, dict],
    locked_prefix_lengths: List[int],
    max_candidate_routes: int,
    max_insert_positions: int,
    disturbance_weight: float,
    time_shift_weight: float,
) -> Tuple[List[dict], List[dict], str, int, int]:
    """
    先做“受影响路径局部修复”，失败后再回退到“全剩余任务修复”。

    这比原来的全量剩余任务重插更符合实际动态调度：
    小事件优先只改受影响车辆和新增订单，不轻易打乱所有车辆。
    """
    target_nodes = get_event_target_nodes(event_notes)
    added_nodes = [str(x) for x in event_notes.get("added_nodes", []) or [] if str(x) in event_instance["node_dict"]]

    affected_routes = affected_route_indices_by_nodes(event_solution, target_nodes)
    if event.get("type") == "new_customer" and not affected_routes:
        affected_routes = []

    partial, fixed_nodes, remaining = build_targeted_partial_solution(
        solver,
        event_solution,
        event_instance,
        affected_routes,
        locked_prefix_lengths,
        additional_remaining_nodes=added_nodes,
    )

    # 对于没有受影响路线但有新增节点的情形，partial就是完整原方案，remaining就是新增节点。
    try:
        new_solution, repair_logs = repair_remaining_tasks(
            solver,
            event_instance,
            partial,
            remaining,
            baseline_map,
            max_candidate_routes=max_candidate_routes,
            max_insert_positions=max_insert_positions,
            disturbance_weight=disturbance_weight,
            time_shift_weight=time_shift_weight,
        )
        ok, _ = solution_complete_feasible(solver, new_solution, event_instance)
        if ok:
            return new_solution, repair_logs, "targeted_local_repair", len(fixed_nodes), len(remaining)
    except Exception as e:
        first_error = repr(e)
    else:
        first_error = "局部修复后覆盖不完整或不可行。"

    # 回退：如果局部修复失败，再扩大到所有冻结窗口外剩余任务。
    partial_all, fixed_nodes_all = build_locked_partial_solution(
        solver,
        event_solution,
        event_instance,
        locked_prefix_lengths,
    )
    all_nodes = expected_service_nodes(event_instance)
    remaining_all = sorted(list(all_nodes - fixed_nodes_all))

    repair_logs_prefix = [{
        "repair_order": 0,
        "node_id": "",
        "status": "fallback",
        "action": "targeted_failed_then_global_remaining_repair",
        "delta_cost": "",
        "message": first_error,
    }]

    new_solution, repair_logs = repair_remaining_tasks(
        solver,
        event_instance,
        partial_all,
        remaining_all,
        baseline_map,
        max_candidate_routes=max_candidate_routes,
        max_insert_positions=max_insert_positions,
        disturbance_weight=disturbance_weight,
        time_shift_weight=time_shift_weight,
    )
    repair_logs = repair_logs_prefix + repair_logs

    return new_solution, repair_logs, "fallback_global_remaining_repair", len(fixed_nodes_all), len(remaining_all)


def build_result_row(
    solver,
    before_instance: dict,
    before_solution: List[dict],
    after_instance: dict,
    after_solution: List[dict],
    event: dict,
    event_notes: dict,
    completed_count: int,
    frozen_count: int,
    locked_count: int,
    remaining_count: int,
    response_seconds: float,
    repair_strategy: str,
) -> dict:
    before_summary = summarize_solution_basic(solver, before_solution, before_instance)
    after_summary = summarize_solution_basic(solver, after_solution, after_instance)
    disturbance = compute_disturbance_metrics(
        solver,
        before_solution,
        after_solution,
        before_instance,
        after_instance,
    )

    return {
        "scenario": event.get("scenario", ""),
        "event_id": event.get("event_id", ""),
        "event_name": event.get("name", ""),
        "event_type": event.get("type", ""),
        "event_time": min_to_clock(event.get("time_min", 0)),
        "event_description": event.get("description", ""),
        "freeze_min": event.get("freeze_min", ""),
        "completed_node_count": completed_count,
        "frozen_node_count": frozen_count,
        "locked_node_count": locked_count,
        "remaining_node_count": remaining_count,
        "repair_strategy": repair_strategy,

        "before_total_cost": before_summary["total_cost"],
        "after_total_cost": after_summary["total_cost"],
        "delta_total_cost": after_summary["total_cost"] - before_summary["total_cost"],

        "before_distance_km": before_summary["distance_km"],
        "after_distance_km": after_summary["distance_km"],
        "delta_distance_km": after_summary["distance_km"] - before_summary["distance_km"],

        "before_wait_minutes": before_summary["wait_minutes"],
        "after_wait_minutes": after_summary["wait_minutes"],
        "delta_wait_minutes": after_summary["wait_minutes"] - before_summary["wait_minutes"],

        "before_late_minutes": before_summary["late_minutes"],
        "after_late_minutes": after_summary["late_minutes"],
        "delta_late_minutes": after_summary["late_minutes"] - before_summary["late_minutes"],

        "before_carbon_cost": before_summary["carbon_cost"],
        "after_carbon_cost": after_summary["carbon_cost"],
        "delta_carbon_cost": after_summary["carbon_cost"] - before_summary["carbon_cost"],

        "before_used_vehicle_count": before_summary["used_vehicle_count"],
        "after_used_vehicle_count": after_summary["used_vehicle_count"],
        "delta_used_vehicle_count": after_summary["used_vehicle_count"] - before_summary["used_vehicle_count"],

        "after_feasible": after_summary["feasible"],
        "after_missing_node_count": after_summary["missing_node_count"],
        "response_time_seconds": round(response_seconds, 3),

        "event_applied": event_notes.get("applied"),
        "event_notes": event_notes.get("message"),
        **disturbance,
    }

def dynamic_reschedule_once(
    solver,
    current_instance: dict,
    current_solution: List[dict],
    event: dict,
    freeze_min: int,
    max_candidate_routes: int,
    max_insert_positions: int,
    improve_trials: int,
    disturbance_weight: float,
    time_shift_weight: float,
) -> Tuple[dict, List[dict], dict, List[dict]]:
    """
    对一个事件进行状态截断与重调度。

    新版逻辑更符合真实调度：
    1. 先应用事件并检查原计划是否仍可执行；
    2. 若原计划仍可执行，则直接接受，不重排所有剩余任务；
    3. 若不可执行，优先只修复受影响车辆/新增节点；
    4. 局部修复失败时，才回退到冻结窗口外的较大范围修复。
    """
    t0 = time.time()
    event_time = int(event["time_min"])
    event = dict(event)
    event["freeze_min"] = freeze_min

    before_instance = copy.deepcopy(current_instance)
    before_solution = clone_solution(current_solution)
    baseline_map = build_node_plan_map(solver, before_solution, before_instance)

    locked_lens, completed_nodes, frozen_nodes = compute_locked_nodes(
        solver, before_solution, before_instance, event_time, freeze_min
    )

    event_instance, event_solution, event_notes = apply_event(
        solver,
        before_instance,
        before_solution,
        event,
        completed_nodes,
        frozen_nodes,
    )

    locked_lens2, completed_nodes2, frozen_nodes2 = compute_locked_nodes(
        solver, event_solution, event_instance, event_time, freeze_min
    )

    # 先检查事件应用后是否可以直接接受
    accept_direct, accept_reason = should_accept_event_solution_directly(
        solver, event, event_solution, event_instance, event_notes
    )

    if accept_direct:
        repair_logs = [{
            "repair_order": 0,
            "node_id": "",
            "status": "accepted_without_repair",
            "action": "direct_accept",
            "delta_cost": 0,
            "message": accept_reason,
        }]
        result = build_result_row(
            solver,
            before_instance,
            before_solution,
            event_instance,
            event_solution,
            event,
            event_notes,
            completed_count=len(completed_nodes),
            frozen_count=len(frozen_nodes),
            locked_count=len(frozen_nodes2),
            remaining_count=0,
            response_seconds=time.time() - t0,
            repair_strategy="direct_accept_no_global_reschedule",
        )
        result["event_notes"] = str(result.get("event_notes", "")) + "；" + accept_reason
        return event_instance, event_solution, result, repair_logs

    # 不能直接接受时，先做受影响局部修复
    new_solution, repair_logs, repair_strategy, locked_count, remaining_count = targeted_repair_after_event(
        solver,
        event,
        event_instance,
        event_solution,
        event_notes,
        baseline_map,
        locked_prefix_lengths=locked_lens2,
        max_candidate_routes=max_candidate_routes,
        max_insert_positions=max_insert_positions,
        disturbance_weight=disturbance_weight,
        time_shift_weight=time_shift_weight,
    )

    # 轻量局部改进，不改变冻结前缀
    new_solution = light_relocate_improve(
        solver,
        event_instance,
        new_solution,
        max_trials=improve_trials,
    )

    result = build_result_row(
        solver,
        before_instance,
        before_solution,
        event_instance,
        new_solution,
        event,
        event_notes,
        completed_count=len(completed_nodes),
        frozen_count=len(frozen_nodes),
        locked_count=locked_count,
        remaining_count=remaining_count,
        response_seconds=time.time() - t0,
        repair_strategy=repair_strategy,
    )

    return event_instance, new_solution, result, repair_logs


# =============================================================================
# 十、场景定义
# =============================================================================

def build_scenario_events() -> Dict[str, List[dict]]:
    """
    构造建模手给出的 2 个单事件场景 + 1 个综合事件场景。
    时间统一为相对 8:00 的分钟数。
    """
    s1 = [
        {
            "scenario": "S1_已有客户新增订单",
            "event_id": "S1",
            "name": "午间客户25追加订单",
            "type": "add_to_existing_customer",
            "time_min": clock_to_min("12:00"),
            "customer_id": 25,
            "add_weight": 180.0,
            "add_volume": 0.45,
            "service_time": 20,
            "description": "12:00 客户25追加订单，新增180kg、0.45m³，要求与客户25原配送任务合并完成。",
        }
    ]

    s2 = [
        {
            "scenario": "S2_客户时间窗提前",
            "event_id": "S2",
            "name": "客户16临时要求提前收货",
            "type": "time_window",
            "time_min": clock_to_min("09:30"),
            "customer_id": 16,
            "new_earliest": clock_to_min("08:50"),
            "new_latest": clock_to_min("09:40"),
            "description": "09:30 客户16将时间窗调整为[08:50,09:40]。",
        }
    ]

    s3 = [
        {
            "scenario": "S3_综合多事件联动",
            "event_id": "E1",
            "name": "客户16时间窗提前",
            "type": "time_window",
            "time_min": clock_to_min("09:30"),
            "customer_id": 16,
            "new_earliest": clock_to_min("08:50"),
            "new_latest": clock_to_min("09:40"),
            "description": "E1 09:30 客户16时间窗提前为[08:50,09:40]。",
        },
        {
            "scenario": "S3_综合多事件联动",
            "event_id": "E2",
            "name": "客户48订单取消",
            "type": "cancel_customer",
            "time_min": clock_to_min("10:30"),
            "customer_id": 48,
            "description": "E2 10:30 客户48取消订单，若尚未服务且不在冻结窗口则删除。",
        },
        {
            "scenario": "S3_综合多事件联动",
            "event_id": "E3",
            "name": "绿色区新增急单客户X",
            "type": "new_customer",
            "time_min": clock_to_min("11:00"),
            "customer_id": 10001,
            "node_prefix": "X",
            "x": 3.5,
            "y": 6.8,
            "weight": 210.0,
            "volume": 0.52,
            "earliest": clock_to_min("12:00"),
            "latest": clock_to_min("13:30"),
            "service_time": 20,
            "description": "E3 11:00 新增绿色区急单客户X，坐标(3.5,6.8)，时间窗[12:00,13:30]。",
        },
        {
            "scenario": "S3_综合多事件联动",
            "event_id": "E4",
            "name": "客户25地址变更",
            "type": "address_change",
            "time_min": clock_to_min("14:00"),
            "customer_id": 25,
            "new_x": 9.2,
            "new_y": 10.8,
            "description": "E4 14:00 客户25地址变更为(9.2,10.8)，时间窗和需求不变。",
        },
        {
            "scenario": "S3_综合多事件联动",
            "event_id": "E5",
            "name": "区外新增普通订单客户Y",
            "type": "new_customer",
            "time_min": clock_to_min("15:30"),
            "customer_id": 10002,
            "node_prefix": "Y",
            "x": -12.0,
            "y": 8.0,
            "weight": 350.0,
            "volume": 0.88,
            "earliest": clock_to_min("17:00"),
            "latest": clock_to_min("20:30"),
            "service_time": 20,
            "description": "E5 15:30 新增区外普通订单客户Y，坐标(-12.0,8.0)，时间窗[17:00,20:30]。",
        },
    ]

    return {
        "S1_已有客户新增订单": s1,
        "S2_客户时间窗提前": s2,
        "S3_综合多事件联动": s3,
    }


# =============================================================================
# 十一、输出
# =============================================================================

def save_solution_files(solver, instance: dict, solution: List[dict], folder: Path, prefix: str):
    ensure_dir(folder)
    try:
        route_df, summary = solver.summarize_solution(solution, instance)
        route_df.to_excel(folder / f"{prefix}_路线汇总.xlsx", index=False)
        with open(folder / f"{prefix}_结果摘要.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        # 兜底输出
        rows = []
        for i, r in enumerate(solution, start=1):
            ev = solver.evaluate_route(r, instance)
            rows.append({
                "route_id": i,
                "vehicle_type": r["vehicle_type"],
                "route": "0-" + "-".join(r["stops"]) + "-0",
                "cost": ev.total_cost,
                "distance_km": ev.total_distance,
                "travel_minutes": ev.total_travel_minutes,
            })
        pd.DataFrame(rows).to_excel(folder / f"{prefix}_路线汇总.xlsx", index=False)

    # 弧段明细
    detail_rows = []
    for i, r in enumerate(solution, start=1):
        ev = solver.evaluate_route(r, instance)
        for seq, det in enumerate(ev.details, start=1):
            row = {
                "route_id": i,
                "vehicle_type": r["vehicle_type"],
                "seq": seq,
                **det,
            }
            detail_rows.append(row)
    pd.DataFrame(detail_rows).to_excel(folder / f"{prefix}_弧段明细.xlsx", index=False)


def save_event_notes(folder: Path, event: dict, result: dict, repair_logs: List[dict]):
    ensure_dir(folder)
    with open(folder / "事件设定.json", "w", encoding="utf-8") as f:
        json.dump(event, f, ensure_ascii=False, indent=2)
    pd.DataFrame([result]).to_excel(folder / "事件结果指标.xlsx", index=False)
    pd.DataFrame(repair_logs).to_excel(folder / "重调度修复日志.xlsx", index=False)


# =============================================================================
# 十二、主程序
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="问题三：按建模手指定情景假设进行动态调度。")
    parser.add_argument("--data-dir", type=str, default=r"D:\A_Question", help="数据根目录")
    parser.add_argument("--solver", type=str, default=r"D:\A_Question\q2.py", help="q2求解器路径")
    parser.add_argument("--base-output-dir", type=str, default=r"D:\A_Question\问题二输出", help="问题二初始方案输出目录")
    parser.add_argument("--outdir", type=str, default=r"D:\A_Question\问题三输出", help="问题三输出目录")

    parser.add_argument("--base-seed", type=int, default=3407, help="读取既有方案失败时，用该seed重新生成基准方案")
    parser.add_argument("--base-iterations", type=int, default=120, help="读取既有方案失败时，基准方案迭代次数")
    parser.add_argument("--split-policy", type=str, default="balanced", choices=["safe", "balanced", "compact"])
    parser.add_argument("--green-policy-mode", type=str, default="strict", choices=["strict", "wait"])

    parser.add_argument("--freeze-min", type=int, default=30, help="冻结窗口长度，单位分钟")
    parser.add_argument("--max-candidate-routes", type=int, default=12, help="每个节点最多尝试的候选路线数")
    parser.add_argument("--max-insert-positions", type=int, default=5, help="每条候选路线最多尝试的关键插入位置数")
    parser.add_argument("--improve-trials", type=int, default=30, help="轻量局部改进最大尝试次数")
    parser.add_argument("--disturbance-weight", type=float, default=30.0, help="换车扰动惩罚权重")
    parser.add_argument("--time-shift-weight", type=float, default=0.1, help="到达时间偏移惩罚权重")

    parser.add_argument("--skip-s3", action="store_true", help="只运行S1、S2，不运行综合场景S3")
    parser.add_argument("--scenario-limit", type=int, default=0, help="仅调试时使用：限制运行前N个场景，0表示不限制")
    parser.add_argument("--no-clean-outdir", action="store_true", help="不清理旧的问题三输出目录；默认会清理旧结果以避免不同版本输出混杂")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    solver_path = Path(args.solver)
    base_output_dir = Path(args.base_output_dir)
    outdir = Path(args.outdir)
    if not args.no_clean_outdir:
        clean_q3_output_dir(outdir)
    outdir = ensure_dir(outdir)

    solver = import_solver(solver_path)

    # 设置 q2 求解器全局参数
    try:
        solver.SPLIT_POLICY = args.split_policy
        solver.GREEN_POLICY_MODE = args.green_policy_mode
    except Exception:
        pass

    print("[1/6] 读取数据并构造问题二政策实例...")
    instance = solver.load_instance(data_dir)
    rebuild_green_metadata_if_possible(solver, instance)

    print("[2/6] 读取或生成问题二初始调度方案...")
    base_solution = load_solution_from_output(solver, instance, base_output_dir)
    if base_solution is None:
        print("  未能读取既有方案，开始重新求解基准方案...")
        base_solution = build_base_solution_by_solver(
            solver,
            instance,
            seed=args.base_seed,
            iterations=args.base_iterations,
            skip_local_search=False,
        )

    solver.refresh_solution_cache(base_solution, instance)
    base_summary = summarize_solution_basic(solver, base_solution, instance)
    print(f"  基准方案成本: {base_summary['total_cost']:.2f} | 车辆数: {base_summary['used_vehicle_count']} | 缺失节点: {base_summary['missing_node_count']}")

    baseline_dir = ensure_dir(outdir / "阶段0_问题二初始方案")
    save_solution_files(solver, instance, base_solution, baseline_dir, "阶段0_问题二初始方案")

    print("[3/6] 构造指定动态事件场景...")
    scenarios = build_scenario_events()
    if args.skip_s3:
        scenarios = {k: v for k, v in scenarios.items() if not k.startswith("S3")}
    if args.scenario_limit and args.scenario_limit > 0:
        items = list(scenarios.items())[:args.scenario_limit]
        scenarios = dict(items)

    for name, events in scenarios.items():
        print(f"  - {name}: {len(events)} 个事件")

    print("[4/6] 逐场景执行动态重调度...")
    all_results = []
    all_repair_logs = []

    for scenario_name, events in scenarios.items():
        print(f"\n>>> 正在处理场景：{scenario_name}")
        scenario_dir = ensure_dir(outdir / scenario_name)

        current_instance = copy.deepcopy(instance)
        current_solution = clone_solution(base_solution)

        for stage_idx, event in enumerate(events, start=1):
            event_folder = ensure_dir(scenario_dir / f"阶段{stage_idx}_{event['event_id']}_{event['name']}")
            print(f"  [{scenario_name}] {event['event_id']} {event['name']} @ {min_to_clock(event['time_min'])}")

            try:
                new_instance, new_solution, result, repair_logs = dynamic_reschedule_once(
                    solver,
                    current_instance,
                    current_solution,
                    event,
                    freeze_min=args.freeze_min,
                    max_candidate_routes=args.max_candidate_routes,
                    max_insert_positions=args.max_insert_positions,
                    improve_trials=args.improve_trials,
                    disturbance_weight=args.disturbance_weight,
                    time_shift_weight=args.time_shift_weight,
                )

                save_solution_files(solver, new_instance, new_solution, event_folder, f"{event['event_id']}_{event['name']}")
                save_event_notes(event_folder, event, result, repair_logs)

                for r in repair_logs:
                    rr = dict(r)
                    rr["scenario"] = scenario_name
                    rr["event_id"] = event["event_id"]
                    all_repair_logs.append(rr)

                all_results.append(result)

                print(
                    f"    完成：调整后成本={result['after_total_cost']:.2f} | "
                    f"成本变化={result['delta_total_cost']:.2f} | "
                    f"迟到变化={result['delta_late_minutes']:.2f}min | "
                    f"响应时间={result['response_time_seconds']:.2f}s | "
                    f"缺失节点={result['after_missing_node_count']}"
                )

                # 综合场景下一阶段基于上一阶段方案继续运行
                current_instance = new_instance
                current_solution = new_solution

            except Exception as e:
                print(f"    失败：{repr(e)}")
                fail_row = {
                    "scenario": scenario_name,
                    "event_id": event.get("event_id", ""),
                    "event_name": event.get("name", ""),
                    "event_type": event.get("type", ""),
                    "event_time": min_to_clock(event.get("time_min", 0)),
                    "after_feasible": False,
                    "error": repr(e),
                }
                all_results.append(fail_row)

    print("\n[5/6] 输出全局汇总表...")
    result_df = pd.DataFrame(all_results)
    result_df.to_excel(outdir / "问题三_动态事件结果汇总.xlsx", index=False)

    if all_repair_logs:
        pd.DataFrame(all_repair_logs).to_excel(outdir / "问题三_全部重调度修复日志.xlsx", index=False)

    print("[6/6] 输出说明文件...")
    lines = []
    lines.append("问题三输出说明")
    lines.append("=" * 60)
    lines.append("本代码按照建模手指定的 2 个单事件场景 + 1 个综合事件场景进行动态重调度。")
    lines.append("主线方法：事件驱动滚动时域重调度 + 先保持原方案 + 受影响车辆局部修复 + 必要时扩大重调度。")
    lines.append("")
    lines.append(f"初始方案目录：{base_output_dir}")
    lines.append(f"绿色区政策模式：{args.green_policy_mode}")
    lines.append(f"冻结窗口：{args.freeze_min} 分钟")
    lines.append(f"候选路线数上限：{args.max_candidate_routes}")
    lines.append(f"每条路线候选插入位置上限：{args.max_insert_positions}")
    lines.append(f"轻量局部改进尝试次数：{args.improve_trials}")
    lines.append("")
    lines.append("核心文件：")
    lines.append("1. 问题三_动态事件结果汇总.xlsx：所有场景/阶段的成本、距离、等待、迟到、碳排、扰动和响应时间。")
    lines.append("2. 各场景文件夹：每个事件阶段的路线汇总、弧段明细、结果指标、修复日志。")
    lines.append("3. 阶段0_问题二初始方案：问题三使用的问题二初始调度方案。")
    lines.append("")
    lines.append("新增客户 X、Y 说明：题目原距离矩阵没有新客户行列，代码采用“欧氏距离 × 既有道路系数中位数”的近似方法补充距离矩阵。")
    (outdir / "问题三_输出说明.txt").write_text("\n".join(lines), encoding="utf-8")

    print("完成。结果已输出到：", outdir)


if __name__ == "__main__":
    main()
