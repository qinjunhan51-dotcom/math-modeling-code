# -*- coding: utf-8 -*-
"""
问题二：城市绿色物流配送调度（绿色配送区限行情景）

实现思路：
1. 标准化读取四张原始表（订单、距离矩阵、坐标、时间窗）；
2. 进行订单层清洗与缺失值修复；
3. 仅对“超容量客户”做订单级双容量拆分，生成虚拟服务节点；
4. 构造统一实例对象（服务节点、距离矩阵、车辆参数、速度参数、成本参数）；
5. 使用“改进 ALNS”做主求解；
6. 使用 VNS/TS 风格的局部搜索做精修；
7. 对固定路径方案做鲁棒仿真，检验速度波动下的稳定性；
8. 输出路径结果、成本分解、图件和汇总文件。

说明：
- 本文件重在“可运行的工程实现”，对应论文中的“改进 ALNS + 局部优化 + 鲁棒仿真”三层。
- MILP 作为论文中的理论模型主体，建议在论文中单独写出；程序中不强制大规模精确求解。
- 路径采用“单车单趟闭合路径”：0 -> ... -> 0。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 全局常量（与题意一致）
# =========================
MAX_WEIGHT = 3000.0
MAX_VOLUME = 15.0
# 按建模手最终口径：仅当客户汇总需求超过最大车型容量时才做虚拟拆分。
# 普通客户保持“一个客户一个服务节点”；超容量客户进行订单级双容量分组拆分。
SPLIT_TRIGGER_WEIGHT = MAX_WEIGHT
SPLIT_TRIGGER_VOLUME = MAX_VOLUME
# 拆分策略说明：
# safe：稳定兼容型。所有虚拟节点尽量控制在 1500kg / 10.8m³ 内，车型兼容性最高。
# balanced：均衡降成本型。先按 safe 生成安全小包，再在同一原客户内部有限合并，尝试减少虚拟节点数。
# compact：激进压缩型。直接按 3000kg / 13.5m³ 生成较大包，仅用于对比试验，可能造成 3t 车辆不足。
SPLIT_POLICY = "balanced"

# safe 模式容量，也是 Fuel_1500 的容量；有利于保留中型车可用性。
SAFE_CAP_WEIGHT = 1500.0
SAFE_CAP_VOLUME = 10.8

# balanced / compact 中的大包上限优先按 Fuel_3000 容积 13.5 控制，避免过度依赖仅 10 辆的 EV_3000。
FUEL3000_CAP_WEIGHT = 3000.0
FUEL3000_CAP_VOLUME = 13.5
EV3000_CAP_WEIGHT = 3000.0
EV3000_CAP_VOLUME = 15.0

# 兼容旧变量名：旧拆分函数仍保留，但新版本默认不再依赖“软 target 自动跳到 3000/15”的逻辑。
SPLIT_GROUP_TARGET_WEIGHT = SAFE_CAP_WEIGHT
SPLIT_GROUP_TARGET_VOLUME = SAFE_CAP_VOLUME
SERVICE_MIN = 20
WAIT_COST_PER_HOUR = 20.0
LATE_COST_PER_HOUR = 50.0
FUEL_PRICE = 7.61
ELEC_PRICE = 1.64
CARBON_PRICE = 0.65
FUEL_CARBON_COEF = 2.547   # kg/L
ELEC_CARBON_COEF = 0.501   # kg/kWh

# =========================
# 问题二新增：绿色配送区限行政策参数
# =========================
# 时间原点仍沿用问题一：8:00 记为 0 分钟。
# 因此 8:00—16:00 对应 [0, 480)。
GREEN_CENTER_X = 0.0
GREEN_CENTER_Y = 0.0
GREEN_RADIUS_KM = 10.0
GREEN_BAN_START_MIN = 0.0
GREEN_BAN_END_MIN = 480.0

# 政策处理方式：
# "wait"：燃油车若将在禁行时段进入绿色区节点或穿越绿色区弧段，则等待到 16:00 后再执行；
# "strict"：直接判该路线不可行。
# 当前最终论文口径采用 strict：限行窗口内燃油车涉及入区/穿区直接判不可行。
GREEN_POLICY_MODE = "strict"



# =========================
# 数据结构
# =========================
@dataclass
class VehicleType:
    vehicle_type: str
    fuel_type: str  # fuel / electric
    capacity_weight: float
    capacity_volume: float
    count: int
    fixed_cost: float
    load_alpha: float  # 满载相对空载增耗比例


@dataclass
class Node:
    node_id: str
    original_customer_id: int
    sub_id: int
    weight: float
    volume: float
    earliest: int   # 相对 8:00 的分钟数
    latest: int
    service_time: int
    x: float
    y: float
    is_virtual: bool


@dataclass
class RouteEval:
    feasible: bool
    total_cost: float
    fixed_cost: float
    wait_cost: float
    late_cost: float
    energy_cost: float
    carbon_cost: float
    total_distance: float
    total_travel_minutes: float
    departure_time: int
    return_time: float
    arrivals: List[float]
    waits: List[float]
    lates: List[float]
    details: List[dict]
    message: str = ""

    # 问题二新增：绿色区政策相关统计
    policy_wait_min: float = 0.0
    policy_wait_cost: float = 0.0
    policy_affected_arc_count: int = 0
    green_service_count: int = 0
    cross_green_arc_count: int = 0


# =========================
# 工具函数
# =========================
def set_chinese_font():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def find_existing_file(base_dir: Path, candidates: List[str]) -> Path:
    for name in candidates:
        p = base_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到文件：{candidates}")


def to_numeric_int(x, default=0):
    if pd.isna(x):
        return default
    return int(round(float(x)))


def time_to_rel_minutes(x, base_hour=8) -> int:
    if pd.isna(x):
        raise ValueError("时间窗存在空值")
    if isinstance(x, pd.Timestamp):
        h, m = int(x.hour), int(x.minute)
    else:
        s = str(x).strip()
        if ":" not in s:
            raise ValueError(f"无法解析时间：{x}")
        parts = s.split(":")
        h, m = int(parts[0]), int(parts[1])
    return (h - base_hour) * 60 + m


def rel_minutes_to_clock_str(m: float, base_hour=8) -> str:
    m = int(round(m))
    hour = base_hour + m // 60
    minute = m % 60
    return f"{hour:02d}:{minute:02d}"


def is_in_green_ban_window(t_min: float) -> bool:
    """判断相对 8:00 的分钟数是否处于绿色配送区限行时段 [8:00, 16:00)。"""
    return GREEN_BAN_START_MIN <= float(t_min) < GREEN_BAN_END_MIN


def point_in_green_zone(x: float, y: float) -> bool:
    """判断坐标点是否位于绿色配送区内。"""
    dx = float(x) - GREEN_CENTER_X
    dy = float(y) - GREEN_CENTER_Y
    return dx * dx + dy * dy <= GREEN_RADIUS_KM * GREEN_RADIUS_KM + 1e-9


def segment_crosses_green_zone(x1: float, y1: float, x2: float, y2: float) -> bool:
    """
    判断线段是否穿越或触及绿色配送区圆。
    说明：这是基于坐标的几何近似，仅用于政策属性识别；
    实际距离、行驶时间、能耗与碳排放仍使用题目距离矩阵。
    """
    vx = float(x2) - float(x1)
    vy = float(y2) - float(y1)
    wx = GREEN_CENTER_X - float(x1)
    wy = GREEN_CENTER_Y - float(y1)

    c2 = vx * vx + vy * vy
    if c2 <= 1e-12:
        px, py = float(x1), float(y1)
    else:
        t = (vx * wx + vy * wy) / c2
        t = max(0.0, min(1.0, t))
        px = float(x1) + t * vx
        py = float(y1) + t * vy

    dx = px - GREEN_CENTER_X
    dy = py - GREEN_CENTER_Y
    return dx * dx + dy * dy <= GREEN_RADIUS_KM * GREEN_RADIUS_KM + 1e-9


def is_fuel_vehicle_type(vehicle_type: str) -> bool:
    return str(vehicle_type).startswith("Fuel_")


def iqr_upper(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return float(q3 + 1.5 * iqr)


def fuel_consumption_per_100km(v_kmh: float) -> float:
    return 0.0025 * (v_kmh ** 2) - 0.2554 * v_kmh + 31.75


def elec_consumption_per_100km(v_kmh: float) -> float:
    return 0.0014 * (v_kmh ** 2) - 0.12 * v_kmh + 36.19


# =========================
# 参数表
# =========================
def build_vehicle_types() -> List[VehicleType]:
    return [
        VehicleType("Fuel_3000", "fuel", 3000.0, 13.5, 60, 400.0, 0.40),
        VehicleType("Fuel_1500", "fuel", 1500.0, 10.8, 50, 400.0, 0.40),
        VehicleType("Fuel_1250", "fuel", 1250.0, 6.5, 50, 400.0, 0.40),
        VehicleType("EV_3000", "electric", 3000.0, 15.0, 10, 400.0, 0.35),
        VehicleType("EV_1250", "electric", 1250.0, 8.5, 15, 400.0, 0.35),
    ]


def build_speed_periods() -> List[dict]:
    # 相对 8:00 的分钟数；主模型取均值速度
    # 标准差按官方补充说明修正：顺畅0.10，一般5.20，拥堵4.70
    return [
        {"period_id": 1, "start": 0,   "end": 60,  "state": "拥堵", "mean_speed": 9.8,  "std": 4.70},   # 8:00-9:00
        {"period_id": 2, "start": 60,  "end": 120, "state": "顺畅", "mean_speed": 55.3, "std": 0.10},   # 9:00-10:00
        {"period_id": 3, "start": 120, "end": 210, "state": "一般", "mean_speed": 35.4, "std": 5.20},   # 10:00-11:30
        {"period_id": 4, "start": 210, "end": 300, "state": "拥堵", "mean_speed": 9.8,  "std": 4.70},   # 11:30-13:00
        {"period_id": 5, "start": 300, "end": 420, "state": "顺畅", "mean_speed": 55.3, "std": 0.10},   # 13:00-15:00
        {"period_id": 6, "start": 420, "end": 540, "state": "一般", "mean_speed": 35.4, "std": 5.20},   # 15:00-17:00
        {"period_id": 7, "start": 540, "end": 660, "state": "拥堵", "mean_speed": 9.8,  "std": 4.70},   # 17:00-19:00
        {"period_id": 8, "start": 660, "end": 780, "state": "顺畅", "mean_speed": 55.3, "std": 0.10},   # 19:00-21:00
        {"period_id": 9, "start": 780, "end": 1440, "state": "顺畅", "mean_speed": 55.3, "std": 0.10},  # 21:00以后做延拓
    ]


def build_cost_params() -> dict:
    return {
        "wait_cost_per_hour": WAIT_COST_PER_HOUR,
        "late_cost_per_hour": LATE_COST_PER_HOUR,
        "fuel_price": FUEL_PRICE,
        "elec_price": ELEC_PRICE,
        "fuel_carbon_coef": FUEL_CARBON_COEF,
        "elec_carbon_coef": ELEC_CARBON_COEF,
        "carbon_price": CARBON_PRICE,
        "service_time": SERVICE_MIN,
    }


# =========================
# 预处理：订单清洗、缺失值修复
# =========================
def impute_orders(orders: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    df = orders.copy()
    df["重量_原始"] = df["重量"]
    df["体积_原始"] = df["体积"]
    df["is_imputed"] = False
    df["impute_note"] = ""

    complete = df[df["重量"].notna() & df["体积"].notna() & (df["重量"] > 0) & (df["体积"] > 0)].copy()
    complete["体积重量比"] = complete["体积"] / complete["重量"]
    global_ratio = float(complete["体积重量比"].median())

    customer_stats = (
        complete.groupby("目标客户编号")["体积重量比"]
        .agg(["median", "count"]).reset_index()
        .rename(columns={"median": "客户比值中位数", "count": "客户完整样本数"})
    )
    df = df.merge(customer_stats, on="目标客户编号", how="left")

    for idx, row in df.iterrows():
        ratio = row["客户比值中位数"] if pd.notna(row["客户比值中位数"]) and row["客户完整样本数"] >= 2 else global_ratio
        level = "客户内中位数" if pd.notna(row["客户比值中位数"]) and row["客户完整样本数"] >= 2 else "全局中位数"

        if pd.isna(row["重量"]) and pd.notna(row["体积"]):
            df.at[idx, "重量"] = row["体积"] / ratio
            df.at[idx, "is_imputed"] = True
            df.at[idx, "impute_note"] = f"体积反推重量({level})"
        elif pd.isna(row["体积"]) and pd.notna(row["重量"]):
            df.at[idx, "体积"] = row["重量"] * ratio
            df.at[idx, "is_imputed"] = True
            df.at[idx, "impute_note"] = f"重量反推体积({level})"

    df.drop(columns=["客户比值中位数", "客户完整样本数"], inplace=True)
    df["is_valid"] = (~df["目标客户编号"].isna()) & (df["重量"] > 0) & (df["体积"] > 0)
    return df, global_ratio


# =========================
# 超容量客户：订单级双容量拆分
# =========================
def split_customer_orders_bfd(
    customer_orders: pd.DataFrame,
    target_weight: float = SPLIT_GROUP_TARGET_WEIGHT,
    target_volume: float = SPLIT_GROUP_TARGET_VOLUME,
    hard_weight: float = MAX_WEIGHT,
    hard_volume: float = MAX_VOLUME,
) -> List[pd.DataFrame]:
    """
    对“已判定为超容量客户”的订单集合做双容量分组。

    建模口径：
    - 只有客户汇总需求超过 MAX_WEIGHT/MAX_VOLUME 时才进入本函数；
    - 本函数优先保留原始订单结构，不拆分单笔可由最大车型承载的订单；
    - 若单笔订单本身超过最大车型容量，则做必要的输入层切分；
    - 最终每个子需求包必须满足硬约束：重量<=MAX_WEIGHT，体积<=MAX_VOLUME。

    实现策略：
    - 排序键为 max(weight/target_weight, volume/target_volume)，优先处理最紧张订单；
    - Best Fit：在可放入的组中，选择放入后利用率最高的组；
    - 对“小订单组合”优先控制在 target_weight/target_volume 内，以保留大车资源；
      对单笔已经超过 target 但未超过 MAX 的订单，允许其作为“大包”存在并按硬约束合并。
    """
    temp = customer_orders.copy()

    expanded_rows = []
    for _, row in temp.iterrows():
        w = float(row["重量"])
        v = float(row["体积"])
        # 只有单笔订单本身超过最大车型容量时才切分单笔订单。
        parts = int(max(math.ceil(w / hard_weight), math.ceil(v / hard_volume)))
        if parts <= 1:
            r = row.to_dict()
            r["_piece_of_order"] = 0
            expanded_rows.append(r)
        else:
            for k in range(parts):
                r = row.to_dict()
                if k < parts - 1:
                    pw = w / parts
                    pv = v / parts
                else:
                    pw = w - (parts - 1) * (w / parts)
                    pv = v - (parts - 1) * (v / parts)
                r["重量"] = float(pw)
                r["体积"] = float(pv)
                r["_piece_of_order"] = k + 1
                expanded_rows.append(r)

    temp = pd.DataFrame(expanded_rows)
    temp["_key"] = temp.apply(lambda r: max(r["重量"] / target_weight, r["体积"] / target_volume), axis=1)
    temp = temp.sort_values("_key", ascending=False).drop(columns=["_key"])

    bins = []
    for _, row in temp.iterrows():
        w = float(row["重量"])
        v = float(row["体积"])
        row_big = (w > target_weight + 1e-9) or (v > target_volume + 1e-9)
        best_idx = None
        best_score = None
        for bi, b in enumerate(bins):
            nw = b["w"] + w
            nv = b["v"] + v
            bin_big = (b["w"] > target_weight + 1e-9) or (b["v"] > target_volume + 1e-9)
            # 小订单组合优先不超过 target；若该组或该订单本身已经是“大包”，则使用最大车型硬上限。
            cap_w = hard_weight if (bin_big or row_big) else target_weight
            cap_v = hard_volume if (bin_big or row_big) else target_volume
            if nw <= cap_w + 1e-9 and nv <= cap_v + 1e-9:
                score = max(nw / cap_w, nv / cap_v)
                if (best_score is None) or (score > best_score):
                    best_score = score
                    best_idx = bi
        if best_idx is None:
            # 新开组。单组可能超过 target，但绝不能超过最大车型硬约束。
            if w > hard_weight + 1e-9 or v > hard_volume + 1e-9:
                raise ValueError("存在切分后仍超过最大车型容量的订单片段，请检查订单拆分逻辑。")
            bins.append({"w": w, "v": v, "rows": [row.to_dict()]})
        else:
            bins[best_idx]["w"] += w
            bins[best_idx]["v"] += v
            bins[best_idx]["rows"].append(row.to_dict())

    return [pd.DataFrame(b["rows"]) for b in bins]



def split_customer_orders_bfd_strict(
    customer_orders: pd.DataFrame,
    cap_weight: float,
    cap_volume: float,
) -> List[pd.DataFrame]:
    """
    严格双容量装箱函数。

    与旧版 split_customer_orders_bfd() 的区别：
    - cap_weight / cap_volume 是真正硬上限；
    - 不会因为某个订单或某个箱子超过 target，就自动切换到 3000kg / 15m³；
    - 因此可以稳定控制虚拟服务节点的车型兼容性。
    """
    expanded_rows = []

    for _, row in customer_orders.iterrows():
        w = float(row["重量"])
        v = float(row["体积"])
        parts = int(max(math.ceil(w / cap_weight), math.ceil(v / cap_volume)))

        if parts <= 1:
            r = row.to_dict()
            r["_piece_of_order"] = 0
            expanded_rows.append(r)
        else:
            # 说明：对超过当前策略容量的单笔订单进行等比例切片。
            # 这样可保证每个订单片段都可被当前策略容量承载。
            for k in range(parts):
                r = row.to_dict()
                r["重量"] = w / parts
                r["体积"] = v / parts
                r["_piece_of_order"] = k + 1
                expanded_rows.append(r)

    temp = pd.DataFrame(expanded_rows)
    if temp.empty:
        return []

    temp["_key"] = temp.apply(
        lambda r: max(float(r["重量"]) / cap_weight, float(r["体积"]) / cap_volume),
        axis=1
    )
    temp = temp.sort_values("_key", ascending=False).drop(columns=["_key"])

    bins = []
    for _, row in temp.iterrows():
        w = float(row["重量"])
        v = float(row["体积"])

        best_idx = None
        best_score = None
        for bi, b in enumerate(bins):
            nw = b["w"] + w
            nv = b["v"] + v
            if nw <= cap_weight + 1e-9 and nv <= cap_volume + 1e-9:
                score = max(nw / cap_weight, nv / cap_volume)
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = bi

        if best_idx is None:
            if w > cap_weight + 1e-9 or v > cap_volume + 1e-9:
                raise ValueError(
                    f"订单片段仍超过策略容量：w={w:.3f}, v={v:.3f}, "
                    f"cap_weight={cap_weight}, cap_volume={cap_volume}"
                )
            bins.append({"w": w, "v": v, "rows": [row.to_dict()]})
        else:
            bins[best_idx]["w"] += w
            bins[best_idx]["v"] += v
            bins[best_idx]["rows"].append(row.to_dict())

    return [pd.DataFrame(b["rows"]) for b in bins]


def merge_groups_for_cost_reduction(
    groups: List[pd.DataFrame],
    max_weight: float = FUEL3000_CAP_WEIGHT,
    max_volume: float = FUEL3000_CAP_VOLUME,
    big_group_limit: Optional[int] = None,
) -> List[pd.DataFrame]:
    """
    同一原客户内部的有限合并函数。

    用途：
    - 先用 safe 模式生成中型车兼容的小包；
    - 再尝试把同一客户下的两个小包合并成更大的包；
    - 合并后不超过 Fuel_3000 容量 3000kg / 13.5m³，避免生成只能依赖 EV_3000 的大体积节点。

    big_group_limit：
    - 可选参数，用于限制该客户内部生成的大包数量；
    - 当前默认 None，先让算法尽量合法合并；后续通过全局车型兼容性检查判断是否可采用。
    """
    bins = []
    for g in groups:
        if g is None or len(g) == 0:
            continue
        bins.append({
            "w": float(g["重量"].sum()),
            "v": float(g["体积"].sum()),
            "df": g.copy(),
        })

    def is_big(b):
        return (b["w"] > SAFE_CAP_WEIGHT + 1e-9) or (b["v"] > SAFE_CAP_VOLUME + 1e-9)

    changed = True
    while changed:
        changed = False
        best_pair = None
        best_score = None

        current_big_count = sum(1 for b in bins if is_big(b))

        for i in range(len(bins)):
            for j in range(i + 1, len(bins)):
                nw = bins[i]["w"] + bins[j]["w"]
                nv = bins[i]["v"] + bins[j]["v"]

                if nw <= max_weight + 1e-9 and nv <= max_volume + 1e-9:
                    new_big = (nw > SAFE_CAP_WEIGHT + 1e-9) or (nv > SAFE_CAP_VOLUME + 1e-9)
                    old_big = int(is_big(bins[i])) + int(is_big(bins[j]))
                    next_big_count = current_big_count - old_big + int(new_big)
                    if big_group_limit is not None and next_big_count > big_group_limit:
                        continue

                    # score 越大，合并后越接近容量上限，越能减少碎片化。
                    score = max(nw / max_weight, nv / max_volume)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_pair = (i, j)

        if best_pair is not None:
            i, j = best_pair
            new_df = pd.concat([bins[i]["df"], bins[j]["df"]], ignore_index=True)
            new_bin = {
                "w": bins[i]["w"] + bins[j]["w"],
                "v": bins[i]["v"] + bins[j]["v"],
                "df": new_df,
            }
            for idx in sorted([i, j], reverse=True):
                bins.pop(idx)
            bins.append(new_bin)
            changed = True

    return [b["df"] for b in bins]


def split_customer_orders_by_policy(customer_orders: pd.DataFrame) -> List[pd.DataFrame]:
    """
    根据 SPLIT_POLICY 选择超容量客户拆分策略。

    safe：严格按 1500kg / 10.8m³ 拆分，最稳。
    balanced：先 safe，再在同一客户内部有限合并到 3000kg / 13.5m³。
    compact：严格按 3000kg / 13.5m³ 拆分，仅建议做对比试验。
    """
    policy = str(SPLIT_POLICY).lower().strip()

    if policy == "safe":
        return split_customer_orders_bfd_strict(
            customer_orders,
            cap_weight=SAFE_CAP_WEIGHT,
            cap_volume=SAFE_CAP_VOLUME,
        )

    if policy == "balanced":
        safe_groups = split_customer_orders_bfd_strict(
            customer_orders,
            cap_weight=SAFE_CAP_WEIGHT,
            cap_volume=SAFE_CAP_VOLUME,
        )
        return merge_groups_for_cost_reduction(
            safe_groups,
            max_weight=FUEL3000_CAP_WEIGHT,
            max_volume=FUEL3000_CAP_VOLUME,
            # 为避免生成过多“必须由3000kg级车辆服务”的大包，
            # balanced 模式默认每个超容量客户最多合并出 1 个大包。
            # 这比 compact 激进压缩更稳，也比 safe 完全不合并更有降成本潜力。
            big_group_limit=1,
        )

    if policy == "compact":
        return split_customer_orders_bfd_strict(
            customer_orders,
            cap_weight=FUEL3000_CAP_WEIGHT,
            cap_volume=FUEL3000_CAP_VOLUME,
        )

    raise ValueError(f"未知 SPLIT_POLICY={SPLIT_POLICY}，可选 safe / balanced / compact")


def add_vehicle_compatibility_columns(service_nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    为服务节点表增加车型兼容性检查字段。
    这部分是预处理层面的可行性诊断，不改变订单重量和体积。
    """
    df = service_nodes_df.copy()

    def feasible_types(row):
        out = []
        w = float(row["weight"])
        v = float(row["volume"])

        if w <= 3000.0 + 1e-9 and v <= 13.5 + 1e-9:
            out.append("Fuel_3000")
        if w <= 1500.0 + 1e-9 and v <= 10.8 + 1e-9:
            out.append("Fuel_1500")
        if w <= 1250.0 + 1e-9 and v <= 6.5 + 1e-9:
            out.append("Fuel_1250")
        if w <= 3000.0 + 1e-9 and v <= 15.0 + 1e-9:
            out.append("EV_3000")
        if w <= 1250.0 + 1e-9 and v <= 8.5 + 1e-9:
            out.append("EV_1250")

        return ",".join(out)

    df["feasible_vehicle_types"] = df.apply(feasible_types, axis=1)
    df["feasible_type_count"] = df["feasible_vehicle_types"].apply(
        lambda x: 0 if str(x).strip() == "" else len(str(x).split(","))
    )
    df["need_3000_vehicle"] = (
        (df["weight"] > SAFE_CAP_WEIGHT + 1e-9) |
        (df["volume"] > SAFE_CAP_VOLUME + 1e-9)
    )
    df["only_ev3000"] = (
        (df["weight"] <= EV3000_CAP_WEIGHT + 1e-9) &
        (df["volume"] > FUEL3000_CAP_VOLUME + 1e-9) &
        (df["volume"] <= EV3000_CAP_VOLUME + 1e-9)
    )
    return df


def service_node_compatibility_summary(service_nodes_df: pd.DataFrame) -> dict:
    """生成服务节点车型兼容性摘要。"""
    df = service_nodes_df
    combo_counts = df["feasible_vehicle_types"].value_counts(dropna=False).to_dict() if "feasible_vehicle_types" in df.columns else {}
    return {
        "split_policy": SPLIT_POLICY,
        "service_node_count": int(len(df)),
        "no_feasible_vehicle_node_count": int((df.get("feasible_type_count", pd.Series(dtype=int)) == 0).sum()) if len(df) else 0,
        "need_3000_vehicle_node_count": int(df.get("need_3000_vehicle", pd.Series(dtype=bool)).sum()) if len(df) else 0,
        "only_ev3000_node_count": int(df.get("only_ev3000", pd.Series(dtype=bool)).sum()) if len(df) else 0,
        "feasible_vehicle_type_combo_counts": combo_counts,
    }


def build_service_nodes(
    orders_clean: pd.DataFrame,
    coords: pd.DataFrame,
    tw: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    输出：
    1. service_nodes_df: 真正进入模型的服务节点（普通客户节点 + 虚拟节点）
    2. split_summary_df: 超容量客户拆分情况汇总
    """
    coord_customer = coords[coords["类型"] == "客户"].copy()
    coord_map = coord_customer.set_index("ID")[["X (km)", "Y (km)"]].to_dict("index")
    tw_map = tw.set_index("客户编号")[["earliest", "latest"]].to_dict("index")

    valid_orders = orders_clean[orders_clean["is_valid"]].copy()
    grouped = valid_orders.groupby("目标客户编号")

    node_rows = []
    split_rows = []

    for cust_id, g in grouped:
        total_w = float(g["重量"].sum())
        total_v = float(g["体积"].sum())
        xy = coord_map[int(cust_id)]
        twi = tw_map[int(cust_id)]
        # 按建模手口径：只有超过最大车型容量的客户才拆分；普通客户不做策略性预拆。
        over_cap = (total_w > SPLIT_TRIGGER_WEIGHT) or (total_v > SPLIT_TRIGGER_VOLUME)

        if not over_cap:
            node_rows.append({
                "node_id": str(int(cust_id)),
                "original_customer_id": int(cust_id),
                "sub_id": 0,
                "weight": total_w,
                "volume": total_v,
                "earliest": int(twi["earliest"]),
                "latest": int(twi["latest"]),
                "service_time": SERVICE_MIN,
                "x": float(xy["X (km)"]),
                "y": float(xy["Y (km)"]),
                "is_virtual": False,
                "order_count": int(len(g)),
            })
            split_rows.append({
                "customer_id": int(cust_id),
                "lower_bound": int(max(math.ceil(total_w / SPLIT_TRIGGER_WEIGHT), math.ceil(total_v / SPLIT_TRIGGER_VOLUME))),
                "final_groups": 1,
                "total_weight": total_w,
                "total_volume": total_v,
                "is_split": False,
            })
        else:
            lower_bound = int(max(math.ceil(total_w / SPLIT_TRIGGER_WEIGHT), math.ceil(total_v / SPLIT_TRIGGER_VOLUME)))
            groups = split_customer_orders_by_policy(g)
            final_groups = len(groups)
            split_rows.append({
                "customer_id": int(cust_id),
                "lower_bound": lower_bound,
                "final_groups": final_groups,
                "total_weight": total_w,
                "total_volume": total_v,
                "is_split": True,
            })
            for gid, gg in enumerate(groups, start=1):
                gw = float(gg["重量"].sum())
                gv = float(gg["体积"].sum())
                node_rows.append({
                    "node_id": f"{int(cust_id)}_{gid}",
                    "original_customer_id": int(cust_id),
                    "sub_id": gid,
                    "weight": gw,
                    "volume": gv,
                    "earliest": int(twi["earliest"]),
                    "latest": int(twi["latest"]),
                    "service_time": SERVICE_MIN,
                    "x": float(xy["X (km)"]),
                    "y": float(xy["Y (km)"]),
                    "is_virtual": True,
                    "order_count": int(len(gg)),
                })

    service_nodes_df = pd.DataFrame(node_rows)
    service_nodes_df = add_vehicle_compatibility_columns(service_nodes_df)
    split_summary_df = pd.DataFrame(split_rows)
    split_summary_df["split_policy"] = SPLIT_POLICY
    return service_nodes_df, split_summary_df


# =========================
# 数据读取与实例构造
# =========================
def load_instance(base_dir: Path) -> dict:
    order_path = find_existing_file(base_dir, ["订单信息.xlsx"])
    dist_path = find_existing_file(base_dir, ["距离矩阵.xlsx"])
    coord_path = find_existing_file(base_dir, ["客户坐标信息.xlsx", "客户坐标信息(2).xlsx"])
    tw_path = find_existing_file(base_dir, ["时间窗.xlsx"])

    orders = pd.read_excel(order_path)
    dist_raw = pd.read_excel(dist_path)
    coords = pd.read_excel(coord_path)
    tw = pd.read_excel(tw_path)

    orders.columns = [str(c).strip() for c in orders.columns]
    dist_raw.columns = [str(c).strip() for c in dist_raw.columns]
    coords.columns = [str(c).strip() for c in coords.columns]
    tw.columns = [str(c).strip() for c in tw.columns]

    # 基本类型
    orders["订单编号"] = pd.to_numeric(orders["订单编号"], errors="coerce").astype("Int64")
    orders["目标客户编号"] = pd.to_numeric(orders["目标客户编号"], errors="coerce").astype("Int64")
    orders["重量"] = pd.to_numeric(orders["重量"], errors="coerce")
    orders["体积"] = pd.to_numeric(orders["体积"], errors="coerce")

    coords["ID"] = pd.to_numeric(coords["ID"], errors="coerce").astype("Int64")
    coords["X (km)"] = pd.to_numeric(coords["X (km)"], errors="coerce")
    coords["Y (km)"] = pd.to_numeric(coords["Y (km)"], errors="coerce")

    tw["客户编号"] = pd.to_numeric(tw["客户编号"], errors="coerce").astype("Int64")
    tw["earliest"] = tw["开始时间"].apply(lambda x: time_to_rel_minutes(x, 8))
    tw["latest"] = tw["结束时间"].apply(lambda x: time_to_rel_minutes(x, 8))

    # 距离矩阵
    row_ids = pd.to_numeric(dist_raw["客户"], errors="coerce")
    col_ids = pd.to_numeric(pd.Index(dist_raw.columns[1:]), errors="coerce")
    dist_mat = dist_raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    dist_mat.index = row_ids.astype(int)
    dist_mat.columns = col_ids.astype(int)

    # 清洗与缺失修复
    orders_clean, global_ratio = impute_orders(orders)

    # 异常标签（仅标记，不直接删除）
    wt_upper = iqr_upper(orders_clean["重量"])
    vol_upper = iqr_upper(orders_clean["体积"])
    orders_clean["weight_outlier"] = orders_clean["重量"] > wt_upper
    orders_clean["volume_outlier"] = orders_clean["体积"] > vol_upper
    orders_clean["single_over_weight_cap"] = orders_clean["重量"] > MAX_WEIGHT
    orders_clean["single_over_volume_cap"] = orders_clean["体积"] > MAX_VOLUME

    # 服务节点（普通节点 + 虚拟节点）
    service_nodes_df, split_summary_df = build_service_nodes(orders_clean, coords, tw)

    # 节点映射表
    mapping_df = service_nodes_df[["node_id", "original_customer_id", "sub_id", "is_virtual"]].copy()

    # 配送中心坐标
    depot_row = coords[coords["类型"] == "配送中心"].iloc[0]
    depot = {"node_id": "0", "x": float(depot_row["X (km)"]), "y": float(depot_row["Y (km)"])}

    # 问题二新增：绿色区节点标记与穿区弧段标记
    green_nodes, cross_green_arcs, green_arc_df = build_green_zone_metadata(service_nodes_df, depot)
    service_nodes_df["in_green_zone"] = service_nodes_df["node_id"].astype(str).map(green_nodes).fillna(False)

    green_zone_summary = {
        "green_center_x": GREEN_CENTER_X,
        "green_center_y": GREEN_CENTER_Y,
        "green_radius_km": GREEN_RADIUS_KM,
        "green_ban_start_min": GREEN_BAN_START_MIN,
        "green_ban_end_min": GREEN_BAN_END_MIN,
        "green_ban_start_clock": rel_minutes_to_clock_str(GREEN_BAN_START_MIN),
        "green_ban_end_clock": rel_minutes_to_clock_str(GREEN_BAN_END_MIN),
        "green_service_node_count": int(service_nodes_df["in_green_zone"].sum()),
        "green_original_customer_count": int(service_nodes_df.loc[service_nodes_df["in_green_zone"], "original_customer_id"].nunique()),
        "cross_green_arc_count": int(green_arc_df["cross_green_zone"].sum()) if not green_arc_df.empty else 0,
        "policy_mode": GREEN_POLICY_MODE,
    }

    vehicles_list = build_vehicle_types()
    node_dict = build_node_dict(service_nodes_df)
    speed_periods_list = build_speed_periods()
    distance_numpy = dist_mat.to_numpy(dtype=float)
    distance_id_to_pos = {str(int(idx)): pos for pos, idx in enumerate(dist_mat.index)}

    instance = {
        "orders_clean": orders_clean,
        "service_nodes": service_nodes_df,
        "service_node_compatibility_summary": service_node_compatibility_summary(service_nodes_df),
        "green_nodes": green_nodes,
        "cross_green_arcs": cross_green_arcs,
        "green_arc_df": green_arc_df,
        "green_zone_summary": green_zone_summary,
        "split_summary": split_summary_df,
        "split_policy": SPLIT_POLICY,
        "mapping": mapping_df,
        "coordinates": coords,
        "time_windows": tw[["客户编号", "earliest", "latest"]].copy(),
        "distance_matrix": dist_mat,
        "distance_numpy": distance_numpy,
        "distance_id_to_pos": distance_id_to_pos,
        "vehicles": pd.DataFrame([asdict(v) for v in vehicles_list]),
        "vehicle_types_list": vehicles_list,
        "vehicle_map": vehicle_lookup(vehicles_list),
        "vehicle_stock": {v.vehicle_type: v.count for v in vehicles_list},
        "speed_periods": pd.DataFrame(speed_periods_list),
        "speed_periods_records": speed_periods_list,
        "cost_params": build_cost_params(),
        "depot": depot,
        "global_ratio": global_ratio,
        "node_dict": node_dict,
        "source_files": {
            "orders": order_path.name,
            "distance": dist_path.name,
            "coords": coord_path.name,
            "time_windows": tw_path.name,
        }
    }
    return instance


# =========================
# 统一访问接口：节点、距离、速度
# =========================
def build_node_dict(service_nodes_df: pd.DataFrame) -> Dict[str, Node]:
    nd = {}
    for _, r in service_nodes_df.iterrows():
        nd[str(r["node_id"])] = Node(
            node_id=str(r["node_id"]),
            original_customer_id=int(r["original_customer_id"]),
            sub_id=int(r["sub_id"]),
            weight=float(r["weight"]),
            volume=float(r["volume"]),
            earliest=int(r["earliest"]),
            latest=int(r["latest"]),
            service_time=int(r["service_time"]),
            x=float(r["x"]),
            y=float(r["y"]),
            is_virtual=bool(r["is_virtual"]),
        )
    return nd


def build_green_zone_metadata(service_nodes_df: pd.DataFrame, depot: dict) -> Tuple[Dict[str, bool], Dict[Tuple[str, str], bool], pd.DataFrame]:
    """
    生成问题二绿色配送区节点标记和穿区弧段标记。
    返回：
        green_nodes: node_id -> 是否位于绿色区内；
        cross_green_arcs: (from_id, to_id) -> 弧段是否穿越绿色区；
        arc_df: 弧段标记明细表，用于输出。
    """
    coord = {"0": (float(depot["x"]), float(depot["y"]))}

    for _, r in service_nodes_df.iterrows():
        coord[str(r["node_id"])] = (float(r["x"]), float(r["y"]))

    green_nodes = {
        nid: point_in_green_zone(x, y)
        for nid, (x, y) in coord.items()
        if nid != "0"
    }

    cross_green_arcs = {}
    arc_rows = []

    node_ids = list(coord.keys())
    for a in node_ids:
        for b in node_ids:
            if a == b:
                cross = False
            else:
                x1, y1 = coord[a]
                x2, y2 = coord[b]
                cross = segment_crosses_green_zone(x1, y1, x2, y2)

            cross_green_arcs[(str(a), str(b))] = bool(cross)

            if a != b:
                arc_rows.append({
                    "from_node": str(a),
                    "to_node": str(b),
                    "from_in_green_zone": bool(green_nodes.get(str(a), False)),
                    "to_in_green_zone": bool(green_nodes.get(str(b), False)),
                    "cross_green_zone": bool(cross),
                })

    arc_df = pd.DataFrame(arc_rows)
    return green_nodes, cross_green_arcs, arc_df


def get_original_customer_id(node_id: str, node_dict: Dict[str, Node]) -> int:
    if str(node_id) == "0":
        return 0
    return node_dict[str(node_id)].original_customer_id


def lookup_distance(node_a: str, node_b: str, node_dict: Dict[str, Node], distance_numpy: np.ndarray, distance_id_to_pos: Dict[str, int]) -> float:
    a = str(get_original_customer_id(str(node_a), node_dict))
    b = str(get_original_customer_id(str(node_b), node_dict))
    ia = distance_id_to_pos[a]
    ib = distance_id_to_pos[b]
    return float(distance_numpy[ia, ib])


def route_total_weight(route: dict, node_dict: Dict[str, Node]) -> float:
    return float(route.get("weight_sum", sum(node_dict[n].weight for n in route["stops"])))


def route_total_volume(route: dict, node_dict: Dict[str, Node]) -> float:
    return float(route.get("volume_sum", sum(node_dict[n].volume for n in route["stops"])))


def route_original_customer_set(route: dict, node_dict: Dict[str, Node]) -> set:
    if "original_customer_set" in route and route["original_customer_set"] is not None:
        return set(route["original_customer_set"])
    return {node_dict[n].original_customer_id for n in route["stops"]}


def rebuild_route_cache(route: dict, node_dict: Dict[str, Node]) -> dict:
    route["weight_sum"] = sum(node_dict[n].weight for n in route["stops"])
    route["volume_sum"] = sum(node_dict[n].volume for n in route["stops"])
    route["original_customer_set"] = {node_dict[n].original_customer_id for n in route["stops"]}
    route["cached_eval"] = None
    return route


def make_route(vehicle_type: str, stops: List[str], node_dict: Dict[str, Node]) -> dict:
    return rebuild_route_cache({"vehicle_type": vehicle_type, "stops": list(stops)}, node_dict)


def insert_node_to_route(route: dict, pos: int, node_id: str, node_dict: Dict[str, Node]) -> dict:
    """插入节点并同步更新路线缓存，避免提速缓存与 stops 不一致。"""
    old_weight = route.get("weight_sum")
    old_volume = route.get("volume_sum")
    route["stops"].insert(pos, node_id)
    if old_weight is None or old_volume is None:
        rebuild_route_cache(route, node_dict)
    else:
        route["weight_sum"] = float(old_weight) + float(node_dict[node_id].weight)
        route["volume_sum"] = float(old_volume) + float(node_dict[node_id].volume)
        route["original_customer_set"] = set(route.get("original_customer_set", set()))
        route["original_customer_set"].add(node_dict[node_id].original_customer_id)
        route["cached_eval"] = None
    return route


def remove_node_from_route(route: dict, node_id: str, node_dict: Dict[str, Node]) -> dict:
    route["stops"].remove(node_id)
    rebuild_route_cache(route, node_dict)
    return route


def clone_route(route: dict) -> dict:
    """
    克隆路线时只复制路线结构和轻量缓存，不复制 cached_eval。
    原因：局部搜索会改变 stops，如果继续复用旧 cached_eval，
    就可能出现“路线已经变了，但成本仍然是旧路线成本”的问题。
    """
    new_route = {
        "vehicle_type": route["vehicle_type"],
        "stops": list(route["stops"]),
    }
    if "weight_sum" in route:
        new_route["weight_sum"] = float(route["weight_sum"])
    if "volume_sum" in route:
        new_route["volume_sum"] = float(route["volume_sum"])
    if "original_customer_set" in route:
        new_route["original_customer_set"] = set(route["original_customer_set"])

    # 关键：克隆后不继承旧成本缓存
    new_route["cached_eval"] = None
    return new_route


def clone_solution(solution: List[dict]) -> List[dict]:
    return [clone_route(r) for r in solution]


def remove_empty_routes(solution: List[dict]) -> List[dict]:
    return [r for r in solution if len(r["stops"]) > 0]

def refresh_solution_cache(solution: List[dict], instance: dict) -> List[dict]:
    """
    强制重建每条路线的缓存。
    只要经过 2-opt、relocate、swap 等直接修改 stops 的操作，
    都应该调用这个函数，避免 weight_sum / volume_sum / original_customer_set / cached_eval 失效。
    """
    node_dict = instance["node_dict"]
    for r in solution:
        rebuild_route_cache(r, node_dict)
    return solution

def speed_at_time(minute: float, speed_periods: List[dict], sampled_speeds: Optional[Dict[int, float]] = None) -> Tuple[float, float]:
    for p in speed_periods:
        if p["start"] <= minute < p["end"]:
            spd = sampled_speeds.get(p["period_id"], p["mean_speed"]) if sampled_speeds else p["mean_speed"]
            return max(1.0, float(spd)), float(p["end"])
    # 理论上不会到这里，兜底用最后一个时段
    p = speed_periods[-1]
    spd = sampled_speeds.get(p["period_id"], p["mean_speed"]) if sampled_speeds else p["mean_speed"]
    return max(1.0, float(spd)), float(p["end"])


def travel_minutes_piecewise(distance_km: float, depart_minute: float, speed_periods: List[dict], sampled_speeds: Optional[Dict[int, float]] = None) -> Tuple[float, float]:
    """
    返回：
    - 行驶时间（分钟）
    - 加权平均速度（km/h）
    """
    remain = float(distance_km)
    t = float(depart_minute)
    total_minutes = 0.0
    weighted_speed_dist = 0.0

    while remain > 1e-9:
        spd, period_end = speed_at_time(t, speed_periods, sampled_speeds)
        available_minutes = max(1e-9, period_end - t)
        can_go = spd * (available_minutes / 60.0)
        if remain <= can_go + 1e-9:
            need_minutes = remain / spd * 60.0
            total_minutes += need_minutes
            weighted_speed_dist += remain
            t += need_minutes
            remain = 0.0
        else:
            total_minutes += available_minutes
            weighted_speed_dist += can_go
            remain -= can_go
            t = period_end

    avg_speed = distance_km / (total_minutes / 60.0) if total_minutes > 0 else speed_periods[0]["mean_speed"]
    return total_minutes, avg_speed


# =========================
# 成本与路径评估
# =========================
def route_capacity_ok(route: dict, node_dict: Dict[str, Node], vehicle: VehicleType) -> bool:
    w = route_total_weight(route, node_dict)
    v = route_total_volume(route, node_dict)
    return (w <= vehicle.capacity_weight + 1e-9) and (v <= vehicle.capacity_volume + 1e-9)


def no_duplicate_original_customer(route: dict, node_dict: Dict[str, Node]) -> bool:
    origs = route_original_customer_set(route, node_dict)
    return len(origs) == len(route["stops"])


def simulate_route(route: dict, node_dict: Dict[str, Node], distance_numpy: np.ndarray,
                   distance_id_to_pos: Dict[str, int], speed_periods: List[dict], vehicle: VehicleType, cost_params: dict,
                   sampled_speeds: Optional[Dict[int, float]] = None, fixed_departure: Optional[int] = None,
                   green_policy: Optional[dict] = None) -> RouteEval:
    if not route["stops"]:
        return RouteEval(True, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, [], [], [], [], "空路径")

    # 基础可行性：双容量 + 同一路径同原客户不重复
    if not route_capacity_ok(route, node_dict, vehicle):
        return RouteEval(False, 1e18, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], [], [], [], "容量超限")
    if not no_duplicate_original_customer(route, node_dict):
        return RouteEval(False, 1e18, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], [], [], [], "同一路径中同一原客户重复出现")
    # 问题二新增：绿色区政策信息
    green_policy = green_policy or {}
    green_nodes = green_policy.get("green_nodes", {})
    cross_green_arcs = green_policy.get("cross_green_arcs", {})
    policy_mode = green_policy.get("policy_mode", GREEN_POLICY_MODE)
    ban_start = float(green_policy.get("ban_start", GREEN_BAN_START_MIN))
    ban_end = float(green_policy.get("ban_end", GREEN_BAN_END_MIN))
    is_fuel = vehicle.fuel_type == "fuel"

    def policy_zone_flags(prev_node: str, next_node: str) -> Tuple[bool, bool]:
        """
        返回：
            to_green: 下一服务节点是否位于绿色区；
            cross_green: 当前弧段是否穿越绿色区。
        """
        to_green = bool(green_nodes.get(str(next_node), False)) if str(next_node) != "0" else False
        cross_green = bool(cross_green_arcs.get((str(prev_node), str(next_node)), False))
        return to_green, cross_green

    def interval_overlaps_ban_window(depart_t: float, arrival_t: float) -> bool:
        """
        判断弧段行驶时间 [depart_t, arrival_t] 是否与禁行窗口 [ban_start, ban_end) 有交集。
        这样 15:55 出发、16:05 到达也会被判为涉及限行时段。
        """
        return not (float(arrival_t) <= ban_start or float(depart_t) >= ban_end)

    def policy_action_forbidden(prev_node: str, next_node: str, depart_t: float, arrival_t: float) -> Tuple[bool, bool, bool]:
        """
        动态禁行判定：
        燃油车 + 弧段涉及绿色区节点/穿区 + 弧段时间与限行窗口有交集 => 禁行。
        16:00 后出发且整个弧段不与限行窗口重叠，则允许通行。
        """
        to_green, cross_green = policy_zone_flags(prev_node, next_node)
        forbidden = (
            is_fuel
            and (to_green or cross_green)
            and interval_overlaps_ban_window(depart_t, arrival_t)
        )
        return forbidden, to_green, cross_green

    # 粗略估计一组候选发车时刻，减少大额等待成本
    if fixed_departure is None:
        approx_speed = 35.4
        candidates = {0}
        cum = 0.0
        prev = "0"
        # 判断该路线是否涉及绿色区节点或穿区弧段。
        route_policy_risk = False
        prev_for_policy = "0"

        for idx, nid in enumerate(route["stops"]):
            node = node_dict[nid]
            d = lookup_distance(prev, nid, node_dict, distance_numpy, distance_id_to_pos)
            cum += d / approx_speed * 60.0
            if idx > 0:
                cum += node_dict[route["stops"][idx-1]].service_time
            start_guess = max(0.0, node.earliest - cum)
            candidates.add(int(round(start_guess / 5.0) * 5))

            to_green_tmp, cross_green_tmp = policy_zone_flags(prev_for_policy, nid)
            if to_green_tmp or cross_green_tmp:
                route_policy_risk = True
            prev_for_policy = nid
            prev = nid

        # 返回配送中心弧段也可能穿区。
        _, cross_green_tmp = policy_zone_flags(prev_for_policy, "0")
        if cross_green_tmp:
            route_policy_risk = True

        # 政策感知发车时间：若燃油车路线涉及绿色区/穿区，则主动尝试 16:00 后发车。
        # 这比“15:55 出发、16:05 到达后放行”更严格，也符合小组讨论的 ban 规则。
        if is_fuel and route_policy_risk:
            candidates.update([int(ban_end), int(ban_end + 10), int(ban_end + 20), int(ban_end + 30), int(ban_end + 60)])

        candidate_departures = sorted(c for c in candidates if c >= 0)
    else:
        candidate_departures = [fixed_departure]

    best_eval = None
    total_route_weight = route_total_weight(route, node_dict)
    for departure in candidate_departures:
        remain_weight = total_route_weight
        total_distance = 0.0
        total_travel_minutes = 0.0
        wait_cost = 0.0
        late_cost = 0.0
        energy_cost = 0.0
        carbon_cost = 0.0
        policy_wait_cost = 0.0
        policy_wait_min_total = 0.0
        policy_affected_arc_count = 0
        green_service_count = 0
        cross_green_arc_count = 0
        arrivals, waits, lates, details = [], [], [], []

        current_time = float(departure)
        prev = "0"
        feasible = True
        message = ""

        for nid in route["stops"]:
            node = node_dict[nid]
            d = lookup_distance(prev, nid, node_dict, distance_numpy, distance_id_to_pos)

            # 先按当前出发时刻估计该弧段行驶时间，再判断整条弧段是否与限行窗口重叠。
            policy_wait_min = 0.0
            travel_min, avg_speed = travel_minutes_piecewise(d, current_time, speed_periods, sampled_speeds)
            arc_depart_time = current_time
            arc_arrival_time = current_time + travel_min

            restricted, to_green, cross_green = policy_action_forbidden(prev, nid, arc_depart_time, arc_arrival_time)

            if cross_green:
                cross_green_arc_count += 1
            if to_green:
                green_service_count += 1

            if restricted:
                if policy_mode == "strict":
                    feasible = False
                    message = "燃油车在限行时段访问绿色区节点或穿越绿色区弧段"
                    break

                # 保留 wait 作为对照模式：等待到 16:00 后重新计算该弧段行驶时间。
                policy_wait_min = max(0.0, ban_end - current_time)
                if policy_wait_min > 1e-9:
                    current_time += policy_wait_min
                    policy_wait_min_total += policy_wait_min
                    policy_wait_cost += policy_wait_min / 60.0 * cost_params["wait_cost_per_hour"]
                    wait_cost += policy_wait_min / 60.0 * cost_params["wait_cost_per_hour"]
                    policy_affected_arc_count += 1
                    travel_min, avg_speed = travel_minutes_piecewise(d, current_time, speed_periods, sampled_speeds)

            load_ratio = 0.0 if vehicle.capacity_weight <= 0 else max(0.0, min(1.0, remain_weight / vehicle.capacity_weight))
            if vehicle.fuel_type == "fuel":
                base = fuel_consumption_per_100km(avg_speed)
                actual_cons = base * (1.0 + vehicle.load_alpha * load_ratio)
                litres = d / 100.0 * actual_cons
                arc_energy_cost = litres * cost_params["fuel_price"]
                emission = litres * cost_params["fuel_carbon_coef"]
            else:
                base = elec_consumption_per_100km(avg_speed)
                actual_cons = base * (1.0 + vehicle.load_alpha * load_ratio)
                kwh = d / 100.0 * actual_cons
                arc_energy_cost = kwh * cost_params["elec_price"]
                emission = kwh * cost_params["elec_carbon_coef"]

            arc_carbon_cost = emission * cost_params["carbon_price"]

            current_time += travel_min
            arrival = current_time
            wait = max(0.0, node.earliest - arrival)
            late = max(0.0, arrival - node.latest)
            start_service = arrival + wait
            current_time = start_service + node.service_time

            arrivals.append(arrival)
            waits.append(wait)
            lates.append(late)

            wait_cost += wait / 60.0 * cost_params["wait_cost_per_hour"]
            late_cost += late / 60.0 * cost_params["late_cost_per_hour"]
            energy_cost += arc_energy_cost
            carbon_cost += arc_carbon_cost
            total_distance += d
            total_travel_minutes += travel_min

            details.append({
                "from": prev,
                "to": nid,
                "distance_km": d,
                "depart_time": current_time - node.service_time - wait - travel_min,
                "arrival_time": arrival,
                "wait_min": wait,
                "late_min": late,
                "service_start": start_service,
                "service_end": current_time,
                "avg_speed_kmh": avg_speed,
                "arc_energy_cost": arc_energy_cost,
                "arc_carbon_cost": arc_carbon_cost,
                "remaining_weight_before_arc": remain_weight,
                "to_in_green_zone": bool(to_green),
                "arc_cross_green_zone": bool(cross_green),
                "policy_restricted_action": bool(restricted),
                "policy_wait_min": policy_wait_min,
            })

            remain_weight -= node.weight
            prev = nid

        d = lookup_distance(prev, "0", node_dict, distance_numpy, distance_id_to_pos)

        return_policy_wait_min = 0.0
        travel_min_probe, _ = travel_minutes_piecewise(d, current_time, speed_periods, sampled_speeds)
        restricted, to_green, cross_green = policy_action_forbidden(prev, "0", current_time, current_time + travel_min_probe)

        if cross_green:
            cross_green_arc_count += 1

        if restricted:
            if policy_mode == "strict":
                feasible = False
                message = "燃油车在限行时段返回时穿越绿色区弧段"
            else:
                return_policy_wait_min = max(0.0, ban_end - current_time)
                if return_policy_wait_min > 1e-9:
                    current_time += return_policy_wait_min
                    policy_wait_min_total += return_policy_wait_min
                    policy_wait_cost += return_policy_wait_min / 60.0 * cost_params["wait_cost_per_hour"]
                    wait_cost += return_policy_wait_min / 60.0 * cost_params["wait_cost_per_hour"]
                    policy_affected_arc_count += 1

        if not feasible:
            total_cost = 1e18
            result = RouteEval(
                feasible=False, total_cost=1e18,
                fixed_cost=0.0, wait_cost=0.0, late_cost=0.0,
                energy_cost=0.0, carbon_cost=0.0,
                total_distance=0.0, total_travel_minutes=0.0,
                departure_time=int(departure), return_time=current_time,
                arrivals=arrivals, waits=waits, lates=lates, details=details,
                message=message,
                policy_wait_min=policy_wait_min_total,
                policy_wait_cost=policy_wait_cost,
                policy_affected_arc_count=policy_affected_arc_count,
                green_service_count=green_service_count,
                cross_green_arc_count=cross_green_arc_count,
            )
            if best_eval is None or result.total_cost < best_eval.total_cost:
                best_eval = result
            continue

        travel_min, avg_speed = travel_minutes_piecewise(d, current_time, speed_periods, sampled_speeds)
        if vehicle.fuel_type == "fuel":
            base = fuel_consumption_per_100km(avg_speed)
            actual_cons = base
            litres = d / 100.0 * actual_cons
            arc_energy_cost = litres * cost_params["fuel_price"]
            emission = litres * cost_params["fuel_carbon_coef"]
        else:
            base = elec_consumption_per_100km(avg_speed)
            actual_cons = base
            kwh = d / 100.0 * actual_cons
            arc_energy_cost = kwh * cost_params["elec_price"]
            emission = kwh * cost_params["elec_carbon_coef"]
        arc_carbon_cost = emission * cost_params["carbon_price"]
        total_distance += d
        total_travel_minutes += travel_min
        energy_cost += arc_energy_cost
        carbon_cost += arc_carbon_cost
        return_time = current_time + travel_min

        fixed_cost = vehicle.fixed_cost
        total_cost = fixed_cost + wait_cost + late_cost + energy_cost + carbon_cost

        result = RouteEval(
            feasible=feasible,
            total_cost=total_cost,
            fixed_cost=fixed_cost,
            wait_cost=wait_cost,
            late_cost=late_cost,
            energy_cost=energy_cost,
            carbon_cost=carbon_cost,
            total_distance=total_distance,
            total_travel_minutes=total_travel_minutes,
            departure_time=int(departure),
            return_time=return_time,
            arrivals=arrivals,
            waits=waits,
            lates=lates,
            details=details,
            message=message,
            policy_wait_min=policy_wait_min_total,
            policy_wait_cost=policy_wait_cost,
            policy_affected_arc_count=policy_affected_arc_count,
            green_service_count=green_service_count,
            cross_green_arc_count=cross_green_arc_count,
        )
        if best_eval is None or result.total_cost < best_eval.total_cost:
            best_eval = result

    return best_eval


def evaluate_route(route: dict, instance: dict, sampled_speeds: Optional[Dict[int, float]] = None, fixed_departure: Optional[int] = None) -> RouteEval:
    if sampled_speeds is None and route.get("cached_eval") is not None and fixed_departure is None:
        return route["cached_eval"]
    node_dict = instance["node_dict"]
    distance_numpy = instance["distance_numpy"]
    distance_id_to_pos = instance["distance_id_to_pos"]
    speed_periods = instance["speed_periods_records"]
    cost_params = instance["cost_params"]
    vehicle = instance["vehicle_map"][route["vehicle_type"]]
    green_policy = {
        "green_nodes": instance.get("green_nodes", {}),
        "cross_green_arcs": instance.get("cross_green_arcs", {}),
        "policy_mode": instance.get("green_zone_summary", {}).get("policy_mode", GREEN_POLICY_MODE),
        "ban_start": GREEN_BAN_START_MIN,
        "ban_end": GREEN_BAN_END_MIN,
    }
    ev = simulate_route(route, node_dict, distance_numpy, distance_id_to_pos, speed_periods, vehicle, cost_params, sampled_speeds, fixed_departure, green_policy)
    if sampled_speeds is None and fixed_departure is None:
        route["cached_eval"] = ev
    return ev


def solution_vehicle_counts(solution: List[dict]) -> Dict[str, int]:
    counts = {}
    for r in solution:
        vt = r["vehicle_type"]
        counts[vt] = counts.get(vt, 0) + 1
    return counts


def vehicle_lookup(vehicles: List[VehicleType]) -> Dict[str, VehicleType]:
    return {v.vehicle_type: v for v in vehicles}


def solution_cost(solution: List[dict], instance: dict, sampled_speeds: Optional[Dict[int, float]] = None) -> Tuple[float, List[RouteEval], bool]:
    used = solution_vehicle_counts(solution)
    stock = instance["vehicle_stock"]
    for k, c in used.items():
        if c > stock.get(k, 0):
            return 1e18, [], False

    evals = []
    total = 0.0
    feasible = True
    for r in solution:
        ev = evaluate_route(r, instance, sampled_speeds=sampled_speeds)
        evals.append(ev)
        total += ev.total_cost
        feasible = feasible and ev.feasible
    return total, evals, feasible


# =========================
# 初始解：贪心插入
# =========================
# =========================
# 初始解：贪心插入
# =========================
def feasible_vehicle_types_for_node(node: Node, vehicles: List[VehicleType]) -> List[VehicleType]:
    out = []
    for v in vehicles:
        if node.weight <= v.capacity_weight + 1e-9 and node.volume <= v.capacity_volume + 1e-9:
            out.append(v)
    return out


def best_new_route_for_node(node_id: str, current_solution: List[dict], instance: dict) -> Optional[Tuple[dict, float]]:
    node_dict = instance["node_dict"]
    node = node_dict[node_id]
    vehicles = build_vehicle_types()
    used = solution_vehicle_counts(current_solution)
    green_nodes = instance.get("green_nodes", {})
    vehicle_candidates = feasible_vehicle_types_for_node(node, vehicles)
    if bool(green_nodes.get(str(node_id), False)):
        vehicle_candidates = sorted(vehicle_candidates, key=lambda vv: (0 if vv.fuel_type == "electric" else 1, vv.capacity_weight))
    candidates = []
    for v in vehicle_candidates:
        if used.get(v.vehicle_type, 0) >= v.count:
            continue
        route = {"vehicle_type": v.vehicle_type, "stops": [node_id]}
        cost, _, feasible = solution_cost(current_solution + [route], instance)
        if feasible:
            candidates.append((route, cost))
    if not candidates:
        return None
    base_cost, _, _ = solution_cost(current_solution, instance)
    route, tot = min(candidates, key=lambda x: x[1])
    return route, (tot - base_cost)


def construct_initial_solution(instance: dict) -> List[dict]:
    """
    速度优先的初始解构造：
    - 先按容量占用和时间窗紧迫度排序；
    - 大节点优先开路线，小节点尽量追加到已有路线尾部；
    - 初始解阶段不过度追求最优，只追求“尽快得到可行解”。
    """
    nodes_df = instance["service_nodes"].copy()
    node_dict = instance["node_dict"]
    distance_numpy = instance["distance_numpy"]
    distance_id_to_pos = instance["distance_id_to_pos"]
    nodes_df["dist0"] = nodes_df["original_customer_id"].apply(lambda x: float(distance_numpy[distance_id_to_pos["0"], distance_id_to_pos[str(int(x))]]))
    nodes_df["cap_key"] = nodes_df.apply(lambda r: max(r["weight"] / MAX_WEIGHT, r["volume"] / MAX_VOLUME), axis=1)
    # 问题二新增：政策风险高的节点优先安排。
    # 绿色区节点若迟迟未被新能源车或合适路线吸收，后续容易造成燃油车禁行冲突。
    if "in_green_zone" in nodes_df.columns:
        nodes_df["policy_risk"] = nodes_df["in_green_zone"].astype(int)
    else:
        nodes_df["policy_risk"] = 0
    nodes_df = nodes_df.sort_values(["policy_risk", "cap_key", "latest", "earliest", "dist0"], ascending=[False, False, True, True, False])

    solution: List[dict] = []
    vehicles = instance["vehicle_types_list"]
    stock = {v.vehicle_type: v.count for v in vehicles}
    veh_map = instance["vehicle_map"]

    for node_id in nodes_df["node_id"].astype(str).tolist():
        node = node_dict[node_id]
        placed = False
        current_total, current_evals, _ = solution_cost(solution, instance) if solution else (0.0, [], True)

        candidate_routes = []
        for ridx, route in enumerate(solution):
            vt = veh_map[route["vehicle_type"]]
            if route_total_weight(route, node_dict) + node.weight > vt.capacity_weight + 1e-9 or route_total_volume(route, node_dict) + node.volume > vt.capacity_volume + 1e-9:
                continue
            if node.original_customer_id in route_original_customer_set(route, node_dict):
                continue
            last = route["stops"][-1] if route["stops"] else "0"
            approx_delta = (
                lookup_distance(last, node_id, node_dict, distance_numpy, distance_id_to_pos)
                + lookup_distance(node_id, "0", node_dict, distance_numpy, distance_id_to_pos)
                - lookup_distance(last, "0", node_dict, distance_numpy, distance_id_to_pos)
            )
            candidate_routes.append((approx_delta, ridx))

        candidate_routes.sort(key=lambda x: x[0])
        for _, ridx in candidate_routes[:6]:
            old_ev = current_evals[ridx]
            cand_route = clone_route(solution[ridx])
            insert_node_to_route(cand_route, len(cand_route["stops"]), node_id, node_dict)
            new_ev = evaluate_route(cand_route, instance)
            if new_ev.feasible:
                solution[ridx] = cand_route
                placed = True
                break

        if placed:
            continue

        feasible_v = [v for v in vehicles if node.weight <= v.capacity_weight + 1e-9 and node.volume <= v.capacity_volume + 1e-9 and stock[v.vehicle_type] > 0]
        if not feasible_v:
            raise RuntimeError(f"节点 {node_id} 无可用车型，请检查拆分策略或车队配置。")

        # 建模口径不变：只是初始解构造策略调整。
        # 固定启动成本相同，因此新开路线时优先选择“能装下该节点的最小车型”，
        # 以保留大容量车辆给真正的大需求节点，避免初始解阶段过早耗尽大车。
        chosen = min(
            feasible_v,
            key=lambda v: (
                v.capacity_weight / MAX_WEIGHT + v.capacity_volume / MAX_VOLUME,
                v.capacity_weight,
                v.capacity_volume,
                -stock[v.vehicle_type],
            ),
        )
        solution.append(make_route(chosen.vehicle_type, [node_id], node_dict))
        stock[chosen.vehicle_type] -= 1

    return solution


# =========================
# ALNS 主框架
# =========================
def flatten_nodes(solution: List[dict]) -> List[str]:
    out = []
    for r in solution:
        out.extend(r["stops"])
    return out

def expected_service_node_set(instance: dict) -> set:
    return set(instance["service_nodes"]["node_id"].astype(str).tolist())


def solution_node_set(solution: List[dict]) -> set:
    return set(flatten_nodes(solution))


def solution_covers_all_nodes(solution: List[dict], instance: dict) -> bool:
    expected = expected_service_node_set(instance)
    actual_list = flatten_nodes(solution)
    actual_set = set(actual_list)

    # 1. 是否有重复服务节点
    if len(actual_list) != len(actual_set):
        return False

    # 2. 是否正好覆盖全部服务节点
    return actual_set == expected


def coverage_report(solution: List[dict], instance: dict) -> dict:
    """
    输出服务节点覆盖情况和重量/体积核验。
    这是本版最关键的结果可信性检查：最终路线必须完整覆盖全部服务节点。
    """
    expected = expected_service_node_set(instance)
    actual_list = flatten_nodes(solution)
    actual_set = set(actual_list)
    node_dict = instance["node_dict"]
    vehicle_map = instance["vehicle_map"]

    expected_weight = sum(node_dict[n].weight for n in expected)
    actual_weight = sum(node_dict[n].weight for n in actual_list if n in node_dict)
    expected_volume = sum(node_dict[n].volume for n in expected)
    actual_volume = sum(node_dict[n].volume for n in actual_list if n in node_dict)

    used_capacity_weight = sum(vehicle_map[r["vehicle_type"]].capacity_weight for r in solution)
    used_capacity_volume = sum(vehicle_map[r["vehicle_type"]].capacity_volume for r in solution)

    return {
        "expected_node_count": len(expected),
        "actual_node_count": len(actual_list),
        "unique_actual_node_count": len(actual_set),
        "missing_node_count": len(expected - actual_set),
        "extra_node_count": len(actual_set - expected),
        "has_duplicate_node": len(actual_list) != len(actual_set),
        "missing_nodes": sorted(list(expected - actual_set)),
        "extra_nodes": sorted(list(actual_set - expected)),
        "expected_weight": float(expected_weight),
        "actual_weight": float(actual_weight),
        "expected_volume": float(expected_volume),
        "actual_volume": float(actual_volume),
        "used_vehicle_count": len(solution),
        "used_capacity_weight": float(used_capacity_weight),
        "used_capacity_volume": float(used_capacity_volume),
    }


def random_remove(solution: List[dict], q: int, instance: dict) -> Tuple[List[dict], List[str]]:
    """随机移除节点，并同步更新路线缓存。"""
    sol = clone_solution(solution)
    node_dict = instance["node_dict"]
    all_nodes = flatten_nodes(sol)
    if not all_nodes:
        return sol, []
    q = min(q, len(all_nodes))
    removed = random.sample(all_nodes, q)
    for n in removed:
        for r in sol:
            if n in r["stops"]:
                remove_node_from_route(r, n, node_dict)
                break
    return remove_empty_routes(sol), removed


def worst_remove(solution: List[dict], q: int, instance: dict) -> Tuple[List[dict], List[str]]:
    sol = clone_solution(solution)
    node_dict = instance["node_dict"]
    distance_numpy = instance["distance_numpy"]
    distance_id_to_pos = instance["distance_id_to_pos"]
    contributions = []
    for ridx, r in enumerate(sol):
        seq = ["0"] + r["stops"] + ["0"]
        for i in range(1, len(seq) - 1):
            a, b, c = seq[i-1], seq[i], seq[i+1]
            saving = lookup_distance(a, b, node_dict, distance_numpy, distance_id_to_pos) + lookup_distance(b, c, node_dict, distance_numpy, distance_id_to_pos) - lookup_distance(a, c, node_dict, distance_numpy, distance_id_to_pos)
            contributions.append((saving, ridx, b))
    contributions.sort(reverse=True, key=lambda x: x[0])
    q = min(q, len(contributions))
    removed = []
    for _, ridx, n in contributions[:q]:
        if n in sol[ridx]["stops"]:
            remove_node_from_route(sol[ridx], n, node_dict)
            removed.append(n)
    return remove_empty_routes(sol), removed


def late_remove(solution: List[dict], q: int, instance: dict, current_evals: Optional[List[RouteEval]] = None) -> Tuple[List[dict], List[str]]:
    """
    删除迟到惩罚较高的节点。

    提速版修复说明：
    - 优先复用当前解已有的 route_evals，避免重复整解评估；
    - 若 evals 与当前 solution 长度不一致，则自动回退重算，避免 ALNS 接受新解后索引错位。
    """
    sol = clone_solution(solution)
    evals = current_evals

    if evals is None or len(evals) != len(sol):
        _, evals, _ = solution_cost(sol, instance)

    score = []
    for ridx, ev in enumerate(evals):
        if ridx >= len(sol):
            continue
        stops = sol[ridx]["stops"]
        for pos, late in enumerate(ev.lates):
            if pos < len(stops) and late > 1e-9:
                score.append((late, ridx, stops[pos]))

    score.sort(reverse=True, key=lambda x: x[0])
    removed = []
    removed_set = set()
    for _, ridx, n in score:
        if len(removed) >= min(q, len(score)):
            break
        if ridx < len(sol) and n in sol[ridx]["stops"] and n not in removed_set:
            remove_node_from_route(sol[ridx], n, instance["node_dict"])
            removed.append(n)
            removed_set.add(n)

    if not removed:
        return random_remove(solution, q, instance)
    return remove_empty_routes(sol), removed


def candidate_positions_by_distance(route: dict, node_id: str, instance: dict, top_k: int = 3) -> List[int]:
    node_dict = instance["node_dict"]
    distance_numpy = instance["distance_numpy"]
    distance_id_to_pos = instance["distance_id_to_pos"]
    seq = ["0"] + route["stops"] + ["0"]
    scored = []
    for pos in range(len(route["stops"]) + 1):
        a = seq[pos]
        b = seq[pos + 1]
        delta = (
            lookup_distance(a, node_id, node_dict, distance_numpy, distance_id_to_pos)
            + lookup_distance(node_id, b, node_dict, distance_numpy, distance_id_to_pos)
            - lookup_distance(a, b, node_dict, distance_numpy, distance_id_to_pos)
        )
        scored.append((delta, pos))
    scored.sort(key=lambda x: x[0])
    return [p for _, p in scored[:max(1, top_k)]]


def quick_route_filter(route: dict, node_id: str, instance: dict) -> bool:
    node_dict = instance["node_dict"]
    veh = instance["vehicle_map"][route["vehicle_type"]]
    node = node_dict[node_id]
    if node.original_customer_id in route_original_customer_set(route, node_dict):
        return False
    if route_total_weight(route, node_dict) + node.weight > veh.capacity_weight + 1e-9:
        return False
    if route_total_volume(route, node_dict) + node.volume > veh.capacity_volume + 1e-9:
        return False
    return True


def best_new_route_for_node(node_id: str, current_solution: List[dict], current_total_cost: float, instance: dict) -> Optional[Tuple[dict, float]]:
    node_dict = instance["node_dict"]
    node = node_dict[node_id]
    vehicles = instance["vehicle_types_list"]
    used = solution_vehicle_counts(current_solution)
    green_nodes = instance.get("green_nodes", {})
    vehicle_candidates = feasible_vehicle_types_for_node(node, vehicles)
    if bool(green_nodes.get(str(node_id), False)):
        vehicle_candidates = sorted(vehicle_candidates, key=lambda vv: (0 if vv.fuel_type == "electric" else 1, vv.capacity_weight))
    best_route = None
    best_delta = None
    for v in vehicle_candidates:
        if used.get(v.vehicle_type, 0) >= v.count:
            continue
        route = make_route(v.vehicle_type, [node_id], node_dict)
        ev = evaluate_route(route, instance)
        if ev.feasible:
            delta = ev.total_cost
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_route = route
    if best_route is None:
        return None
    return best_route, best_delta


def greedy_insert(solution: List[dict], removed: List[str], instance: dict) -> Optional[List[dict]]:
    sol = clone_solution(solution)
    current_total, route_evals, _ = solution_cost(sol, instance)
    node_dict = instance["node_dict"]
    for node_id in removed:
        best_action = None
        best_cost = None
        for ridx, route in enumerate(sol):
            if not quick_route_filter(route, node_id, instance):
                continue
            old_ev = route_evals[ridx]
            for pos in candidate_positions_by_distance(route, node_id, instance, top_k=3):
                cand_route = clone_route(route)
                insert_node_to_route(cand_route, pos, node_id, node_dict)
                new_ev = evaluate_route(cand_route, instance)
                if new_ev.feasible:
                    tot = current_total - old_ev.total_cost + new_ev.total_cost
                    if best_cost is None or tot < best_cost:
                        best_cost = tot
                        best_action = ("replace", ridx, cand_route, new_ev)
        new_route_info = best_new_route_for_node(node_id, sol, current_total, instance)
        if new_route_info is not None:
            route, delta = new_route_info
            tot = current_total + delta
            if best_cost is None or tot < best_cost:
                best_action = ("append", None, route, evaluate_route(route, instance))
                best_cost = tot
        if best_action is None:
            return None
        if best_action[0] == "replace":
            _, ridx, new_route, new_ev = best_action
            sol[ridx] = new_route
            route_evals[ridx] = new_ev
        else:
            _, _, route, new_ev = best_action
            sol.append(route)
            route_evals.append(new_ev)
        current_total = best_cost
    return remove_empty_routes(sol)


def regret_insert(solution: List[dict], removed: List[str], instance: dict, regret_k: int = 2) -> Optional[List[dict]]:
    sol = clone_solution(solution)
    node_dict = instance["node_dict"]
    pending = removed[:]
    current_total, route_evals, _ = solution_cost(sol, instance)
    while pending:
        option_table = []
        for node_id in pending:
            costs = []
            for ridx, route in enumerate(sol):
                if not quick_route_filter(route, node_id, instance):
                    continue
                old_ev = route_evals[ridx]
                for pos in candidate_positions_by_distance(route, node_id, instance, top_k=3):
                    cand_route = clone_route(route)
                    insert_node_to_route(cand_route, pos, node_id, node_dict)
                    new_ev = evaluate_route(cand_route, instance)
                    if new_ev.feasible:
                        delta = new_ev.total_cost - old_ev.total_cost
                        costs.append((delta, ("replace", ridx, cand_route, new_ev)))
            new_route_info = best_new_route_for_node(node_id, sol, current_total, instance)
            if new_route_info is not None:
                route, delta = new_route_info
                new_ev = evaluate_route(route, instance)
                costs.append((delta, ("append", None, route, new_ev)))
            if costs:
                costs.sort(key=lambda x: x[0])
                best = costs[0][0]
                kth = costs[min(regret_k - 1, len(costs) - 1)][0]
                regret = kth - best
                option_table.append((regret, best, node_id, costs[0][1]))
        if not option_table:
            return None
        option_table.sort(reverse=True, key=lambda x: x[0])
        _, best_delta, chosen_node, best_action = option_table[0]
        if best_action[0] == "replace":
            _, ridx, cand_route, new_ev = best_action
            sol[ridx] = cand_route
            route_evals[ridx] = new_ev
        else:
            _, _, route, new_ev = best_action
            sol.append(route)
            route_evals.append(new_ev)
        current_total += best_delta
        pending.remove(chosen_node)
    return remove_empty_routes(sol)


def choose_operator(weights: Dict[str, float]) -> str:
    keys = list(weights.keys())
    vals = np.array([weights[k] for k in keys], dtype=float)
    probs = vals / vals.sum()
    return np.random.choice(keys, p=probs)


def alns_solve(instance: dict, initial_solution: List[dict], iterations: int = 300, seed: int = 42) -> Tuple[List[dict], List[float]]:
    random.seed(seed)
    np.random.seed(seed)

    current = clone_solution(initial_solution)
    current_cost, current_evals, _ = solution_cost(current, instance)

    # 关键：初始解必须覆盖全部服务节点。否则后续 ALNS 即使成本低也没有意义。
    if not solution_covers_all_nodes(current, instance):
        rep = coverage_report(current, instance)
        raise RuntimeError(
            "初始解没有覆盖全部服务节点，请检查初始解构造。"
            f" 应覆盖 {rep['expected_node_count']} 个，实际 {rep['actual_node_count']} 个，"
            f"缺失 {rep['missing_node_count']} 个。"
        )

    best = clone_solution(current)
    best_cost = current_cost

    removal_weights = {"random": 1.0, "worst": 1.0, "late": 1.0}
    insert_weights = {"greedy": 1.0, "regret": 1.0}

    history = [best_cost]
    T = max(1.0, 0.05 * best_cost)
    cooling = 0.995

    for it in range(iterations):
        cand = clone_solution(current)
        n_nodes = len(flatten_nodes(cand))
        q = max(2, int(0.08 * n_nodes))

        rem_op = choose_operator(removal_weights)
        if rem_op == "random":
            partial, removed = random_remove(cand, q, instance)
        elif rem_op == "worst":
            partial, removed = worst_remove(cand, q, instance)
        else:
            partial, removed = late_remove(cand, q, instance, current_evals=current_evals)

        ins_op = choose_operator(insert_weights)
        if ins_op == "greedy":
            repaired = greedy_insert(partial, removed, instance)
        else:
            repaired = regret_insert(partial, removed, instance, regret_k=2)

        # 修复失败，直接拒绝本轮候选解
        if repaired is None:
            history.append(best_cost)
            T *= cooling
            continue

        repaired = remove_empty_routes(repaired)

        # 关键：必须覆盖所有服务节点，否则不能接受
        if not solution_covers_all_nodes(repaired, instance):
            history.append(best_cost)
            T *= cooling
            continue

        new_cost, new_evals, feasible = solution_cost(repaired, instance)
        if not feasible:
            history.append(best_cost)
            T *= cooling
            continue

        accepted = False
        if new_cost <= current_cost:
            accepted = True
        else:
            prob = math.exp(-(new_cost - current_cost) / max(1e-6, T))
            if random.random() < prob:
                accepted = True

        if accepted:
            current = repaired
            current_cost = new_cost
            current_evals = new_evals
            removal_weights[rem_op] += 0.1
            insert_weights[ins_op] += 0.1
        else:
            removal_weights[rem_op] *= 0.999
            insert_weights[ins_op] *= 0.999

        if current_cost < best_cost:
            best = clone_solution(current)
            best_cost = current_cost
            removal_weights[rem_op] += 1.0
            insert_weights[ins_op] += 1.0

        history.append(best_cost)
        T *= cooling

    return best, history


# =========================
# TS/VNS 风格局部优化
# =========================
def local_search_vns(solution: List[dict], instance: dict, max_passes: int = 1) -> List[dict]:
    """
    TS/VNS 风格局部优化。
    修正版重点：
    1. 每次修改 stops 后强制刷新路线缓存；
    2. 不复用旧 cached_eval；
    3. 只接受真实重新评估后可行且成本更低的解；
    4. 返回前再次核验最终解。
    """
    sol = clone_solution(solution)
    refresh_solution_cache(sol, instance)

    best_cost, _, best_feasible = solution_cost(sol, instance)
    if not best_feasible:
        return clone_solution(solution)

    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1

        # 1）单路线 2-opt：反转同一路线中的一段客户顺序
        for ridx in range(len(sol)):
            route = sol[ridx]
            m = len(route["stops"])
            if m < 4:
                continue

            for i in range(m - 1):
                for j in range(i + 1, m):
                    cand = clone_solution(sol)

                    # 修改 stops
                    cand[ridx]["stops"][i:j + 1] = list(reversed(cand[ridx]["stops"][i:j + 1]))

                    # 关键：修改 stops 后必须刷新缓存
                    refresh_solution_cache(cand, instance)

                    if not solution_covers_all_nodes(cand, instance):
                        continue
                    new_cost, _, feasible = solution_cost(cand, instance)
                    if feasible and new_cost + 1e-8 < best_cost:
                        sol = cand
                        best_cost = new_cost
                        improved = True
                        break

                if improved:
                    break

            if improved:
                break

        if improved:
            continue

        # 2）跨路线 relocate：把一条路线中的某个节点移动到另一条路线
        for r1 in range(len(sol)):
            for r2 in range(len(sol)):
                if r1 == r2 or not sol[r1]["stops"]:
                    continue

                for pos1 in range(len(sol[r1]["stops"])):
                    for pos2 in range(len(sol[r2]["stops"]) + 1):
                        cand = clone_solution(sol)

                        moved = cand[r1]["stops"].pop(pos1)
                        cand[r2]["stops"].insert(pos2, moved)

                        cand = remove_empty_routes(cand)

                        # 关键：跨路线移动后，所有路线缓存都要刷新
                        refresh_solution_cache(cand, instance)

                        if not solution_covers_all_nodes(cand, instance):
                            continue
                        new_cost, _, feasible = solution_cost(cand, instance)
                        if feasible and new_cost + 1e-8 < best_cost:
                            sol = cand
                            best_cost = new_cost
                            improved = True
                            break

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

    # 返回前强制重新核验，避免局部搜索产生缓存污染
    refresh_solution_cache(sol, instance)
    final_cost, _, final_feasible = solution_cost(sol, instance)

    if final_feasible and solution_covers_all_nodes(sol, instance):
        return sol

    return clone_solution(solution)



# =========================
# 合法降成本后处理：车型重分配 + 路线合并
# =========================
def choose_best_vehicle_for_stops(stops: List[str], instance: dict, used_counts: Optional[Dict[str, int]] = None) -> Optional[dict]:
    """
    在不改变服务节点集合与路径顺序的前提下，为一组 stops 选择成本最低的可行车型。
    若传入 used_counts，则同时检查该车型剩余库存。
    """
    node_dict = instance["node_dict"]
    vehicles = instance["vehicle_types_list"]
    stock = instance["vehicle_stock"]

    # 同一路线不允许同一原客户重复出现，保持原建模口径不变。
    origs = [node_dict[n].original_customer_id for n in stops]
    if len(origs) != len(set(origs)):
        return None

    total_w = sum(node_dict[n].weight for n in stops)
    total_v = sum(node_dict[n].volume for n in stops)

    best_route = None
    best_cost = None
    for v in vehicles:
        if used_counts is not None and used_counts.get(v.vehicle_type, 0) >= stock.get(v.vehicle_type, 0):
            continue
        if total_w > v.capacity_weight + 1e-9 or total_v > v.capacity_volume + 1e-9:
            continue
        cand = make_route(v.vehicle_type, list(stops), node_dict)
        ev = evaluate_route(cand, instance)
        if ev.feasible and (best_cost is None or ev.total_cost < best_cost):
            best_cost = ev.total_cost
            best_route = cand

    return best_route


def reassign_vehicle_types(solution: List[dict], instance: dict) -> List[dict]:
    """
    车型重分配：
    固定每条路线的服务节点顺序不变，重新选择满足容量与库存约束的最低成本车型。
    该操作不改变覆盖节点，不改变模型约束，只降低车型选择造成的额外成本。
    """
    node_dict = instance["node_dict"]
    stock = instance["vehicle_stock"]

    # 先处理重载/大体积路线，避免后面车型库存被轻载路线占用。
    routes = clone_solution(solution)
    routes.sort(
        key=lambda r: (
            route_total_weight(r, node_dict) / MAX_WEIGHT,
            route_total_volume(r, node_dict) / MAX_VOLUME,
            len(r["stops"]),
        ),
        reverse=True,
    )

    used_counts: Dict[str, int] = {}
    new_solution: List[dict] = []
    for r in routes:
        best = choose_best_vehicle_for_stops(r["stops"], instance, used_counts=used_counts)
        if best is None:
            # 理论上不应发生；若发生则保留原路线，后续由全局可行性检查拦截。
            best = clone_route(r)
        used_counts[best["vehicle_type"]] = used_counts.get(best["vehicle_type"], 0) + 1
        if used_counts[best["vehicle_type"]] > stock.get(best["vehicle_type"], 0):
            raise RuntimeError(f"车型 {best['vehicle_type']} 使用数量超过库存，请检查车型重分配逻辑。")
        new_solution.append(best)

    refresh_solution_cache(new_solution, instance)
    if solution_covers_all_nodes(new_solution, instance):
        new_cost, _, feasible = solution_cost(new_solution, instance)
        old_cost, _, old_feasible = solution_cost(solution, instance)
        if feasible and (not old_feasible or new_cost <= old_cost + 1e-8):
            return new_solution
    return clone_solution(solution)


def merge_routes_postprocess(solution: List[dict], instance: dict, max_passes: int = 1) -> List[dict]:
    """
    路线合并后处理：
    尝试把两条路线合并为一条路线，以减少固定启用成本与回仓距离。
    只在满足容量、库存、同原客户不重复、全节点覆盖且总成本下降时接受。
    """
    sol = clone_solution(solution)
    refresh_solution_cache(sol, instance)

    base_cost, _, base_feasible = solution_cost(sol, instance)
    if not base_feasible or not solution_covers_all_nodes(sol, instance):
        return clone_solution(solution)

    for _ in range(max_passes):
        improved = False
        best_candidate = None
        best_candidate_cost = base_cost

        n = len(sol)
        # 先尝试服务节点少、装载率低的路线，通常更容易合并。
        order = sorted(
            range(n),
            key=lambda i: (
                len(sol[i]["stops"]),
                route_total_weight(sol[i], instance["node_dict"]),
                route_total_volume(sol[i], instance["node_dict"]),
            ),
        )

        for ai in range(len(order)):
            i = order[ai]
            if i >= len(sol):
                continue
            for bj in range(ai + 1, len(order)):
                j = order[bj]
                if j >= len(sol) or i == j:
                    continue

                r1, r2 = sol[i], sol[j]
                # 快速容量下界筛选：合并后至少要能被最大车型承载。
                if route_total_weight(r1, instance["node_dict"]) + route_total_weight(r2, instance["node_dict"]) > MAX_WEIGHT + 1e-9:
                    continue
                if route_total_volume(r1, instance["node_dict"]) + route_total_volume(r2, instance["node_dict"]) > MAX_VOLUME + 1e-9:
                    continue

                old_pair_cost = evaluate_route(r1, instance).total_cost + evaluate_route(r2, instance).total_cost
                remain = [clone_route(r) for k, r in enumerate(sol) if k not in (i, j)]
                used_counts = solution_vehicle_counts(remain)

                for stops in (r1["stops"] + r2["stops"], r2["stops"] + r1["stops"]):
                    merged = choose_best_vehicle_for_stops(stops, instance, used_counts=used_counts)
                    if merged is None:
                        continue
                    merged_cost = evaluate_route(merged, instance).total_cost
                    if merged_cost >= old_pair_cost - 1e-8:
                        continue

                    cand = remain + [merged]
                    refresh_solution_cache(cand, instance)
                    if not solution_covers_all_nodes(cand, instance):
                        continue
                    cand_cost, _, cand_feasible = solution_cost(cand, instance)
                    if cand_feasible and cand_cost + 1e-8 < best_candidate_cost:
                        best_candidate = cand
                        best_candidate_cost = cand_cost

        if best_candidate is not None:
            sol = best_candidate
            base_cost = best_candidate_cost
            improved = True

        if not improved:
            break

    sol = reassign_vehicle_types(sol, instance)
    refresh_solution_cache(sol, instance)
    if solution_covers_all_nodes(sol, instance):
        new_cost, _, feasible = solution_cost(sol, instance)
        old_cost, _, old_feasible = solution_cost(solution, instance)
        if feasible and (not old_feasible or new_cost <= old_cost + 1e-8):
            return sol
    return clone_solution(solution)


# =========================
# 鲁棒仿真
# =========================
def sample_period_speeds(speed_periods: List[dict], rng: np.random.Generator) -> Dict[int, float]:
    sampled = {}
    for p in speed_periods:
        mu, sigma = float(p["mean_speed"]), float(p["std"])
        spd = rng.normal(mu, sigma)
        sampled[p["period_id"]] = max(1.0, float(spd))
    return sampled


def robust_simulation(solution: List[dict], instance: dict, n_sim: int = 50, seed: int = 42) -> dict:
    """
    鲁棒仿真：
    固定最终路径方案不变，只对各时段车速进行随机扰动，
    重新计算每次仿真下的总成本、等待、迟到、碳排放与服务可靠度。

    新增指标：
    1. expected_total_cost：期望总成本，即 total_cost 的均值；
    2. mean_service_reliability：平均服务可靠度，即所有仿真场景下未迟到服务节点比例的均值。
    """
    rng = np.random.default_rng(seed)
    speed_periods = instance["speed_periods_records"]

    vals = []
    late_route_counts = []

    for sim_id in range(1, n_sim + 1):
        sampled = sample_period_speeds(speed_periods, rng)
        total, evals, feasible = solution_cost(solution, instance, sampled_speeds=sampled)

        total_late = sum(sum(ev.lates) for ev in evals)
        total_wait = sum(sum(ev.waits) + getattr(ev, "policy_wait_min", 0.0) for ev in evals)
        total_carbon = sum(ev.carbon_cost / CARBON_PRICE for ev in evals)

        # =========================
        # 新增：节点级服务可靠度
        # =========================
        total_service_nodes = 0
        on_time_nodes = 0
        late_nodes = 0

        for ev in evals:
            for det in ev.details:
                # ev.details 中每条记录对应一次真实服务节点到达，
                # 不包含最后返回配送中心的弧段，因此可以直接作为服务节点统计。
                total_service_nodes += 1

                if det["late_min"] <= 1e-9:
                    on_time_nodes += 1
                else:
                    late_nodes += 1

        service_reliability = (
            on_time_nodes / total_service_nodes
            if total_service_nodes > 0 else 0.0
        )

        vals.append({
            "sim_id": sim_id,
            "feasible": feasible,
            "total_cost": total,
            "total_late_min": total_late,
            "total_wait_min": total_wait,
            "total_carbon_kg": total_carbon,
            "total_service_node_count": total_service_nodes,
            "on_time_node_count": on_time_nodes,
            "late_node_count": late_nodes,
            "service_reliability": service_reliability,
        })

        late_route_counts.append(sum(1 for ev in evals if sum(ev.lates) > 0))

    df = pd.DataFrame(vals)

    expected_cost = float(df["total_cost"].mean())
    service_reliability_mean = float(df["service_reliability"].mean())

    return {
        "n_sim": n_sim,

        # 原有字段，保留，避免影响后续输出
        "mean_total_cost": expected_cost,
        "std_total_cost": float(df["total_cost"].std(ddof=1)),
        "mean_total_late_min": float(df["total_late_min"].mean()),
        "mean_total_wait_min": float(df["total_wait_min"].mean()),
        "mean_total_carbon_kg": float(df["total_carbon_kg"].mean()),
        "mean_late_route_count": float(np.mean(late_route_counts)),

        # 新增字段，方便论文手直接对应公式
        "expected_total_cost": expected_cost,
        "mean_service_reliability": service_reliability_mean,
        "mean_on_time_node_count": float(df["on_time_node_count"].mean()),
        "mean_late_node_count": float(df["late_node_count"].mean()),
        "total_service_node_count": int(df["total_service_node_count"].iloc[0]) if len(df) > 0 else 0,

        # 明细表
        "raw": df,
    }


# =========================
# 输出
# =========================
def route_to_str(route: dict) -> str:
    if not route["stops"]:
        return "0-0"
    return "0-" + "-".join(map(str, route["stops"])) + "-0"


def summarize_solution(solution: List[dict], instance: dict) -> Tuple[pd.DataFrame, dict]:
    total, evals, feasible = solution_cost(solution, instance)
    node_dict = instance["node_dict"]
    rows = []
    type_count = solution_vehicle_counts(solution)
    for idx, (route, ev) in enumerate(zip(solution, evals), start=1):
        rows.append({
            "route_id": idx,
            "vehicle_type": route["vehicle_type"],
            "route": route_to_str(route),
            "stop_count": len(route["stops"]),
            "original_customer_count": len({node_dict[n].original_customer_id for n in route["stops"]}),
            "total_weight": sum(node_dict[n].weight for n in route["stops"]),
            "total_volume": sum(node_dict[n].volume for n in route["stops"]),
            "departure_time": rel_minutes_to_clock_str(ev.departure_time),
            "return_time": rel_minutes_to_clock_str(ev.return_time),
            "distance_km": ev.total_distance,
            "travel_minutes": ev.total_travel_minutes,
            "fixed_cost": ev.fixed_cost,
            "wait_cost": ev.wait_cost,
            "late_cost": ev.late_cost,
            "energy_cost": ev.energy_cost,
            "carbon_cost": ev.carbon_cost,
            "policy_wait_min": getattr(ev, "policy_wait_min", 0.0),
            "policy_wait_cost": getattr(ev, "policy_wait_cost", 0.0),
            "policy_affected_arc_count": getattr(ev, "policy_affected_arc_count", 0),
            "green_service_count": getattr(ev, "green_service_count", 0),
            "cross_green_arc_count": getattr(ev, "cross_green_arc_count", 0),
            "total_cost": ev.total_cost,
            "feasible": ev.feasible,
        })
    route_df = pd.DataFrame(rows)
    summary = {
        "solution_feasible": feasible,
        "total_cost": total,
        "total_routes": len(solution),
        "vehicle_type_count": type_count,
        "route_cost_sum": float(route_df["total_cost"].sum()) if not route_df.empty else 0.0,
        "distance_km_sum": float(route_df["distance_km"].sum()) if not route_df.empty else 0.0,
        "fixed_cost_sum": float(route_df["fixed_cost"].sum()) if not route_df.empty else 0.0,
        "wait_cost_sum": float(route_df["wait_cost"].sum()) if not route_df.empty else 0.0,
        "late_cost_sum": float(route_df["late_cost"].sum()) if not route_df.empty else 0.0,
        "energy_cost_sum": float(route_df["energy_cost"].sum()) if not route_df.empty else 0.0,
        "carbon_cost_sum": float(route_df["carbon_cost"].sum()) if not route_df.empty else 0.0,
        "policy_wait_min_sum": float(route_df["policy_wait_min"].sum()) if "policy_wait_min" in route_df.columns and not route_df.empty else 0.0,
        "policy_wait_cost_sum": float(route_df["policy_wait_cost"].sum()) if "policy_wait_cost" in route_df.columns and not route_df.empty else 0.0,
        "policy_affected_route_count": int((route_df["policy_affected_arc_count"] > 0).sum()) if "policy_affected_arc_count" in route_df.columns and not route_df.empty else 0,
        "policy_affected_arc_count_sum": int(route_df["policy_affected_arc_count"].sum()) if "policy_affected_arc_count" in route_df.columns and not route_df.empty else 0,
    }

    cover = coverage_report(solution, instance)
    summary.update({
        "expected_node_count": cover["expected_node_count"],
        "actual_node_count": cover["actual_node_count"],
        "unique_actual_node_count": cover["unique_actual_node_count"],
        "missing_node_count": cover["missing_node_count"],
        "extra_node_count": cover["extra_node_count"],
        "has_duplicate_node": cover["has_duplicate_node"],
        "expected_weight": cover["expected_weight"],
        "actual_weight": cover["actual_weight"],
        "expected_volume": cover["expected_volume"],
        "actual_volume": cover["actual_volume"],
        "used_vehicle_count": cover["used_vehicle_count"],
        "used_capacity_weight": cover["used_capacity_weight"],
        "used_capacity_volume": cover["used_capacity_volume"],
    })
    return route_df, summary


def draw_routes(solution: List[dict], instance: dict, out_png: Path):
    set_chinese_font()
    service_nodes = instance["service_nodes"]
    node_dict = instance["node_dict"]
    coords_raw = instance["coordinates"]
    depot = coords_raw[coords_raw["类型"] == "配送中心"].iloc[0]
    cust = coords_raw[coords_raw["类型"] == "客户"]

    plt.figure(figsize=(10, 8))
    plt.scatter(cust["X (km)"], cust["Y (km)"], s=18, alpha=0.5, label="客户点")
    plt.scatter([depot["X (km)"]], [depot["Y (km)"]], marker="*", s=220, label="配送中心")

    # 问题二新增：绘制绿色配送区边界
    theta = np.linspace(0, 2 * np.pi, 360)
    plt.plot(
        GREEN_CENTER_X + GREEN_RADIUS_KM * np.cos(theta),
        GREEN_CENTER_Y + GREEN_RADIUS_KM * np.sin(theta),
        linestyle="--",
        linewidth=2.0,
        label="绿色配送区边界"
    )

    if "in_green_zone" in service_nodes.columns:
        green_pts = service_nodes[service_nodes["in_green_zone"] == True]
        if not green_pts.empty:
            plt.scatter(green_pts["x"], green_pts["y"], s=45, marker="o", label="区内服务节点")

    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(solution))))
    for idx, route in enumerate(solution):
        xs = [float(depot["X (km)"])]
        ys = [float(depot["Y (km)"])]
        for nid in route["stops"]:
            xs.append(node_dict[nid].x)
            ys.append(node_dict[nid].y)
        xs.append(float(depot["X (km)"]))
        ys.append(float(depot["Y (km)"]))
        plt.plot(xs, ys, linewidth=1.8, color=colors[idx], label=f"路线{idx+1}:{route['vehicle_type']}")

    plt.title("问题二：配送路径图（含绿色配送区）")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.axis("equal")
    if len(solution) <= 12:
        plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=240)
    plt.close()


def draw_convergence(history: List[float], out_png: Path):
    set_chinese_font()
    plt.figure(figsize=(8, 5))
    plt.plot(history, linewidth=1.8)
    plt.xlabel("迭代次数")
    plt.ylabel("当前最优总成本")
    plt.title("ALNS 收敛曲线")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_png, dpi=240)
    plt.close()


def save_outputs(instance: dict, solution: List[dict], history: List[float], robust: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # 原始和标准化实例
    instance["service_nodes"].to_excel(outdir / "服务节点表.xlsx", index=False)
    instance["service_nodes"].to_csv(outdir / "服务节点表.csv", index=False, encoding="utf-8-sig")
    instance["split_summary"].to_excel(outdir / "超容量客户拆分汇总.xlsx", index=False)
    instance["orders_clean"].to_excel(outdir / "订单清洗表.xlsx", index=False)
    instance["vehicles"].to_excel(outdir / "车辆参数表.xlsx", index=False)
    instance["speed_periods"].to_excel(outdir / "时段速度参数表.xlsx", index=False)
    pd.DataFrame([instance["cost_params"]]).to_excel(outdir / "成本参数表.xlsx", index=False)

    # 问题二新增：绿色区政策标记输出
    if "in_green_zone" in instance["service_nodes"].columns:
        instance["service_nodes"][[
            "node_id", "original_customer_id", "sub_id", "is_virtual",
            "x", "y", "weight", "volume", "earliest", "latest", "in_green_zone"
        ]].to_excel(outdir / "问题二_绿色区节点标记.xlsx", index=False)
    if "green_arc_df" in instance:
        instance["green_arc_df"].to_excel(outdir / "问题二_穿区弧段标记.xlsx", index=False)
    if "green_zone_summary" in instance:
        pd.DataFrame([instance["green_zone_summary"]]).to_excel(outdir / "问题二_绿色区政策摘要.xlsx", index=False)

    # 结果汇总
    route_df, summary = summarize_solution(solution, instance)
    route_df.to_excel(outdir / "问题二_路线汇总.xlsx", index=False)
    route_df.to_csv(outdir / "问题二_路线汇总.csv", index=False, encoding="utf-8-sig")

    # 鲁棒性明细
    robust["raw"].to_excel(outdir / "鲁棒仿真明细.xlsx", index=False)
    robust_to_save = {k: v for k, v in robust.items() if k != "raw"}

    # 结果 JSON
    payload = {
        "summary": summary,
        "robust_summary": robust_to_save,
        "source_files": instance["source_files"],
    }
    with open(outdir / "问题二_结果摘要.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 路线明细（逐节点）
    _, evals, _ = solution_cost(solution, instance)
    detail_rows = []
    for ridx, (route, ev) in enumerate(zip(solution, evals), start=1):
        for seq, det in enumerate(ev.details, start=1):
            detail_rows.append({
                "route_id": ridx,
                "vehicle_type": route["vehicle_type"],
                "seq": seq,
                **det,
                "from_clock": rel_minutes_to_clock_str(det["depart_time"]),
                "arrival_clock": rel_minutes_to_clock_str(det["arrival_time"]),
                "service_start_clock": rel_minutes_to_clock_str(det["service_start"]),
                "service_end_clock": rel_minutes_to_clock_str(det["service_end"]),
            })
    pd.DataFrame(detail_rows).to_excel(outdir / "问题二_弧段与到达时间明细.xlsx", index=False)

    # 图件
    draw_routes(solution, instance, outdir / "问题二_配送路径图.png")
    draw_convergence(history, outdir / "问题二_ALNS收敛曲线.png")

    # 摘要文本
    lines = []
    lines.append("问题二：绿色配送区限行情景下车辆调度——程序输出说明")
    lines.append("=" * 50)
    lines.append(f"总配送成本：{summary['total_cost']:.2f}")
    lines.append(f"启用路线数：{summary['total_routes']}")
    lines.append(f"总距离（km）：{summary['distance_km_sum']:.2f}")
    lines.append(f"启动成本合计：{summary['fixed_cost_sum']:.2f}")
    lines.append(f"等待成本合计：{summary['wait_cost_sum']:.2f}")
    lines.append(f"迟到成本合计：{summary['late_cost_sum']:.2f}")
    lines.append(f"能源成本合计：{summary['energy_cost_sum']:.2f}")
    lines.append(f"碳排放成本合计：{summary['carbon_cost_sum']:.2f}")
    lines.append(f"政策等待总分钟：{summary.get('policy_wait_min_sum', 0):.2f}")
    lines.append(f"政策等待成本合计：{summary.get('policy_wait_cost_sum', 0):.2f}")
    lines.append(f"受政策等待影响路线数：{summary.get('policy_affected_route_count', 0)}")
    lines.append("")
    lines.append("各车辆类型使用数：")
    for k, v in summary["vehicle_type_count"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("鲁棒仿真（固定路径方案，速度扰动）概要：")
    lines.append(f"- 仿真次数：{robust['n_sim']}")
    lines.append(f"- 平均总成本：{robust['mean_total_cost']:.2f}")
    lines.append(f"- 总成本标准差：{robust['std_total_cost']:.2f}")
    lines.append(f"- 平均迟到总分钟：{robust['mean_total_late_min']:.2f}")
    lines.append(f"- 平均等待总分钟：{robust['mean_total_wait_min']:.2f}")
    lines.append(f"- 平均碳排放量(kg)：{robust['mean_total_carbon_kg']:.2f}")
    # 新增：与论文 5.3 鲁棒性验证指标对应
    lines.append(
        f"- 期望总成本 Expected Cost：{robust.get('expected_total_cost', robust.get('mean_total_cost', 0)):.2f}")
    lines.append(f"- 服务可靠度 Service Reliability：{robust.get('mean_service_reliability', 0) * 100:.2f}%")
    lines.append(f"- 平均准时服务节点数：{robust.get('mean_on_time_node_count', 0):.2f}")
    lines.append(f"- 平均迟到服务节点数：{robust.get('mean_late_node_count', 0):.2f}")
    lines.append(f"- 服务节点总数：{robust.get('total_service_node_count', 0)}")
    lines.append("")
    lines.append("输出文件实际意义：")
    lines.append("1. 服务节点表：问题一真正进入模型的节点输入，包含普通客户节点和虚拟拆分节点。")
    lines.append("2. 超容量客户拆分汇总：说明哪些客户发生了虚拟拆分、理论下界与实际分组数是多少。")
    lines.append("3. 订单清洗表：记录缺失值修复与异常标签，便于回溯。")
    lines.append("4. 路线汇总表：给出每条路线的车型、路径、出发回仓时间和成本分解。")
    lines.append("5. 弧段与到达时间明细：给出逐弧段的到达时刻、等待、迟到、能耗与碳排放成本。")
    lines.append("6. 鲁棒仿真明细：检验固定路径在速度波动下的稳定性。")
    lines.append("7. 配送路径图：用于论文和答辩的可视化展示。")
    lines.append("8. ALNS 收敛曲线：展示主求解器的搜索过程。")
    lines.append("9. 绿色区节点标记：说明哪些服务节点位于半径 10 km 的绿色配送区内。")
    lines.append("   本版采用动态禁行：燃油车弧段只要与 8:00—16:00 限行窗口重叠且涉及入区/穿区，即判不可行；16:00 后完整通行的弧段允许。")
    lines.append("10. 穿区弧段标记：说明哪些弧段可能穿越绿色配送区。")
    with open(outdir / "问题二_输出说明.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# 论文补充图表自动生成模块（内嵌 q2_画图）
# =========================
_Q2_EMBEDDED_PLOTTER_CODE = '# -*- coding: utf-8 -*-\nr"""\n问题二 论文补充图表绘制脚本\n\n用途：\n- 读取问题二最终输出文件夹中的结果文件；\n- 生成与问题一风格基本一致的论文补充图表；\n- 输出到“q2_论文补充图表”文件夹。\n\n默认读取文件（优先精确匹配，也支持自动模糊查找）：\n- 问题二_结果摘要.json\n- 问题二_路线汇总.xlsx 或 问题二_路线汇总.csv\n- 问题二_弧段与到达时间明细.xlsx\n- 车辆参数表.xlsx（若存在，用于补算载重利用率）\n\n默认输出：\n- 图A：成本构成环形图 / 饼图\n- 图B：载重利用率直方图\n- 图C：路径距离/时长分布图\n- 表D：典型配送路径微观明细表\n\n使用方法：\n1. 若脚本就放在问题二结果文件夹内：\n   python q2_画图.py\n\n2. 若脚本放在上级目录（如 D:\\A_Question）：\n   python q2_画图.py\n   脚本会自动向下查找问题二输出目录。\n\n3. 手动指定结果目录：\n   python q2_画图.py --root "D:\\A_Question\\问题二_xxx"\n"""\n\nimport argparse\nimport json\nimport math\nimport re\nfrom copy import copy\nfrom pathlib import Path\n\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nfrom matplotlib import font_manager\n\n\nPROB_CN = "问题二"\nPROB_ALIASES = ["问题二", "问题2", "q2", "Q2"]\nDEFAULT_OUTDIR = "q2_论文补充图表"\n\n\n# =========================\n# 1. 基础工具函数\n# =========================\n\ndef set_chinese_font():\n    font_candidates = [\n        "Microsoft YaHei",\n        "SimHei",\n        "SimSun",\n        "Noto Sans CJK SC",\n        "Noto Sans CJK JP",\n        "Source Han Sans SC",\n        "WenQuanYi Micro Hei",\n        "Arial Unicode MS",\n    ]\n    installed = {f.name for f in font_manager.fontManager.ttflist}\n    for name in font_candidates:\n        if name in installed:\n            plt.rcParams["font.sans-serif"] = [name]\n            break\n    plt.rcParams["axes.unicode_minus"] = False\n\n\ndef safe_float(x, default=0.0):\n    try:\n        if pd.isna(x):\n            return default\n        return float(x)\n    except Exception:\n        return default\n\n\ndef money_fmt(x):\n    return f"{safe_float(x):,.2f}"\n\n\ndef percent_fmt(x):\n    return f"{safe_float(x):.2f}%"\n\n\ndef normalize_name(s: str) -> str:\n    return re.sub(r"\\s+", "", str(s)).lower()\n\n\ndef choose_best_candidate(paths, prefer_keywords=None):\n    paths = [Path(p) for p in paths if Path(p).exists()]\n    if not paths:\n        return None\n\n    prefer_keywords = prefer_keywords or []\n\n    def score(p: Path):\n        name = p.name.lower()\n        s = 0\n        for kw in prefer_keywords:\n            if kw.lower() in name:\n                s += 10\n        # 更偏好名字更短、更像正式输出文件的文件\n        s -= len(p.name) * 0.01\n        return s\n\n    return sorted(paths, key=score, reverse=True)[0]\n\n\ndef find_exact_or_fuzzy_file(root: Path, exact_names, fuzzy_patterns, prefer_keywords=None):\n    for name in exact_names:\n        p = root / name\n        if p.exists():\n            return p\n\n    candidates = []\n    for pat in fuzzy_patterns:\n        candidates.extend(root.glob(pat))\n\n    return choose_best_candidate(candidates, prefer_keywords=prefer_keywords)\n\n\ndef read_json(path: Path):\n    if path is None or not path.exists():\n        return {}\n    with open(path, "r", encoding="utf-8") as f:\n        return json.load(f)\n\n\ndef detect_input_files(root: Path):\n    summary_json = find_exact_or_fuzzy_file(\n        root,\n        exact_names=[f"{PROB_CN}_结果摘要.json", "问题2_结果摘要.json"],\n        fuzzy_patterns=["*结果摘要*.json", "*.json"],\n        prefer_keywords=["结果摘要", *PROB_ALIASES],\n    )\n    route_file = find_exact_or_fuzzy_file(\n        root,\n        exact_names=[f"{PROB_CN}_路线汇总.xlsx", f"{PROB_CN}_路线汇总.csv", "问题2_路线汇总.xlsx", "问题2_路线汇总.csv"],\n        fuzzy_patterns=["*路线汇总*.xlsx", "*路线汇总*.csv", "*.xlsx", "*.csv"],\n        prefer_keywords=["路线汇总", *PROB_ALIASES],\n    )\n    arc_file = find_exact_or_fuzzy_file(\n        root,\n        exact_names=[f"{PROB_CN}_弧段与到达时间明细.xlsx", "问题2_弧段与到达时间明细.xlsx"],\n        fuzzy_patterns=["*弧段*到达时间明细*.xlsx", "*到达时间明细*.xlsx", "*.xlsx"],\n        prefer_keywords=["弧段", "到达时间明细", *PROB_ALIASES],\n    )\n    vehicle_file = find_exact_or_fuzzy_file(\n        root,\n        exact_names=["车辆参数表.xlsx"],\n        fuzzy_patterns=["*车辆参数*.xlsx"],\n        prefer_keywords=["车辆参数"],\n    )\n    return summary_json, route_file, arc_file, vehicle_file\n\n\ndef has_required_result_files(folder: Path) -> bool:\n    summary_json, route_file, arc_file, _ = detect_input_files(folder)\n    return summary_json is not None and route_file is not None and arc_file is not None\n\n\ndef find_result_root(start_dir: Path) -> Path:\n    start_dir = start_dir.expanduser().resolve()\n\n    if has_required_result_files(start_dir):\n        return start_dir\n\n    for parent in start_dir.parents:\n        if has_required_result_files(parent):\n            return parent\n\n    candidates = []\n    max_depth = 4\n    for p in list(start_dir.rglob("*路线汇总*.xlsx")) + list(start_dir.rglob("*路线汇总*.csv")):\n        folder = p.parent\n        try:\n            depth = len(folder.relative_to(start_dir).parts)\n        except Exception:\n            depth = 999\n        if depth <= max_depth and has_required_result_files(folder):\n            candidates.append(folder)\n\n    if candidates:\n        def score(folder: Path):\n            name = folder.name.lower()\n            s = 0\n            for kw in PROB_ALIASES:\n                if kw.lower() in name:\n                    s += 10\n            s -= len(folder.parts)\n            return s\n\n        return sorted(set(candidates), key=score, reverse=True)[0]\n\n    raise FileNotFoundError(\n        "未找到问题二结果文件夹。请把脚本放到问题二输出目录内，或使用 --root 手动指定。"\n    )\n\n\ndef read_route_summary(route_file: Path):\n    if route_file is None or not route_file.exists():\n        raise FileNotFoundError("未找到 问题二_路线汇总.xlsx 或 问题二_路线汇总.csv")\n    if route_file.suffix.lower() == ".xlsx":\n        return pd.read_excel(route_file)\n    return pd.read_csv(route_file, encoding="utf-8-sig")\n\n\ndef read_arc_detail(arc_file: Path):\n    if arc_file is None or not arc_file.exists():\n        raise FileNotFoundError("未找到 问题二_弧段与到达时间明细.xlsx")\n    return pd.read_excel(arc_file)\n\n\ndef read_vehicle_params(vehicle_file: Path | None):\n    if vehicle_file is None or not vehicle_file.exists():\n        return None\n    try:\n        return pd.read_excel(vehicle_file)\n    except Exception:\n        return None\n\n\ndef parse_capacity_from_vehicle_type(vehicle_type):\n    m = re.search(r"(\\d+)", str(vehicle_type))\n    if m:\n        return float(m.group(1))\n    return np.nan\n\n\ndef get_existing_col(df: pd.DataFrame, candidates, required=False):\n    norm_map = {normalize_name(c): c for c in df.columns}\n    for c in candidates:\n        c_norm = normalize_name(c)\n        if c_norm in norm_map:\n            return norm_map[c_norm]\n    if required:\n        raise KeyError(f"未找到字段，候选为：{candidates}")\n    return None\n\n\ndef add_capacity_and_load_rate(route_df: pd.DataFrame, vehicle_df: pd.DataFrame | None):\n    df = route_df.copy()\n\n    vehicle_type_col = get_existing_col(df, ["vehicle_type", "车型", "vehicle"])\n    total_weight_col = get_existing_col(df, ["total_weight", "总载重", "总载重_kg", "load_weight", "weight"])\n    total_volume_col = get_existing_col(df, ["total_volume", "总体积", "load_volume", "volume"])\n\n    if vehicle_type_col is None:\n        df["vehicle_type"] = "未知车型"\n        vehicle_type_col = "vehicle_type"\n    elif vehicle_type_col != "vehicle_type":\n        df["vehicle_type"] = df[vehicle_type_col]\n\n    if total_weight_col is not None and total_weight_col != "total_weight":\n        df["total_weight"] = df[total_weight_col]\n\n    if total_volume_col is not None and total_volume_col != "total_volume":\n        df["total_volume"] = df[total_volume_col]\n\n    if "capacity_weight" not in df.columns:\n        df["capacity_weight"] = np.nan\n    if "capacity_volume" not in df.columns:\n        df["capacity_volume"] = np.nan\n\n    if vehicle_df is not None:\n        v_vehicle_type = get_existing_col(vehicle_df, ["vehicle_type", "车型", "vehicle"])\n        v_cap_weight = get_existing_col(vehicle_df, ["capacity_weight", "载重上限", "载重上限_kg", "weight_capacity"])\n        v_cap_volume = get_existing_col(vehicle_df, ["capacity_volume", "容积上限", "volume_capacity"])\n\n        if v_vehicle_type and v_cap_weight:\n            cap_map = vehicle_df.set_index(v_vehicle_type)[v_cap_weight].to_dict()\n            df["capacity_weight"] = df["capacity_weight"].fillna(df["vehicle_type"].map(cap_map))\n        if v_vehicle_type and v_cap_volume:\n            vol_map = vehicle_df.set_index(v_vehicle_type)[v_cap_volume].to_dict()\n            df["capacity_volume"] = df["capacity_volume"].fillna(df["vehicle_type"].map(vol_map))\n\n    df["capacity_weight"] = df["capacity_weight"].fillna(df["vehicle_type"].apply(parse_capacity_from_vehicle_type))\n\n    if "load_rate_weight_percent" not in df.columns:\n        if "total_weight" in df.columns:\n            df["load_rate_weight_percent"] = df["total_weight"] / df["capacity_weight"] * 100\n        else:\n            df["load_rate_weight_percent"] = np.nan\n    else:\n        # 若已有该列但小于等于1，则很可能是比例形式，转换为百分数\n        s = pd.to_numeric(df["load_rate_weight_percent"], errors="coerce")\n        if s.dropna().size > 0 and s.dropna().max() <= 1.0001:\n            df["load_rate_weight_percent"] = s * 100\n\n    df["load_rate_weight_percent"] = pd.to_numeric(df["load_rate_weight_percent"], errors="coerce")\n    df["load_rate_weight_percent"] = df["load_rate_weight_percent"].replace([np.inf, -np.inf], np.nan)\n\n    if "load_rate_volume_percent" not in df.columns and "total_volume" in df.columns:\n        df["load_rate_volume_percent"] = df["total_volume"] / df["capacity_volume"] * 100\n        df["load_rate_volume_percent"] = df["load_rate_volume_percent"].replace([np.inf, -np.inf], np.nan)\n\n    return df\n\n\ndef save_excel(path: Path, data_dict: dict):\n    path.parent.mkdir(parents=True, exist_ok=True)\n    with pd.ExcelWriter(path, engine="openpyxl") as writer:\n        for sheet_name, df in data_dict.items():\n            df.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)\n\n        wb = writer.book\n        for ws in wb.worksheets:\n            ws.freeze_panes = "A2"\n            for col_cells in ws.columns:\n                max_len = 8\n                col_letter = col_cells[0].column_letter\n                for cell in col_cells:\n                    val = "" if cell.value is None else str(cell.value)\n                    max_len = max(max_len, min(len(val), 40))\n                ws.column_dimensions[col_letter].width = max_len + 2\n\n            for row in ws.iter_rows():\n                for cell in row:\n                    new_alignment = copy(cell.alignment)\n                    new_alignment.wrap_text = True\n                    new_alignment.vertical = "center"\n                    cell.alignment = new_alignment\n\n            for cell in ws[1]:\n                new_font = copy(cell.font)\n                new_font.bold = True\n                cell.font = new_font\n\n\n# =========================\n# 2. 图A：成本构成图\n# =========================\n\ndef make_cost_data(summary_json: dict, route_df: pd.DataFrame):\n    s = summary_json.get("summary", {}) if isinstance(summary_json, dict) else {}\n    if not s and isinstance(summary_json, dict):\n        s = summary_json\n\n    cost_items = [\n        ("固定成本", safe_float(s.get("fixed_cost_sum", np.nan), np.nan)),\n        ("能耗成本", safe_float(s.get("energy_cost_sum", np.nan), np.nan)),\n        ("碳排放成本", safe_float(s.get("carbon_cost_sum", np.nan), np.nan)),\n        ("等待成本", safe_float(s.get("wait_cost_sum", np.nan), np.nan)),\n        ("迟到成本", safe_float(s.get("late_cost_sum", np.nan), np.nan)),\n    ]\n\n    valid_sum = sum(v for _, v in cost_items if not pd.isna(v))\n\n    if valid_sum <= 0:\n        fallback_cols = [\n            ("固定成本", ["fixed_cost", "固定成本"]),\n            ("能耗成本", ["energy_cost", "能耗成本"]),\n            ("碳排放成本", ["carbon_cost", "碳排放成本"]),\n            ("等待成本", ["wait_cost", "等待成本"]),\n            ("迟到成本", ["late_cost", "迟到成本"]),\n        ]\n        cost_items = []\n        for name, col_cands in fallback_cols:\n            col = get_existing_col(route_df, col_cands)\n            val = route_df[col].sum() if col else 0.0\n            cost_items.append((name, safe_float(val)))\n\n    cost_df = pd.DataFrame(cost_items, columns=["成本项", "金额_元"])\n    total = safe_float(cost_df["金额_元"].sum())\n    if total <= 0:\n        total = 1.0\n    cost_df["占比"] = cost_df["金额_元"] / total\n    cost_df["金额_元_格式"] = cost_df["金额_元"].map(money_fmt)\n    cost_df["占比_格式"] = (cost_df["占比"] * 100).map(percent_fmt)\n    return cost_df\n\n\ndef draw_cost_donut(cost_df: pd.DataFrame, out_dir: Path):\n    values = cost_df["金额_元"].to_numpy(dtype=float)\n    labels = cost_df["成本项"].tolist()\n    total = max(values.sum(), 1.0)\n\n    color_list = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]\n\n    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=160)\n    wedges, _ = ax.pie(\n        values,\n        startangle=90,\n        counterclock=False,\n        colors=color_list[:len(values)],\n        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=1.5),\n    )\n\n    legend_labels = [\n        f"{name}：{money_fmt(v)} 元（{v / total * 100:.2f}%）"\n        for name, v in zip(labels, values)\n    ]\n    ax.legend(\n        wedges,\n        legend_labels,\n        loc="center left",\n        bbox_to_anchor=(1.02, 0.5),\n        frameon=False,\n        fontsize=10,\n    )\n    ax.text(0, 0.08, "总成本", ha="center", va="center", fontsize=14, fontweight="bold")\n    ax.text(0, -0.08, f"{money_fmt(total)} 元", ha="center", va="center", fontsize=12)\n    ax.set_title(f"图A  {PROB_CN}成本构成环形图", fontsize=14, fontweight="bold", pad=16)\n    plt.tight_layout()\n    plt.savefig(out_dir / f"{PROB_CN}_图A_成本构成环形图.png", dpi=300, bbox_inches="tight")\n    plt.close(fig)\n\n    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=160)\n    ax.pie(\n        values,\n        labels=labels,\n        autopct=lambda pct: f"{pct:.2f}%",\n        startangle=90,\n        counterclock=False,\n        colors=color_list[:len(values)],\n        wedgeprops=dict(edgecolor="white", linewidth=1.2),\n        textprops=dict(fontsize=10),\n    )\n    ax.set_title(f"图A（备选）  {PROB_CN}成本构成饼图", fontsize=14, fontweight="bold", pad=14)\n    plt.tight_layout()\n    plt.savefig(out_dir / f"{PROB_CN}_图A_成本构成饼图.png", dpi=300, bbox_inches="tight")\n    plt.close(fig)\n\n\n# =========================\n# 3. 图B：载重利用率分布\n# =========================\n\ndef make_load_rate_tables(route_df: pd.DataFrame):\n    s = pd.to_numeric(route_df["load_rate_weight_percent"], errors="coerce").dropna().copy()\n    if len(s) == 0:\n        raise ValueError("路线汇总表中缺少可用的 load_rate_weight_percent，无法生成图B。")\n\n    # 将极个别 >100 的情况保留到最后一档，便于检查是否存在轻微超载/数值误差。\n    bins = [0, 20, 40, 60, 80, 90, 100, 110, np.inf]\n    labels = [\n        "0%-20%", "20%-40%", "40%-60%", "60%-80%",\n        "80%-90%", "90%-100%", "100%-110%", ">110%"\n    ]\n    cat = pd.cut(s, bins=bins, labels=labels, right=True, include_lowest=True)\n    dist = cat.value_counts(sort=False).reset_index()\n    dist.columns = ["利用率区间", "车辆数"]\n    dist["占比"] = dist["车辆数"] / max(dist["车辆数"].sum(), 1)\n    dist["占比_格式"] = (dist["占比"] * 100).map(percent_fmt)\n\n    stat = pd.DataFrame({\n        "统计量": ["样本数", "均值", "中位数", "最小值", "最大值", "标准差"],\n        "数值": [\n            len(s), s.mean(), s.median(), s.min(), s.max(), s.std(ddof=1) if len(s) > 1 else 0.0\n        ]\n    })\n    stat["数值_格式"] = stat["数值"].apply(lambda x: f"{x:.2f}")\n    return dist, stat\n\n\ndef draw_load_hist(load_dist: pd.DataFrame, out_dir: Path):\n    fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=160)\n    bars = ax.bar(load_dist["利用率区间"], load_dist["车辆数"], color="#4C78A8", edgecolor="white")\n    ax.set_xlabel("重量利用率区间")\n    ax.set_ylabel("车辆数 / 条")\n    ax.set_title(f"图B  {PROB_CN}载重利用率分布图", fontsize=14, fontweight="bold", pad=12)\n    ax.grid(axis="y", linestyle="--", alpha=0.35)\n    ax.set_axisbelow(True)\n\n    for rect, val in zip(bars, load_dist["车辆数"]):\n        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.3, str(int(val)), ha="center", va="bottom", fontsize=9)\n\n    plt.xticks(rotation=20)\n    plt.tight_layout()\n    plt.savefig(out_dir / f"{PROB_CN}_图B_载重利用率直方图.png", dpi=300, bbox_inches="tight")\n    plt.close(fig)\n\n\n# =========================\n# 4. 图C：路径距离/时长分布图\n# =========================\n\ndef make_route_distribution_summary(route_df: pd.DataFrame):\n    distance_col = get_existing_col(route_df, ["distance_km", "总里程_km", "distance"])\n    time_col = get_existing_col(route_df, ["travel_minutes", "总行驶时长_min", "travel_time", "time_minutes"])\n\n    items = []\n    for name, col in [("路径里程(km)", distance_col), ("行驶时长(min)", time_col)]:\n        if col is None:\n            continue\n        s = pd.to_numeric(route_df[col], errors="coerce").dropna()\n        if len(s) == 0:\n            continue\n        items.extend([\n            [name, "样本数", len(s)],\n            [name, "均值", s.mean()],\n            [name, "中位数", s.median()],\n            [name, "最小值", s.min()],\n            [name, "最大值", s.max()],\n            [name, "标准差", s.std(ddof=1) if len(s) > 1 else 0.0],\n        ])\n\n    stat_df = pd.DataFrame(items, columns=["指标", "统计量", "数值"])\n    stat_df["数值_格式"] = stat_df["数值"].apply(lambda x: f"{x:.2f}")\n    return stat_df\n\n\ndef draw_route_distribution(route_df: pd.DataFrame, out_dir: Path):\n    route_id_col = get_existing_col(route_df, ["route_id", "路径编号", "id"])\n    if route_id_col is None:\n        route_df = route_df.copy()\n        route_df["route_id"] = range(1, len(route_df) + 1)\n        route_id_col = "route_id"\n\n    distance_col = get_existing_col(route_df, ["distance_km", "总里程_km", "distance"])\n    time_col = get_existing_col(route_df, ["travel_minutes", "总行驶时长_min", "travel_time", "time_minutes"])\n\n    if distance_col is None and time_col is None:\n        raise ValueError("路线汇总表中既没有 distance_km，也没有 travel_minutes，无法生成图C。")\n\n    # 图C-1：里程/时长双箱线图（用两个独立子图画在一张图内）\n    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6), dpi=160)\n\n    if distance_col is not None:\n        s1 = pd.to_numeric(route_df[distance_col], errors="coerce").dropna()\n        axes[0].boxplot(s1, vert=True, patch_artist=True, boxprops=dict(facecolor="#4C78A8", alpha=0.6))\n        axes[0].set_title("路径里程分布")\n        axes[0].set_ylabel("里程 / km")\n        axes[0].grid(axis="y", linestyle="--", alpha=0.35)\n    else:\n        axes[0].axis("off")\n\n    if time_col is not None:\n        s2 = pd.to_numeric(route_df[time_col], errors="coerce").dropna()\n        axes[1].boxplot(s2, vert=True, patch_artist=True, boxprops=dict(facecolor="#F58518", alpha=0.6))\n        axes[1].set_title("路径时长分布")\n        axes[1].set_ylabel("时长 / min")\n        axes[1].grid(axis="y", linestyle="--", alpha=0.35)\n    else:\n        axes[1].axis("off")\n\n    fig.suptitle(f"图C-1  {PROB_CN}路径里程/时长箱线图", fontsize=14, fontweight="bold", y=0.98)\n    plt.tight_layout()\n    plt.savefig(out_dir / f"{PROB_CN}_图C_路径距离时长分布图.png", dpi=300, bbox_inches="tight")\n    plt.close(fig)\n\n    # 图C-2：单独里程箱线图\n    if distance_col is not None:\n        s1 = pd.to_numeric(route_df[distance_col], errors="coerce").dropna()\n        fig, ax = plt.subplots(figsize=(5.2, 5.8), dpi=160)\n        ax.boxplot(s1, vert=True, patch_artist=True, boxprops=dict(facecolor="#4C78A8", alpha=0.6))\n        ax.set_ylabel("路径里程 / km")\n        ax.set_title(f"图C-2  {PROB_CN}路径里程箱线图", fontsize=14, fontweight="bold", pad=12)\n        ax.grid(axis="y", linestyle="--", alpha=0.35)\n        plt.tight_layout()\n        plt.savefig(out_dir / f"{PROB_CN}_图C_路径里程箱线图.png", dpi=300, bbox_inches="tight")\n        plt.close(fig)\n\n    # 图C-3：单独时长箱线图\n    if time_col is not None:\n        s2 = pd.to_numeric(route_df[time_col], errors="coerce").dropna()\n        fig, ax = plt.subplots(figsize=(5.2, 5.8), dpi=160)\n        ax.boxplot(s2, vert=True, patch_artist=True, boxprops=dict(facecolor="#F58518", alpha=0.6))\n        ax.set_ylabel("路径行驶时长 / min")\n        ax.set_title(f"图C-3  {PROB_CN}路径时长箱线图", fontsize=14, fontweight="bold", pad=12)\n        ax.grid(axis="y", linestyle="--", alpha=0.35)\n        plt.tight_layout()\n        plt.savefig(out_dir / f"{PROB_CN}_图C_路径时长箱线图.png", dpi=300, bbox_inches="tight")\n        plt.close(fig)\n\n    # 图C-4：里程最长Top30条路径条形图\n    if distance_col is not None:\n        top_n = min(30, len(route_df))\n        top_df = route_df[[route_id_col, distance_col]].copy()\n        top_df[distance_col] = pd.to_numeric(top_df[distance_col], errors="coerce")\n        top_df = top_df.dropna().sort_values(distance_col, ascending=False).head(top_n)\n        top_df = top_df.sort_values(distance_col, ascending=True)\n\n        fig, ax = plt.subplots(figsize=(9.0, 8.0), dpi=160)\n        ax.barh(top_df[route_id_col].astype(str), top_df[distance_col], color="#4C78A8")\n        ax.set_xlabel("路径里程 / km")\n        ax.set_ylabel("路径编号")\n        ax.set_title(f"图C-4  {PROB_CN}里程最长的前30条路径", fontsize=14, fontweight="bold", pad=12)\n        ax.grid(axis="x", linestyle="--", alpha=0.35)\n        plt.tight_layout()\n        plt.savefig(out_dir / f"{PROB_CN}_图C_路径里程条形图_Top30.png", dpi=300, bbox_inches="tight")\n        plt.close(fig)\n\n\n# =========================\n# 5. 表D：典型配送路径微观明细表\n# =========================\n\ndef select_typical_routes(route_df: pd.DataFrame, n=2):\n    df = route_df.copy()\n\n    route_id_col = get_existing_col(df, ["route_id", "路径编号", "id"])\n    if route_id_col and route_id_col != "route_id":\n        df["route_id"] = df[route_id_col]\n    elif route_id_col is None:\n        df["route_id"] = range(1, len(df) + 1)\n\n    late_cost_col = get_existing_col(df, ["late_cost", "迟到成本", "late_penalty"])\n    wait_cost_col = get_existing_col(df, ["wait_cost", "等待成本"])\n    distance_col = get_existing_col(df, ["distance_km", "总里程_km", "distance"])\n\n    if late_cost_col:\n        cand = df[pd.to_numeric(df[late_cost_col], errors="coerce").fillna(0) <= 1e-9].copy()\n    else:\n        cand = df.copy()\n\n    if len(cand) < n:\n        cand = df.copy()\n\n    sort_cols = []\n    ascending = []\n\n    if "load_rate_weight_percent" in cand.columns:\n        sort_cols.append("load_rate_weight_percent")\n        ascending.append(False)\n    if late_cost_col:\n        sort_cols.append(late_cost_col)\n        ascending.append(True)\n    if wait_cost_col:\n        sort_cols.append(wait_cost_col)\n        ascending.append(True)\n    if distance_col:\n        sort_cols.append(distance_col)\n        ascending.append(True)\n\n    if sort_cols:\n        cand = cand.sort_values(sort_cols, ascending=ascending)\n\n    selected = cand.head(n).copy()\n    selected["选取说明"] = "载重利用率较高且迟到较少/无迟到的典型路径"\n    return selected\n\n\ndef make_table_d(selected_routes: pd.DataFrame, arc_df: pd.DataFrame):\n    detail = arc_df.copy()\n    arc_route_id_col = get_existing_col(detail, ["route_id", "路径编号", "id"], required=True)\n    if arc_route_id_col != "route_id":\n        detail["route_id"] = detail[arc_route_id_col]\n\n    selected_ids = selected_routes["route_id"].tolist()\n    detail = detail[detail["route_id"].isin(selected_ids)].copy()\n\n    seq_col = get_existing_col(detail, ["seq", "弧段序号", "step", "order"])\n    if seq_col and seq_col != "seq":\n        detail["seq"] = detail[seq_col]\n\n    detail["_典型路径顺序"] = pd.Categorical(detail["route_id"], categories=selected_ids, ordered=True)\n    if "seq" in detail.columns:\n        detail = detail.sort_values(["_典型路径顺序", "seq"])\n    else:\n        detail = detail.sort_values(["_典型路径顺序"])\n    detail = detail.drop(columns=["_典型路径顺序"])\n\n    route_keep = [\n        "route_id", "vehicle_type", "route", "stop_count", "original_customer_count",\n        "total_weight", "capacity_weight", "load_rate_weight_percent",\n        "total_volume", "departure_time", "return_time", "distance_km",\n        "travel_minutes", "fixed_cost", "wait_cost", "late_cost",\n        "energy_cost", "carbon_cost", "total_cost", "feasible", "选取说明"\n    ]\n    route_keep = [c for c in route_keep if c in selected_routes.columns]\n    summary = selected_routes[route_keep].copy()\n\n    route_info_cols = [\n        "route_id", "route", "total_weight", "capacity_weight",\n        "load_rate_weight_percent", "distance_km", "travel_minutes",\n        "total_cost", "选取说明"\n    ]\n    route_info_cols = [c for c in route_info_cols if c in selected_routes.columns]\n    if route_info_cols:\n        detail = detail.merge(\n            selected_routes[route_info_cols],\n            on="route_id",\n            how="left",\n            suffixes=("", "_route")\n        )\n\n    summary_cn = summary.rename(columns={\n        "route_id": "路径编号",\n        "vehicle_type": "车型",\n        "route": "路径序列",\n        "stop_count": "停靠点数",\n        "original_customer_count": "原始客户数",\n        "total_weight": "总载重_kg",\n        "capacity_weight": "车辆载重上限_kg",\n        "load_rate_weight_percent": "载重利用率_%",\n        "total_volume": "总体积",\n        "departure_time": "出发时刻",\n        "return_time": "返回时刻",\n        "distance_km": "总里程_km",\n        "travel_minutes": "总行驶时长_min",\n        "fixed_cost": "固定成本_元",\n        "wait_cost": "等待成本_元",\n        "late_cost": "迟到成本_元",\n        "energy_cost": "能耗成本_元",\n        "carbon_cost": "碳排放成本_元",\n        "total_cost": "总成本_元",\n        "feasible": "是否可行",\n    })\n\n    detail_cn = detail.rename(columns={\n        "route_id": "路径编号",\n        "vehicle_type": "车型",\n        "seq": "弧段序号",\n        "from": "起点",\n        "to": "终点",\n        "distance_km": "弧段距离_km",\n        "depart_time": "出发时间_min",\n        "arrival_time": "到达时间_min",\n        "wait_min": "等待时间_min",\n        "late_min": "迟到时间_min",\n        "service_start": "服务开始_min",\n        "service_end": "服务结束_min",\n        "avg_speed_kmh": "平均速度_km_h",\n        "arc_energy_cost": "弧段能耗成本_元",\n        "arc_carbon_cost": "弧段碳排放成本_元",\n        "remaining_weight_before_arc": "弧段前剩余载重_kg",\n        "remaining_weight_after_arc": "弧段后剩余载重_kg",\n        "from_clock": "起点离开时刻",\n        "arrival_clock": "到达时刻",\n        "service_start_clock": "服务开始时刻",\n        "service_end_clock": "服务结束时刻",\n        "route": "路径序列",\n        "total_weight": "总载重_kg",\n        "capacity_weight": "车辆载重上限_kg",\n        "load_rate_weight_percent": "载重利用率_%",\n        "distance_km_route": "路径总里程_km",\n        "travel_minutes": "路径总行驶时长_min",\n        "total_cost": "路径总成本_元",\n    })\n\n    return summary_cn, detail_cn\n\n\n# =========================\n# 6. 输出说明\n# =========================\n\ndef write_readme(out_dir: Path, root: Path, input_files: dict, selected_routes: pd.DataFrame):\n    ids = selected_routes["route_id"].tolist() if "route_id" in selected_routes.columns else []\n\n    text = f"""{PROB_CN}补充图表输出说明\n========================================\n输入目录：{root}\n\n读取文件：\n- 结果摘要：{input_files.get(\'summary_json\')}\n- 路线汇总：{input_files.get(\'route_file\')}\n- 弧段明细：{input_files.get(\'arc_file\')}\n- 车辆参数：{input_files.get(\'vehicle_file\')}\n\n已生成图表：\n1. {PROB_CN}_图A_成本构成环形图.png：展示固定成本、能耗成本、碳排放成本、等待成本、迟到成本占比。\n2. {PROB_CN}_图A_成本构成饼图.png：成本构成的饼图版本，可作为备选。\n3. {PROB_CN}_图B_载重利用率直方图.png：展示各配送路径/车辆的载重利用率分布。\n4. {PROB_CN}_图C_路径距离时长分布图.png：展示路径里程与行驶时长的总体分布。\n5. {PROB_CN}_图C_路径里程箱线图.png：展示路径里程的离散程度。\n6. {PROB_CN}_图C_路径时长箱线图.png：展示路径行驶时长的离散程度。\n7. {PROB_CN}_图C_路径里程条形图_Top30.png：展示里程最长的前30条路径。\n\n已生成表格：\n1. {PROB_CN}_图A_成本构成数据.xlsx\n2. {PROB_CN}_图B_载重利用率分布数据.xlsx\n3. {PROB_CN}_图B_载重利用率统计摘要.xlsx\n4. {PROB_CN}_图C_路径里程时长统计摘要.xlsx\n5. {PROB_CN}_表D_典型配送路径微观明细.xlsx\n6. {PROB_CN}_载重利用率增强路线汇总.xlsx\n\n表D自动选取的典型路径 route_id：{ids}\n选取逻辑：优先选取载重利用率较高且迟到成本为0的路径；若不足，则选取载重利用率最高的路径。\n表D包含两个工作表：\n- 典型路径汇总：给出被选中路径的总体成本、载重利用率、里程和时长。\n- 弧段到达明细：给出每条典型路径的弧段、到达时刻、等待/迟到时间、弧段成本等微观信息。\n\n论文使用建议：\n- 图A可放在结果分析的成本构成小节；\n- 图B可用于说明车辆装载效率；\n- 图C可用于说明路径任务分配的均衡性与离散程度；\n- 表D可放在正文或附录，用于展示典型配送路径的微观到达时序。\n"""\n    with open(out_dir / f"{PROB_CN}_画图输出说明.txt", "w", encoding="utf-8") as f:\n        f.write(text)\n\n\n# =========================\n# 7. 主程序\n# =========================\n\ndef main():\n    parser = argparse.ArgumentParser(description=f"生成{PROB_CN}论文补充图表")\n    parser.add_argument(\n        "--root",\n        type=str,\n        default=None,\n        help="问题二输出文件夹路径；不填则从 q2_画图.py 所在目录及其子目录自动查找",\n    )\n    parser.add_argument(\n        "--outdir",\n        type=str,\n        default=DEFAULT_OUTDIR,\n        help=f"输出图表文件夹名称，默认 {DEFAULT_OUTDIR}",\n    )\n    args = parser.parse_args()\n\n    set_chinese_font()\n\n    start_dir = Path(args.root).expanduser().resolve() if args.root else Path(__file__).resolve().parent\n    root = find_result_root(start_dir)\n    out_dir = root / args.outdir\n    out_dir.mkdir(parents=True, exist_ok=True)\n\n    summary_json_file, route_file, arc_file, vehicle_file = detect_input_files(root)\n    input_files = {\n        "summary_json": str(summary_json_file) if summary_json_file else "未找到",\n        "route_file": str(route_file) if route_file else "未找到",\n        "arc_file": str(arc_file) if arc_file else "未找到",\n        "vehicle_file": str(vehicle_file) if vehicle_file else "未找到",\n    }\n\n    print(f">>> 脚本起始查找目录：{start_dir}")\n    print(f">>> 已定位输入目录：{root}")\n    print(f">>> 输出目录：{out_dir}")\n    print(f">>> 结果摘要文件：{summary_json_file}")\n    print(f">>> 路线汇总文件：{route_file}")\n    print(f">>> 弧段明细文件：{arc_file}")\n\n    summary_json = read_json(summary_json_file)\n    route_df = read_route_summary(route_file)\n    arc_df = read_arc_detail(arc_file)\n    vehicle_df = read_vehicle_params(vehicle_file)\n\n    route_df = add_capacity_and_load_rate(route_df, vehicle_df)\n\n    # 图A\n    cost_df = make_cost_data(summary_json, route_df)\n    draw_cost_donut(cost_df, out_dir)\n    cost_out = cost_df[["成本项", "金额_元", "占比_格式"]].rename(columns={"占比_格式": "占比"})\n    save_excel(out_dir / f"{PROB_CN}_图A_成本构成数据.xlsx", {"成本构成": cost_out})\n\n    # 图B\n    load_dist, load_stat = make_load_rate_tables(route_df)\n    draw_load_hist(load_dist, out_dir)\n    save_excel(out_dir / f"{PROB_CN}_图B_载重利用率分布数据.xlsx", {"载重利用率分布": load_dist})\n    save_excel(out_dir / f"{PROB_CN}_图B_载重利用率统计摘要.xlsx", {"统计摘要": load_stat})\n\n    # 图C\n    route_stat = make_route_distribution_summary(route_df)\n    draw_route_distribution(route_df, out_dir)\n    save_excel(out_dir / f"{PROB_CN}_图C_路径里程时长统计摘要.xlsx", {"统计摘要": route_stat})\n\n    # 表D\n    selected_routes = select_typical_routes(route_df, n=2)\n    table_d_summary, table_d_detail = make_table_d(selected_routes, arc_df)\n    save_excel(\n        out_dir / f"{PROB_CN}_表D_典型配送路径微观明细.xlsx",\n        {\n            "典型路径汇总": table_d_summary,\n            "弧段到达明细": table_d_detail,\n        }\n    )\n\n    extra_cols = [\n        "route_id", "vehicle_type", "route", "total_weight", "capacity_weight",\n        "load_rate_weight_percent", "distance_km", "travel_minutes",\n        "wait_cost", "late_cost", "total_cost"\n    ]\n    extra_cols = [c for c in extra_cols if c in route_df.columns]\n    if extra_cols:\n        save_excel(out_dir / f"{PROB_CN}_载重利用率增强路线汇总.xlsx", {"路线汇总": route_df[extra_cols]})\n\n    write_readme(out_dir, root, input_files, selected_routes)\n\n    print(f">>> 已完成{PROB_CN}论文补充图表生成。")\n    print(f">>> 图A、图B、图C、表D均已输出到：{out_dir}")\n    if "route_id" in selected_routes.columns:\n        print(f">>> 表D自动选取的典型路径 route_id：{selected_routes[\'route_id\'].tolist()}")\n\n\nif __name__ == "__main__":\n    main()\n'


def generate_q2_paper_charts(output_dir: Path):
    """
    在问题二主程序完成输出后，自动生成论文补充图表。
    注意：
    - 本函数在主求解、局部搜索、后处理、鲁棒仿真全部结束后才执行；
    - 使用独立命名空间执行画图脚本，避免污染主求解器函数名、随机数和模型逻辑；
    - 读取目录就是本次 q2 输出目录；
    - 输出子目录为 output_dir / "q2_论文补充图表"。
    """
    import sys as _sys

    _plot_ns = {
        "__name__": "__q2_embedded_plotter__",
        "__file__": str(Path(__file__).resolve()),
    }

    exec(_Q2_EMBEDDED_PLOTTER_CODE, _plot_ns)

    _old_argv = list(_sys.argv)
    try:
        _sys.argv = [
            _old_argv[0] if _old_argv else "q2.py",
            "--root", str(output_dir),
            "--outdir", "q2_论文补充图表",
        ]
        _plot_ns["main"]()
    finally:
        _sys.argv = _old_argv


# =========================
# 命令行与主函数
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="问题二：绿色配送区限行情景下车辆调度（改进 ALNS + 局部优化 + 鲁棒仿真）")
    parser.add_argument("--data-dir", type=str, default=".", help="原始数据所在文件夹")
    parser.add_argument("--outdir", type=str, default=r"D:\A_Question\问题二输出", help="结果输出文件夹")
    parser.add_argument("--iterations", type=int, default=2000, help="ALNS 迭代次数")
    parser.add_argument("--seed", type=int, default=3407, help="随机种子")
    parser.add_argument("--sim-n", type=int, default=50, help="鲁棒仿真次数")
    parser.add_argument("--skip-local-search", action="store_true", help="是否跳过局部搜索")
    parser.add_argument(
        "--split-policy",
        type=str,
        default=SPLIT_POLICY,
        choices=["safe", "balanced", "compact"],
        help="超容量客户拆分策略：safe=稳定兼容；balanced=先安全拆分再有限合并；compact=激进压缩"
    )
    parser.add_argument(
        "--green-policy-mode",
        type=str,
        default=GREEN_POLICY_MODE,
        choices=["wait", "strict"],
        help="绿色区限行处理方式：strict=燃油车限行时段入区/穿区直接禁行；wait=等待到16:00后进入（仅作对照）"
    )
    return parser.parse_args()


def main():
    global SPLIT_POLICY, GREEN_POLICY_MODE
    args = parse_args()
    SPLIT_POLICY = args.split_policy
    GREEN_POLICY_MODE = args.green_policy_mode
    base_dir = Path(args.data_dir).resolve()
    outdir = Path(args.outdir).resolve()

    print("[1/6] 读取并构造标准化实例...")
    print(f"  当前超容量客户拆分策略: {SPLIT_POLICY}")
    instance = load_instance(base_dir)
    comp = instance.get("service_node_compatibility_summary", {})
    if comp:
        print("  服务节点车型兼容性摘要:")
        print(f"  - 服务节点数: {comp.get('service_node_count')}")
        print(f"  - 必须使用3000kg级车辆的节点数: {comp.get('need_3000_vehicle_node_count')}")
        print(f"  - 只能依赖EV_3000容积能力的节点数: {comp.get('only_ev3000_node_count')}")
        print(f"  - 无任何车型可服务节点数: {comp.get('no_feasible_vehicle_node_count')}")
    green_summary = instance.get("green_zone_summary", {})
    if green_summary:
        print("  绿色配送区政策摘要:")
        print(f"  - 区内服务节点数: {green_summary.get('green_service_node_count')}")
        print(f"  - 区内原始客户数: {green_summary.get('green_original_customer_count')}")
        print(f"  - 穿区弧段标记数: {green_summary.get('cross_green_arc_count')}")
        print(f"  - 限行时段: {green_summary.get('green_ban_start_clock')}—{green_summary.get('green_ban_end_clock')}")
        print(f"  - 政策处理方式: {green_summary.get('policy_mode')}")

    print("[2/6] 生成初始可行解...")
    init_solution = construct_initial_solution(instance)
    init_cost, _, init_feasible = solution_cost(init_solution, instance)
    print(f"  初始解成本: {init_cost:.2f} | 可行: {init_feasible}")

    print("[3/6] 运行改进 ALNS 主求解器...")
    best_solution, history = alns_solve(instance, init_solution, iterations=args.iterations, seed=args.seed)
    best_cost, _, best_feasible = solution_cost(best_solution, instance)
    print(f"  ALNS 最优成本: {best_cost:.2f} | 可行: {best_feasible}")

    if not args.skip_local_search:
        print("[4/6] 运行 TS/VNS 风格局部精修...")
        improved = local_search_vns(best_solution, instance, max_passes=2)

        # 关键：局部搜索返回后强制刷新缓存再核验
        refresh_solution_cache(improved, instance)
        imp_cost, _, imp_feasible = solution_cost(improved, instance)

        if imp_feasible and imp_cost <= best_cost:
            best_solution = improved
            best_cost = imp_cost
            print(f"  局部搜索后成本: {best_cost:.2f}")
        else:
            print("  局部搜索未取得更优解或不可行，保留 ALNS 结果。")
    else:
        print("[4/6] 已跳过局部搜索。")

    print("[4.5/6] 执行合法降成本后处理（车型重分配 + 路线合并）...")
    reduced = merge_routes_postprocess(best_solution, instance, max_passes=1)
    refresh_solution_cache(reduced, instance)
    reduced_cost, _, reduced_feasible = solution_cost(reduced, instance)
    if reduced_feasible and solution_covers_all_nodes(reduced, instance) and reduced_cost <= best_cost + 1e-8:
        best_solution = reduced
        best_cost = reduced_cost
        print(f"  降成本后处理后成本: {best_cost:.2f}")
    else:
        print("  降成本后处理未取得更优解，保留原结果。")

    print("[5/6] 进行鲁棒仿真验证...")

    # 关键：鲁棒仿真前，强制刷新最终解缓存并重新检查可行性
    refresh_solution_cache(best_solution, instance)
    final_check_cost, _, final_check_feasible = solution_cost(best_solution, instance)
    cover = coverage_report(best_solution, instance)

    print("  服务节点覆盖检查：")
    print(f"  应覆盖节点数: {cover['expected_node_count']}")
    print(f"  实际节点数: {cover['actual_node_count']}")
    print(f"  唯一节点数: {cover['unique_actual_node_count']}")
    print(f"  缺失节点数: {cover['missing_node_count']}")
    print(f"  额外节点数: {cover['extra_node_count']}")
    print(f"  是否有重复: {cover['has_duplicate_node']}")
    print(f"  应配送总重量: {cover['expected_weight']:.2f} kg")
    print(f"  实际路线总重量: {cover['actual_weight']:.2f} kg")
    print(f"  应配送总体积: {cover['expected_volume']:.3f} m³")
    print(f"  实际路线总体积: {cover['actual_volume']:.3f} m³")
    print(f"  实际启用车辆数: {cover['used_vehicle_count']} 辆")
    print(f"  启用车辆总载重能力: {cover['used_capacity_weight']:.2f} kg")
    print(f"  启用车辆总体积能力: {cover['used_capacity_volume']:.3f} m³")

    if not solution_covers_all_nodes(best_solution, instance):
        print("  严重错误：最终方案没有覆盖所有服务节点，当前结果不可用于论文。")
        print("  缺失节点示例:", cover["missing_nodes"][:30])
        return

    if not final_check_feasible:
        print("  警告：最终方案在确定性评估下不可行，跳过鲁棒仿真。")
        robust = {
            "n_sim": 0,
            "mean_total_cost": 1e18,
            "std_total_cost": 0,
            "mean_total_late_min": 0,
            "mean_total_wait_min": 0,
            "mean_total_carbon_kg": 0,
            "mean_late_route_count": 0,

            # 新增字段
            "expected_total_cost": 1e18,
            "mean_service_reliability": 0,
            "mean_on_time_node_count": 0,
            "mean_late_node_count": 0,
            "total_service_node_count": 0,

            "raw": pd.DataFrame(),
        }
    else:
        robust = robust_simulation(best_solution, instance, n_sim=args.sim_n, seed=args.seed)
        print(f"  鲁棒仿真平均总成本: {robust['mean_total_cost']:.2f}")
        print(f"  鲁棒仿真服务可靠度: {robust['mean_service_reliability'] * 100:.2f}%")

    print("[6/6] 输出文件...")
    save_outputs(instance, best_solution, history, robust, outdir)

    print("[6.5/6] 生成论文补充图表...")
    try:
        generate_q2_paper_charts(outdir)
    except Exception as e:
        print(f"  警告：论文补充图表生成失败，但主求解结果已正常输出。错误信息：{e}")

    print(f"完成。结果已输出到：{outdir}")


if __name__ == "__main__":
    main()
