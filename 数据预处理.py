# -*- coding: utf-8 -*-
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


MAX_WEIGHT = 3000.0
MAX_VOLUME = 15.0
GREEN_RADIUS = 10.0


def find_file(base_dir: Path, candidates):
    for name in candidates:
        path = base_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到文件，候选名称：{candidates}")


def time_to_minutes(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, pd.Timestamp):
        return int(x.hour) * 60 + int(x.minute)
    s = str(x).strip()
    if not s:
        return np.nan
    parts = s.split(":")
    if len(parts) >= 2:
        return int(parts[0]) * 60 + int(parts[1])
    raise ValueError(f"无法解析时间：{x}")


def minutes_to_hhmm(x):
    if pd.isna(x):
        return ""
    x = int(round(float(x)))
    h = x // 60
    m = x % 60
    return f"{h:02d}:{m:02d}"


def iqr_thresholds(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = max(0.0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr
    return {"q1": float(q1), "q3": float(q3), "iqr": float(iqr), "lower": float(lower), "upper": float(upper)}


def impute_orders(orders):
    df = orders.copy()

    df["重量_原始"] = df["重量"]
    df["体积_原始"] = df["体积"]
    df["缺失处理方式"] = ""
    df["缺失处理层级"] = ""

    complete = df[df["重量"].notna() & df["体积"].notna() & (df["重量"] > 0) & (df["体积"] > 0)].copy()
    complete["体重比_vw"] = complete["体积"] / complete["重量"]

    global_ratio = float(complete["体重比_vw"].median())

    customer_ratio = (
        complete.groupby("目标客户编号")["体重比_vw"]
        .agg(["median", "count"])
        .reset_index()
        .rename(columns={"median": "客户体重比中位数", "count": "客户完整样本数"})
    )

    df = df.merge(customer_ratio, on="目标客户编号", how="left")

    def choose_ratio(row):
        if pd.notna(row["客户体重比中位数"]) and row["客户完整样本数"] >= 2:
            return float(row["客户体重比中位数"]), "客户内中位数"
        return global_ratio, "全局中位数"

    for idx, row in df.iterrows():
        ratio, level = choose_ratio(row)
        if pd.isna(row["重量"]) and pd.notna(row["体积"]) and ratio > 0:
            df.at[idx, "重量"] = row["体积"] / ratio
            df.at[idx, "缺失处理方式"] = "由体积反推重量"
            df.at[idx, "缺失处理层级"] = level
        elif pd.isna(row["体积"]) and pd.notna(row["重量"]) and ratio > 0:
            df.at[idx, "体积"] = row["重量"] * ratio
            df.at[idx, "缺失处理方式"] = "由重量反推体积"
            df.at[idx, "缺失处理层级"] = level

    del df["客户体重比中位数"]
    del df["客户完整样本数"]
    return df, global_ratio


def preprocess():
    base_dir = Path(__file__).resolve().parent

    order_path = find_file(base_dir, ["订单信息.xlsx"])
    dist_path = find_file(base_dir, ["距离矩阵.xlsx"])
    coord_path = find_file(base_dir, ["客户坐标信息.xlsx", "客户坐标信息(2).xlsx"])
    time_path = find_file(base_dir, ["时间窗.xlsx"])

    orders = pd.read_excel(order_path)
    dist_raw = pd.read_excel(dist_path)
    coords = pd.read_excel(coord_path)
    tw = pd.read_excel(time_path)

    orders.columns = [str(c).strip() for c in orders.columns]
    dist_raw.columns = [str(c).strip() for c in dist_raw.columns]
    coords.columns = [str(c).strip() for c in coords.columns]
    tw.columns = [str(c).strip() for c in tw.columns]

    for c in ["订单编号", "重量", "体积", "目标客户编号"]:
        if c not in orders.columns:
            raise ValueError(f"订单表缺少必要列：{c}")

    for c in ["类型", "ID", "X (km)", "Y (km)"]:
        if c not in coords.columns:
            raise ValueError(f"坐标表缺少必要列：{c}")

    for c in ["客户编号", "开始时间", "结束时间"]:
        if c not in tw.columns:
            raise ValueError(f"时间窗表缺少必要列：{c}")

    if "客户" not in dist_raw.columns:
        raise ValueError("距离矩阵缺少首列“客户”")

    orders["订单编号"] = pd.to_numeric(orders["订单编号"], errors="coerce").astype("Int64")
    orders["目标客户编号"] = pd.to_numeric(orders["目标客户编号"], errors="coerce").astype("Int64")
    orders["重量"] = pd.to_numeric(orders["重量"], errors="coerce")
    orders["体积"] = pd.to_numeric(orders["体积"], errors="coerce")

    coords["ID"] = pd.to_numeric(coords["ID"], errors="coerce").astype("Int64")
    coords["X (km)"] = pd.to_numeric(coords["X (km)"], errors="coerce")
    coords["Y (km)"] = pd.to_numeric(coords["Y (km)"], errors="coerce")

    tw["客户编号"] = pd.to_numeric(tw["客户编号"], errors="coerce").astype("Int64")
    tw["开始分钟"] = tw["开始时间"].apply(time_to_minutes)
    tw["结束分钟"] = tw["结束时间"].apply(time_to_minutes)

    issues = {"明显错误": [], "需要关注的问题": []}

    if orders["订单编号"].duplicated().any():
        issues["明显错误"].append("订单编号存在重复。")
    if orders["目标客户编号"].isna().any():
        issues["明显错误"].append("订单表存在无法识别的目标客户编号。")
    if (coords["ID"].duplicated().any()):
        issues["明显错误"].append("坐标表中的ID存在重复。")
    if (tw["客户编号"].duplicated().any()):
        issues["明显错误"].append("时间窗表中的客户编号存在重复。")
    if ((tw["开始分钟"] >= tw["结束分钟"]).fillna(False)).any():
        issues["明显错误"].append("存在开始时间晚于或等于结束时间的客户时间窗。")

    row_ids = pd.to_numeric(dist_raw["客户"], errors="coerce")
    col_ids = pd.to_numeric(pd.Index(dist_raw.columns[1:]), errors="coerce")
    dist_mat = dist_raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").copy()
    dist_mat.index = row_ids
    dist_mat.columns = col_ids

    if dist_mat.shape[0] != dist_mat.shape[1]:
        issues["明显错误"].append("距离矩阵不是方阵。")
    if set(dist_mat.index.dropna()) != set(dist_mat.columns.dropna()):
        issues["明显错误"].append("距离矩阵行列节点集合不一致。")

    diag = np.diag(dist_mat.to_numpy())
    if not np.allclose(diag, 0, atol=1e-8, equal_nan=False):
        issues["明显错误"].append("距离矩阵主对角线不全为0。")

    symmetry_diff = np.nanmax(np.abs(dist_mat.to_numpy() - dist_mat.to_numpy().T))
    if symmetry_diff > 1e-6:
        issues["需要关注的问题"].append(f"距离矩阵存在不对称，最大差值约为 {symmetry_diff:.6f}。")

    if (dist_mat.to_numpy() < 0).any():
        issues["明显错误"].append("距离矩阵存在负值。")

    missing_before = {
        "订单_重量缺失": int(orders["重量"].isna().sum()),
        "订单_体积缺失": int(orders["体积"].isna().sum()),
        "订单_任一关键字段缺失": int((orders["重量"].isna() | orders["体积"].isna()).sum()),
        "坐标_X缺失": int(coords["X (km)"].isna().sum()),
        "坐标_Y缺失": int(coords["Y (km)"].isna().sum()),
        "时间窗_开始缺失": int(tw["开始分钟"].isna().sum()),
        "时间窗_结束缺失": int(tw["结束分钟"].isna().sum()),
        "距离矩阵缺失": int(dist_mat.isna().sum().sum()),
    }

    orders_clean, global_ratio = impute_orders(orders)

    orders_clean["非正重量标记"] = (orders_clean["重量"] <= 0).fillna(False)
    orders_clean["非正体积标记"] = (orders_clean["体积"] <= 0).fillna(False)

    wt_thr = iqr_thresholds(orders_clean["重量"])
    vol_thr = iqr_thresholds(orders_clean["体积"])

    orders_clean["重量_IQR离群标记"] = orders_clean["重量"] > wt_thr["upper"]
    orders_clean["体积_IQR离群标记"] = orders_clean["体积"] > vol_thr["upper"]
    orders_clean["单笔超最大载重标记"] = orders_clean["重量"] > MAX_WEIGHT
    orders_clean["单笔超最大容积标记"] = orders_clean["体积"] > MAX_VOLUME

    orders_clean["订单问题标签"] = ""
    for idx, row in orders_clean.iterrows():
        tags = []
        if row["缺失处理方式"]:
            tags.append(row["缺失处理方式"])
        if row["非正重量标记"]:
            tags.append("非正重量")
        if row["非正体积标记"]:
            tags.append("非正体积")
        if row["重量_IQR离群标记"]:
            tags.append("重量统计离群")
        if row["体积_IQR离群标记"]:
            tags.append("体积统计离群")
        if row["单笔超最大载重标记"]:
            tags.append("单笔超最大载重")
        if row["单笔超最大容积标记"]:
            tags.append("单笔超最大容积")
        orders_clean.at[idx, "订单问题标签"] = "；".join(tags)

    order_customers = set(orders_clean["目标客户编号"].dropna().astype(int))
    coord_customers = set(coords.loc[coords["类型"] == "客户", "ID"].dropna().astype(int))
    tw_customers = set(tw["客户编号"].dropna().astype(int))

    zero_demand_customers = sorted(coord_customers - order_customers)
    missing_in_coords = sorted(order_customers - coord_customers)
    missing_in_tw = sorted(order_customers - tw_customers)

    if zero_demand_customers:
        issues["需要关注的问题"].append(f"存在坐标表中有但订单表中未出现的客户，共 {len(zero_demand_customers)} 个。")
    if missing_in_coords:
        issues["明显错误"].append(f"订单中存在坐标表未收录的客户编号：{missing_in_coords}")
    if missing_in_tw:
        issues["明显错误"].append(f"订单中存在时间窗表未收录的客户编号：{missing_in_tw}")

    customer_demand = (
        orders_clean.groupby("目标客户编号", dropna=False)
        .agg(
            订单数=("订单编号", "count"),
            总重量=("重量", "sum"),
            总体积=("体积", "sum"),
            缺失修补订单数=("缺失处理方式", lambda s: int((s != "").sum())),
            重量离群订单数=("重量_IQR离群标记", "sum"),
            体积离群订单数=("体积_IQR离群标记", "sum"),
        )
        .reset_index()
        .rename(columns={"目标客户编号": "客户编号"})
    )

    customer_master = (
        coords.loc[coords["类型"] == "客户", ["ID", "X (km)", "Y (km)"]]
        .rename(columns={"ID": "客户编号"})
        .merge(tw[["客户编号", "开始时间", "结束时间", "开始分钟", "结束分钟"]], on="客户编号", how="left")
        .merge(customer_demand, on="客户编号", how="left")
    )

    customer_master["订单数"] = customer_master["订单数"].fillna(0).astype(int)
    customer_master["总重量"] = customer_master["总重量"].fillna(0.0)
    customer_master["总体积"] = customer_master["总体积"].fillna(0.0)
    customer_master["缺失修补订单数"] = customer_master["缺失修补订单数"].fillna(0).astype(int)
    customer_master["重量离群订单数"] = customer_master["重量离群订单数"].fillna(0).astype(int)
    customer_master["体积离群订单数"] = customer_master["体积离群订单数"].fillna(0).astype(int)

    customer_master["时间窗宽度_分钟"] = customer_master["结束分钟"] - customer_master["开始分钟"]
    customer_master["是否零需求客户"] = customer_master["订单数"] == 0
    customer_master["是否超最大单车载重"] = customer_master["总重量"] > MAX_WEIGHT
    customer_master["是否超最大单车容积"] = customer_master["总体积"] > MAX_VOLUME
    customer_master["最低所需服务次数"] = np.maximum(
        np.ceil(customer_master["总重量"] / MAX_WEIGHT),
        np.ceil(customer_master["总体积"] / MAX_VOLUME),
    ).fillna(0).astype(int)
    customer_master["到原点距离"] = np.sqrt(customer_master["X (km)"] ** 2 + customer_master["Y (km)"] ** 2)
    customer_master["是否位于绿色区_原点半径10"] = customer_master["到原点距离"] <= GREEN_RADIUS

    green_count = int(customer_master["是否位于绿色区_原点半径10"].sum())
    if green_count != 30:
        issues["需要关注的问题"].append(
            f"按坐标直接计算，原点半径10圆内客户数为 {green_count}，与题面描述30个客户不一致。"
        )

    preprocess_summary = {
        "输入文件": {
            "订单信息": order_path.name,
            "距离矩阵": dist_path.name,
            "客户坐标信息": coord_path.name,
            "时间窗": time_path.name,
        },
        "缺失值统计_处理前": missing_before,
        "缺失值处理说明": {
            "策略": "优先使用同一客户完整订单的体积/重量比中位数进行定向插补；若该客户完整样本不足2条，则回退到全局体积/重量比中位数。",
            "全局体积重量比中位数": global_ratio,
            "重量缺失补重量": int(((orders["重量"].isna()) & orders["体积"].notna()).sum()),
            "体积缺失补体积": int(((orders["体积"].isna()) & orders["重量"].notna()).sum()),
            "补后重量缺失": int(orders_clean["重量"].isna().sum()),
            "补后体积缺失": int(orders_clean["体积"].isna().sum()),
        },
        "异常阈值说明": {
            "业务硬阈值": {
                "单笔订单超最大载重_kg": MAX_WEIGHT,
                "单笔订单超最大容积_m3": MAX_VOLUME,
                "客户聚合需求超最大单车容量": "用于识别后续需拆分配送的客户，不直接删除",
            },
            "统计阈值_IQR": {
                "重量": wt_thr,
                "体积": vol_thr,
                "说明": "IQR阈值用于打标签，不直接删除离群记录；物流题中的大值可能是真实大订单。",
            },
        },
        "初步检查结果": {
            "零需求客户数": len(zero_demand_customers),
            "零需求客户编号": zero_demand_customers,
            "绿色区客户数_原点半径10": green_count,
            "单笔超最大载重订单数": int(orders_clean["单笔超最大载重标记"].sum()),
            "单笔超最大容积订单数": int(orders_clean["单笔超最大容积标记"].sum()),
            "重量统计离群订单数": int(orders_clean["重量_IQR离群标记"].sum()),
            "体积统计离群订单数": int(orders_clean["体积_IQR离群标记"].sum()),
            "客户聚合后超最大载重客户数": int(customer_master["是否超最大单车载重"].sum()),
            "客户聚合后超最大容积客户数": int(customer_master["是否超最大单车容积"].sum()),
        },
        "问题清单": issues,
    }

    output_dir = base_dir / "预处理结果"
    output_dir.mkdir(exist_ok=True)

    orders_out = orders_clean.copy()
    customer_out = customer_master.copy()
    tw_out = tw.copy()

    tw_out["开始时间_标准"] = tw_out["开始分钟"].apply(minutes_to_hhmm)
    tw_out["结束时间_标准"] = tw_out["结束分钟"].apply(minutes_to_hhmm)

    orders_out.to_excel(output_dir / "订单信息_初步预处理.xlsx", index=False)
    orders_out.to_csv(output_dir / "订单信息_初步预处理.csv", index=False, encoding="utf-8-sig")

    customer_out.to_excel(output_dir / "客户主表_初步预处理.xlsx", index=False)
    customer_out.to_csv(output_dir / "客户主表_初步预处理.csv", index=False, encoding="utf-8-sig")

    tw_out.to_excel(output_dir / "时间窗_标准化.xlsx", index=False)
    tw_out.to_csv(output_dir / "时间窗_标准化.csv", index=False, encoding="utf-8-sig")

    dist_raw.to_excel(output_dir / "距离矩阵_原样导出.xlsx", index=False)

    with open(output_dir / "预处理摘要.json", "w", encoding="utf-8") as f:
        json.dump(preprocess_summary, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("A题问题一：初步数据预处理摘要")
    lines.append("=" * 40)
    lines.append("")
    lines.append("1. 已执行的预处理操作")
    lines.append(" - 统一四张表字段名与数据类型。")
    lines.append(" - 将时间窗转换为分钟制，便于后续路径时间递推。")
    lines.append(" - 检查订单、坐标、时间窗、距离矩阵之间的编号与结构一致性。")
    lines.append(" - 对订单表中的重量/体积单变量缺失进行定向插补。")
    lines.append(" - 使用业务阈值与IQR阈值对异常值做标记，不直接删除。")
    lines.append(" - 将订单表聚合为客户主表，用于后续问题一建模。")
    lines.append("")
    lines.append("2. 缺失值处理策略")
    lines.append(" - 优先使用同一客户完整订单的体积/重量比中位数进行插补。")
    lines.append(" - 若该客户完整样本不足2条，则使用全局体积/重量比中位数回退插补。")
    lines.append(" - 本阶段原则：能补则补，不轻易删除订单。")
    lines.append("")
    lines.append("3. 异常值处理策略")
    lines.append(f" - 业务硬阈值：单笔重量>{MAX_WEIGHT:.0f}kg、单笔体积>{MAX_VOLUME:.0f}m³，先打标签，不直接删除。")
    lines.append(" - 统计阈值：使用IQR上界识别重量/体积统计离群点，仅作风险标记。")
    lines.append(" - 客户聚合需求超过最大车型容量时，不视为脏数据，而视为后续模型需拆分配送的信号。")
    lines.append("")
    lines.append("4. 本阶段发现的主要问题")
    for k, vals in preprocess_summary["问题清单"].items():
        lines.append(f"【{k}】")
        if vals:
            for v in vals:
                lines.append(f" - {v}")
        else:
            lines.append(" - 无")
    lines.append("")
    lines.append("5. 关键统计")
    key = preprocess_summary["初步检查结果"]
    for kk, vv in key.items():
        lines.append(f" - {kk}: {vv}")
    lines.append("")
    lines.append("6. 输出文件")
    lines.append(f" - {output_dir / '订单信息_初步预处理.xlsx'}")
    lines.append(f" - {output_dir / '客户主表_初步预处理.xlsx'}")
    lines.append(f" - {output_dir / '时间窗_标准化.xlsx'}")
    lines.append(f" - {output_dir / '预处理摘要.json'}")

    with open(output_dir / "预处理摘要.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("")
    print("预处理完成。")


if __name__ == "__main__":
    preprocess()
