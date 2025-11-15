"""Utility functions for the route planning heuristics module."""

from __future__ import annotations

import math
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import folium
import numpy as np
import pandas as pd

from pyvrp import Model
from pyvrp.stop import MaxRuntime, MultipleCriteria, NoImprovement

from ..api_config import (
    DIST_MATRIX_TYPE,
    EXTRACT_INFO_TYPE,
    MAX_CUS_VISITED,
    PYVRP_MAX_NO_IMPROVE_ITERS,
    PYVRP_MAX_RUNTIME,
    VEH_CAPACITY,
)
from ..log_service import add_n35_log_info
from .logger import LOGGER

def plan_distribution_routes01(
    n34_id: str,
    df_cus: pd.DataFrame,
    df_depot: pd.DataFrame,
    station_code: str,
    *,
    vehicle_capacity: int = 100,
    demand_col: Optional[str] = None,
    lon_col_cus: str = "MD03_RETAIL_CUST_LON",
    lat_col_cus: str = "MD03_RETAIL_CUST_LAT",
    lon_col_dep: str = "MD03_SHIPMENT_STATION_LON",
    lat_col_dep: str = "MD03_SHIPMENT_STATION_LAT",
    cust_code_col: str = "BB_RETAIL_CUSTOMER_CODE",
    algo: str = "nearest",            # "nearest" | "random"
    rng_seed: Optional[int] = 42,      # 随机算法的种子
    vehicle_speed_kmh: float = 50.0,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """对 *指定发货点* 的客户进行车辆与访问顺序分配（简单启发式：最近邻 / 随机装载）。

    参数：
    - algo: "nearest"（最近邻，O(n^2)）或 "random"（随机顺序装载，O(n)）
    - rng_seed: 随机算法的种子，便于复现实验
    约定：
    - 每个客户需求默认为 1（若提供 `demand_col` 则使用该列，否则按 1 计）。
    - 车辆容量默认为 100，车辆数量不限（装满/无可装后自动开新车）。
    - 距离采用 Haversine 球面距离（单位：km），总距离=出车→首点 + 点间 + 末点→回仓。
    - 车辆速度默认为 50 km/h，用于估算配送时长。

    返回：
    - routes_dict: {route_id: { 'station': str, 'vehicle_idx': int, 'sequence': [cust_code,...], 'load': int, 'distance_km': float }}
    - routes_df:   路线汇总表（DataFrame）
    """
    import time
    t0 = time.time()
    if algo not in {"nearest", "random"}:
        raise ValueError(f"未知 algo={algo}，可选 'nearest' 或 'random'")
    # add_n35_log_info(n34_id, "ALG_05", f"线路规划-开始 最近邻启发式 station={station_code}, cap={vehicle_capacity}")

    # ==== 1) 取该发货点坐标 ====
    dep = df_depot[df_depot["MD03_SHIPMENT_STATION_CODE"].astype(str) == str(station_code)]
    if dep.empty:
        raise ValueError(f"未找到发货点(站) station_code={station_code}")
    dep = dep.iloc[0]
    dep_lon = float(dep[lon_col_dep])
    dep_lat = float(dep[lat_col_dep])

    # ==== 2) 取该发货点下客户 ====
    # 若 df_cus 有分配列，优先过滤；否则默认取所有客户（由调用方保证只给本仓的客户）
    if "MD03_DIST_STATION_CODE" in df_cus.columns:
        cus_all = df_cus[df_cus["MD03_DIST_STATION_CODE"].astype(str) == str(station_code)].copy()
    else:
        cus_all = df_cus.copy()
    if cus_all.empty:
        return {}, pd.DataFrame(columns=["route_id","station","vehicle_idx","load","distance_km","num_stops"])  # 无客户

    # 需求：默认 1，或取指定列
    if demand_col and demand_col in cus_all.columns:
        demands = cus_all[demand_col].fillna(1).astype(float).clip(lower=0.0)
    else:
        demands = pd.Series(1.0, index=cus_all.index)

    # 坐标清洗：只保留有坐标的客户
    mask_xy = cus_all[lon_col_cus].notna() & cus_all[lat_col_cus].notna()
    if not mask_xy.all():
        # 记录/剔除缺坐标
        missing = cus_all.loc[~mask_xy, cust_code_col].astype(str).tolist()
        # if missing:
        #     add_n35_log_info(n34_id, "ALG_05_WARN", f"剔除缺坐标客户 {len(missing)} 个：{missing[:5]}{'...' if len(missing)>5 else ''}")
        cus_all = cus_all.loc[mask_xy].copy()
        demands = demands.loc[cus_all.index]
    if cus_all.empty:
        return {}, pd.DataFrame(columns=["route_id","station","vehicle_idx","load","distance_km","num_stops"])  # 无可达客户

    # 提取需要的列
    cus_all["_lon"] = cus_all[lon_col_cus].astype(float)
    cus_all["_lat"] = cus_all[lat_col_cus].astype(float)
    cus_all["_code"] = cus_all[cust_code_col].astype(str)

    # ==== 3) 距离函数 ====
    def haversine_km(lon1, lat1, lon2, lat2):
        R = 6371.0
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return 2 * R * math.asin(math.sqrt(a))

    def route_distance_and_segments_km(seq_idx_local):
        """返回 (total_km, segments_km)
        segments_km 仅包括：仓->首点 + 点间，**不包含**末点->回仓。
        total_km 则包含回仓距离，用于总里程与总时长计算。
        """
        if not seq_idx_local:
            return 0.0, []
        dist_sum = 0.0
        segs = []
        # 仓到首点
        lon1, lat1 = dep_lon, dep_lat
        lon2 = float(cus_all.at[seq_idx_local[0], "_lon"])
        lat2 = float(cus_all.at[seq_idx_local[0], "_lat"])
        d01 = haversine_km(lon1, lat1, lon2, lat2)
        dist_sum += d01
        segs.append(d01)
        # 点间
        for a, b in zip(seq_idx_local, seq_idx_local[1:]):
            dab = haversine_km(
                float(cus_all.at[a, "_lon"]), float(cus_all.at[a, "_lat"]),
                float(cus_all.at[b, "_lon"]), float(cus_all.at[b, "_lat"]),
            )
            dist_sum += dab
            segs.append(dab)
        # 末点回仓（不加入 segs，只加入总和）
        lon_last = float(cus_all.at[seq_idx_local[-1], "_lon"])
        lat_last = float(cus_all.at[seq_idx_local[-1], "_lat"])
        dist_sum += haversine_km(lon_last, lat_last, dep_lon, dep_lat)
        return dist_sum, segs

    # ==== 4) 构造多条路线（支持最近邻 / 随机） ====
    unserved = set(cus_all.index.tolist())
    vehicle_idx = 0
    routes = []  # list of dicts

    if algo == "nearest":
        # 最近邻 + 容量约束
        while unserved:
            vehicle_idx += 1
            load = 0.0
            seq_idx = []  # 索引序列
            seq_codes = []
            distance = 0.0
            prev_segs = []  # 逐段距离（仓->首点、点间），不含回仓
            # 从仓出发
            cur_lon, cur_lat = dep_lon, dep_lat
            while unserved:
                feas = [i for i in unserved if load + float(demands.loc[i]) <= vehicle_capacity]
                if not feas:
                    break
                best_i = None
                best_d = float("inf")
                for i in feas:
                    lon_i, lat_i = float(cus_all.at[i, "_lon"]), float(cus_all.at[i, "_lat"])
                    d = haversine_km(cur_lon, cur_lat, lon_i, lat_i)
                    if d < best_d:
                        best_d, best_i = d, i
                seq_idx.append(best_i)
                seq_codes.append(cus_all.at[best_i, "_code"])
                load += float(demands.loc[best_i])
                distance += best_d
                prev_segs.append(best_d)
                cur_lon, cur_lat = float(cus_all.at[best_i, "_lon"]), float(cus_all.at[best_i, "_lat"])
                unserved.remove(best_i)
            distance += haversine_km(cur_lon, cur_lat, dep_lon, dep_lat)
            load_ratio = float(round((load / vehicle_capacity * 100.0) if vehicle_capacity > 0 else 0.0, 2))
            duration_h = float(round(distance / vehicle_speed_kmh, 3))
            prev_durations = [round(d / vehicle_speed_kmh, 3) for d in prev_segs]
            route_id = f"{station_code}-V{vehicle_idx:03d}"
            route_name = route_id
            vehicle_id = f"{station_code}-VEH{vehicle_idx:03d}"
            routes.append({
                "route_id": route_id,
                "route_name": route_name,
                "station": str(station_code),
                "vehicle_idx": vehicle_idx,
                "vehicle_id": vehicle_id,
                "sequence": seq_codes,
                "load": int(load),
                "vehicle_capacity": int(vehicle_capacity),
                "load_ratio": load_ratio,
                "distance_km": float(round(distance, 3)),
                "duration_h": duration_h,
                "num_stops": len(seq_codes),
                "prev_distance_km": [round(x, 3) for x in prev_segs],
                "prev_duration_h": prev_durations,
            })
    else:
        # 随机顺序装载 + 简单分车（更快，序列随机）
        rng = np.random.default_rng(rng_seed)
        order = list(cus_all.index.tolist())
        rng.shuffle(order)
        cur_route_idx = []
        cur_codes = []
        cur_load = 0.0
        for i in order:
            d_i = float(demands.loc[i])
            if cur_load + d_i <= vehicle_capacity:
                cur_route_idx.append(i)
                cur_codes.append(cus_all.at[i, "_code"])
                cur_load += d_i
            else:
                total_km, segs = route_distance_and_segments_km(cur_route_idx)
                load_ratio = float(round((cur_load / vehicle_capacity * 100.0) if vehicle_capacity > 0 else 0.0, 2))
                duration_h = float(round(total_km / vehicle_speed_kmh, 3))
                prev_durations = [round(d / vehicle_speed_kmh, 3) for d in segs]
                route_id = f"{station_code}-V{vehicle_idx+1:03d}"
                route_name = route_id
                vehicle_id = f"{station_code}-VEH{vehicle_idx+1:03d}"
                routes.append({
                    "route_id": route_id,
                    "route_name": route_name,
                    "station": str(station_code),
                    "vehicle_idx": vehicle_idx + 1,
                    "vehicle_id": vehicle_id,
                    "sequence": cur_codes,
                    "load": int(cur_load),
                    "vehicle_capacity": int(vehicle_capacity),
                    "load_ratio": load_ratio,
                    "distance_km": float(round(total_km, 3)),
                    "duration_h": duration_h,
                    "num_stops": len(cur_codes),
                    "prev_distance_km": [round(x, 3) for x in segs],
                    "prev_duration_h": prev_durations,
                })
                # 开新车并把当前客户装进去
                vehicle_idx += 1
                cur_route_idx = [i]
                cur_codes = [cus_all.at[i, "_code"]]
                cur_load = d_i
        # 收尾最后一辆
        if cur_codes:
            total_km, segs = route_distance_and_segments_km(cur_route_idx)
            load_ratio = float(round((cur_load / vehicle_capacity * 100.0) if vehicle_capacity > 0 else 0.0, 2))
            duration_h = float(round(total_km / vehicle_speed_kmh, 3))
            prev_durations = [round(d / vehicle_speed_kmh, 3) for d in segs]
            route_id = f"{station_code}-V{vehicle_idx+1:03d}"
            route_name = route_id
            vehicle_id = f"{station_code}-VEH{vehicle_idx+1:03d}"
            routes.append({
                "route_id": route_id,
                "route_name": route_name,
                "station": str(station_code),
                "vehicle_idx": vehicle_idx + 1,
                "vehicle_id": vehicle_id,
                "sequence": cur_codes,
                "load": int(cur_load),
                "vehicle_capacity": int(vehicle_capacity),
                "load_ratio": load_ratio,
                "distance_km": float(round(total_km, 3)),
                "duration_h": duration_h,
                "num_stops": len(cur_codes),
                "prev_distance_km": [round(x, 3) for x in segs],
                "prev_duration_h": prev_durations,
            })

    # 汇总表
    routes_df = pd.DataFrame(routes, columns=[
        "route_id","route_name","station","vehicle_idx","vehicle_id",
        "load","vehicle_capacity","load_ratio","distance_km","duration_h","num_stops","sequence",
        "prev_distance_km","prev_duration_h"
    ])
    routes_df = routes_df.sort_values(["vehicle_idx"]).reset_index(drop=True)

    # 转换成 dict 便于后续 out_n36_n37 使用
    routes_dict = {r["route_id"]: r for r in routes}

    elapsed = time.time() - t0
    n = len(cus_all)
    algoname = "最近邻" if algo == "nearest" else "随机装载"
    complexity = "O(n^2)（逐次找最近）" if algo == "nearest" else "O(n)（一次洗牌+线性装载）"
    add_n35_log_info(
        n34_id,
        "ALG_06",
        f"线路规划-完成 {algoname} station={station_code}, 车辆数={len(routes)}, 客户数={n}, 复杂度≈{complexity}, 用时={elapsed:.3f}s"
    )
    return routes_dict, routes_df

def _scale_to_uint(x, xmin, xmax, bits=31):
    if pd.isna(x) or xmax == xmin:
        return 0
    x = (x - xmin) / (xmax - xmin)
    x = min(max(x, 0.0), 1.0)
    return int(round(x * ((1 << bits) - 1)))

def _zorder_key(lon, lat, xmin, xmax, ymin, ymax, bits=31):
    # Morton (Z-order) key for空间填充排序
    xi = _scale_to_uint(lon, xmin, xmax, bits)
    yi = _scale_to_uint(lat, ymin, ymax, bits)
    key = 0
    for b in range(bits):
        key |= ((xi >> b) & 1) << (2 * b + 1)
        key |= ((yi >> b) & 1) << (2 * b)
    return key

def _haversine_km_fast(lon1, lat1, lon2, lat2):
    import math
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def _allocate_counts_by_weight(weights, total_k):
    """
    按权重比例把 total_k 条线路分配到各分组；返回每个分组的条数（整数），和为 total_k。
    """
    if len(weights) == 0:
        return []
    w = np.array(weights, dtype=float)
    w[w <= 0] = 0.0
    if w.sum() == 0:
        base = total_k // len(w)
        arr = [base] * len(w)
        for i in range(total_k - base * len(w)):
            arr[i % len(w)] += 1
        return arr
    ratios = w / w.sum()
    raw = ratios * total_k
    floor = np.floor(raw).astype(int)
    remain = total_k - floor.sum()
    frac = raw - floor
    order = np.argsort(-frac)
    for i in range(remain):
        floor[order[i]] += 1
    # 确保有权重的分组至少分到1条
    for i, wi in enumerate(w):
        if wi > 0 and floor[i] == 0:
            j = int(np.argmax(floor))
            floor[j] -= 1
            floor[i] = 1
    return floor.tolist()

def repartition_customers_to_n_routes(
    db_n28: pd.DataFrame,
    db_n26: pd.DataFrame,
    n_routes: int,
    *,
    balance: str = "送货户数均衡",      # "送货户数均衡" | "送货数量均衡" | "送货里程均衡"
    group_by_station: bool = True,
    speed_kmh: float = 50.0,
) -> pd.DataFrame:
    """
    将零售户重分为 n_routes 条线路，支持三种均衡：
    - 送货户数均衡：每线客户数均衡
    - 送货数量均衡：每线需求量(AVG_ORDER_QTY)均衡
    - 送货里程均衡：每线“工作量”均衡（用 站点->客户 的一程哈弗辛距离作为快速代理）

    若 group_by_station=True，先按发货点拆分，再把 n_routes 按各发货点总权重比例分配。
    返回列：route_id, station, sequence, prev_distance_km, prev_duration_h, distance_km, duration_h,
           vehicle_id, vehicle_capacity, load_ratio, route_name
    （车辆与装载率先留空，后续你可以再补算）
    """
    df = db_n28.copy()

    need = ["MD03_DIST_STATION_CODE","BB_RETAIL_CUSTOMER_CODE",
            "MD03_RETAIL_CUST_LON","MD03_RETAIL_CUST_LAT",
            "MD03_SHIPMENT_STATION_LON","MD03_SHIPMENT_STATION_LAT"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"db_n28 缺少必要列: {miss}")

    for c in ["MD03_RETAIL_CUST_LON","MD03_RETAIL_CUST_LAT",
              "MD03_SHIPMENT_STATION_LON","MD03_SHIPMENT_STATION_LAT"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if balance == "送货数量均衡":
        if "AVG_ORDER_QTY" not in df.columns:
            raise ValueError("balance='送货数量均衡' 需要列 AVG_ORDER_QTY")
        df["__weight__"] = pd.to_numeric(df["AVG_ORDER_QTY"], errors="coerce").fillna(0.0)
    elif balance == "送货里程均衡":
        df["__weight__"] = df.apply(
            lambda r: _haversine_km_fast(r["MD03_SHIPMENT_STATION_LON"], r["MD03_SHIPMENT_STATION_LAT"],
                                         r["MD03_RETAIL_CUST_LON"], r["MD03_RETAIL_CUST_LAT"])
                      if pd.notna(r["MD03_SHIPMENT_STATION_LON"]) and pd.notna(r["MD03_SHIPMENT_STATION_LAT"])
                      and pd.notna(r["MD03_RETAIL_CUST_LON"]) and pd.notna(r["MD03_RETAIL_CUST_LAT"]) else 0.0,
            axis=1
        )
    elif balance == "送货户数均衡":
        df["__weight__"] = 1.0
    elif balance == "送货时长均衡":
        df["__weight__"] = 1.0 # todo
    else:
        df["__weight__"] = 1.0

    groups = list(df.groupby("MD03_DIST_STATION_CODE")) if group_by_station else [(None, df)]
    group_weights = [g["__weight__"].sum() for _, g in groups]
    group_counts = _allocate_counts_by_weight(group_weights, n_routes)

    route_rows = []
    route_index_global = 1

    for (gkey, gdf), k in zip(groups, group_counts):
        if k == 0 or gdf.empty:
            continue

        lon_min, lon_max = float(gdf["MD03_RETAIL_CUST_LON"].min()), float(gdf["MD03_RETAIL_CUST_LON"].max())
        lat_min, lat_max = float(gdf["MD03_RETAIL_CUST_LAT"].min()), float(gdf["MD03_RETAIL_CUST_LAT"].max())

        gdf = gdf.copy()
        gdf["__z__"] = gdf.apply(
            lambda r: _zorder_key(r["MD03_RETAIL_CUST_LON"], r["MD03_RETAIL_CUST_LAT"],
                                  lon_min, lon_max, lat_min, lat_max, bits=20)
                      if pd.notna(r["MD03_RETAIL_CUST_LON"]) and pd.notna(r["MD03_RETAIL_CUST_LAT"]) else 0,
            axis=1
        )
        gdf = gdf.sort_values(["__z__"]).reset_index(drop=True)

        total_w = float(gdf["__weight__"].sum())
        target_w = [(i+1) * (total_w / k) for i in range(k-1)] if total_w > 0 else []
        cum_w = gdf["__weight__"].cumsum().to_numpy()
        cut_idx = [int(np.searchsorted(cum_w, tw, side="left")) for tw in target_w]
        segments = np.split(gdf, cut_idx)

        if len(segments) < k:
            segments += [gdf.iloc[0:0]] * (k - len(segments))
        elif len(segments) > k:
            tail = pd.concat(segments[k-1:], axis=0)
            segments = segments[:k-1] + [tail]

        for i, seg in enumerate(segments):
            if seg.empty:
                continue
            seq_ids = seg["BB_RETAIL_CUSTOMER_CODE"].astype(str).tolist()

            st_code = str(seg["MD03_DIST_STATION_CODE"].astype(str).mode(dropna=True).iloc[0]) \
                      if not seg["MD03_DIST_STATION_CODE"].dropna().empty else ""
            st_lon = seg["MD03_SHIPMENT_STATION_LON"].dropna().iloc[0] if not seg["MD03_SHIPMENT_STATION_LON"].dropna().empty else np.nan
            st_lat = seg["MD03_SHIPMENT_STATION_LAT"].dropna().iloc[0] if not seg["MD03_SHIPMENT_STATION_LAT"].dropna().empty else np.nan

            prev_dists, prev_times = [], []
            prev_lon = prev_lat = None

            if len(seq_ids) > 0:
                first_row = seg.loc[seg["BB_RETAIL_CUSTOMER_CODE"].astype(str) == seq_ids[0]]
                if not first_row.empty and pd.notna(st_lon) and pd.notna(st_lat):
                    lon1 = float(first_row["MD03_RETAIL_CUST_LON"].iloc[0])
                    lat1 = float(first_row["MD03_RETAIL_CUST_LAT"].iloc[0])
                    d0 = _haversine_km_fast(st_lon, st_lat, lon1, lat1)
                    prev_dists.append(round(d0, 2))
                    prev_times.append(round(d0 / speed_kmh * 60, 2))
                    prev_lon, prev_lat = lon1, lat1
                else:
                    prev_dists.append(0.0); prev_times.append(0.0)

            for cid in seq_ids[1:]:
                row = seg.loc[seg["BB_RETAIL_CUSTOMER_CODE"].astype(str) == cid]
                if not row.empty and prev_lon is not None and prev_lat is not None:
                    lon2 = float(row["MD03_RETAIL_CUST_LON"].iloc[0])
                    lat2 = float(row["MD03_RETAIL_CUST_LAT"].iloc[0])
                    d = _haversine_km_fast(prev_lon, prev_lat, lon2, lat2)
                    prev_dists.append(round(d, 2))
                    prev_times.append(round(d / speed_kmh * 60, 2))
                    prev_lon, prev_lat = lon2, lat2
                else:
                    prev_dists.append(0.0); prev_times.append(0.0)

            tail_back_km = tail_back_min = 0.0
            if len(seq_ids) > 0 and prev_lon is not None and prev_lat is not None and pd.notna(st_lon) and pd.notna(st_lat):
                tail_back_km = round(_haversine_km_fast(prev_lon, prev_lat, st_lon, st_lat), 2)
                tail_back_min = round(tail_back_km / speed_kmh * 60, 2)

            total_km = round(sum(prev_dists) + tail_back_km, 2)
            total_min = round(sum(prev_times) + tail_back_min, 2)

            route_id = f"{st_code}-{i+1:03d}" if group_by_station else f"R{route_index_global:04d}"
            route_index_global += 1

            route_rows.append({
                "route_id": route_id,
                "station": st_code,
                "vehicle_id": "",
                "vehicle_capacity": "",
                "load_ratio": "",
                "distance_km": total_km,
                "duration_h": total_min,   # 注意：这里单位仍是“分钟”，为兼容你现有字段命名
                "sequence": ",".join(seq_ids),
                "prev_distance_km": ",".join(map(str, prev_dists)),
                "prev_duration_h": ",".join(map(str, prev_times)),
                "route_name": f"线路{route_id}",
            })

    routes_df_balanced = pd.DataFrame(route_rows).sort_values(["station","route_id"]).reset_index(drop=True)
    return routes_df_balanced

def add_route_distances(df_res, df_n28):
    import math
    def haversine_km(lon1, lat1, lon2, lat2):
        R = 6371.0
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return 2 * R * math.asin(math.sqrt(a))

    # 站点到首客户距离和时间
    df_res["station_to_first_customer_distance_km"] = df_res.apply(
        lambda row: round(haversine_km(
            row["MD03_SHIPMENT_STATION_LON"], row["MD03_SHIPMENT_STATION_LAT"],
            row["first_customer_lon"], row["first_customer_lat"]
        ), 2) if pd.notna(row["MD03_SHIPMENT_STATION_LON"]) and pd.notna(row["MD03_SHIPMENT_STATION_LAT"]) and
                     pd.notna(row["first_customer_lon"]) and pd.notna(row["first_customer_lat"]) else 0.0,
        axis=1
    )
    df_res["station_to_first_customer_time_min"] = df_res["station_to_first_customer_distance_km"].apply(
        lambda x: round(x / 50 * 60, 2)
    )
    # 末客户到站点距离和时间
    df_res["last_customer_to_station_distance_km"] = df_res.apply(
        lambda row: round(haversine_km(
            row["last_customer_lon"], row["last_customer_lat"],
            row["MD03_SHIPMENT_STATION_LON"], row["MD03_SHIPMENT_STATION_LAT"]
        ), 2) if pd.notna(row["MD03_SHIPMENT_STATION_LON"]) and pd.notna(row["MD03_SHIPMENT_STATION_LAT"]) and
                     pd.notna(row["last_customer_lon"]) and pd.notna(row["last_customer_lat"]) else 0.0,
        axis=1
    )
    df_res["last_customer_to_station_time_min"] = df_res["last_customer_to_station_distance_km"].apply(
        lambda x: round(x / 10, 2)
    )
    df_res.drop(columns=["first_customer_code", "last_customer_code", "first_customer_lon", "first_customer_lat", "last_customer_lon", "last_customer_lat"], inplace=True)

    # 计算每个客户与上一个客户的距离和时间
    df_res["PREV_CUSTOMER_DISTANCE_KM"] = 0.0
    df_res["PREV_CUSTOMER_TIME_MIN"] = 0.0
    for idx, row in df_res.iterrows():
        cust_codes = row["BB_RETAIL_CUSTOMER_CODE"].split(",")
        cust_coords = df_n28[df_n28["BB_RETAIL_CUSTOMER_CODE"].isin(cust_codes)][["BB_RETAIL_CUSTOMER_CODE", "MD03_RETAIL_CUST_LON", "MD03_RETAIL_CUST_LAT"]]
        cust_coords = cust_coords.dropna(subset=["MD03_RETAIL_CUST_LON", "MD03_RETAIL_CUST_LAT"])
        cust_coords = cust_coords.set_index("BB_RETAIL_CUSTOMER_CODE").astype({"MD03_RETAIL_CUST_LON": float, "MD03_RETAIL_CUST_LAT": float})
        prev_lon, prev_lat = None, None
        distances = []
        times = []
        for c in cust_codes:
            if c in cust_coords.index:
                lon = cust_coords.at[c, "MD03_RETAIL_CUST_LON"]
                lat = cust_coords.at[c, "MD03_RETAIL_CUST_LAT"]
                if prev_lon is not None and prev_lat is not None:
                    dist = haversine_km(prev_lon, prev_lat, lon, lat)
                    time_min = dist / 10
                    distances.append(round(dist, 2))
                    times.append(round(time_min, 2))
                else:
                    distances.append(row["station_to_first_customer_distance_km"])
                    times.append(row["station_to_first_customer_time_min"])
                prev_lon, prev_lat = lon, lat
            else:
                distances.append(0.0)
                times.append(0.0)
        df_res.at[idx, "PREV_CUSTOMER_DISTANCE_KM"] = ",".join(map(str, distances))
        df_res.at[idx, "PREV_CUSTOMER_TIME_MIN"] = ",".join(map(str, times))

    # 总距离和总时间
    df_res["distance_km"] = df_res["PREV_CUSTOMER_DISTANCE_KM"].apply(
        lambda x: sum([float(i) for i in x.split(",") if i.replace(".","",1).isdigit()]) if pd.notna(x) and x else 0.0
    ) + df_res["last_customer_to_station_distance_km"]
    df_res["time_min"] = df_res["PREV_CUSTOMER_TIME_MIN"].apply(
        lambda x: sum([float(i) for i in x.split(",") if i.replace(".","",1).isdigit()]) if pd.notna(x) and x else 0.0
    ) + df_res["last_customer_to_station_time_min"]
    df_res["distance_km"] = df_res["distance_km"].round(2)
    df_res["time_min"] = df_res["time_min"].round(2)
    return df_res

def _haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def reorder_lines_by_nearest(
    df_n28: pd.DataFrame,
    *,
    line_col: str = "LINE_CODES",
    cust_id_col: str = "BB_RETAIL_CUSTOMER_CODE",
    cust_lon_col: str = "MD03_RETAIL_CUST_LON",
    cust_lat_col: str = "MD03_RETAIL_CUST_LAT",
    station_col: str = "MD03_DIST_STATION_CODE",
    station_lon_col: str = "MD03_SHIPMENT_STATION_LON",
    station_lat_col: str = "MD03_SHIPMENT_STATION_LAT",
    keep_unlocated: bool = True,
    rng_seed: int = 0
) -> pd.DataFrame:
    """
    基于 LINE_CODE 对每条线内的零售户按最近邻重排序：
    - 起点=对应发货点坐标
    - 首点=离发货点最近的零售户
    - 之后每步选择距离当前点最近的未访问零售户
    输出：在原 df_n28 基础上新增 ORDER_IDX（从1开始），并提供每线的 LINE_SEQUENCE 汇总列。

    参数：
      keep_unlocated: 若某些零售户缺坐标，设为 True 则将其保留到该线的末尾（原相对顺序不变）；
                      False 则丢弃这些点（只在排序结果里）。
    """
    df = df_n28.copy()
    # 坐标清洗
    df[cust_lon_col] = pd.to_numeric(df[cust_lon_col], errors="coerce")
    df[cust_lat_col] = pd.to_numeric(df[cust_lat_col], errors="coerce")
    df[station_lon_col] = pd.to_numeric(df[station_lon_col], errors="coerce")
    df[station_lat_col] = pd.to_numeric(df[station_lat_col], errors="coerce")

    # 容器：每条线的排序结果（id 列表）与顺序号
    ordered_idx_all = []
    rng = np.random.default_rng(rng_seed)

    # 逐线处理
    for line, g in df.groupby(line_col, sort=False):
        g = g.copy()

        # 该线的发货点（假设同线同仓；如一线多仓，取出现频次最高的一仓）
        st_counts = g[station_col].astype(str).value_counts(dropna=True)
        if st_counts.empty:
            # 无仓ID，则无法从仓出发；直接按经纬度最近邻从“任一点”启动
            depot_lon = depot_lat = None
        else:
            st_pick = st_counts.index[0]
            g_st = g[g[station_col].astype(str) == st_pick]
            # 仓坐标：取该线中该仓的第一行仓坐标
            depot_lon = g_st[station_lon_col].dropna().iloc[0] if not g_st[station_lon_col].dropna().empty else None
            depot_lat = g_st[station_lat_col].dropna().iloc[0] if not g_st[station_lat_col].dropna().empty else None

        # 可参与距离计算的客户
        mask_loc = g[cust_lon_col].notna() & g[cust_lat_col].notna()
        g_loc = g[mask_loc].copy()
        g_noloc = g[~mask_loc].copy()

        ordered_ids = []

        if not g_loc.empty:
            # 构建点集
            pts = g_loc[[cust_lon_col, cust_lat_col]].to_numpy(float)
            ids = g_loc[cust_id_col].astype(str).to_numpy()

            # 选择起点：若有仓坐标，以仓为起点并选择“离仓最近”的首点；否则随机挑一个首点
            if depot_lon is not None and depot_lat is not None:
                d0 = np.array([_haversine_km(depot_lon, depot_lat, x, y) for x, y in pts])
                cur_idx = int(np.argmin(d0))
            else:
                cur_idx = int(rng.integers(len(ids)))  # 无仓坐标时随机选首点以稳定

            visited = np.zeros(len(ids), dtype=bool)
            visited[cur_idx] = True
            ordered_ids.append(ids[cur_idx])

            # 迭代选择最近的未访问点
            cur_lon, cur_lat = pts[cur_idx]
            for _ in range(len(ids) - 1):
                # 计算到所有未访问点的距离
                dists = np.array([_haversine_km(cur_lon, cur_lat, x, y) if not v else np.inf
                                  for (x, y), v in zip(pts, visited)])
                nxt = int(np.argmin(dists))
                if not np.isfinite(dists[nxt]):  # 理论上不会出现
                    break
                visited[nxt] = True
                ordered_ids.append(ids[nxt])
                cur_lon, cur_lat = pts[nxt]

        # 追加缺坐标客户
        if keep_unlocated and not g_noloc.empty:
            # 保留其在原 line 内的相对顺序，且排在最后
            tail_ids = g_noloc[cust_id_col].astype(str).tolist()
            ordered_ids.extend(tail_ids)

        # 将顺序写回
        if ordered_ids:
            order_map = {cid: i+1 for i, cid in enumerate(ordered_ids)}  # 从1开始
            part = g.copy()
            part["ORDER_IDX"] = part[cust_id_col].astype(str).map(order_map).astype("Int64")
        else:
            part = g.copy()
            part["ORDER_IDX"] = pd.NA

        ordered_idx_all.append(part)

    # 汇总
    out = pd.concat(ordered_idx_all, axis=0).sort_values([line_col, "ORDER_IDX"], na_position="last")

    # 每条线增加“LINE_SEQUENCE”（逗号分隔的客户顺序）
    seq = (
        out.sort_values([line_col, "ORDER_IDX"], na_position="last")
          .groupby(line_col, as_index=False)[cust_id_col]
          .apply(lambda s: ",".join(s.astype(str).tolist()))
          .rename(columns={cust_id_col: "LINE_SEQUENCE"})
    )
    out = out.merge(seq, on=line_col, how="left")

    return out

def _to_int_coords(lon, lat, scale=1_000_000):
    # PyVRP 需要整数坐标，放大经纬度后取整
    return int(round(lon * scale)), int(round(lat * scale))

def _euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def reorder_lines_by_pyvrp(
    df_n28: pd.DataFrame,
    *,
    line_col: str = "LINE_CODES",
    cust_id_col: str = "BB_RETAIL_CUSTOMER_CODE",
    cust_lon_col: str = "MD03_RETAIL_CUST_LON",
    cust_lat_col: str = "MD03_RETAIL_CUST_LAT",
    station_col: str = "MD03_DIST_STATION_CODE",
    station_lon_col: str = "MD03_SHIPMENT_STATION_LON",
    station_lat_col: str = "MD03_SHIPMENT_STATION_LAT",
    demand_col: str = None,                # 如 "AVG_ORDER_QTY"；为空则当作无容量约束
    vehicle_capacity = None, # 若提供则开启容量约束，否则给超大容量
    runtime_s: float = 1.0,                # 每条线最大求解时间
    no_improve_iters: int = 200,           # 无改进终止迭代
    keep_unlocated: bool = True,           # 无坐标客户放末尾
    return_to_depot: bool = True,          # 是否回仓
) -> pd.DataFrame:
    """
    用 PyVRP 对每条 LINE 求解访问顺序（单车），写回 ORDER_IDX 与 LINE_SEQUENCE。
    """
    df = df_n28.copy()

    # 基础清洗
    for c in (cust_lon_col, cust_lat_col, station_lon_col, station_lat_col):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if demand_col is not None and demand_col in df.columns:
        df[demand_col] = pd.to_numeric(df[demand_col], errors="coerce").fillna(0.0)
    else:
        demand_col = None  # 视为无容量约束

    parts = []
    for line_id, g in df.groupby(line_col, sort=False):
        g = g.copy()

        # —— 仓信息 —— #
        st_counts = g[station_col].astype(str).value_counts(dropna=True)
        if st_counts.empty:
            # 没有仓 ID：退化为“从任一点出发”的近似（用 PyVRP 也能跑，但没起点含义）
            depot_lon = depot_lat = None
        else:
            st_pick = st_counts.index[0]
            g_st = g[g[station_col].astype(str) == st_pick]
            depot_lon = g_st[station_lon_col].dropna().iloc[0] if not g_st[station_lon_col].dropna().empty else None
            depot_lat = g_st[station_lat_col].dropna().iloc[0] if not g_st[station_lat_col].dropna().empty else None

        # —— 可求解客户 / 无坐标客户 —— #
        mask_loc = g[cust_lon_col].notna() & g[cust_lat_col].notna()
        g_loc = g[mask_loc].copy()
        g_noloc = g[~mask_loc].copy()

        ordered_ids = []

        if not g_loc.empty:
            # 1) 组织整数坐标
            cus_ids = g_loc[cust_id_col].astype(str).tolist()
            cus_xy_f = list(zip(g_loc[cust_lon_col].astype(float), g_loc[cust_lat_col].astype(float)))
            cus_xy_i = [_to_int_coords(lon, lat) for lon, lat in cus_xy_f]

            if depot_lon is not None and depot_lat is not None:
                depot_xy_i = _to_int_coords(float(depot_lon), float(depot_lat))
                has_depot = True
            else:
                # 无仓坐标：临时以首客户当作仓（仅为跑求解；结果仍只取访问客户顺序）
                depot_xy_i = cus_xy_i[0]
                has_depot = False

            # 2) 建模（每条线单车）
            m = Model()
            cap = int(vehicle_capacity) if (vehicle_capacity is not None) else 10**9
            m.add_vehicle_type(1, capacity=cap)

            depot = m.add_depot(x=depot_xy_i[0], y=depot_xy_i[1])

            # 客户需求：若无容量约束就给 delivery=0；否则用实际需求
            deliveries = (g_loc[demand_col].astype(float).tolist() if demand_col else [0]*len(cus_ids))
            clients = []
            for (x, y), dem in zip(cus_xy_i, deliveries):
                clients.append(m.add_client(x=x, y=y, delivery=int(round(dem))))

            # 3) 添加边（欧氏距离；也可改 Manhattan）
            locs = [depot] + clients
            for i, frm in enumerate(locs):
                for j, to in enumerate(locs):
                    d = _euclid((frm.x, frm.y), (to.x, to.y))
                    m.add_edge(frm, to, distance=int(round(d)))  # 用整数距离

            # 4) 求解
            stop = MultipleCriteria([NoImprovement(no_improve_iters),
                                     MaxRuntime(runtime_s)])
            res = m.solve(stop=stop, display=False)
            if not res.is_feasible():
                # 不可行：退回最近邻
                order = list(range(len(cus_ids)))
            else:
                # 取唯一一条路线
                route = res.best.routes()[0]
                visit_idx = list(route.visits())  # 1..n（PyVRP 的“客户索引”从1起）
                order = [i-1 for i in visit_idx]  # 转为 0..n-1

                # 若不回仓：忽略末端回仓（PyVRP 里路线天然起终点为仓）
                # 我们这里只取 visits()，本就不含 0，所以无需处理 return_to_depot

            ordered_ids = [cus_ids[i] for i in order]

        # 5) 追加无坐标客户到末尾
        if keep_unlocated and not g_noloc.empty:
            ordered_ids += g_noloc[cust_id_col].astype(str).tolist()

        # 6) 写回顺序
        if ordered_ids:
            order_map = {cid: i+1 for i, cid in enumerate(ordered_ids)}  # 从1开始
            part = g.copy()
            part["ORDER_IDX"] = part[cust_id_col].astype(str).map(order_map).astype("Int64")
        else:
            part = g.copy()
            part["ORDER_IDX"] = pd.NA

        parts.append(part)

    out = pd.concat(parts, axis=0)

    # 生成 LINE_SEQUENCE
    tmp = (
        out.sort_values([line_col, "ORDER_IDX"], na_position="last")
           .groupby(line_col, as_index=False)[cust_id_col]
           .apply(lambda s: ",".join(s.astype(str).tolist()))
           .rename(columns={cust_id_col: "LINE_SEQUENCE"})
    )
    out = out.merge(tmp, on=line_col, how="left")
    return out

def solve_vrp(gdf_ty_grid, D, G):

    def cov_sol_list(solution_list):
        """
        将 solution_list 转为 DataFrame。

        参数:
        - solution_list: 包含每条路线信息的列表

        返回:
        - DataFrame: 转换后的 DataFrame
        """
        rows = []
        for solution in solution_list:
            # solution多个值时才需要取出
            route = solution.get('route', [])
            route_city = solution.get('rCity', pd.Series(dtype=str))
            route_name = solution.get('rName', pd.Series(dtype=str))
            route_dist_m = solution.get('distance', (0, []))
            route_cost = solution.get('routeCost', (0, 0, 0, (0)))
            route_dist = solution.get('routeDist', (0, []))
            route_time = solution.get('routeTime', (0, []))
            route_stat = solution.get('routeStat', {})
            route_city2 = solution.get('routeCity', pd.Series(dtype=str))
            route_custDemand = solution.get('routeCustDemand', pd.Series(dtype=float))
            rows.append({

                # 装载相关
                'vType': solution.get('vType', 0),
                'vCapacity': solution.get('vCapacity', 0),  # 车载量
                'load_rate': route_stat.get('load_rate', 0),  # 装载率

                # 城市区域\供应商名\客户需求
                # 'route_city': ', '.join(route_city),  # 'route_city2': ', '.join(route_city2.astype(str)),
                # 'route_name': ', '.join(route_name),
                'route_custDemand': ', '.join((route_custDemand).astype(str)),

                # 供应商点数、城市点数相关
                'nb_cust': route_stat.get('nb_cust', 0),
                'nb_city': route_stat.get('nb_city', 0),

                # 费用相关
                'route_cost_total': route_cost[0],  # 费用
                'route_custCost': route_cost[3],  # 费用
                # 'route_cost_fixed': route_cost[1],
                # 'route_cost_variable': route_cost[2],

                # 距离/时间相关
                # 'route_dist_total_model': route_dist_m,     # 待fix model的距离 计算用 结果可能用不到
                'route_dist_total': route_dist[0],  # 距离
                'route_dist_segments': route_dist[1],
                'route_time_total': route_time[0],  # 距离
                'route_time_segments': route_time[1],

                'route_seq': solution.get('routeSeq', []),

                # 体积相关
                'volume': solution.get('routeVolume', 0),  # 体积
                'load_rate_vol': route_stat.get('load_rate_vol', 0),  # 体积
            })
        return pd.DataFrame(rows)

    def plot_data(routes, customer_district, customer_township, customer_code, customer_area, customer_xy, customer_grid, customer_points, customer_demands, sol_list, rid=None, vid=None):
        # 中心设定为 计算所有坐标的中心点作为地图初始位置
        lons = [pt[0] for pt in customer_points]
        lats = [pt[1] for pt in customer_points]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        # 中心设定为 depot点
        # center_lat = customer_points[0][1]
        # center_lon = customer_points[0][0]

        # 创建地图对象，设置初始视角和缩放级别
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='OpenStreetMap')
        # m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles=None)
        # folium.TileLayer(
        #     tiles='https://webrd04.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
        #     # 高德 矢量
        #     attr='高德地图',
        #     name='Gaode Map',
        #     subdomains='1234',
        #     overlay=False,
        #     control=True
        # ).add_to(m)

        with open("./notebook/graph/G_taiyuan.pkl", "rb") as f:
            G = pickle.load(f)

        # 添加 OSM 路网到 folium 地图
        for u, v in G.edges():
            try:
                x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
                x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
                folium.PolyLine(
                    locations=[(y1, x1), (y2, x2)],
                    color='gray',
                    weight=1,
                    opacity=0.5
                ).add_to(m)
            except Exception:
                continue  # 跳过缺失坐标的边

        # 指定颜色调色板
        colormap_list = ['Purple', 'Blue', 'Green', 'Orange', 'Red', 'DarkRed', 'DarkBlue', 'DarkGreen', 'Grey',
                         'Purple', 'Blue', 'Green', 'Orange', 'Red', 'DarkRed', 'DarkBlue', 'DarkGreen']

        # 生成区域到颜色的映射
        unique_areas = list(set(customer_area))
        area_color_map = {area: colormap_list[i] for i, area in enumerate(unique_areas)}  # 颜色映射

        # 绘制多条路线
        for idx, route in enumerate(routes):
            if idx != 0 :
                continue
            folium.PolyLine(
                locations=route,
                weight=3,
                color=colormap_list[idx],
                opacity=0.8,
                popup=f"Route {idx + 1}"
            ).add_to(m)


        # 为每个客户添加圆形标记
        if 1:
            for idx, (dist, town, code, area, xy, grid, point, demand, r, v) in enumerate(
                    zip(customer_district, customer_township, customer_code, customer_area, customer_xy, customer_grid, customer_points, customer_demands, rid, vid)):

                if EXTRACT_INFO_TYPE == 'Osm' or EXTRACT_INFO_TYPE == 'Node':
                    radius = 5 + float(demand) / 2000 * 0.01
                    popup_text = f"索引号:{idx} <br>客户代码: {code}<br>区划: {dist} <br>街道: {town} <br>需求量: {demand}<br>线路号: {r}<br>车辆类型: {v}"
                    folium.CircleMarker(
                        location=[point[1], point[0]],
                        radius=radius,
                        popup=folium.Popup(popup_text, max_width=300),
                        color=area_color_map[area],
                        weight=4,
                        fill=False,
                        fill_opacity=0.99,
                        fill_color=area_color_map[area],
                        opacity=0.99
                    ).add_to(m)

                    popup_text = f"客户代码: {code}"
                    radius = 2
                    folium.CircleMarker(
                        location=[xy[1], xy[0]],
                        radius=radius,
                        popup=folium.Popup(popup_text, max_width=300),
                        color='grey',
                        weight=2,
                        fill=False,
                        fill_opacity=0.99,
                        fill_color='grey',
                        opacity=0.99
                    ).add_to(m)

                    if EXTRACT_INFO_TYPE == 'Node':
                        popup_text = f"索引号:{idx}"
                        radius = 8
                        folium.CircleMarker(
                            location=[grid[1], grid[0]],
                            radius=radius,
                            popup=folium.Popup(popup_text, max_width=300),
                            color='grey',
                            weight=3,
                            fill=False,
                            fill_opacity=0.01,
                            fill_color='grey',
                            opacity=0.01
                        ).add_to(m)

                elif EXTRACT_INFO_TYPE == 'Grid':
                    popup_text = f"索引号:{idx}"
                    radius = 8
                    folium.CircleMarker(
                        location=[grid[1], grid[0]],
                        radius=radius,
                        popup=folium.Popup(popup_text, max_width=300),
                        color=area_color_map[area],
                        weight=3,
                        fill=False,
                        fill_opacity=0.99,
                        fill_color=area_color_map[area],
                        opacity=0.99
                    ).add_to(m)


        # 添加自定义的动态 marker 插件
        # dynamic_marker_click = DynamicMarkerOnClick(customer_points, sol_list)
        dynamic_marker_click = DynamicMarkerOnSpace(customer_points, sol_list)

        m.add_child(dynamic_marker_click)

        import os
        from datetime import datetime
        try:
            save_dir = "/app/logs"
            os.makedirs(save_dir, exist_ok=True)
        except OSError:
            save_dir = "app/logs"
            os.makedirs(save_dir, exist_ok=True)

        time_str = datetime.now().strftime("%Y-%m-%d")
        save_path = os.path.join(save_dir, f"fig_{time_str}.html")
        m.save(save_path)

        return m

    def folium_map(info, solution_list, LOGGER):

        cus_info = info["CUS"]
        customer_code = cus_info['NAMES']
        customer_township = cus_info['TOWNSHIPS']  # 或者使用 'CITIES' 或 'DISTRICTS' 根据需要
        customer_district = cus_info['DISTRICTS']
        customer_points = cus_info['COORDS_ORG'] #[list(point) for point in zip(cus_info['COORDS'][0].tolist(), cus_info['COORDS'][1].tolist())]
        customer_xy = cus_info['COORDS_node']
        customer_grid = cus_info['COORDS_grid']

        customer_demands = cus_info['DEMANDS']

        sol_list = [sol['routeSeq'] for sol in solution_list]
        # 找出所有客户的最大编号
        max_customer_id = max(max(route['routeSeq']) for route in solution_list)

        # 初始化结果列表，默认都为0（代表depot或未分配）
        route_id_list = [0] * (max_customer_id + 1)
        veh_type_list = [0] * (max_customer_id + 1)
        # 分配线路编号
        routes = []
        for route_num, route_info in enumerate(solution_list, start=1):
            r = [(customer_points[i][1], customer_points[i][0]) for i in route_info['routeSeq']]
            r.insert(0, (customer_points[0][1], customer_points[0][0]))
            routes.append(r)
            for customer in route_info['routeSeq']:
                route_id_list[customer] = route_num
                veh_type_list[customer] = route_info['vType']
        customer_area = route_id_list

        map_obj = plot_data(routes, customer_district, customer_township, customer_code, customer_area, customer_xy, customer_grid, customer_points, customer_demands, sol_list,
                            rid=route_id_list, vid=veh_type_list)

        # customer_code = ['586829172726835901', 'KC691CZ', 'KC307WX-1', 'KC307WX-2', 'KC132CZ', 'KC641CZ',
        #                  'KC701WX', 'KC771SZ', 'KC303SZ', 'KC132SJ', 'KC343ZJ', 'KC589WX', 'KC174WX', 'KC421TC',
        #                  'KC719SH', 'KC584SH', 'KC015SH', 'KC683SH', 'KC114SH', 'KC440SH', 'KC787SH', 'KC521SH',
        #                  'KC106SH', 'KC757SH', 'KC149SH', 'KC655SH', 'KC111SH', 'KC411JX', 'KC587JX', 'KC678JX',
        #                  'KC409JX1']
        # customer_area = ['DC', '常州', '无锡', '无锡', '常州', '常州', '宜兴', '苏州', '苏州', '苏州', '镇江',
        #                  '无锡', '无锡', '太仓', '嘉定区', '青浦区', '嘉定区', '松江区', '松江区', '松江区',
        #                  '奉贤区', '奉贤区', '奉贤区', '浦东新区', '浦东新区', '浦东新区', '嘉定区', '嘉兴市',
        #                  '嘉兴市', '嘉兴市', '嘉兴市']
        # customer_demands = [0, 4000, 5000, 5000, 0, 2000, 1000, 8000, 1000, 0, 2000, 2000, 3000, 2000, 4000,
        #                     7000, 1000, 2000, 2000, 13000, 2000, 1000, 7000, 2000, 0, 0, 6000, 14000, 2000,
        #                     1000, 2000]
        # customer_points = [[118.809767, 31.902278], [119.844626, 32.019164], [120.142616, 31.644711],
        #                    [120.155533, 31.619288], [119.921002, 31.661418], [119.972139, 31.844131],
        #                    [119.709864, 31.525442], [120.484538, 31.357411], [121.033029, 31.356837],
        #                    [120.83186, 31.326278], [119.722785, 31.992168], [120.454308, 31.526769],
        #                    [120.386116, 31.528741], [121.122, 31.480362], [121.198094, 31.31372],
        #                    [121.075976, 31.252772], [120.281994, 30.19795], [121.256826, 31.026634],
        #                    [121.181974, 30.973608], [121.210802, 30.98602], [121.454201, 30.95818],
        #                    [121.45167, 30.95474], [121.455059, 30.950802], [121.689229, 31.211621],
        #                    [121.627223, 31.12997], [121.634304, 31.253107], [121.231682, 31.362006],
        #                    [120.766697, 30.801929], [120.797641, 30.787966], [120.716773, 30.7661],
        #                    [120.835855, 30.738595]]
        # sol_list = sol_list = [[1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19],
        #                        [20, 21, 22, 23, 24, 25, 26], [27, 28, 29, 30]]
        #
        # # 调用函数并保存地图到HTML文件
        # map_obj = plot_data(customer_code, customer_area, customer_points, customer_demands, sol_list)
        print("地图已保存为customer_map.html")

    def extract_solution(res, m, sub_info):
        """
        从求解器结果对象中提取主路线 solution 及详细 solution_list。

        参数:
            sub_info: res对应的问题信息字典。
            res: 求解器返回的结果对象，包含最佳路线。
            sub_info: 子问题对应的 info 字典（如拆分客户后）。
            sub_cus_indices: 子问题客户索引到原始索引的映射列表，默认为空。

        返回:
            solution: 完整的客户访问顺序（含仓库起终点）。
            solution_list: 每条路线的详细信息（客户、车型、容量等）。
        """

        sub_cus_indices = sub_info.get("CUS", {}).get("ID", [])

        # 1 获取所有可用车型（COUNT>0）的容量列表
        active_vehicle_capacities = [sub_info["VEH"]["CAPACITY"][idx] for idx, value in
                                     enumerate(sub_info["VEH"]["COUNT"]) if
                                     value >= 0]

        # 2 提取路线信息
        routes = res.best.routes()
        for idx, r in enumerate(routes):
            delivery = r.delivery()[0]
            print(f"{idx}: 线路 visits 客户 {r.visits()}.")
            print(
                f"   Trip total delivery {delivery} vehicle capacity {active_vehicle_capacities[r.vehicle_type()]}, distance is {r.distance()}.")

        # 3 获取 solution
        solution = [0]  # 起点仓库
        for r in routes:
            solution.extend(r)  # 添加客户点序列
            solution.append(0)  # 终点仓库
        if sub_cus_indices:
            solution = [sub_cus_indices[idx] if idx != 0 else 0 for idx in solution]  # 映射回原始索引

        # 增加sub_info
        # sub_info

        # 4 获取 solution_list
        solution_list = []
        for r in routes:
            cust_idx = list(r.visits()) + [0]  # 获取客户点列表并添加仓库点 0
            if sub_cus_indices:
                cust_idx_org = [sub_cus_indices[idx] if idx != 0 else 0 for idx in cust_idx]  # 映射回原始索引
            solution_list.append({
                'route': cust_idx_org,
                'routeSeq': cust_idx_org[:-1],  # 去掉末尾的 0，保留客户点序列
                'rName': [sub_info["CUS"]["NAMES"][i] for i in cust_idx[:-1]],
                'rCity': [sub_info["CUS"]["CITIES"][i] for i in cust_idx[:-1]],
                'vType': active_vehicle_capacities[r.vehicle_type()],  # 车辆类型
                'vCapacity': r.delivery()[0],  # 车辆实际装载量
                'rDist': r.distance()
            })

        return solution, solution_list

    def extract_cus_veh_info(gdf_ty_grid, D, G, depot_node):

        # # nid = [0] + gdf_ty_grid["nid"].tolist()
        # if EXTRACT_INFO_TYPE == "Node":
        #     # Do Nothing
        #     h=1
        #     # OSM_NID = [depot_node] + gdf_ty_grid["customer_osmid_mapping"].tolist()
        # elif EXTRACT_INFO_TYPE == "Osm":
        #     gdf_ty_grid = gdf_ty_grid.groupby('customer_osmid_mapping').agg({
        #         'nid': lambda x: ', '.join(str(i) for i in x.unique()),
        #         'group': lambda x: ', '.join(str(i) for i in x.unique()),
        #         'BB_RETAIL_CUSTOMER_NAME': lambda x: ', '.join(x.unique()),
        #         'city': lambda x: ', '.join(x.unique()),
        #         'district': lambda x: ', '.join(x.unique()),
        #         'township': lambda x: ', '.join(x.unique()),
        #         'avg_order_qty': 'sum',
        #         'OSM_X': 'mean',
        #         'OSM_Y': 'mean',
        #         'lon_84': 'mean',
        #         'lat_84': 'mean',
        #         'LonGridCenter': 'mean',
        #         'LatGridCenter': 'mean'
        #     }).reset_index()
        #     # 从G的nodes对应的OSM_NID提取x和y值 赋值为COORDS
        #     # OSM_NID = [depot_node] + gdf_ty_grid["customer_osmid_mapping"].tolist()
        # elif EXTRACT_INFO_TYPE == "Grid":
        #     gdf_ty_grid = gdf_ty_grid.groupby('grid_id').agg({
        #         'nid': lambda x: ', '.join(str(i) for i in x.unique()),
        #         'group': lambda x: ', '.join(str(i) for i in x.unique()),
        #         'BB_RETAIL_CUSTOMER_NAME': lambda x: ', '.join(x.unique()),
        #         'city': lambda x: ', '.join(x.unique()),
        #         'district': lambda x: ', '.join(x.unique()),
        #         'township': lambda x: ', '.join(x.unique()),
        #         'avg_order_qty': 'sum',
        #         'OSM_X': 'mean',
        #         'OSM_Y': 'mean',
        #         'lon_84': 'mean',
        #         'lat_84': 'mean',
        #         'LonGridCenter': 'mean',
        #         'LatGridCenter': 'mean'
        #     }).reset_index()
        #     # 从G的nodes对应的OSM_NID提取x和y值 赋值为COORDS
        #     # OSM_NID = [depot_node] + gdf_ty_grid["customer_osmid_mapping"].tolist()
        # else:
        #     raise ValueError(f"未知的 EXTRACT_INFO_TYPE: {EXTRACT_INFO_TYPE}. 仅支持 'Node' 或 'Grid'.")

        # COORDS = [(int(G.nodes[node]['x']* 1000000), int(G.nodes[node]['y'] * 1000000)) for node in OSM_NID]
        # COORDS_ORG = [(G.nodes[node]['x'], G.nodes[node]['y']) for node in OSM_NID]  # 原始坐标

        COORDS_ORG = [(G.nodes[depot_node]['x'], G.nodes[depot_node]['y'])] + list(zip(gdf_ty_grid['OSM_X'], gdf_ty_grid['OSM_Y']))
        COORDS = [(int(x * 1000000), int(y * 1000000)) for x, y in COORDS_ORG]

        COORDS_node = [COORDS_ORG[0]] + list(zip(gdf_ty_grid['lon_84'], gdf_ty_grid['lat_84']))
        COORDS_grid = [COORDS_ORG[0]] + list(zip(gdf_ty_grid['LonGridCenter'], gdf_ty_grid['LatGridCenter']))

        NAMES = ['太原配送点'] + gdf_ty_grid['BB_RETAIL_CUSTOMER_NAME'].tolist()
        DEMANDS = [0] + gdf_ty_grid["avg_order_qty"].astype(int).tolist()
        CITIES = ['太原市'] + gdf_ty_grid['city'].tolist()
        DISTRICTS = ['配送区'] + gdf_ty_grid['district'].tolist()
        TOWNSHIPS = ['配送街道'] + gdf_ty_grid['township'].tolist()
        GROUPS = ['配送组'] + gdf_ty_grid['group'].astype(str).tolist()
        CUSID = list(range(len(DEMANDS)))

        CAPACITY = [VEH_CAPACITY]
        COUNT = [10]
        VEHID = list(range(len(CAPACITY)))  # 车辆类型ID从0开始

        CUS = {"COORDS": COORDS, "COORDS_ORG": COORDS_ORG, "COORDS_node": COORDS_node,"COORDS_grid": COORDS_grid, "DEMANDS": DEMANDS, "GROUPS": GROUPS, "CITIES": CITIES, "DISTRICTS": DISTRICTS, "TOWNSHIPS": TOWNSHIPS, "ID": CUSID,
               "NAMES": NAMES}
        VEH = {"CAPACITY": CAPACITY, "COUNT": COUNT, "ID": VEHID}

        return {"CUS": CUS, "VEH": VEH, "D": D}

    def extract_sub_info_from_indices(info, indices):
        """
        从 info(是剩余) 中提取 indices 对应的客户信息，车辆信息保持不变。

        参数:
            info (dict): 包含客户和车辆信息的字典，需包含 'CUS' 和 'VEH' 键。
            indices (list): 需要提取的客户索引列表。

        返回:
            dict: 包含提取的客户信息和原始车辆信息的字典。
        """
        # 1 提取客户信息
        cus_info = info["CUS"]

        # 找出indices对应的cus_info的ID的位置[从绝对位置索引到相对索引!!!]
        indices_idx = [idx for idx, id_ in enumerate(cus_info["ID"]) if id_ in indices]

        sub_cus = {k: [v[idx] for idx in indices_idx] for k, v in cus_info.items()}

        remain_cus = {k: [v[idx] for idx in range(len(v)) if idx not in indices_idx or idx == 0] for k, v in
                      cus_info.items()}

        # 2 保持车辆信息不变
        # veh_info = info["VEH"].copy()

        # 3 距离提取索引
        D = info["D"]

        # D是距离矩阵，类似sub_cus从D内提取对应索引好的子矩阵sub_D
        sub_D = [[D[i][j] for j in indices_idx] for i in indices_idx]

        # D是距离矩阵，类似remain_cuss从D内提取对应剩余索引的子矩阵到remain_D
        remain_D = [[D[i][j] for j in range(len(D)) if j not in indices_idx or j == 0] for i in range(len(D)) if
                    i not in indices_idx or i == 0]

        # 3 返回提取后的信息
        sub_info = {"CUS": sub_cus, "VEH": info["VEH"].copy(), "D": sub_D}

        remaining_info = {"CUS": remain_cus, "VEH": info["VEH"].copy(), "D": remain_D}

        return sub_info, remaining_info

    def update_remaining_veh(res, m, remaining_info):
        """
        根据求解结果res，减少remaining_info中对应车型的车辆数。
        """

        v = m.data().vehicle_types()
        for r in res.best.routes():
            # 1 获取车辆类型名称veh_name转化为int
            veh_name = v[r.vehicle_type()].name  # 获取车辆类型名称[即长度]
            if not veh_name.isdigit():
                raise ValueError(f"车辆类型名称 {veh_name} 不是数字，无法转换为索引。")
            veh_name = int(veh_name)

            # 2 remaining_info减少对应车辆数量
            if veh_name not in remaining_info["VEH"]["CAPACITY"]:
                raise ValueError(f"车辆类型 {veh_name} 不在剩余车辆信息中: {remaining_info['VEH']['CAPACITY']}")
            # veh_name在remaining_info["VEH"]["CAPACITY"内的索引
            veh_type = remaining_info["VEH"]["CAPACITY"].index(veh_name)
            if remaining_info["VEH"]["COUNT"][veh_type] > 0:
                remaining_info["VEH"]["COUNT"][veh_type] -= 1
            else:
                raise ValueError(f"车辆类型 {veh_name} 的剩余数量为0，无法减少。")

        return remaining_info

    def _build_model_add_priority(COORDS, DEMANDS, CITIES, NAMES, veh_count, veh_capacity, LOGGER, D=[],
                                  group_indices=[], conflict_indices=[]):

        MAX_CUS_VISITED = 1000
        MAX_ROUTE_DURATION = 100000
        SERVICE_DURATION = 1
        IS_PRIORITY_CUS = False
        IS_CONFLICT_CUS = False
        VEHICLE_SPEED = 1

        LOGGER.info(f"模型构建中: ")
        m = Model()

        # 1 添加profiles
        LOGGER.info(f"  构建模型 add_vehicle_type")
        profiles = {}
        for capacity in veh_capacity:
            profile_name = f"pro_{capacity}"
            profiles[profile_name] = m.add_profile()

        for count, capacity in zip(veh_count, veh_capacity):
            m.add_vehicle_type(count,
                               capacity=[capacity, MAX_CUS_VISITED],
                               max_duration=MAX_ROUTE_DURATION,
                               profile=profiles[f"pro_{capacity}"],
                               name=f"{capacity}")  # ,initial_load=[0, 0]

        # 2 添加仓库和客户
        LOGGER.info(f"  构建模型 add_client")
        depot = m.add_depot(x=COORDS[0][0], y=COORDS[0][1], name='DC')
        clients = [
            m.add_client(x=COORDS[idx][0], y=COORDS[idx][1],
                         delivery=[DEMANDS[idx], 1], service_duration=SERVICE_DURATION,
                         name=NAMES[idx])
            for idx in range(1, len(COORDS))
        ]

        LOGGER.info(f"  构建模型 add_edge")
        # 3 添加边的距离（曼哈顿距离）
        if not IS_PRIORITY_CUS:
            group_indices = []

        if not IS_CONFLICT_CUS:
            conflict_indices = []

        # 同group[城市区域]优先配送[即距离为0]
        locations = [depot] + clients
        same_group_pairs = set()
        for group in group_indices:
            for i in group:
                for j in group:
                    if i != j:
                        same_group_pairs.add((i, j))

        # 同group[城市区域]互斥配送[即距离为10 **6 ]
        same_conflict_pairs = set()
        for group in conflict_indices:
            for i in group:
                for j in group:
                    if i != j:
                        same_conflict_pairs.add((i, j))

        for i, frm in enumerate(locations):
            for j, to in enumerate(locations):
                if (i, j) in same_group_pairs or (j, i) in same_group_pairs:
                    distance = 0
                elif (i, j) in same_conflict_pairs or (j, i) in same_conflict_pairs:
                    distance = 10 ** 6  # 不同城市的客户之间距离为无穷大
                else:
                    if DIST_MATRIX_TYPE == 'Manhattan':
                        distance = (abs(frm.x - to.x) + abs(frm.y - to.y)) * 1
                    elif DIST_MATRIX_TYPE == 'Euclidean':
                        distance = ((frm.x - to.x) ** 2 + (frm.y - to.y) ** 2) ** 0.5
                    elif DIST_MATRIX_TYPE == 'GaoDe':
                        distance = D[i][j]
                    else:
                        raise ValueError(f"未知的距离矩阵类型: {DIST_MATRIX_TYPE}")

                # # 要求[todo 改为参数化 非name？或name也可]：i和j强制拼载 【可行, 除i和j外，其他到j的距离为♾️ 如3个，除j和k外，其他到k的距离为♾️】
                if frm.name != 'KC303SZ' and frm.name != 'KC132SJ' and to.name == 'KC132SJ':
                    distance = 10 ** 6
                if frm.name != 'KC111SH' and frm.name != 'KC114SH' and to.name == 'KC114SH':
                    distance = 10 ** 6
                # add_edge增加边
                m.add_edge(frm, to, distance=distance, duration=distance / VEHICLE_SPEED)

                # # # 要求：j的访问被某些车型禁用 todo
                # # 1 得到j的可用车型
                # j

                # 2 从profiles找到不可用车型的profile

                # 3 增加这个车型的距离为10**6
                # m.add_edge(frm, to, distance=10 ** 6, profile=profile)

                # # # 要求：某些边禁用特殊车型
                # if i != j:
                #     frm_city = CITIES[i] if i > 0 else None
                #     to_city = CITIES[j] if j > 0 else None
                #
                #     for pro, profile in profiles.items():
                #         hate_cities = VEHICLE_CITY_HATE_MAP.get(pro, [])  # 指定车型的hate_cities读取
                #         like_cities = VEHICLE_CITY_LIKE_MAP.get(pro, [])
                #         if to_city in hate_cities:
                #             LOGGER.info(
                #                 f"添加边({frm}, {to})，距离为10^6，车型为{pro}，因为{to_city}在{pro}的hate_cities中")
                #             m.add_edge(frm, to, distance=10 ** 6, profile=profile)
                #         if to_city in like_cities:
                #             LOGGER.info(f"添加边({frm}, {to})，距离为0，车型为{pro}，因为{to_city}在{pro}的like_cities中")
                #             m.add_edge(frm, to, distance=0, profile=profile)
                #
                #     # VEHICLE_CITY_HATE_MAP
                #     # if in_city(CITIES[j]):
                #     #     m.add_edge(frm, to, distance=10 ** 6,profile = list(profiles.values())[-1])
                #     # if in_city_like(CITIES[j]):
                #     #     m.add_edge(frm, to, distance=0, profile=list(profiles.values())[-1])

        return m


    def _build_model_route(COORDS, DEMANDS, CITIES, NAMES, veh_count, veh_capacity, LOGGER, D=[]):

        MAX_CUS_VISITED = 1000
        MAX_ROUTE_DURATION = 100000
        SERVICE_DURATION = 1
        VEHICLE_SPEED = 1

        LOGGER.info(f"模型构建中: ")
        m = Model()

        # 1 添加profiles
        for count, capacity in zip(veh_count, veh_capacity):
            m.add_vehicle_type(count,
                               capacity=[capacity, MAX_CUS_VISITED],
                               max_duration=MAX_ROUTE_DURATION,
                               profile=profiles[f"pro_{capacity}"],
                               name=f"{capacity}")  # ,initial_load=[0, 0]

        # 2 添加仓库和客户
        LOGGER.info(f"  构建模型 add_client")
        depot = m.add_depot(x=COORDS[0][0], y=COORDS[0][1], name='DC')
        clients = [
            m.add_client(x=COORDS[idx][0], y=COORDS[idx][1],
                         delivery=[DEMANDS[idx], 1], service_duration=SERVICE_DURATION,
                         name=NAMES[idx])
            for idx in range(1, len(COORDS))
        ]

        LOGGER.info(f"  构建模型 add_edge")
        # 3 添加边的距离（曼哈顿距离）
        if not IS_PRIORITY_CUS:
            group_indices = []

        if not IS_CONFLICT_CUS:
            conflict_indices = []

        # 同group[城市区域]优先配送[即距离为0]
        locations = [depot] + clients
        same_group_pairs = set()
        for group in group_indices:
            for i in group:
                for j in group:
                    if i != j:
                        same_group_pairs.add((i, j))

        # 同group[城市区域]互斥配送[即距离为10 **6 ]
        same_conflict_pairs = set()
        for group in conflict_indices:
            for i in group:
                for j in group:
                    if i != j:
                        same_conflict_pairs.add((i, j))

        for i, frm in enumerate(locations):
            for j, to in enumerate(locations):
                if (i, j) in same_group_pairs or (j, i) in same_group_pairs:
                    distance = 0
                elif (i, j) in same_conflict_pairs or (j, i) in same_conflict_pairs:
                    distance = 10 ** 6  # 不同城市的客户之间距离为无穷大
                else:
                    if DIST_MATRIX_TYPE == 'Manhattan':
                        distance = (abs(frm.x - to.x) + abs(frm.y - to.y)) * 1
                    elif DIST_MATRIX_TYPE == 'Euclidean':
                        distance = ((frm.x - to.x) ** 2 + (frm.y - to.y) ** 2) ** 0.5
                    elif DIST_MATRIX_TYPE == 'GaoDe':
                        distance = D[i][j]
                    else:
                        raise ValueError(f"未知的距离矩阵类型: {DIST_MATRIX_TYPE}")

                # add_edge增加边
                m.add_edge(frm, to, distance=distance, duration=distance / VEHICLE_SPEED)

        return m


    def _pyvrp_solve_MultipleCriteria(info, LOGGER, runtime=1, iters=100, is_display=False):

        COORDS = info["CUS"]["COORDS"]
        DEMANDS = info["CUS"]["DEMANDS"]
        CITIES = info["CUS"]["CITIES"]
        NAMES = info["CUS"]["NAMES"]
        D = info["D"]
        veh_capacity = info["VEH"]["CAPACITY"]
        veh_count = info["VEH"]["COUNT"]
        veh_capacity = [veh_capacity[i] for i, c in enumerate(veh_count) if c > 0]
        veh_count = [c for c in veh_count if c > 0]

        group_indices = []
        conflict_indices = []  # 示例：假设1和3、2和4是冲突的客户 [[1,3], [1,4], [1,5]]

        m = _build_model_add_priority(COORDS, DEMANDS, CITIES, NAMES, veh_count, veh_capacity, LOGGER,
                                      D=D, group_indices=group_indices, conflict_indices=conflict_indices)

        res = m.solve(stop=MultipleCriteria([NoImprovement(iters), MaxRuntime(runtime)]), display=is_display)

        LOGGER.info(f"\n{res}")

        if 1: # IS_PLOT_SOLUTION
            import os
            import matplotlib.pyplot as plt
            from pyvrp.plotting import plot_solution
            from datetime import datetime
            _, ax = plt.subplots(figsize=(8, 8))
            plot_solution(res.best, m.data(), plot_clients=True, ax=ax)

            # 保存为文件（可以是 PNG、PDF、SVG 等）
            try:
                save_dir = "/app/logs"
                os.makedirs(save_dir, exist_ok=True)
            except OSError:
                save_dir = "app/logs"
                os.makedirs(save_dir, exist_ok=True)

            time_str = datetime.now().strftime("%Y-%m-%d")
            save_path = os.path.join(save_dir, f"fig_{time_str}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

        return res, m

    def solve_final_customers(
            remaining_info, all_solutions, all_solution_lists, LOGGER
    ):

        # 2.0 抽取final_indices
        final_indices = remaining_info["CUS"]["ID"]
        final_info, remaining_info = extract_sub_info_from_indices(remaining_info, final_indices)

        # 2.1 抽取suggest_vehicle_combinations
        final_res, final_m = _pyvrp_solve_MultipleCriteria(final_info, LOGGER, runtime=PYVRP_MAX_RUNTIME,
                                                           iters=PYVRP_MAX_NO_IMPROVE_ITERS, is_display=True)
        if final_res.is_feasible():  # 检查是否为可行解
            if final_res.best.distance() > 10 ** 6:  # 如果距离过大, 认为是不可行解
                LOGGER.info(f"<UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK>")

            remaining_info = update_remaining_veh(final_res, final_m, remaining_info)

            [sol, sol_list] = extract_solution(final_res, final_m, final_info)

            all_solutions = all_solutions[:-1] + sol  # 合并solution
            all_solution_lists.extend(sol_list)

            # # to del start
            from tabulate import tabulate
            df_sol = cov_sol_list(sol_list)
            LOGGER.info(f"\ndf_sol:\n{tabulate(df_sol, headers='keys', tablefmt='grid')}")
            # # to del end
        else:
            # 提示错误 时间不够
            LOGGER.error(f"求解失败，未找到可行解。请检查输入数据或调整参数。")


        return all_solutions, all_solution_lists, remaining_info


    # def get_distance(distance_dict, node1, node2):
    #     """
    #     根据距离字典获取两点距离
    #     """
    #     key = tuple(sorted((node1, node2)))
    #     return distance_dict.get(key, 9999999)
    #
    # def calculate_euclidean_distance(point1, point2):
    #     """
    #     计算两个点的欧几里得距离。
    #     point1, point2: 包含 x/y 坐标的字典（如 G.nodes[node]）
    #     """
    #     x1, y1 = point1[0], point1[1]
    #     x2, y2 = point2[0], point2[1]
    #     return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    #
    # def calculate_haversine_distance(point1, point2):
    #     """
    #     计算两个点的欧几里得距离。
    #     point1, point2: 包含 x/y 坐标的字典（如 G.nodes[node]）
    #     """
    #     lon1, lat1 = point1[0], point1[1]
    #     lon2, lat2 = point2[0], point2[1]
    #     return haversine_distance(lat1, lon1, lat2, lon2)
    #
    # # 定义 Haversine 距离计算函数（单位：米）
    # def haversine_distance(lat1, lon1, lat2, lon2):
    #     R = 6371000  # 地球平均半径（米）
    #     # 转换为弧度
    #     phi1 = np.radians(lat1)
    #     phi2 = np.radians(lat2)
    #     delta_phi = np.radians(lat2 - lat1)
    #     delta_lambda = np.radians(lon2 - lon1)
    #     # 公式计算
    #     a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    #     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    #     return R * c

    LOGGER.info(f"\n--------------- Part3 算法求解开始 ---------------")

    # 1 数据预处理 提取pyvrp计算必要数据
    info = extract_cus_veh_info(gdf_ty_grid, D, G, depot_node = 7839518188)

    # 3 初始化剩余信息[后续全面适用remaining]
    remaining_info = info.copy()

    all_solutions = []
    all_solution_lists = []

    if 0:
        LOGGER.info(f"分区1求解开始： 求解无法拼车客户组")
        all_solutions, all_solution_lists, remaining_info = (
            solve_isolated_customers(remaining_info, all_solutions, all_solution_lists, LOGGER))

    if 0:
        LOGGER.info(f"分区2求解开始： 同city组 改为同发货点组 或 同小区组")
        for cities in splitCityArea:
            LOGGER.info(f"当前分区 cities: {cities}")
            all_solutions, all_solution_lists, remaining_info = (
                solve_cityarea_customers(remaining_info, all_solutions, all_solution_lists, cities, LOGGER)
            )

    if 1:
        LOGGER.info(f"分区N求解开始： 求解最终剩余客户组")
        all_solutions, all_solution_lists, remaining_info = (
            solve_final_customers(remaining_info, all_solutions, all_solution_lists, LOGGER))

    if 1:
        folium_map(info, all_solution_lists, LOGGER)

    return all_solutions, all_solution_lists
