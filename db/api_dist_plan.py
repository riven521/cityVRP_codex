from typing import Dict, Optional, Tuple, Iterable, Any
import pandas as pd
import numpy as np
import math
from db.log_service import add_n35_log_info
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import json
import hashlib

# === Local cache I/O helpers ===
def _coerce_decimal_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Decimal objects to float so parquet/json writers won't choke."""
    if df is None or df.empty:
        return df
    for col in df.columns:
        s = df[col]
        try:
            if s.dtype == 'object' and s.map(lambda x: isinstance(x, Decimal)).any():
                df[col] = s.map(lambda x: float(x) if isinstance(x, Decimal) else x)
        except Exception:
            # Best-effort; keep original col on failure
            pass
    return df

def _df_to_parquet(df: pd.DataFrame, path: Path) -> None:
    df = _coerce_decimal_to_float(df.copy())
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)  # requires pyarrow or fastparquet
    except Exception:
        # Fallback: pickle when parquet engine is missing
        df.to_pickle(path.with_suffix('.pkl'))


def _df_from_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            # Fallback: try pkl side-by-side
            pkl = path.with_suffix('.pkl')
            if pkl.exists():
                return pd.read_pickle(pkl)
            raise
    else:
        # allow .pkl-only cache
        pkl = path.with_suffix('.pkl')
        if pkl.exists():
            return pd.read_pickle(pkl)
        raise FileNotFoundError(f"Cache file not found: {path}")


def _schema_fingerprint(dfs: dict) -> str:
    """Make a light schema fingerprint for quick compatibility checks."""
    items = []
    for name, df in dfs.items():
        cols = [(c, str(df[c].dtype)) for c in df.columns] if isinstance(df, pd.DataFrame) else []
        items.append((name, tuple(cols)))
    raw = json.dumps(items, ensure_ascii=False)
    return hashlib.md5(raw.encode('utf-8')).hexdigest()

def save_tables_local(n34_id: str, tables: dict, cache_dir: str = 'cache_vrp') -> Path:
    """Persist tables to local parquet under cache_dir/n34_id/.
    `tables` should be a dict name->DataFrame (e.g., {'db_n34': df, ...}).
    Returns the folder path used.
    """
    root = Path(cache_dir) / str(n34_id)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        'n34_id': n34_id,
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'tables': {},
        'schema_fingerprint': None,
        'pandas_version': pd.__version__,
    }
    for name, df in tables.items():
        if df is None:
            continue
        out_path = root / f"{name}.parquet"
        _df_to_parquet(df, out_path)
        manifest['tables'][name] = str(out_path.name)
    manifest['schema_fingerprint'] = _schema_fingerprint(tables)
    with open(root / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return root


def load_tables_local(n34_id: str, cache_dir: str = 'cache_vrp', *, strict: bool = False) -> dict:
    """Load previously saved tables from cache_dir/n34_id.
    If `strict` is True, raise if any table is missing; otherwise missing ones are ignored.
    Returns a dict name->DataFrame.
    """
    root = Path(cache_dir) / str(n34_id)
    manifest_path = root / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest for n34_id={n34_id} under {root}")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    out = {}
    for name, filename in manifest.get('tables', {}).items():
        try:
            out[name] = _df_from_parquet(root / filename)
        except Exception as e:
            if strict:
                raise
            else:
                print(f"[load_tables_local] skip {name}: {e}")
    return out

def prepare_vrp_inputs(
    n34_id: str,
    df_26: pd.DataFrame,
    df_28: pd.DataFrame,
    df_27: Optional[pd.DataFrame] = None,
    df_31: Optional[pd.DataFrame] = None,
    df_19: Optional[pd.DataFrame] = None,
    df_32: Optional[pd.DataFrame] = None,
    df_33: Optional[pd.DataFrame] = None,
    logger: Optional[Any] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    将 26/28（必需）及若干可选源表，转为 VRP 求解所需输入：df_cus, df_veh, D。

    参数
    ----
    df_26 : 仓库/线路或车辆配置相关表（必需）
    df_28 : 客户/需求相关表（必需）
    df_27, df_31, df_19, df_32, df_33 : 可选的辅助数据源
    logger : 可选日志器，需支持 .info/.warning/.error

    返回
    ----
    (df_cus, df_veh, D)
      df_cus : 客户表（含坐标/需求/服务时长/时间窗等）
      df_veh : 车型表（含容量/数量/费用参数等）
      D      : 距离或时间矩阵（numpy.ndarray 或 pd.DataFrame，视你的实现而定）

    约定
    ----
    - 仅对 df_26 / df_28 做最小列校验（计划中心与方案号两列），其他可选表按需使用。
    - 具体字段映射、构图/路网距离计算请在标注的 “TODO” 区域落地。
    """
    add_n35_log_info(n34_id, "ALG_01",
                     f"范围规划算法-数据准备-开始构建 VRP 输入数据（客户/车辆/距离矩阵）")
    # ---- 简易 logger ----
    class _DummyLogger:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass

    logger = logger or _DummyLogger()

    # ---- 最小校验：必需列 ----
    _need_cols = {"MD03_DIST_PLAN_SCHEME_CODE", "MD03_DIST_CENTER_CODE"}
    for name, df in (("df_26", df_26), ("df_28", df_28)):
        missing = _need_cols - set(df.columns)
        if missing:
            raise KeyError(f"{name} 缺少必要列: {sorted(missing)}")

    if df_26.empty or df_28.empty:
        raise ValueError("df_26 或 df_28 为空，无法构建 VRP 输入。")

    logger.info("prepare_vrp_inputs: 基础数据校验通过。")

    # =========================================================
    # 1) 必需步骤：由 df_28 构建 df_cus；由 df_26 构建 df_depot
    # =========================================================
    # TODO: 根据你的业务规则，把 df_28 映射为 VRP 的客户表 df_cus
    # 示意：选取/重命名/计算需求、经纬度、时间窗、服务时长等
    df_cus = df_28.copy()
    df_cus['MD03_AVG_ORDER_QTY'] = 45
    df_cus['MD03_SERVICE_DURATION'] = 10

    # 例如（请按你真实字段替换）：
    # df_cus = df_28.rename(columns={
    #     "城市经度": "lon",
    #     "城市纬度": "lat",
    #     "托数": "demand",
    # }).loc[:, ["客户ID", "lon", "lat", "demand", "service_time", "tw_start", "tw_end"]]

    # TODO: 根据你的业务规则，把 df_26 映射为 VRP 的车辆表 df_veh
    df_depot = df_26.copy()
    # 例如（请按你真实字段替换）：
    # df_veh = df_26.rename(columns={
    #     "vehicleModel": "vehicleModel",
    #     "vehicleCount": "vehicleCount",
    #     "capacity_weight": "cap_weight",
    #     ...
    # }).loc[:, ["vehicleModel", "vehicleCount", "cap_weight", "cap_volume", ...]]

    df_veh = df_27.copy()

    # =========================================================
    # 2) 距离/时间矩阵 D 的构建（必需）
    # =========================================================
    # TODO: 用 df_cus(坐标) 及需要的话 df_26/路网，生成 D
    # - 若有路网图（OSMnx 等）走“最短路时间/距离”
    # - 若没有，先用球面距离/直线距离 + 系数近似
    # 下面示意：占位用空矩阵；请替换为你的实际实现
    import numpy as np
    n = len(df_cus)
    D = np.zeros((n, n), dtype=float)

    # =========================================================
    # 3) 可选增强：若传入其他数据表，则做补充/校正
    # =========================================================
    # 3.1 df_27（例如车型维度或价格参数的补充）
    if df_27 is not None and not df_27.empty:
        logger.info("应用 df_27 增强车辆/费用参数 ...")
        # TODO: 按你的键（如车型/方案号/中心代码）去 merge/enrich df_veh

    # 3.2 df_31 / df_19 / df_32 / df_33（例如：禁行、回仓、时间窗策略、固定线路等）
    if df_31 is not None and not df_31.empty:
        logger.info("应用 df_31 增强客户/约束 ...")
        # TODO: enrich df_cus / 追加约束字段

    if df_19 is not None and not df_19.empty:
        logger.info("应用 df_19 增强全局/车辆参数 ...")
        # TODO

    if df_32 is not None and not df_32.empty:
        logger.info("应用 df_32 增强路线/时间窗策略 ...")
        # TODO

    if df_33 is not None and not df_33.empty:
        logger.info("应用 df_33 增强固定/动态线路等约束 ...")
        # TODO

    # 末尾快速健诊（可选）
    if len(df_cus) == 0:
        raise ValueError("构建后的 df_cus 为空，请检查输入或过滤条件。")
    if len(df_depot) == 0:
        logger.warning("构建后的 df_veh 为空，后续求解可能无法进行。")

    add_n35_log_info(n34_id, "ALG_02",
                     f"范围规划算法-数据准备-完成 VRP 输入数据构建，客户数={len(df_cus)}，发货点数={len(df_depot)}")
    return df_cus, df_depot, df_veh, D


def plan_distribution_regions(
    n34_id: str,
    df_cus: pd.DataFrame,
    df_depot: pd.DataFrame,
    D_cd: Optional[pd.DataFrame] = None,  # 行=发货点(MD03_DIST_PLAN_DIST_SITE_ID), 列=客户(BB_RETAIL_CUSTOMER_CODE)
    *,
    return_unassigned: bool = True,
    assign_unreachable_to_nearest: bool = False,  # 新增：是否把不可达/缺坐标客户也强行分配
    assign_strategy: str = "nearest",             # 新增：分配策略
    depot_capacity: Optional[Dict[str, float]] = None,  # 新增：仓运力（单位：户数），如 {"1402000101": 1200, ...}
) -> Tuple[Dict[str, set], Optional[Iterable[str]]]:
    """基于距离将客户分配到最近的发货点。
    新增策略：
    - assign_strategy="township_capacity"：两阶段策略。
      阶段1（同街道一致）：将同一街道（`township` 列）内所有零售户分配到同一发货点；该发货点由“多数服从多数的最近仓”确定：先计算每个零售户最近仓，再在街道内投票选出出现次数最多的最近仓（并列时选择该街道到该仓的平均距离最小者）。
      阶段2（按运力再平衡）：给定每个发货点的运力（`depot_capacity` 参数，单位按“户数”计），若某发货点分配超载，则以“整街道”为单位，将其迁移到其它有剩余运力的发货点，优先选择导致该街道平均距离增量最小的发货点。迁移遵守“同一街道必须在同一发货点”的约束。
    - assign_strategy="district_capacity"：两阶段策略。
      阶段1（同区域一致）：将同一区域（`district` 列）内所有零售户分配到同一发货点；该发货点由“多数服从多数的最近仓”确定：先计算每个零售户最近仓，再在区域内投票选出出现次数最多的最近仓（并列时选择该区域到该仓的平均距离最小者）。
      阶段2（按运力再平衡）：给定每个发货点的运力（`depot_capacity` 参数，单位按“户数”计），若某发货点分配超载，则以“整区域”为单位，将其迁移到其它有剩余运力的发货点，优先选择导致该区域平均距离增量最小的发货点。迁移遵守“同一区域必须在同一发货点”的约束。

    额外参数：
    - depot_capacity: dict 映射 {发货点编码: 可承载户数}。仅 `township_capacity` 策略需要；未提供则报错。
    返回:
      dict_dist_plan: { depot_id: set(customer_codes) }
      unassigned: 无法分配客户列表（当 return_unassigned=True）
    """
    add_n35_log_info(n34_id, "ALG_03",
                     f"范围规划算法-范围划分-开始执行客户到发货点的分配")
    # ==== 列校验 ====
    need_cus = {"BB_RETAIL_CUSTOMER_CODE", "MD03_RETAIL_CUST_LON", "MD03_RETAIL_CUST_LAT"}
    miss = need_cus - set(df_cus.columns)
    if miss:
        raise KeyError(f"df_cus 缺少列: {sorted(miss)}")

    need_dep = {"MD03_SHIPMENT_STATION_CODE", "MD03_SHIPMENT_STATION_LON", "MD03_SHIPMENT_STATION_LAT"}
    miss = need_dep - set(df_depot.columns)
    if miss:
        raise KeyError(f"df_depot 缺少列: {sorted(miss)}")

    # 清洗：去掉无发货点ID的行
    df_depot = df_depot[df_depot["MD03_SHIPMENT_STATION_CODE"].notna()].copy()
    if df_depot.empty:
        raise ValueError("df_depot 中发货点ID皆为空。")

    # 标准化为字符串 ID
    cus_ids_all = df_cus["BB_RETAIL_CUSTOMER_CODE"].astype(str)
    depot_ids = df_depot["MD03_SHIPMENT_STATION_CODE"].astype(str).tolist()
    first_depot = depot_ids[0]  # 作为缺坐标时的回退仓（保持确定性）

    # ==== 若未提供 D_cd，则用坐标计算 ====
    if D_cd is None:
        # 坐标齐全的客户/仓
        cus_ok_mask = df_cus["MD03_RETAIL_CUST_LON"].notna() & df_cus["MD03_RETAIL_CUST_LAT"].notna()
        depot_ok_mask = df_depot["MD03_SHIPMENT_STATION_LON"].notna() & df_depot["MD03_SHIPMENT_STATION_LAT"].notna()

        missing_coord_customers = df_cus.loc[~cus_ok_mask, "BB_RETAIL_CUSTOMER_CODE"].astype(str).tolist()
        df_cus2 = df_cus.loc[cus_ok_mask].copy()
        df_depot2 = df_depot.loc[depot_ok_mask].copy()

        # 若仓坐标全缺，无法计算任何距离
        if df_depot2.empty:
            dict_dist_plan = {d: set() for d in depot_ids}
            if assign_unreachable_to_nearest:
                # 全部指派到第一个仓
                for c in cus_ids_all.tolist():
                    dict_dist_plan[first_depot].add(c)
                return (dict_dist_plan, None) if return_unassigned else (dict_dist_plan, None)
            else:
                if return_unassigned:
                    return dict_dist_plan, cus_ids_all.tolist()
                return dict_dist_plan, None

        cus_ids = df_cus2["BB_RETAIL_CUSTOMER_CODE"].astype(str).tolist()

        # Haversine
        def haversine(lon1, lat1, lon2, lat2):
            R = 6371.0
            lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
            dlon, dlat = lon2 - lon1, lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
            return 2 * R * math.asin(math.sqrt(a))

        cus_xy = df_cus2[["MD03_RETAIL_CUST_LON", "MD03_RETAIL_CUST_LAT"]].to_numpy(float)
        depot_xy = df_depot2[["MD03_SHIPMENT_STATION_LON", "MD03_SHIPMENT_STATION_LAT"]].to_numpy(float)
        depot_ids_ok = df_depot2["MD03_SHIPMENT_STATION_CODE"].astype(str).tolist()

        dist = np.empty((len(depot_ids_ok), len(cus_ids)), dtype=float)
        for i, (dlon, dlat) in enumerate(depot_xy):
            for j, (clon, clat) in enumerate(cus_xy):
                dist[i, j] = haversine(dlon, dlat, clon, clat)

        D_cd = pd.DataFrame(dist, index=depot_ids_ok, columns=cus_ids)

        # 先对“坐标齐全”的客户完成常规分配；坐标缺失者后续按参数处理
        post_assign_missing_coords = missing_coord_customers
    else:
        # 使用外部距离矩阵
        D_cd = D_cd.copy()
        D_cd.index = D_cd.index.astype(str)
        D_cd.columns = D_cd.columns.astype(str)
        post_assign_missing_coords = []  # 有 D_cd 时不区分是否缺坐标

        # 补齐缺失仓行为 +inf
        missing_rows = set(depot_ids) - set(D_cd.index)
        if missing_rows:
            D_cd = pd.concat(
                [D_cd, pd.DataFrame(np.inf, index=sorted(missing_rows), columns=D_cd.columns)],
                axis=0
            )
        # 仅保留本次仓顺序
        D_cd = D_cd.loc[depot_ids]

    # ==== 统一：NaN→+inf，避免 idxmin NaN ====
    D_cd = D_cd.astype(float).replace([np.inf, -np.inf], np.inf).fillna(np.inf)

    # 判定“不可达”（整列为 +inf）的客户
    all_inf_mask = np.isinf(D_cd).all(axis=0)
    unreachable_cols = D_cd.columns[all_inf_mask].tolist()

    # 只对可达客户做最近仓分配
    feasible_cols = D_cd.columns[~all_inf_mask]
    # nearest_depot = D_cd.loc[:, feasible_cols].idxmin(axis=0) if len(feasible_cols) else pd.Series(dtype=str)

    # assigned_depot: pd.Series
    random_state = 42  # 用于随机分配的随机种子

    if len(feasible_cols) == 0:
        assigned_depot = pd.Series(dtype=str)
    else:
        # 每个客户的可达仓（距离非 inf）
        D_feas = D_cd.loc[:, feasible_cols]
        feas_mask = ~np.isinf(D_feas)
        rng = np.random.default_rng(random_state)

        if assign_strategy == "nearest":
            # 与旧逻辑一致：各列取最小值的行索引
            assigned_depot = D_feas.idxmin(axis=0)

        elif assign_strategy == "random":
            # 在每一列的可达仓中随机选择一个
            choices = {}
            for c in D_feas.columns:
                rows_ok = feas_mask.loc[:, c]
                depots_ok = D_feas.index[rows_ok].tolist()
                if depots_ok:
                    choices[c] = rng.choice(depots_ok)
            assigned_depot = pd.Series(choices, name="depot")

        elif assign_strategy == "balanced":
            # 数量均衡：贪心地把每个客户分配给当前负载最小的可达仓；并列时选距离最近
            load = {d: 0 for d in D_feas.index.tolist()}
            assign_list = []
            # 为了稳定性，按照客户列名排序遍历
            for c in sorted(D_feas.columns.tolist()):
                rows_ok = feas_mask.loc[:, c]
                depots_ok = D_feas.index[rows_ok].tolist()
                if not depots_ok:
                    continue
                # 找到当前负载最小的可达仓集合
                min_load = min(load[d] for d in depots_ok)
                candidate_depots = [d for d in depots_ok if load[d] == min_load]
                if len(candidate_depots) == 1:
                    d_star = candidate_depots[0]
                else:
                    # 并列时选距离最近的仓
                    d_star = D_feas.loc[candidate_depots, c].idxmin()
                load[d_star] += 1
                assign_list.append((c, d_star))
            assigned_depot = pd.Series({c: d for c, d in assign_list}, name="depot")

        elif assign_strategy == "township_capacity":
            # === 阶段1：同街道一致，按“多数的最近仓”确定街道归属 ===
            # 依赖列：df_cus2 内需包含 'township'
            if 'township' not in df_cus2.columns:
                raise KeyError("使用 'township_capacity' 策略需要 df_cus 中包含列 'township'")
            # 计算每个可达客户的最近仓
            nearest_each = D_feas.idxmin(axis=0)  # Series: col(customer)->row(depot)
            # 建立客户->街道映射，仅限可达客户列
            cus_meta = df_cus2.set_index('BB_RETAIL_CUSTOMER_CODE')[['township']].copy()
            cus_meta['township'] = cus_meta['township'].astype(str)
            cus_meta.index = cus_meta.index.astype(str)
            # 仅保留可达客户
            cus_meta = cus_meta.loc[nearest_each.index.intersection(cus_meta.index)]
            # 按街道投票：选择出现次数最多的最近仓；平局时选平均距离最小者
            township_to_depot = {}
            for t, idx in cus_meta.groupby('township').groups.items():
                cust_ids = list(map(str, idx.tolist()))
                votes = nearest_each.loc[cust_ids].value_counts()
                top_cnt = votes.max()
                candidates = votes[votes == top_cnt].index.tolist()
                if len(candidates) == 1:
                    chosen = candidates[0]
                else:
                    # 平局：按该街道到候选仓的平均距离最小选择
                    # 先从 D_feas 取出该街道客户列
                    subD = D_feas.loc[:, cust_ids]
                    avg_by_depot = subD.loc[candidates].replace(np.inf, np.nan).mean(axis=1)
                    chosen = avg_by_depot.idxmin()
                township_to_depot[str(t)] = str(chosen)
            # 将可达客户按街道一并指派
            assign_pairs = {}
            for c in nearest_each.index.tolist():
                t = cus_meta.at[c, 'township']
                assign_pairs[c] = township_to_depot.get(str(t))
            assigned_depot = pd.Series(assign_pairs, name="depot")

            # === 阶段2：按运力再平衡（整街道迁移） ===
            if depot_capacity is None:
                raise ValueError("使用 'township_capacity' 策略需要提供 depot_capacity（{仓: 可承载户数}）")
            # 正规化运力字典（仅保留本次仓）
            cap = {str(d): float(depot_capacity.get(str(d), np.inf)) for d in D_feas.index.tolist()}

            # 预计算 距离矩阵 的 (街道, 仓) 平均距离，便于快速评估迁移代价
            # 准备：每个街道包含的客户列集合
            township_customers = {str(t): list(map(str, idx.tolist()))
                                  for t, idx in cus_meta.groupby('township').groups.items()}
            # 街道规模（户数）
            township_sizes = {t: len(cols) for t, cols in township_customers.items()}

            # 当前分配统计（仅包含可达客户）
            load = {d: 0.0 for d in D_feas.index.tolist()}
            depot_towns = {d: [] for d in D_feas.index.tolist()}
            for t, d in township_to_depot.items():
                sz = float(township_sizes.get(t, 0))
                load[d] += sz
                depot_towns[d].append(t)

            def town_avg_dist(t: str, d: str) -> float:
                cols = township_customers.get(t, [])
                if not cols:
                    return np.inf
                sub = D_feas.loc[d, cols].astype(float).replace([np.inf, -np.inf], np.nan)
                return float(np.nanmean(sub.values))

            # 贪心迁移：从超载仓向有余量的仓，按“平均距离增量”从小到大移动整街道
            def total_overload():
                return sum(max(0.0, load[d] - cap.get(d, np.inf)) for d in load.keys())

            # 预先计算候选接收仓列表（有剩余或容量为inf）
            def best_receiver_for_town(t: str, cur_d: str) -> Optional[Tuple[str, float]]:
                # 返回 (receiver_depot, penalty)
                candidates = []
                base = town_avg_dist(t, cur_d)
                for d2 in D_feas.index.tolist():
                    if d2 == cur_d:
                        continue
                    remaining = cap.get(d2, np.inf) - load[d2]
                    if remaining <= 0:
                        continue
                    avg2 = town_avg_dist(t, d2)
                    penalty = avg2 - base
                    candidates.append((d2, penalty))
                if not candidates:
                    return None
                # 优先选择 penalty 最小者（可为负）
                return sorted(candidates, key=lambda x: (x[1], x[0]))[0]

            safety = 0
            while True:
                safety += 1
                if safety > 10000:
                    break
                # 找到任一超载仓
                over_depots = [d for d in load if load[d] > cap.get(d, np.inf)]
                if not over_depots:
                    break
                # 选择一个超载仓（按超载量降序处理）
                over_depots.sort(key=lambda d: load[d] - cap.get(d, np.inf), reverse=True)
                d0 = over_depots[0]
                # 在该仓的街道中，选择“可被接受且代价最小”的街道
                movable = []
                for t in list(depot_towns.get(d0, [])):
                    sz = township_sizes.get(t, 0.0)
                    # 找最优接收仓
                    best = best_receiver_for_town(t, d0)
                    if best is None:
                        continue
                    recv, penalty = best
                    # 仅当目标仓有足够剩余容量容纳整街道时才允许迁移
                    if (cap.get(recv, np.inf) - load[recv]) >= sz:
                        movable.append((t, recv, penalty, sz))
                if not movable:
                    # 无可迁移街道，退出（可能无法完全满足容量约束）
                    break
                # 选择 penalty 最小的街道进行迁移
                t_move, recv, pen, sz = sorted(movable, key=lambda x: (x[2], -x[3]))[0]
                # 应用迁移
                township_to_depot[t_move] = recv
                depot_towns[d0].remove(t_move)
                depot_towns[recv].append(t_move)
                load[d0] -= sz
                load[recv] += sz

            # 迁移结束后，更新 assigned_depot（仅对可达客户）
            new_assign_pairs = {}
            for t, d in township_to_depot.items():
                for c in township_customers.get(t, []):
                    new_assign_pairs[c] = d
            assigned_depot = pd.Series(new_assign_pairs, name="depot")

        elif assign_strategy == "district_capacity":
            # === 阶段1：同区域一致，按“多数的最近仓”确定区域归属 ===
            if 'district' not in df_cus2.columns:
                raise KeyError("使用 'district_capacity' 策略需要 df_cus 中包含列 'district'")
            nearest_each = D_feas.idxmin(axis=0)  # Series: customer -> nearest depot
            cus_meta = df_cus2.set_index('BB_RETAIL_CUSTOMER_CODE')[['district']].copy()
            cus_meta['district'] = cus_meta['district'].astype(str)
            cus_meta.index = cus_meta.index.astype(str)
            cus_meta = cus_meta.loc[nearest_each.index.intersection(cus_meta.index)]
            district_to_depot = {}
            for t, idx in cus_meta.groupby('district').groups.items():
                cust_ids = list(map(str, idx.tolist()))
                votes = nearest_each.loc[cust_ids].value_counts()
                top_cnt = votes.max()
                candidates = votes[votes == top_cnt].index.tolist()
                if len(candidates) == 1:
                    chosen = candidates[0]
                else:
                    subD = D_feas.loc[:, cust_ids]
                    avg_by_depot = subD.loc[candidates].replace(np.inf, np.nan).mean(axis=1)
                    chosen = avg_by_depot.idxmin()
                district_to_depot[str(t)] = str(chosen)
            assign_pairs = {}
            for c in nearest_each.index.tolist():
                t = cus_meta.at[c, 'district']
                assign_pairs[c] = district_to_depot.get(str(t))
            assigned_depot = pd.Series(assign_pairs, name="depot")

            # === 阶段2：按运力再平衡（整区域迁移） ===
            if depot_capacity is None:
                raise ValueError("使用 'district_capacity' 策略需要提供 depot_capacity（{仓: 可承载户数}）")
            cap = {str(d): float(depot_capacity.get(str(d), np.inf)) for d in D_feas.index.tolist()}
            district_customers = {str(t): list(map(str, idx.tolist()))
                                  for t, idx in cus_meta.groupby('district').groups.items()}
            district_sizes = {t: len(cols) for t, cols in district_customers.items()}
            load = {d: 0.0 for d in D_feas.index.tolist()}
            depot_dists = {d: [] for d in D_feas.index.tolist()}
            for t, d in district_to_depot.items():
                sz = float(district_sizes.get(t, 0))
                load[d] += sz
                depot_dists[d].append(t)

            def group_avg_dist(t: str, d: str) -> float:
                cols = district_customers.get(t, [])
                if not cols:
                    return np.inf
                sub = D_feas.loc[d, cols].astype(float).replace([np.inf, -np.inf], np.nan)
                return float(np.nanmean(sub.values))

            def best_receiver_for_group(t: str, cur_d: str) -> Optional[Tuple[str, float]]:
                candidates = []
                base = group_avg_dist(t, cur_d)
                for d2 in D_feas.index.tolist():
                    if d2 == cur_d:
                        continue
                    remaining = cap.get(d2, np.inf) - load[d2]
                    if remaining <= 0:
                        continue
                    avg2 = group_avg_dist(t, d2)
                    penalty = avg2 - base
                    candidates.append((d2, penalty))
                if not candidates:
                    return None
                return sorted(candidates, key=lambda x: (x[1], x[0]))[0]

            safety = 0
            while True:
                safety += 1
                if safety > 10000:
                    break
                over_depots = [d for d in load if load[d] > cap.get(d, np.inf)]
                if not over_depots:
                    break
                over_depots.sort(key=lambda d: load[d] - cap.get(d, np.inf), reverse=True)
                d0 = over_depots[0]
                movable = []
                for t in list(depot_dists.get(d0, [])):
                    sz = district_sizes.get(t, 0.0)
                    best = best_receiver_for_group(t, d0)
                    if best is None:
                        continue
                    recv, penalty = best
                    if (cap.get(recv, np.inf) - load[recv]) >= sz:
                        movable.append((t, recv, penalty, sz))
                if not movable:
                    break
                t_move, recv, pen, sz = sorted(movable, key=lambda x: (x[2], -x[3]))[0]
                district_to_depot[t_move] = recv
                depot_dists[d0].remove(t_move)
                depot_dists[recv].append(t_move)
                load[d0] -= sz
                load[recv] += sz

            new_assign_pairs = {}
            for t, d in district_to_depot.items():
                for c in district_customers.get(t, []):
                    new_assign_pairs[c] = d
            assigned_depot = pd.Series(new_assign_pairs, name="depot")

        else:
            raise ValueError(f"未知 assign_strategy: {assign_strategy}. 允许: 'nearest'|'random'|'balanced'|'township_capacity'|'district_capacity'")


    # 结果容器
    dict_dist_plan: Dict[str, set] = {d: set() for d in depot_ids}
    unassigned: list[str] = []

    # 1) 正常分配
    for c, d in assigned_depot.items():
        if pd.isna(d):
            continue
        dict_dist_plan[str(d)].add(str(c))

    # 2) 对“不可达”客户的处理
    if assign_unreachable_to_nearest:
        # 有 D_cd 的情况下：对每个不可达客户列，直接选该列的 idxmin（全 inf 时取第一个仓）
        for c in unreachable_cols:
            # 全 inf → idxmin 返回第一个索引；保证确定性
            d_star = D_cd.loc[:, c].idxmin()
            dict_dist_plan[str(d_star)].add(str(c))
    else:
        unassigned.extend(unreachable_cols)

    # 3) 对“缺坐标且没有 D_cd”客户的处理
    if post_assign_missing_coords:
        if assign_unreachable_to_nearest:
            # 若使用“按行政单元一致”的策略，优先按该客户所属单元的已定归属落盘
            group_field = None
            if assign_strategy == "township_capacity" and ('township' in df_cus.columns):
                group_field = "township"
            elif assign_strategy == "district_capacity" and ('district' in df_cus.columns):
                group_field = "district"
            if (group_field is not None) and (not assigned_depot.empty):
                _group_map = df_cus.set_index('BB_RETAIL_CUSTOMER_CODE')[group_field]
                unit2depot = {}
                for c_id, d_id in assigned_depot.items():
                    u = str(_group_map.get(c_id, ''))
                    if u:
                        unit2depot[u] = str(d_id)
                for c in map(str, post_assign_missing_coords):
                    u = str(_group_map.get(c, ''))
                    if u and u in unit2depot:
                        dict_dist_plan[unit2depot[u]].add(c)
                    else:
                        dict_dist_plan[first_depot].add(c)
            else:
                # 无法依据行政单元，则回退到第一个仓
                for c in map(str, post_assign_missing_coords):
                    dict_dist_plan[first_depot].add(c)
        else:
            unassigned.extend(map(str, post_assign_missing_coords))

    # 去重 & 返回
    unassigned = sorted(set(unassigned))

    add_n35_log_info(n34_id, "ALG_04",
                     f"范围规划算法-范围划分-完成客户到发货点的分配，共分配客户数={sum(len(v) for v in dict_dist_plan.values())}，未分配客户数={len(unassigned)}")

    if return_unassigned:
        return dict_dist_plan, (unassigned or None)

    return dict_dist_plan, None


# === folium 可视化：发货点(正方形) — 零售户(圆点) 分配结果 ===
def visualize_dist_plan_folium(
    dict_dist_plan: Dict[str, set],
    df_cus: pd.DataFrame,
    df_depot: pd.DataFrame,
    html_path: str = None,
    tiles: str = "OpenStreetMap",
    draw_lines: bool = False,
) -> str:
    """
    用 folium 展示分配结果：发货点用正方形，零售户用圆点。

    新增功能
    ----
    1) 左上角下拉选择框：可选择单个发货点或“全部发货点”进行展示；
       同时仍保留传统的图层复选框（LayerControl），支持多选开关。
    2) 统计信息：
       - 右下角统计面板显示：发货点数量、零售户总数；
       - 当选择某个发货点时，显示该发货点包含的零售户数量；
       - 图例中各发货点项后附带零售户数量。

    参数
    ----
    dict_dist_plan : {depot_code: set(customer_codes)}
    df_cus : 需包含 [BB_RETAIL_CUSTOMER_CODE, MD03_RETAIL_CUST_LON, MD03_RETAIL_CUST_LAT]
    df_depot : 需包含 [MD03_SHIPMENT_STATION_CODE, MD03_SHIPMENT_STATION_LON, MD03_SHIPMENT_STATION_LAT]
    html_path : 输出 HTML 路径，默认根据时间与仓数量自动命名
    tiles : folium 瓦片底图
    draw_lines : 是否绘制“客户→仓”的连线（细线，便于查看归属）

    返回
    ----
    生成的 HTML 路径
    """
    import folium
    from folium import FeatureGroup
    from branca.element import MacroElement
    from jinja2 import Template

    # 统一 ID 为字符串，便于 join
    df_cus_ = df_cus.copy()
    df_cus_['BB_RETAIL_CUSTOMER_CODE'] = df_cus_['BB_RETAIL_CUSTOMER_CODE'].astype(str)
    df_depot_ = df_depot.copy()
    df_depot_['MD03_SHIPMENT_STATION_CODE'] = df_depot_['MD03_SHIPMENT_STATION_CODE'].astype(str)

    # 过滤出坐标齐全的数据
    cus_ok = df_cus_.dropna(subset=['MD03_RETAIL_CUST_LON', 'MD03_RETAIL_CUST_LAT'])
    dep_ok = df_depot_.dropna(subset=['MD03_SHIPMENT_STATION_LON', 'MD03_SHIPMENT_STATION_LAT'])

    # 保证存在 township/district 列，便于后续可视化（如缺失则以 NA 占位）
    if 'township' not in cus_ok.columns:
        cus_ok['township'] = pd.NA
    if 'district' not in cus_ok.columns:
        cus_ok['district'] = pd.NA

    if cus_ok.empty or dep_ok.empty:
        raise ValueError("可视化失败：客户或发货点缺少有效坐标。")

    # 计算视图中心与边界
    all_lats = pd.concat([
        cus_ok['MD03_RETAIL_CUST_LAT'].astype(float),
        dep_ok['MD03_SHIPMENT_STATION_LAT'].astype(float)
    ])
    all_lons = pd.concat([
        cus_ok['MD03_RETAIL_CUST_LON'].astype(float),
        dep_ok['MD03_SHIPMENT_STATION_LON'].astype(float)
    ])
    center = [all_lats.mean(), all_lons.mean()]

    m = folium.Map(location=center, zoom_start=11, tiles=tiles, control_scale=True)

    # 颜色生成：使用固定的深色调色板 + 按仓编码排序的确定性映射（不使用 hash）
    strong_palette = [
        "red", "blue", "green", "purple", "orange", "darkred", "darkblue",
        "darkgreen", "cadetblue", "darkpurple", "black", "brown",
        "darkorange", "gray", "navy", "maroon", "teal"
    ]
    # 按仓编码排序，固定顺序分配颜色，保证每次运行一致；超过颜色数则循环
    depot_order = [str(k) for k in sorted(dict_dist_plan.keys(), key=lambda x: str(x))]
    color_map = {code: strong_palette[i % len(strong_palette)] for i, code in enumerate(depot_order)}
    def depot_color(depot_code: str) -> str:
        return color_map.get(str(depot_code), strong_palette[0])

    # === 分仓分组：每个发货点一个 FeatureGroup，便于选择和统计 ===
    depot_groups = {}  # depot_code -> FeatureGroup
    depot_counts = {}  # depot_code -> int 客户数量

    # 独立的发货点总图层，便于单独控制显示与否
    depots_layer = FeatureGroup(name="发货点（总开关）", show=True)

    # 先准备一个查找用的发货点坐标索引
    dep_xy = dep_ok.set_index('MD03_SHIPMENT_STATION_CODE')[['MD03_SHIPMENT_STATION_LON','MD03_SHIPMENT_STATION_LAT']].astype(float)

    # 为每个发货点创建图层
    for d_code, customers in dict_dist_plan.items():
        d_code = str(d_code)
        color = depot_color(d_code)
        fg = FeatureGroup(name=f"{d_code}")

        # 发货点正方形（现在加入 depots_layer，而不是 fg）
        if d_code in dep_xy.index:
            d_lon, d_lat = dep_xy.loc[d_code, ['MD03_SHIPMENT_STATION_LON','MD03_SHIPMENT_STATION_LAT']]
            folium.RegularPolygonMarker(
                location=[d_lat, d_lon],
                number_of_sides=4,
                radius=15,
                rotation=45,
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=folium.Popup(f"发货点: {d_code}", max_width=250)
            ).add_to(depots_layer)
        else:
            d_lon = d_lat = None

        # 客户点
        sub = cus_ok[cus_ok['BB_RETAIL_CUSTOMER_CODE'].isin({str(c) for c in customers})]
        depot_counts[d_code] = int(len(sub))
        for _, crow in sub.iterrows():
            c_code = str(crow['BB_RETAIL_CUSTOMER_CODE'])
            c_lon = float(crow['MD03_RETAIL_CUST_LON'])
            c_lat = float(crow['MD03_RETAIL_CUST_LAT'])
            # 组装包含 township / district 的弹窗
            popup_html = f"客户: {c_code}&lt;br&gt;发货点: {d_code}"
            _tw = crow.get('township', None)
            _ds = crow.get('district', None)
            try:
                import pandas as _pd
                if _pd.notna(_tw):
                    popup_html += f"&lt;br&gt;街道: {_tw}"
                if _pd.notna(_ds):
                    popup_html += f"&lt;br&gt;区域: {_ds}"
            except Exception:
                # 容错：即便 pandas 不可用也不影响渲染
                if _tw not in (None, "", "nan"):
                    popup_html += f"&lt;br&gt;街道: {_tw}"
                if _ds not in (None, "", "nan"):
                    popup_html += f"&lt;br&gt;区域: {_ds}"

            folium.CircleMarker(
                location=[c_lat, c_lon],
                radius=4,
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(fg)
            if draw_lines and d_lat is not None and d_lon is not None:
                folium.PolyLine(
                    locations=[[c_lat, c_lon], [d_lat, d_lon]],
                    color=color,
                    weight=1,
                    opacity=0.35
                ).add_to(fg)

        fg.add_to(m)
        depot_groups[d_code] = fg

    # 将发货点总图层添加到地图
    depots_layer.add_to(m)

    # 适配边界
    m.fit_bounds([[all_lats.min(), all_lons.min()], [all_lats.max(), all_lons.max()]])

    # === 统计面板 & 图例（含数量） ===
    # 图例（右下角）：显示颜色 + 发货点编码 + 数量
    items = []
    for d_code in dict_dist_plan.keys():
        items.append((str(d_code), depot_color(d_code), depot_counts.get(str(d_code), 0)))

    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="position: fixed; bottom: 20px; right: 20px; z-index: 9999; background: rgba(255,255,255,0.92); padding: 10px 12px; border: 1px solid #bbb; border-radius: 4px; font-size: 12px; max-height: 40vh; overflow:auto;">
      <div style="font-weight:600; margin-bottom:6px;">图例：发货点颜色（含零售户数）</div>
      {% for name, color, cnt in this.items %}
        <div style="margin:2px 0; display:flex; align-items:center;">
          <span style="display:inline-block; width:14px; height:14px; background: {{color}}; border:1px solid #555; margin-right:6px;"></span>
          <span>{{name}}（{{cnt}}）</span>
        </div>
      {% endfor %}
    </div>
    {% endmacro %}
    """
    class _Legend(MacroElement):
        def __init__(self, items):
            super().__init__()
            self._name = 'Legend'
            self.items = items
            self.template = Template(legend_html)
        def render(self, **kwargs):
            macro = self.template.module.html
            figure = self.get_root()
            html = macro(self, kwargs)
            figure.header.add_child(folium.Element(html))
            return super().render(**kwargs)

    m.add_child(_Legend(items))

    # 统计信息面板（左下角）
    total_depots = len(depot_groups)
    total_customers = int(sum(depot_counts.values()))
    stats_html = f"""
    {{% macro html(this, kwargs) %}}
    <div id="dist-stats" style="position: fixed; bottom: 20px; left: 20px; z-index: 9999; background: rgba(255,255,255,0.92); padding: 10px 12px; border: 1px solid #bbb; border-radius: 4px; font-size: 12px;">
      <div style="font-weight:600; margin-bottom:6px;">统计信息</div>
      <div>发货点数量：{total_depots}</div>
      <div>零售户总数：{total_customers}</div>
      <div id="stats-selected" style="margin-top:4px; color:#333;">当前选择：全部发货点</div>
    </div>
    {{% endmacro %}}
    """
    class _Stats(MacroElement):
        def __init__(self, html_tpl):
            super().__init__()
            self._name = 'Stats'
            self.template = Template(html_tpl)
        def render(self, **kwargs):
            macro = self.template.module.html
            figure = self.get_root()
            html = macro(self, kwargs)
            figure.header.add_child(folium.Element(html))
            return super().render(**kwargs)

    m.add_child(_Stats(stats_html))


    # === 按街道(township) 与 按区域(district) 的可视化图层 ===
    # 说明：为避免初始渲染过慢，这两组图层默认 show=False，用户可在右上角图层面板中按需开启。
    # 数据准备（容错处理列缺失）
    cus_ext = cus_ok.copy()
    if 'township' not in cus_ext.columns:
        cus_ext['township'] = _pd.NA if ' _pd' in globals() else None
    if 'district' not in cus_ext.columns:
        cus_ext['district'] = _pd.NA if ' _pd' in globals() else None

    # 规范显示值
    def _norm_str(s):
        try:
            return str(s) if (s is not None and not (isinstance(s, float) and math.isnan(s))) else "未知"
        except Exception:
            return str(s) if s not in (None, "", "nan") else "未知"

    cus_ext['__township_disp__'] = cus_ext['township'].map(_norm_str)
    cus_ext['__district_disp__'] = cus_ext['district'].map(_norm_str)

    # 固定深色调色板并生成确定性的颜色映射
    palette_town = [
        "red", "blue", "green", "purple", "orange", "darkred", "darkblue",
        "darkgreen", "cadetblue", "darkpurple", "black", "brown",
        "darkorange", "gray", "navy", "maroon", "teal"
    ]
    palette_dist = [
        "black", "darkblue", "darkgreen", "darkred", "darkpurple", "orange",
        "cadetblue", "brown", "navy", "teal", "gray", "purple", "green", "red"
    ]

    # === 1) 按街道分层 ===
    town_layers = {}
    towns = sorted(cus_ext['__township_disp__'].astype(str).unique())
    town_color = {t: palette_town[i % len(palette_town)] for i, t in enumerate(towns)}
    for t in towns:
        sub = cus_ext[cus_ext['__township_disp__'] == t]
        fg_t = FeatureGroup(name=f"街道 | {t}（{len(sub)}）", show=False)
        for _, r in sub.iterrows():
            try:
                c_lon = float(r['MD03_RETAIL_CUST_LON'])
                c_lat = float(r['MD03_RETAIL_CUST_LAT'])
            except Exception:
                continue
            c_code = str(r['BB_RETAIL_CUSTOMER_CODE'])
            # 弹窗：客户 + 街道 + 区域
            pop = f"客户: {c_code}"
            if _norm_str(r.get('township', None)) != "未知":
                pop += f"&lt;br&gt;街道: {r.get('township')}"
            if _norm_str(r.get('district', None)) != "未知":
                pop += f"&lt;br&gt;区域: {r.get('district')}"
            folium.CircleMarker(
                location=[c_lat, c_lon],
                radius=4,
                color=town_color[t],
                weight=1,
                fill=True,
                fill_color=town_color[t],
                fill_opacity=0.75,
                popup=folium.Popup(pop, max_width=300)
            ).add_to(fg_t)
        fg_t.add_to(m)
        town_layers[t] = fg_t

    # === 2) 按区域分层 ===
    dist_layers = {}
    dists = sorted(cus_ext['__district_disp__'].astype(str).unique())
    dist_color = {d: palette_dist[i % len(palette_dist)] for i, d in enumerate(dists)}
    for d in dists:
        sub = cus_ext[cus_ext['__district_disp__'] == d]
        fg_d = FeatureGroup(name=f"区域 | {d}（{len(sub)}）", show=False)
        for _, r in sub.iterrows():
            try:
                c_lon = float(r['MD03_RETAIL_CUST_LON'])
                c_lat = float(r['MD03_RETAIL_CUST_LAT'])
            except Exception:
                continue
            c_code = str(r['BB_RETAIL_CUSTOMER_CODE'])
            pop = f"客户: {c_code}"
            if _norm_str(r.get('township', None)) != "未知":
                pop += f"&lt;br&gt;街道: {r.get('township')}"
            if _norm_str(r.get('district', None)) != "未知":
                pop += f"&lt;br&gt;区域: {r.get('district')}"
            folium.CircleMarker(
                location=[c_lat, c_lon],
                radius=4,
                color=dist_color[d],
                weight=1,
                fill=True,
                fill_color=dist_color[d],
                fill_opacity=0.75,
                popup=folium.Popup(pop, max_width=300)
            ).add_to(fg_d)
        fg_d.add_to(m)
        dist_layers[d] = fg_d


    # === 统一的左上角控制面板：发货点选择 + 街道/区域 全选/全不选 ===
    map_var = m.get_name()
    options = ["<option value='__ALL__' selected>全部发货点</option>"] + [f"<option value='{d}'>{d}（{depot_counts.get(d,0)}）</option>" for d in depot_groups.keys()]
    combined_panel_html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="position: fixed; top: 20px; left: 60px; z-index: 10000; background: rgba(255,255,255,0.95); padding: 8px 10px; border: 1px solid #bbb; border-radius: 4px; font-size: 12px;">
      <div style="display:flex; gap:14px; align-items:center; flex-wrap:wrap;">
        <div>
          <label style="margin-right:6px;">发货点：</label>
          <select id="depotSelect" style="min-width: 220px;">
            {''.join(options)}
          </select>
        </div>
        <label style="display:flex; align-items:center; gap:6px; white-space:nowrap;">
          <input type="checkbox" id="toggleDepots" checked /> 显示发货点
        </label>
        <label style="display:flex; align-items:center; gap:6px; white-space:nowrap;">
          <input type="checkbox" id="toggleTownAll" /> 街道 全选/全不选
        </label>
        <label style="display:flex; align-items:center; gap:6px; white-space:nowrap;">
          <input type="checkbox" id="toggleDistAll" /> 区域 全选/全不选
        </label>
      </div>
    </div>
    <script>
      (function() {{
        // 绑定 Python 端创建的图层到 JS 变量
        var depotGroups = {{}};
        {''.join([f"depotGroups['{k}'] = {depot_groups[k].get_name()};" for k in depot_groups.keys()])}
        var depotsLayer = {depots_layer.get_name()};
  
        var townGroups = {{}};
        {''.join([f"townGroups['{k}'] = {town_layers[k].get_name()};" for k in town_layers.keys()])}
        var distGroups = {{}};
        {''.join([f"distGroups['{k}'] = {dist_layers[k].get_name()};" for k in dist_layers.keys()])}
  
        function setAll(groups, on) {{
          for (var k in groups) {{
            var layer = groups[k];
            var has = {map_var}.hasLayer(layer);
            if (on && !has) {{
              {map_var}.addLayer(layer);
            }} else if (!on && has) {{
              {map_var}.removeLayer(layer);
            }}
          }}
        }}
  
        function showOnlyDepot(value) {{
          var statsSel = document.getElementById('stats-selected');
          if (value === '__ALL__') {{
            for (var k in depotGroups) {{
              if (!{map_var}.hasLayer(depotGroups[k])) {{ {map_var}.addLayer(depotGroups[k]); }}
            }}
            if (statsSel) statsSel.textContent = '当前选择：全部发货点';
          }} else {{
            for (var k in depotGroups) {{
              if (k === value) {{
                if (!{map_var}.hasLayer(depotGroups[k])) {{ {map_var}.addLayer(depotGroups[k]); }}
              }} else {{
                if ({map_var}.hasLayer(depotGroups[k])) {{ {map_var}.removeLayer(depotGroups[k]); }}
              }}
            }}
            if (statsSel) statsSel.textContent = '当前选择：' + value + '（' + {depot_counts!r}[value] + ' 户）';
          }}
        }}
  
        var sel = document.getElementById('depotSelect');
        sel.addEventListener('change', function() {{ showOnlyDepot(this.value); }});
  
        var chkDepots = document.getElementById('toggleDepots');
        chkDepots.addEventListener('change', function() {{
          if (this.checked) {{
            if (!{map_var}.hasLayer(depotsLayer)) {{ {map_var}.addLayer(depotsLayer); }}
          }} else {{
            if ({map_var}.hasLayer(depotsLayer)) {{ {map_var}.removeLayer(depotsLayer); }}
          }}
        }});
  
        var chkTown = document.getElementById('toggleTownAll');
        var chkDist = document.getElementById('toggleDistAll');
        chkTown.checked = false;  // 与 LayerControl 初始一致（show=False）
        chkDist.checked = false;
        chkTown.addEventListener('change', function() {{ setAll(townGroups, this.checked); }});
        chkDist.addEventListener('change', function() {{ setAll(distGroups, this.checked); }});
      }})();
    </script>
    {{% endmacro %}}
    """
    class _CombinedPanel(MacroElement):
        def __init__(self, html_tpl):
            super().__init__()
            self._name = 'CombinedPanel'
            self.template = Template(html_tpl)
        def render(self, **kwargs):
            macro = self.template.module.html
            figure = self.get_root()
            html = macro(self, kwargs)
            figure.header.add_child(folium.Element(html))
            return super().render(**kwargs)

    m.add_child(_CombinedPanel(combined_panel_html))

    # 加一个图层控制器（右上角），支持复选切换
    folium.LayerControl(collapsed=False).add_to(m)

    # 保存
    if html_path is None:
        import datetime as _dt
        html_path = f"dist_plan_map_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    m.save(html_path)
    return html_path



def _normalize_code(series: pd.Series) -> pd.Series:
    """将编码转为可匹配的标准字符串：去空格、去尾部'.0'等。"""
    return (
        series.astype(str)
              .str.strip()
              .str.replace(r'\.0$', '', regex=True)
    )

def round_list_keep_sum(lst):
    """
    对列表中的每个元素进行四舍五入，使得结果的总和与原始总和保持一致。
    优化点：
    - 先对每个元素取整（四舍五入）
    - 计算总和误差
    - 按照小数部分大小排序，优先调整误差
    """
    rounded = [int(round(x)) for x in lst]
    diff = int(round(sum(lst) - sum(rounded)))
    if diff == 0:
        return rounded
    # 计算每个元素的小数部分
    decimals = [x - int(x) for x in lst]
    # 获取调整顺序：diff>0时优先加，diff<0时优先减
    if diff > 0:
        # 需要加1，优先选小数部分最大的
        adjust_indices = sorted(range(len(lst)), key=lambda i: (-(lst[i] - int(lst[i]))))
    else:
        # 需要减1，优先选小数部分最小的
        adjust_indices = sorted(range(len(lst)), key=lambda i: (lst[i] - int(lst[i])))
    for i in adjust_indices[:abs(diff)]:
        rounded[i] += 1 if diff > 0 else -1
    return rounded




def combine_table28(db_n28: pd.DataFrame, customers_path: str) -> pd.DataFrame:
    """
    将 customers_with_address_complete_deduplicated_29868_osmid 表（文件）与 db_n28 按编码关联，
    向 db_n28 增加列：district, township, city, distance_to_station, OSM_X, OSM_Y。

    参数
    ----
    db_n28 : DataFrame，包含列 BB_RETAIL_CUSTOMER_CODE
    customers_path : str，customers 文件路径（支持 .csv/.parquet）

    返回
    ----
    DataFrame：在 db_n28 基础上补充了目标字段；同时在控制台打印未匹配上的行数与详情。
    """
    # --- 读取 customers 文件（支持 csv/parquet；容错 BOM） ---
    import os
    _, ext = os.path.splitext(str(customers_path).lower())
    if ext in {".parquet", ".pq"}:
        customers = pd.read_parquet(customers_path)
    else:
        # 默认按 CSV 读取，处理 BOM
        customers = pd.read_csv(customers_path, encoding="utf-8-sig")

    # 需从 customers 带过来的列
    take_cols = [
        'customer_code',           # 先带上以便合并
        'district',
        'township',
        'city',
        'distance_to_station',
        'OSM_X',
        'OSM_Y',
    ]

    # 只保留所需列并去重（以 customer_code 为键保留第一条）
    customers_slim = customers.loc[:, [c for c in take_cols if c in customers.columns]].copy()
    if 'customer_code' not in customers_slim.columns:
        raise KeyError("customers 文件缺少列 'customer_code'")

    customers_slim = (
        customers_slim
        .assign(customer_code_norm=_normalize_code(customers_slim['customer_code']))
        .drop_duplicates(subset=['customer_code_norm'], keep='first')
    )

    # 规范化 db_n28 的键
    if 'BB_RETAIL_CUSTOMER_CODE' not in db_n28.columns:
        raise KeyError("db_n28 缺少列 'BB_RETAIL_CUSTOMER_CODE'")
    db_n28_norm = db_n28.copy()
    db_n28_norm['BB_RETAIL_CUSTOMER_CODE_norm'] = _normalize_code(db_n28_norm['BB_RETAIL_CUSTOMER_CODE'])

    # 组装 customers 侧用于合并的列并改名
    add_cols = ['district', 'township', 'city', 'distance_to_station', 'OSM_X', 'OSM_Y']
    right_for_merge = customers_slim.rename(
        columns={'customer_code_norm': 'BB_RETAIL_CUSTOMER_CODE_norm'}
    )[ ['BB_RETAIL_CUSTOMER_CODE_norm'] + [c for c in add_cols if c in customers_slim.columns] ]

    # 合并（以 db_n28 为主表）
    merged = db_n28_norm.merge(
        right_for_merge,
        on='BB_RETAIL_CUSTOMER_CODE_norm',
        how='left',
        indicator=True
    )

    # 统计与打印未匹配行（来自 db_n28 的 left_only）
    unmatched_mask = merged['_merge'].eq('left_only')
    unmatched_rows = merged.loc[unmatched_mask, db_n28.columns]  # 仅显示原 db_n28 列作为“详情”
    print(f"[combine_table28] 未匹配到的行数：{unmatched_rows.shape[0]}")
    if unmatched_rows.shape[0] > 0:
        print("[combine_table28] 未匹配到的行详情：")
        # 如数据很多，可替换为 unmatched_rows.head(200).to_string(...)
        print(unmatched_rows.to_string(index=False))

    # 将新增列放回（保持原列不动，仅追加/覆盖新增列）
    for col in add_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    # 清理辅助列
    merged = merged.drop(columns=['BB_RETAIL_CUSTOMER_CODE_norm', '_merge'], errors='ignore')

    # 返回包含新增列的 DataFrame
    return merged


def calculate_depot_capacity(df_veh, df_depot, df_cus, ratio_scale=1.05):
    """
    计算每个发货点的运力(capacity)，按车辆发货量比例分配客户数，返回cap字典。
    参数:
        df_veh: 车辆信息DataFrame，需包含MD03_SHIPMENT_STATION_CODE和MD03_MAX_LODING_CARTON_QTY
        df_depot: 发货点信息DataFrame，需包含MD03_SHIPMENT_STATION_CODE
        df_cus: 客户信息DataFrame
        ratio_scale: 比例放大系数，默认1.05
    返回:
        cap: {发货点编码: 容量}
    """
    df_depot_capacity = (
        df_veh.groupby('MD03_SHIPMENT_STATION_CODE', as_index=False)
              .agg({'MD03_MAX_LODING_CARTON_QTY': 'sum'})
              .rename(columns={'MD03_MAX_LODING_CARTON_QTY': 'total_capacity'})
    )
    total_capacity_sum = df_depot_capacity['total_capacity'].sum()
    df_depot_capacity['capacity_ratio'] = df_depot_capacity['total_capacity'] / total_capacity_sum

    total_customers = df_cus.shape[0]
    ratio = df_depot_capacity['capacity_ratio']
    total_ratio = sum(ratio)
    cap_values = [total_customers * r / total_ratio for r in ratio]
    cap_values_rounded = round_list_keep_sum(cap_values)
    cap = {str(df_depot.iloc[i]['MD03_SHIPMENT_STATION_CODE']): cap_values_rounded[i] * ratio_scale
           for i in range(len(cap_values_rounded))}
    print(f"总客户数: {total_customers}, 按比例分配的容量: {cap_values}, 四舍五入后: {cap_values_rounded}")
    return cap


# --- Reusable address component parsing ---
def parse_md03_address_component(
    df: pd.DataFrame,
    source_col: str = 'MD03_ADDRESS_COMPONENT',
    *,
    city_key: str = 'city',
    district_key: str = 'district',
    township_key: str = 'township',
    overwrite: bool = True,
) -> pd.DataFrame:
    """Parse JSON in `source_col` and populate `city`, `district`, `township`.

    - Robust to dict/str/bytes values and UTF-8 BOM.
    - If a row fails to parse, outputs NA for that row.
    - Set `overwrite=False` to keep existing non-null values in target columns.
    """
    import json, ast
    if df is None or df.empty or source_col not in df.columns:
        return df
    df_out = df.copy()

    def _to_dict(val):
        # Already a mapping
        if isinstance(val, dict):
            return val
        # Bytes → decode with utf-8-sig then parse
        if isinstance(val, (bytes, bytearray)):
            try:
                s = val.decode('utf-8-sig', errors='ignore')
                return json.loads(s)
            except Exception:
                return {}
        # Strings → try json, then literal_eval fallback
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return {}
            # remove possible BOM
            if s and s[0] == '\ufeff':
                s = s.lstrip('\ufeff')
            try:
                return json.loads(s)
            except Exception:
                try:
                    obj = ast.literal_eval(s)
                    return obj if isinstance(obj, dict) else {}
                except Exception:
                    return {}
        # Other types → empty
        return {}

    parsed = df_out[source_col].map(_to_dict)
    city_series = parsed.map(lambda x: x.get(city_key) if isinstance(x, dict) else None)
    district_series = parsed.map(lambda x: x.get(district_key) if isinstance(x, dict) else None)
    township_series = parsed.map(lambda x: x.get(township_key) if isinstance(x, dict) else None)

    for col_name, series in (
        ('city', city_series),
        ('district', district_series),
        ('township', township_series),
    ):
        if overwrite or (col_name not in df_out.columns):
            df_out[col_name] = series
        else:
            # only fill NA if keeping existing values
            df_out[col_name] = df_out[col_name].where(df_out[col_name].notna(), series)

    return df_out


# 新增个main函数，方便直接运行调试
if __name__ == "__main__":
    from db.range_data_service import pre_deal_range, out_n36_n37
    import datetime as _dt

    # === Cache options ===
    USE_LOCAL_CACHE = True   # True: load from local parquet; False: fetch from DB then save
    CACHE_DIR = 'cache_vrp'   # folder to store parquet files

    # API参数 全局规划
    n34_id = "29374109947317922"
    plan_type = "1"  # 1全局规划 2局部规划
    n36_range_ids_str = "" # 101510486268551704

    # API参数 局部规划 【fix 会对某一发货点的零售户 要求分配多个发货点发货】
    # n34_id = "29374109947317922"
    # plan_type = "2"  # 1全局规划 2局部规划
    # n36_range_ids_str = "101546763139245980"
    # n36_range_ids_str = "101510486268541377,29374109947317927" 101510486268541377 101510486268551704
    # customer_path = "./data/customers_with_address_complete_deduplicated_29868_osmid.csv"

    """ 前置方法 """
    if USE_LOCAL_CACHE:
        tables = load_tables_local(n34_id, CACHE_DIR)
        # Expect these keys were saved previously
        db_n34 = tables.get('db_n34')
        db_n19 = tables.get('db_n19')
        db_n26 = tables.get('db_n26')
        db_n27 = tables.get('db_n27')
        db_n28 = tables.get('db_n28')
        db_n31 = tables.get('db_n31')
        db_n32 = tables.get('db_n32')
        db_n33 = tables.get('db_n33')
    else:
        status, message, db_tables = pre_deal_range(n34_id, plan_type, n36_range_ids_str)
        db_n34, db_n19, db_n26, db_n27, db_n28, db_n31, db_n32, db_n33 = db_tables.values()
        # 取到后立即保存到本地，便于后续直接读取
        save_tables_local(
            n34_id,
            {
                'db_n34': db_n34,
                'db_n19': db_n19,
                'db_n26': db_n26,
                'db_n27': db_n27,
                'db_n28': db_n28,
                'db_n31': db_n31,
                'db_n32': db_n32,
                'db_n33': db_n33,
            },
            CACHE_DIR,
        )

    # 解析 MD03_ADDRESS_COMPONENT 字段，提取 city / district / township
    db_n28 = parse_md03_address_component(db_n28)
    # db_n28 = combine_table28(db_n28, customer_path)
    df_cus, df_depot, df_veh, _ = prepare_vrp_inputs(n34_id, db_n26, db_n28, db_n27)

    # 运力计算：df_veh可能为空 如不空仅当df_veh对每个应发货点均有数据时才计算，否则不传cap参数
    if df_veh.empty:
        print("警告：车辆信息表为空，跳过运力计算。")
        cap = None
    else:
        dep_codes_with_veh = set(df_veh['MD03_SHIPMENT_STATION_CODE'].astype(str).unique())
        dep_codes_in_depot = set(df_depot['MD03_SHIPMENT_STATION_CODE'].astype(str).unique())
        if not dep_codes_in_depot.issubset(dep_codes_with_veh):
            print(f"警告：并非所有发货点均有车辆数据，跳过运力计算。发货点数={len(dep_codes_in_depot)}，有车辆数据的发货点数={len(dep_codes_with_veh)}")
            cap = None
        else:
            cap = calculate_depot_capacity(df_veh, df_depot, df_cus, ratio_scale=1.05)

    if cap is not None:
        assign_strategy = "township_capacity"
        dict_dist_plan, unassigned = plan_distribution_regions(
            n34_id,
            df_cus,  # 需包含 'township'、经纬度与客户编码
            df_depot,  # 发货点表
            D_cd=None,  # 或传入预先算好的 仓-客户 距离矩阵
            assign_unreachable_to_nearest=True,
            assign_strategy=assign_strategy, # district_capacity  township_capacity
            depot_capacity=cap,
        )
        html_map = visualize_dist_plan_folium(dict_dist_plan, df_cus, df_depot,
                                              html_path=f"dist_plan_{assign_strategy}_{n34_id}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                              draw_lines=False)
    else:
        dict_dist_plan = {}
        add_n35_log_info(n34_id, "ALG_ERROR",f"车辆信息不足，无法进行CAP求解。")

    """ 后置方法输出 """
    # out_n36_n37(n34_id, plan_type, dict_dist_plan, n36_range_ids_str)
    print("Done dist!")
