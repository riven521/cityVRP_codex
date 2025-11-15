"""Utility functions for the route planning service module."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd
from tabulate import tabulate

from ..log_service import add_n35_log_info
from .heuristics import (
    add_route_distances,
    repartition_customers_to_n_routes,
    reorder_lines_by_pyvrp,
)
from .logger import LOGGER

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

def get_customer_avg_order_qty(db_n32, db_n28, db_n26):
    """
    1 从db_n32抽取零售户BB_RETAIL_CUSTOMER_CODE的平均订单需求量MD03_ORDER_TOTAL_QTY，
    并关联到db_n28，形成新的df_n28表，增加AVG_ORDER_QTY列。
    2 增加发货点的经纬度MD03_SHIPMENT_STATION_LON MD03_SHIPMENT_STATION_LAT
    """
    df_cus_demand = db_n32[["BB_RETAIL_CUSTOMER_CODE", "MD03_ORDER_TOTAL_QTY"]].groupby("BB_RETAIL_CUSTOMER_CODE").agg(
        AVG_ORDER_QTY=("MD03_ORDER_TOTAL_QTY", "mean")
    ).reset_index()
    LOGGER.debug("Computed AVG_ORDER_QTY dtype: %s", df_cus_demand["AVG_ORDER_QTY"].dtype)
    df_cus_demand["AVG_ORDER_QTY"] = pd.to_numeric(df_cus_demand["AVG_ORDER_QTY"], errors='coerce')
    df_cus_demand["AVG_ORDER_QTY"] = df_cus_demand["AVG_ORDER_QTY"].round(2)
    df_n28 = pd.merge(db_n28, df_cus_demand, how="left", left_on="BB_RETAIL_CUSTOMER_CODE", right_on="BB_RETAIL_CUSTOMER_CODE")
    avg_order_qty_mean = df_cus_demand["AVG_ORDER_QTY"].mean()
    df_n28["AVG_ORDER_QTY"].fillna(avg_order_qty_mean, inplace=True)

    # 2 df_n28增加发货点经纬度: 从df_n26表提取增加MD03_DIST_STATION_CODE对应的经纬度
    station_coords = db_n26[["MD03_SHIPMENT_STATION_CODE", "MD03_SHIPMENT_STATION_LON", "MD03_SHIPMENT_STATION_LAT"]].drop_duplicates()
    station_coords = station_coords.set_index("MD03_SHIPMENT_STATION_CODE").astype({"MD03_SHIPMENT_STATION_LON": float, "MD03_SHIPMENT_STATION_LAT": float})
    df_n28 = pd.merge(df_n28, station_coords, how="left", left_on="MD03_DIST_STATION_CODE", right_on="MD03_SHIPMENT_STATION_CODE")

    # 3 df_n28增加客户经纬度MD03_RETAIL_CUST_LON MD03_RETAIL_CUST_LAT, 前向填充
    df_n28["MD03_RETAIL_CUST_LON"] = df_n28["MD03_RETAIL_CUST_LON"].fillna(method='ffill')
    df_n28["MD03_RETAIL_CUST_LAT"] = df_n28["MD03_RETAIL_CUST_LAT"].fillna(method='ffill')

    return df_n28

def parse_line_info(db_n32, db_n28):
    '''
    从db_n32抽取线路信息LINE_CODES和LINE_SEQ，合并到df_n28
    '''
    df_cus_vehicle = db_n32[
        ["BB_RETAIL_CUSTOMER_CODE", "CD_LOGT_COM_DEY_VEHS_CODE", "MD03_DIST_LINE_CODE", "MD03_DELIVER_DELIVER_NO"]
    ].groupby("BB_RETAIL_CUSTOMER_CODE").agg(
        LINE_CODES=("MD03_DIST_LINE_CODE", lambda x: list(x.unique())),
        LINE_SEQ=("MD03_DELIVER_DELIVER_NO", lambda x: list(x.unique())),
    ).reset_index()
    df_cus_vehicle["LINE_CODES"] = df_cus_vehicle["LINE_CODES"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else "")
    df_cus_vehicle["LINE_SEQ"] = df_cus_vehicle["LINE_SEQ"].apply(
        lambda x: ",".join(map(str, x)) if isinstance(x, list) else "")

    return pd.merge(db_n28, df_cus_vehicle, how="left", left_on="BB_RETAIL_CUSTOMER_CODE",
                    right_on="BB_RETAIL_CUSTOMER_CODE")

def parse_veh_info(db_n32, df_n28, db_n27):
    '''
    从db_n32抽取车辆VEHICLE_CODES和VEHICLE_MAX_CARTON_QTY，合并到df_n28
    '''
    # 增加客户车辆和线路信息列
    df_cus_vehicle = db_n32[
        ["BB_RETAIL_CUSTOMER_CODE", "CD_LOGT_COM_DEY_VEHS_CODE", "MD03_DIST_LINE_CODE", "MD03_DELIVER_DELIVER_NO"]
    ].groupby("BB_RETAIL_CUSTOMER_CODE").agg(
        VEHICLE_CODES=("CD_LOGT_COM_DEY_VEHS_CODE", lambda x: list(x.unique())),
    ).reset_index()

    df_cus_vehicle["VEHICLE_CODES"] = df_cus_vehicle["VEHICLE_CODES"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else "")

    # df_cus_vehicle从db_n27内抽取列MD03_MAX_LODING_CARTON_QTY到df_cus_vehicle内，形成新的df_cus_vehicle表
    df_max_carton = db_n27[["CD_LOGT_COM_DEY_VEHS_CODE", "MD03_MAX_LODING_CARTON_QTY"]].drop_duplicates()
    df_cus_vehicle = pd.merge(df_cus_vehicle, df_max_carton, how="left", left_on="VEHICLE_CODES",
                              right_on="CD_LOGT_COM_DEY_VEHS_CODE")
    df_cus_vehicle.rename(columns={"MD03_MAX_LODING_CARTON_QTY": "VEHICLE_MAX_CARTON_QTY"}, inplace=True)
    df_cus_vehicle.drop(columns=["CD_LOGT_COM_DEY_VEHS_CODE"], inplace=True)

    return pd.merge(df_n28, df_cus_vehicle, how="left", left_on="BB_RETAIL_CUSTOMER_CODE",
                    right_on="BB_RETAIL_CUSTOMER_CODE")

def build_route_summary(df_n28, db_n26):
    """
    根据客户和站点信息，构建线路汇总表。
    """
    # 提取相关列
    df_res = df_n28[["MD03_DIST_STATION_CODE", "LINE_CODES", "BB_RETAIL_CUSTOMER_CODE", "LINE_SEQ", "AVG_ORDER_QTY", "township", "district", "city"]].copy()
    # 爆炸线路号和序号
    df_res = df_res.explode(["LINE_CODES", "LINE_SEQ"])
    # 序号转为数值并排序
    df_res["LINE_SEQ"] = pd.to_numeric(df_res["LINE_SEQ"], errors='coerce')
    df_res = df_res.sort_values(by=["LINE_CODES", "LINE_SEQ"])
    # 按线路号分组聚合
    df_res = df_res.groupby("LINE_CODES").agg(
        MD03_DIST_STATION_CODE=("MD03_DIST_STATION_CODE", "first"),
        BB_RETAIL_CUSTOMER_CODE=("BB_RETAIL_CUSTOMER_CODE", lambda x: ",".join(x)),
        AVG_ORDER_QTY=("AVG_ORDER_QTY", lambda x: ",".join(map(str, x))),
        LINE_SEQ=("LINE_SEQ", lambda x: ",".join(map(str, x))),
        township=("township", lambda x: ",".join([str(i) if i is not None else "" for i in x])),
        district=("district", lambda x: ",".join([str(i) if i is not None else "" for i in x])),
        city=("city", lambda x: ",".join([str(i) if i is not None else "" for i in x])),
    ).reset_index()
    # 统计客户数和总需求
    df_res["NUM_CUSTOMERS"] = df_res["BB_RETAIL_CUSTOMER_CODE"].apply(lambda x: len(x.split(",")) if pd.notna(x) and x else 0)
    df_res["TOTAL_AVG_ORDER_QTY"] = df_res["AVG_ORDER_QTY"].apply(
        lambda x: sum([float(i) for i in x.split(",") if i.replace(".","",1).isdigit()]) if pd.notna(x) and x else 0.0
    )
    # 关联站点经纬度
    station_coords = db_n26[["MD03_SHIPMENT_STATION_CODE", "MD03_SHIPMENT_STATION_LON", "MD03_SHIPMENT_STATION_LAT"]].drop_duplicates()
    station_coords = station_coords.set_index("MD03_SHIPMENT_STATION_CODE").astype({"MD03_SHIPMENT_STATION_LON": float, "MD03_SHIPMENT_STATION_LAT": float})
    df_res = pd.merge(df_res, station_coords, how="left", left_on="MD03_DIST_STATION_CODE", right_on="MD03_SHIPMENT_STATION_CODE")
    # 提取首末客户编码
    df_res["first_customer_code"] = df_res["BB_RETAIL_CUSTOMER_CODE"].apply(lambda x: x.split(",")[0] if pd.notna(x) and x else "")
    df_res["last_customer_code"] = df_res["BB_RETAIL_CUSTOMER_CODE"].apply(lambda x: x.split(",")[-1] if pd.notna(x) and x else "")
    # 关联首末客户经纬度
    cust_coords = df_n28[["BB_RETAIL_CUSTOMER_CODE", "MD03_RETAIL_CUST_LON", "MD03_RETAIL_CUST_LAT"]].drop_duplicates()
    cust_coords = cust_coords.set_index("BB_RETAIL_CUSTOMER_CODE").astype({"MD03_RETAIL_CUST_LON": float, "MD03_RETAIL_CUST_LAT": float})
    df_res = pd.merge(df_res, cust_coords, how="left", left_on="first_customer_code", right_on="BB_RETAIL_CUSTOMER_CODE")
    df_res.rename(columns={"MD03_RETAIL_CUST_LON": "first_customer_lon", "MD03_RETAIL_CUST_LAT": "first_customer_lat"}, inplace=True)
    df_res = pd.merge(df_res, cust_coords, how="left", left_on="last_customer_code", right_on="BB_RETAIL_CUSTOMER_CODE")
    df_res.rename(columns={"MD03_RETAIL_CUST_LON": "last_customer_lon", "MD03_RETAIL_CUST_LAT": "last_customer_lat"}, inplace=True)
    return df_res

def build_route_veh_summary(df_n28, db_n26):
    """
    根据客户和站点信息，构建线路汇总表。
    """
    # 提取相关列
    df_res = df_n28[["MD03_DIST_STATION_CODE", "VEHICLE_CODES", "LINE_CODES", "BB_RETAIL_CUSTOMER_CODE", "LINE_SEQ", "VEHICLE_MAX_CARTON_QTY", "AVG_ORDER_QTY", "township", "district", "city"]].copy()
    # 爆炸线路号和序号
    df_res = df_res.explode(["LINE_CODES", "LINE_SEQ"])
    # 序号转为数值并排序
    df_res["LINE_SEQ"] = pd.to_numeric(df_res["LINE_SEQ"], errors='coerce')
    df_res = df_res.sort_values(by=["LINE_CODES", "LINE_SEQ"])
    # 按线路号分组聚合
    df_res = df_res.groupby("LINE_CODES").agg(
        MD03_DIST_STATION_CODE=("MD03_DIST_STATION_CODE", "first"),
        VEHICLE_CODES=("VEHICLE_CODES", "first"),
        VEHICLE_MAX_CARTON_QTY=("VEHICLE_MAX_CARTON_QTY", "first"),
        BB_RETAIL_CUSTOMER_CODE=("BB_RETAIL_CUSTOMER_CODE", lambda x: ",".join(x)),
        AVG_ORDER_QTY=("AVG_ORDER_QTY", lambda x: ",".join(map(str, x))),
        LINE_SEQ=("LINE_SEQ", lambda x: ",".join(map(str, x))),
        township=("township", lambda x: ",".join([str(i) if i is not None else "" for i in x])),
        district=("district", lambda x: ",".join([str(i) if i is not None else "" for i in x])),
        city=("city", lambda x: ",".join([str(i) if i is not None else "" for i in x])),
    ).reset_index()
    # 统计客户数和总需求
    df_res["NUM_CUSTOMERS"] = df_res["BB_RETAIL_CUSTOMER_CODE"].apply(lambda x: len(x.split(",")) if pd.notna(x) and x else 0)
    df_res["TOTAL_AVG_ORDER_QTY"] = df_res["AVG_ORDER_QTY"].apply(
        lambda x: sum([float(i) for i in x.split(",") if i.replace(".","",1).isdigit()]) if pd.notna(x) and x else 0.0
    )
    # 计算装载率
    df_res["LOAD_RATIO"] = df_res.apply(
        lambda row: round(row["TOTAL_AVG_ORDER_QTY"] / float(row["VEHICLE_MAX_CARTON_QTY"]), 4)
        if row["VEHICLE_MAX_CARTON_QTY"] and row["VEHICLE_MAX_CARTON_QTY"] > 0 else 0.0,
        axis=1
    )
    # 关联站点经纬度
    station_coords = db_n26[["MD03_SHIPMENT_STATION_CODE", "MD03_SHIPMENT_STATION_LON", "MD03_SHIPMENT_STATION_LAT"]].drop_duplicates()
    station_coords = station_coords.set_index("MD03_SHIPMENT_STATION_CODE").astype({"MD03_SHIPMENT_STATION_LON": float, "MD03_SHIPMENT_STATION_LAT": float})
    df_res = pd.merge(df_res, station_coords, how="left", left_on="MD03_DIST_STATION_CODE", right_on="MD03_SHIPMENT_STATION_CODE")
    # 提取首末客户编码
    df_res["first_customer_code"] = df_res["BB_RETAIL_CUSTOMER_CODE"].apply(lambda x: x.split(",")[0] if pd.notna(x) and x else "")
    df_res["last_customer_code"] = df_res["BB_RETAIL_CUSTOMER_CODE"].apply(lambda x: x.split(",")[-1] if pd.notna(x) and x else "")
    # 关联首末客户经纬度
    cust_coords = df_n28[["BB_RETAIL_CUSTOMER_CODE", "MD03_RETAIL_CUST_LON", "MD03_RETAIL_CUST_LAT"]].drop_duplicates()
    cust_coords = cust_coords.set_index("BB_RETAIL_CUSTOMER_CODE").astype({"MD03_RETAIL_CUST_LON": float, "MD03_RETAIL_CUST_LAT": float})
    df_res = pd.merge(df_res, cust_coords, how="left", left_on="first_customer_code", right_on="BB_RETAIL_CUSTOMER_CODE")
    df_res.rename(columns={"MD03_RETAIL_CUST_LON": "first_customer_lon", "MD03_RETAIL_CUST_LAT": "first_customer_lat"}, inplace=True)
    df_res = pd.merge(df_res, cust_coords, how="left", left_on="last_customer_code", right_on="BB_RETAIL_CUSTOMER_CODE")
    df_res.rename(columns={"MD03_RETAIL_CUST_LON": "last_customer_lon", "MD03_RETAIL_CUST_LAT": "last_customer_lat"}, inplace=True)
    return df_res

def plan_distribution_routes(
    db_n19: pd.DataFrame,
    db_n26: pd.DataFrame,
    db_n27: pd.DataFrame,
    db_n28: pd.DataFrame,
    db_n31: pd.DataFrame,
    db_n32: pd.DataFrame,
    db_n33: pd.DataFrame,
    db_n34: pd.DataFrame,
    n34_id: str,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Dict[str, Any]]]:
    """High level route planning entry point used by the API service."""
    LOGGER.info(f" ALG_0: n34_id is {n34_id}")
    LOGGER.info(f"\ndb_n34:\n{tabulate(db_n34, headers='keys', tablefmt='grid')}")
    LOGGER.info(f"\ndb_n19:\n{tabulate(db_n19, headers='keys', tablefmt='grid')}")
    LOGGER.info(f"\ndb_n31:\n{tabulate(db_n31, headers='keys', tablefmt='grid')}")
    LOGGER.info(f"\ndb_n33:\n{tabulate(db_n33, headers='keys', tablefmt='grid')}")
    LOGGER.info(f"\ndb_n28:\n{tabulate(db_n28.head(10), headers='keys', tablefmt='grid')}")
    add_n35_log_info(n34_id, "ALG_0", f" ALG_0: n34_id is {n34_id}")

    # 1.1 增加地址解析（行政区划）
    LOGGER.info(f" ALG_0: 路线进入plan_distribution_routes函数")
    add_n35_log_info(n34_id, "ALG_0", f"路线进入plan_distribution_routes函数")
    db_n28 = parse_md03_address_component(db_n28)
    LOGGER.info(f" ALG_0: 路线增加address解析成功")
    add_n35_log_info(n34_id, "ALG_0", f"路线增加address解析成功")
    # 1.2 增加人工线路
    db_n28 = parse_line_info(db_n32, db_n28)
    LOGGER.info(f" ALG_0: 路线增加line解析成功")
    add_n35_log_info(n34_id, "ALG_0", f"路线增加line解析成功")
    # 1.3 增加客户需求和发货点经纬度,填充零售户经纬度
    db_n28 = get_customer_avg_order_qty(db_n32, db_n28, db_n26)
    LOGGER.info(f" ALG_0: 路线增加增加客户需求和处理latlon解析成功")
    add_n35_log_info(n34_id, "ALG_0", f"路线增加latlon解析成功")
    db_n28 = db_n28[[
        "MD03_DIST_STATION_CODE", "MD03_SHIPMENT_STATION_LON", "MD03_SHIPMENT_STATION_LAT",
        "BB_RETAIL_CUSTOMER_CODE", "MD03_RETAIL_CUST_LON", "MD03_RETAIL_CUST_LAT",
        "AVG_ORDER_QTY", "LINE_CODES", "LINE_SEQ", "township", "district", "city"
    ]].copy()
    LOGGER.info(f"\ndb_n28:\n{tabulate(db_n28.head(10), headers='keys', tablefmt='grid')}")

    # 2.1 发货点选择算法: 固定/动态/混合
    algo_code = db_n19["MD03_DIST_PLAN_ALGO_CODE"].iloc[0] if len(db_n19) > 0 else None
    # 如果algo_code为None, 则抛出异常 且logger记录
    if algo_code not in {"VRP1", "VRP2", "VRP3"}:
        LOGGER.error(f" ALG_0: 未指定发货点选择算法")
        add_n35_log_info(n34_id, "ALG_1", f"未指定发货点选择算法")
        raise ValueError(f"未知的发货点选择算法: {algo_code}")
    else:
        LOGGER.info(f" ALG_1: 发货点选择算法 is {algo_code}")
        add_n35_log_info(n34_id, "ALG_1", f"发货点选择算法 is {algo_code}")


    # 2.2 路线规划算法执行: 固定/动态/混合
    if algo_code == "VRP1": # 固定
        isFix = 1
        db_n28 = parse_veh_info(db_n32, db_n28, db_n27)
    elif algo_code == "VRP2": # 动态
        isFix = 2
    elif algo_code == "VRP3": # 混合
        isFix = 3
    routes_df = None

    if isFix == 2:
        # 动态下: 读取参数线路数n31 param_nRoutes
        param_row = db_n31[db_n31["MD03_DIST_PLAN_PARAM_CODE"] == "XLGH0010"]
        param_nRoutes = int(param_row["MD03_DIST_PLAN_PARAM_VAL"].iloc[0]) if not param_row.empty else None
        # 如果param_nRoutes为空, logger报错且跑出异常
        if param_nRoutes is None or param_nRoutes <= 0:
            LOGGER.error(f" ALG_2: 未指定合理的线路数参数")
            add_n35_log_info(n34_id, "ALG_2", f"未指定合理的线路数参数")
            raise ValueError(f"未知的线路数参数: {param_nRoutes}")
        else:
            LOGGER.info(f" ALG_2: 线路数参数 is {param_nRoutes}")
            add_n35_log_info(n34_id, "ALG_2", f"线路数参数 is {param_nRoutes}")

        # 动态下: 读取参数平衡方式n33 balance_type
        # 读取db_n33的MD03_BALANCE_MODE_NAME和MD03_PRIORITY_LEVEL列的取值分别赋值给balance_type和priority_level
        balance_type = db_n33["MD03_BALANCE_MODE_NAME"] if "MD03_BALANCE_MODE_NAME" in db_n33.columns and not db_n33["MD03_BALANCE_MODE_NAME"].isna().all() else "count"
        priority_level = db_n33["MD03_PRIORITY_LEVEL"] if "MD03_PRIORITY_LEVEL" in db_n33.columns and not db_n33["MD03_PRIORITY_LEVEL"].isna().all() else "station"
        if isinstance(priority_level, pd.Series) and not priority_level.empty:
            min_priority_idx = priority_level.idxmin()
            balance_type = db_n33.at[min_priority_idx, "MD03_BALANCE_MODE_NAME"]
        elif isinstance(balance_type, pd.Series) and not balance_type.empty:
            balance_type = balance_type.iloc[0]
        else:
            balance_type = "送货户数均衡"
        LOGGER.info(f" ALG_2: 平衡方式参数 is {balance_type}")

        # 动态平衡
        USE_BALANCED_PARTITION = 1
        try:
            routes_df_balanced = repartition_customers_to_n_routes(
                db_n28, db_n26, param_nRoutes, balance=balance_type,
                group_by_station=True, speed_kmh=50.0
            )
            LOGGER.info(f"[Balanced] Built {len(routes_df_balanced)} routes with balance='{balance_type}', nRoutes={param_nRoutes}")
            add_n35_log_info(n34_id, "ALG_3", f"平衡类型={balance_type}, 线路数={param_nRoutes}")

            # 统计线路信息（每条线路的户数、需求量、里程、时长）赋值给routes_df_balanced
            routes_df_balanced["num_customers"] = routes_df_balanced["sequence"].apply(
                lambda x: len(x.split(",")) if pd.notna(x) and x else 0
            )
            routes_df_balanced["total_demand"] = routes_df_balanced["sequence"].apply(
                lambda x: db_n28[db_n28["BB_RETAIL_CUSTOMER_CODE"].isin(x.split(","))]["AVG_ORDER_QTY"].sum() if pd.notna(x) and x else 0.0
            )

            if USE_BALANCED_PARTITION:
                routes_df = routes_df_balanced.copy()
            LOGGER.info(f" ALG_3: 动态路线规划算法执行成功")
            add_n35_log_info(n34_id, "ALG_3", f"动态路线规划算法执行成功")

        except Exception as e:
            LOGGER.error(f"[Balanced] 生成均衡线路失败: {e}")
            add_n35_log_info(n34_id, "ALG_3", f"动态路线规划算法执行成功")

    elif isFix == 1:

        db_n28_sorted = reorder_lines_by_pyvrp(
            db_n28,
            line_col="LINE_CODES",
            cust_id_col="BB_RETAIL_CUSTOMER_CODE",
            cust_lon_col="MD03_RETAIL_CUST_LON",
            cust_lat_col="MD03_RETAIL_CUST_LAT",
            station_col="MD03_DIST_STATION_CODE",
            station_lon_col="MD03_SHIPMENT_STATION_LON",
            station_lat_col="MD03_SHIPMENT_STATION_LAT",
            demand_col="AVG_ORDER_QTY",
            vehicle_capacity=None,
            runtime_s=1.0,
            no_improve_iters=200,
            keep_unlocated=True,
            return_to_depot=True,
        )
        db_n28_sorted["LINE_SEQ"] = db_n28_sorted["ORDER_IDX"].astype("Int64").astype(str)
        db_n28 = db_n28_sorted

        routes_df = build_route_veh_summary(db_n28, db_n26)
        routes_df = add_route_distances(routes_df, db_n28)
        routes_df = routes_df[
            ["LINE_CODES", "MD03_DIST_STATION_CODE", "VEHICLE_CODES", "VEHICLE_MAX_CARTON_QTY", "LOAD_RATIO",
             "distance_km", "time_min", "BB_RETAIL_CUSTOMER_CODE", "PREV_CUSTOMER_DISTANCE_KM",
             "PREV_CUSTOMER_TIME_MIN"]].copy()
        routes_df.rename(columns={
            "LINE_CODES": "route_id",
            "MD03_DIST_STATION_CODE": "station",
            "VEHICLE_CODES": "vehicle_id",
            "VEHICLE_MAX_CARTON_QTY": "vehicle_capacity",
            "LOAD_RATIO": "load_ratio",
            "distance_km": "distance_km",
            "time_min": "duration_h",
            "BB_RETAIL_CUSTOMER_CODE": "sequence",
            "PREV_CUSTOMER_DISTANCE_KM": "prev_distance_km",
            "PREV_CUSTOMER_TIME_MIN": "prev_duration_h"
        }, inplace=True)
        routes_df["route_name"] = "线路" + routes_df["route_id"].astype(str)

        LOGGER.info(f" ALG_3: 固定路线规划算法执行成功")
        add_n35_log_info(n34_id, "ALG_3", f"固定路线规划算法执行成功")

    # 3. 输出结果整理
    # LOGGER输出routes_df前10行（如有）
    LOGGER.info(f"\nroutes_df:\n{tabulate(routes_df.head(10), headers='keys', tablefmt='grid') if routes_df is not None else 'No routes generated.'}")
    if routes_df is not None:
        # 删除车容量字段
        if "vehicle_capacity" in routes_df.columns:
            routes_df.drop(columns=["vehicle_capacity"], inplace=True)
        # 装载率为null时默认-1
        routes_df["load_ratio"] = pd.to_numeric(routes_df["load_ratio"], errors="coerce").fillna(-1.0)
        routes_dict = {row["route_id"]: row.to_dict() for _, row in routes_df.iterrows()}
    else:
        routes_dict = {}

    LOGGER.info(f" ALG_4: 路线规划算法执行结束")
    add_n35_log_info(n34_id, "ALG_4", f"路线规划算法执行结束")
    return routes_df, routes_dict
