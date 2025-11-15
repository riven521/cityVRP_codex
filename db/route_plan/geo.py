"""Utility functions for the route planning geo module."""

from __future__ import annotations

from datetime import datetime
import os
import pickle

import geopandas as gpd
import numpy as np
import pandas as pd

from ..api_config import EXTRACT_INFO_TYPE
from .logger import LOGGER

def attach_osm_coords(gdf, G, node_col="customer_osmid_mapping", x_col="OSM_X", y_col="OSM_Y", fill_from_geometry=True):
    """
    将图 G 中节点的 x/y 坐标安全地映射到 gdf 的两列中，并对缺失进行容错与填充。

    参数:
        gdf: GeoDataFrame，包含节点列 node_col
        G: networkx/OSMnx 图对象，节点需含 'x'/'y'
        node_col: gdf 中存储图节点 id 的列名
        x_col, y_col: 输出列名
        fill_from_geometry: 缺失时是否尝试使用 geometry 的经纬度进行填充

    返回:
        gdf: 增加了 x_col/y_col 的 GeoDataFrame（原地赋值并返回）
    """
    osm_x = []
    osm_y = []
    missing_rows = []
    num_from_graph = 0
    num_from_geometry = 0
    num_missing = 0

    # 安全遍历并抓取坐标
    for idx, node_id in gdf[node_col].items():
        val_x = np.nan
        val_y = np.nan

        try:
            # 允许 node_id 为 float/str，尽量转为 int
            nid = node_id
            if pd.isna(nid):
                raise KeyError("NaN node id")
            if isinstance(nid, str) and nid.isdigit():
                nid = int(nid)
            # 读取图节点坐标
            if nid in G.nodes:
                node_data = G.nodes[nid]
                if 'x' in node_data and 'y' in node_data:
                    val_x = node_data['x']
                    val_y = node_data['y']
                    num_from_graph += 1
                else:
                    raise KeyError(f"node {nid} missing x/y")
            else:
                raise KeyError(f"node {nid} not in graph")
        except Exception:
            missing_rows.append(idx)
            # 缺失时尝试从 geometry 填充
            if fill_from_geometry and 'geometry' in gdf.columns:
                geom = gdf.at[idx, 'geometry']
                if geom is not None and hasattr(geom, 'x') and hasattr(geom, 'y'):
                    val_x = float(geom.x)
                    val_y = float(geom.y)
                    num_from_geometry += 1
                else:
                    num_missing += 1
            else:
                num_missing += 1

        osm_x.append(val_x)
        osm_y.append(val_y)

    gdf[x_col] = osm_x
    gdf[y_col] = osm_y

    # 记录映射统计
    total = len(gdf)
    LOGGER.info(
        f"attach_osm_coords: total={total}, from_graph={num_from_graph}, from_geometry={num_from_geometry}, remaining_missing={num_missing}")
    if len(missing_rows) > 0:
        LOGGER.warning(
            f"attach_osm_coords: {len(missing_rows)} rows missing graph coordinates in '{node_col}'.")

    return gdf

def load_and_prepare_taiyuan_grid(
    G_path="./notebook/graph/G_taiyuan.pkl",
    address_csv_path="./data/customers_with_address_complete_deduplicated_29868.csv",
    pkl_path="./data/taiyuan_customer_info.pkl",
    city_filter="太原市",
    district_filters=("阳曲县", "尖草坪区"),
    township_filters=("黄寨镇", "光社街道"),
):
    """
    读取 Taiyuan 客户数据（CSV + PKL），注入 OSM 坐标，合并行政区/需求字段，清洗并返回 GeoDataFrame。
    过程中输出核心变化日志（行数/匹配/过滤统计）。
    """
    # 读取
    with open(G_path, "rb") as f:
        G = pickle.load(f)

    df_addr = pd.read_csv(address_csv_path, encoding='utf-8')
    LOGGER.info(f"Read address CSV: rows={df_addr.shape[0]}")

    df_grid = pd.read_pickle(pkl_path)
    LOGGER.info(f"Read grid PKL: rows={df_grid.shape[0]}")

    # 坐标注入
    df_grid = attach_osm_coords(
        df_grid, G,
        node_col='customer_osmid_mapping', x_col='OSM_X', y_col='OSM_Y', fill_from_geometry=True,
    )

    # 合并地址/行政区等属性
    before_merge_rows = df_grid.shape[0]
    df_addr = df_addr.rename(columns={'customer_code': 'BB_RETAIL_CUSTOMER_CODE'})
    merged = df_grid.merge(
        df_addr[['BB_RETAIL_CUSTOMER_CODE', 'avg_order_qty', 'city', 'district', 'township']],
        on='BB_RETAIL_CUSTOMER_CODE', how='left', indicator=True, validate='m:1'
    )
    vc = merged['_merge'].value_counts(dropna=False).to_dict()
    LOGGER.info(f"Merge result: total={merged.shape[0]}, both={vc.get('both', 0)}, left_only={vc.get('left_only', 0)}, right_only={vc.get('right_only', 0)}")
    merged = merged.drop(columns=['_merge'])

    # 缺失行政区清洗
    before_dropna = merged.shape[0]
    merged = merged.dropna(subset=['district', 'township'])
    LOGGER.info(f"Drop NA (district/township): before={before_dropna}, after={merged.shape[0]}, removed={before_dropna - merged.shape[0]}")

    # 条件过滤
    before_filter = merged.shape[0]
    filtered = merged[
        (merged['city'] == city_filter)
        & (merged['district'].isin(list(district_filters)))
        & (merged['township'].isin(list(township_filters)))
    ]
    LOGGER.info(f"Filter rows: before={before_filter}, after={filtered.shape[0]}, removed={before_filter - filtered.shape[0]}")

    # filtered增加'nid'列,从1开始按1递增计数
    filtered['nid'] = range(1, len(filtered) + 1)

    # 转为 GeoDataFrame
    gdf = gpd.GeoDataFrame(filtered, geometry='geometry')


    # 汇总过滤后的零售户需求结果
    total_qty = gdf['avg_order_qty'].sum()
    LOGGER.info(f"Total avg_order_qty after filtering: {total_qty}")

    return gdf, G

def build_distance_matrix_from_graph(gdf, G, depot_node=7839518188, weight='length'):
    """
    基于路网图 G，直接计算包含仓库节点与客户节点的最短路距离矩阵。

    参数:
        gdf: GeoDataFrame，需包含列 'customer_osmid_mapping'
        G: networkx 图
        depot_node: 仓库 OSM 节点 id
        weight: 边权字段（默认 'length'）

    返回:
        numpy.ndarray: 距离矩阵 (N x N)，节点顺序为 [depot] + gdf['customer_osmid_mapping']
    """
    import networkx as nx
    import time

    # 目标节点序列（保留顺序，可含重复）
    target_nodes = [depot_node] + list(gdf["customer_osmid_mapping"].tolist())
    num_nodes = len(target_nodes)

    # 初始化距离矩阵
    distance_matrix = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(distance_matrix, 0.0)

    # 统计信息
    missing_in_graph = 0
    t0 = time.time()

    # 为每个源节点执行一次单源最短路
    for i, source in enumerate(target_nodes):
        if source not in G.nodes:
            missing_in_graph += 1
            continue
        paths = nx.single_source_dijkstra_path_length(G, source, weight=weight)
        for j, target in enumerate(target_nodes):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                if target == source:
                    distance_matrix[i, j] = 0.0
                elif target in paths:
                    distance_matrix[i, j] = paths[target]

    # 检查distance_matrix内是否存在inf值;如(i,j)为inf,但(j,i)不是inf,则赋值(i,j)的为(j,i)的值
    for i in range(num_nodes):
        for j in range(num_nodes):
            if np.isinf(distance_matrix[i, j]) and not np.isinf(distance_matrix[j, i]):
                distance_matrix[i, j] = distance_matrix[j, i]


    t1 = time.time()
    LOGGER.info(
        f"build_distance_matrix_from_graph: nodes={num_nodes}, missing_sources_in_graph={missing_in_graph}, elapsed_sec={t1 - t0:.2f}")

    return distance_matrix

def build_or_load_distance_matrix(
    gdf,
    G,
    cache_path="../../data/taiyuan_customer_distance.pkl",
    depot_node=7839518188,
    weight='length',
    force_recompute=False,
):
    """
    优先从缓存读取距离矩阵；如缓存缺失或不匹配则计算并保存到缓存。

    缓存格式兼容：
      - 旧版: 直接存储 numpy.ndarray
      - 新版: dict 包含 {matrix, nodes, depot_node, weight, created_at}
    """
    target_nodes = [depot_node] + list(gdf["customer_osmid_mapping"].tolist())

    if (not force_recompute) and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache_obj = pickle.load(f)
            # 新版缓存
            if isinstance(cache_obj, dict) and 'matrix' in cache_obj and 'nodes' in cache_obj:
                if (
                    cache_obj.get('nodes') == target_nodes
                    and cache_obj.get('depot_node') == depot_node
                    and cache_obj.get('weight') == weight
                ):
                    LOGGER.info(f"Loaded distance_matrix from cache (match): {cache_path}")
                    return cache_obj['matrix']
                else:
                    LOGGER.warning("Cache mismatch detected (nodes/depot/weight). Recomputing distance matrix.")
            # 旧版缓存（仅矩阵）
            elif hasattr(cache_obj, 'shape') and isinstance(cache_obj.shape, tuple):
                if cache_obj.shape == (len(target_nodes), len(target_nodes)):
                    LOGGER.info(f"Loaded distance_matrix from legacy cache (shape match): {cache_path}")
                    return cache_obj
                else:
                    LOGGER.warning("Legacy cache shape mismatch. Recomputing distance matrix.")
        except Exception as e:
            LOGGER.warning(f"Failed to load distance_matrix cache: {e}. Recomputing.")

    # 计算并写入缓存
    matrix = build_distance_matrix_from_graph(gdf, G, depot_node=depot_node, weight=weight)
    try:
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)
        cache_obj = {
            'matrix': matrix,
            'nodes': target_nodes,
            'depot_node': depot_node,
            'weight': weight,
            'created_at': datetime.now().isoformat(),
            'version': '1.0',
        }
        with open(cache_path, "wb") as f:
            pickle.dump(cache_obj, f)
        LOGGER.info(f"Saved distance_matrix to cache: {cache_path}")
    except Exception as e:
        LOGGER.warning(f"Failed to save distance_matrix cache: {e}")

    # 判断distance_matrix内是否有inf值，并指出具体索引号
    if np.isinf(matrix).any():
        inf_indices = np.argwhere(np.isinf(matrix))
        LOGGER.warning(f"Distance matrix contains inf values at indices: {inf_indices.tolist()}")

    return matrix

def read_input():

    # ① 读取山西G,grid,addr的值 并过滤（统一日志输出）
    gdf, G = load_and_prepare_taiyuan_grid(
        G_path="./notebook/graph/G_taiyuan.pkl",
        address_csv_path="./data/customers_with_address_complete_deduplicated_29868.csv",
        pkl_path="./data/taiyuan_customer_info.pkl",
        city_filter="太原市",
        district_filters=("阳曲县", ), #, "尖草坪区"
        township_filters=("黄寨镇", ),  # , "光社街道"
    )

    # 构建不同层级的gdf[Node, Osm, Grid] Node是默认
    if EXTRACT_INFO_TYPE == "Node":
        # Do Nothing
        gdf = gdf
    elif EXTRACT_INFO_TYPE == "Osm":
        gdf = gdf.groupby('customer_osmid_mapping').agg({
            'nid': lambda x: ', '.join(str(i) for i in x.unique()),
            'group': lambda x: ', '.join(str(i) for i in x.unique()),
            'BB_RETAIL_CUSTOMER_NAME': lambda x: ', '.join(x.unique()),
            'city': lambda x: ', '.join(x.unique()),
            'district': lambda x: ', '.join(x.unique()),
            'township': lambda x: ', '.join(x.unique()),
            'avg_order_qty': 'sum',
            'OSM_X': 'mean',
            'OSM_Y': 'mean',
            'lon_84': 'mean',
            'lat_84': 'mean',
            'LonGridCenter': 'mean',
            'LatGridCenter': 'mean'
        }).reset_index()
    elif EXTRACT_INFO_TYPE == "Grid":
        gdf = gdf.groupby('grid_id').agg({
            'nid': lambda x: ', '.join(str(i) for i in x.unique()),
            'group': lambda x: ', '.join(str(i) for i in x.unique()),
            'BB_RETAIL_CUSTOMER_NAME': lambda x: ', '.join(x.unique()),
            'city': lambda x: ', '.join(x.unique()),
            'district': lambda x: ', '.join(x.unique()),
            'township': lambda x: ', '.join(x.unique()),
            'avg_order_qty': 'sum',
            'OSM_X': 'mean',
            'OSM_Y': 'mean',
            'lon_84': 'mean',
            'lat_84': 'mean',
            'LonGridCenter': 'mean',
            'LatGridCenter': 'mean'
        }).reset_index()
    else:
        raise ValueError(f"未知的 EXTRACT_INFO_TYPE: {EXTRACT_INFO_TYPE}. 仅支持 'Node' 或 'Grid'.")


    # ② 基于 G 直接计算距离矩阵（带缓存）
    if EXTRACT_INFO_TYPE == "Node" or EXTRACT_INFO_TYPE == "Osm":
        distance_matrix = build_or_load_distance_matrix(
            gdf,
            G,
            cache_path="../../data/taiyuan_customer_distance.pkl",
            depot_node=7839518188,
            weight='length',
            force_recompute=False,
        )
    elif EXTRACT_INFO_TYPE == "Grid":
        # 计算d，是基于gdf的LonGridCenter和LatGridCenter是每个点的x和y，曼哈顿距离
        depot_node = 7839518188
        d0 = [G.nodes[depot_node]['x'], G.nodes[depot_node]['y']]
        d = gdf[['LonGridCenter', 'LatGridCenter']].values
        d = np.vstack((d0, d))
        distance_matrix = np.zeros((len(d), len(d)))
        for i in range(len(d)):
            for j in range(len(d)):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    distance_matrix[i, j] = abs(d[i][0] - d[j][0]) + abs(d[i][1] - d[j][1])
        # 将distance_matrix转换为numpy.ndarray
        distance_matrix = np.array(distance_matrix)
    else:
        raise ValueError(f"未知的 EXTRACT_INFO_TYPE: {EXTRACT_INFO_TYPE}. 仅支持 'Node' 或 'Grid'.")

    return gdf, distance_matrix, G
