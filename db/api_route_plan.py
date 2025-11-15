from typing import Dict, Optional, Tuple, Iterable, Any
import pandas as pd
import numpy as np
import math
from db.log_service import add_n35_log_info
from branca.element import MacroElement
from jinja2 import Template

from pathlib import Path
from datetime import datetime
from decimal import Decimal
import json
import hashlib
import loguru
import os
import pickle
import geopandas as gpd
from db.api_config import EXTRACT_INFO_TYPE, VEH_CAPACITY, MAX_CUS_VISITED, PYVRP_MAX_RUNTIME, PYVRP_MAX_NO_IMPROVE_ITERS, DIST_MATRIX_TYPE
import folium
from pyvrp import Model
from pyvrp.stop import MaxRuntime, NoImprovement, MultipleCriteria

# 250913
# 1.完成线路规划相关基础工作：搭建线路规划的初步算法，并实现其可视化功能。
# 2.完善客户数据表格：计算客户点的平均订单量，将该数据整合至客户表中。
# 3.整理车辆信息表格：提取车辆的最大装载量等核心数据，补充到车辆表内。
# 4.转换算法输入数据：将人工计算得出的结果，转换为符合算法计算要求的格式并呈现。
# 5.调整数据展示形式：将“每一行对应一个客户”的表格，优化为“每一行对应一条线路及该线路所访问客户”的表格。
# 6.开展folium可视化工作：以单个发货点为单位，随机抽取若干条线路，据此完成volume的可视化展示。


# === logger helpers ===
loguruer = loguru.logger
if os.name == 'nt':  # Windows
    log_path = f"Log/log_{datetime.now().strftime('%Y%m%d')}.log"
else:  # Linux or MacOS (or Docker)
    log_path = "/app/logs/log_{time:YYYY-MM-DD}.log"

try:
    loguruer.add(log_path, rotation="00:00", encoding="utf-8", retention="7 days", enqueue=True, mode="w", backtrace=True, diagnose=True)
except OSError:
    log_path = "app/logs/log_{time:YYYY-MM-DD}.log"
    loguruer.add(log_path, rotation="00:00", encoding="utf-8", retention="7 days", enqueue=True, mode="w", backtrace=True,
               diagnose=True)

# === OSM helpers ===
# 自定义一个 MacroElement，通过键盘空格事件按顺序展示 marker
class DynamicMarkerOnSpace(MacroElement):
    def __init__(self, customer_points, sol_list):
        super().__init__()
        self.customer_points = customer_points
        self.sol_list = sol_list
        self._template = Template(u"""
            {% macro script(this, kwargs) %}
                // 预设的点数据，每个点为 [纬度, 经度]
                var customer_points = {{ this.customer_points }};
                var sol_list = {{ this.sol_list }};
                var colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'yellow', 'cyan', 'magenta', 'brown'];
                var colorIndex = 0;
                var solIndex = 0; // 用于记录当前顺序下的点
                var isSpacePressed = false; // 用于确保空格按下后执行一次

                // 监听键盘事件，按空格触发
                document.addEventListener('keydown', function(e) {
                    if (e.code === 'Space') {
                        if (solIndex < sol_list.length) {
                            var group = sol_list[solIndex];
                            if (group.length > 0 && !isSpacePressed) {
                                var ptIndex = group.shift(); // 从当前组获取点
                                var pt = customer_points[ptIndex];
                                // 为不同的组分配不同的颜色
                                var color = colors[colorIndex % colors.length];
                                // 添加 CircleMarker 到地图
                                L.circleMarker([pt[1], pt[0]], {
                                    radius: 8,
                                    color: color,
                                    fill: true,
                                    fillOpacity: 0.5
                                }).addTo({{ this._parent.get_name() }});

                                if (group.length === 0) {
                                    solIndex++; // 如果当前组已显示完，切换到下一个组
                                    colorIndex++; // 切换颜色
                                }
                                isSpacePressed = true; // 标记空格已被按下，避免多次触发
                            }
                        }
                    }
                });

                // 监听空格释放事件，用于重置按键状态
                document.addEventListener('keyup', function(e) {
                    if (e.code === 'Space') {
                        isSpacePressed = false; // 释放空格键后允许重新触发
                    }
                });
            {% endmacro %}
        """)


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
    loguruer.info(
        f"attach_osm_coords: total={total}, from_graph={num_from_graph}, from_geometry={num_from_geometry}, remaining_missing={num_missing}")
    if len(missing_rows) > 0:
        loguruer.warning(
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
    loguruer.info(f"Read address CSV: rows={df_addr.shape[0]}")

    df_grid = pd.read_pickle(pkl_path)
    loguruer.info(f"Read grid PKL: rows={df_grid.shape[0]}")

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
    loguruer.info(f"Merge result: total={merged.shape[0]}, both={vc.get('both', 0)}, left_only={vc.get('left_only', 0)}, right_only={vc.get('right_only', 0)}")
    merged = merged.drop(columns=['_merge'])

    # 缺失行政区清洗
    before_dropna = merged.shape[0]
    merged = merged.dropna(subset=['district', 'township'])
    loguruer.info(f"Drop NA (district/township): before={before_dropna}, after={merged.shape[0]}, removed={before_dropna - merged.shape[0]}")

    # 条件过滤
    before_filter = merged.shape[0]
    filtered = merged[
        (merged['city'] == city_filter)
        & (merged['district'].isin(list(district_filters)))
        & (merged['township'].isin(list(township_filters)))
    ]
    loguruer.info(f"Filter rows: before={before_filter}, after={filtered.shape[0]}, removed={before_filter - filtered.shape[0]}")

    # filtered增加'nid'列,从1开始按1递增计数
    filtered['nid'] = range(1, len(filtered) + 1)

    # 转为 GeoDataFrame
    gdf = gpd.GeoDataFrame(filtered, geometry='geometry')


    # 汇总过滤后的零售户需求结果
    total_qty = gdf['avg_order_qty'].sum()
    loguruer.info(f"Total avg_order_qty after filtering: {total_qty}")

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
    loguruer.info(
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
                    loguruer.info(f"Loaded distance_matrix from cache (match): {cache_path}")
                    return cache_obj['matrix']
                else:
                    loguruer.warning("Cache mismatch detected (nodes/depot/weight). Recomputing distance matrix.")
            # 旧版缓存（仅矩阵）
            elif hasattr(cache_obj, 'shape') and isinstance(cache_obj.shape, tuple):
                if cache_obj.shape == (len(target_nodes), len(target_nodes)):
                    loguruer.info(f"Loaded distance_matrix from legacy cache (shape match): {cache_path}")
                    return cache_obj
                else:
                    loguruer.warning("Legacy cache shape mismatch. Recomputing distance matrix.")
        except Exception as e:
            loguruer.warning(f"Failed to load distance_matrix cache: {e}. Recomputing.")

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
        loguruer.info(f"Saved distance_matrix to cache: {cache_path}")
    except Exception as e:
        loguruer.warning(f"Failed to save distance_matrix cache: {e}")

    # 判断distance_matrix内是否有inf值，并指出具体索引号
    if np.isinf(matrix).any():
        inf_indices = np.argwhere(np.isinf(matrix))
        loguruer.warning(f"Distance matrix contains inf values at indices: {inf_indices.tolist()}")

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

def save_tables_or_str_local(n34_id: str, tables: dict, cache_dir: str = 'cache_vrp') -> Path:
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
    # Filter tables to separate DataFrames from strings
    df_tables = {}
    string_tables = {}
    object_tables = {}  # 需要在循环前定义这个字典

    for name, obj in tables.items():
        if obj is None:
            continue
        if isinstance(obj, pd.DataFrame):
            df_tables[name] = obj
        elif isinstance(obj, str):
            string_tables[name] = obj
        else:
            object_tables[name] = obj

    for name, obj in object_tables.items():
        out_path = root / f"{name}.pkl"
        try:
            with open(out_path, 'wb') as f:  # 注意是 'wb' 模式，即二进制写入
                pickle.dump(obj, f)
            manifest['tables'][name] = {
                'file': str(out_path.name),
                'type': 'binary'  # 可以命名为 'pickle' 或 'binary'
            }
            print(f"Successfully saved object to {out_path}")
        except Exception as e:
            print(f"Error saving pickle file {out_path}: {e}")

    # Save DataFrames as parquet files
    for name, df in df_tables.items():
        out_path = root / f"{name}.parquet"
        _df_to_parquet(df, out_path)
        manifest['tables'][name] = {
            'file': str(out_path.name),
            'type': 'parquet'
        }

    # Save strings as text files [2,3](@ref)
    for name, text_content in string_tables.items():
        out_path = root / f"{name}.txt"
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            manifest['tables'][name] = {
                'file': str(out_path.name),
                'type': 'text'
            }
            print(f"Successfully saved string content to {out_path}")
        except Exception as e:
            print(f"Error saving string file {out_path}: {e}")

    # Calculate schema fingerprint only for DataFrame tables
    if df_tables:
        manifest['schema_fingerprint'] = _schema_fingerprint(df_tables)

    # Save manifest
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


def load_tables_or_str_local(n34_id: str, cache_dir: str = 'cache_vrp', *, strict: bool = False) -> dict:
    """Load previously saved tables from cache_dir/n34_id.
    Now supports both DataFrame (parquet) and string (text) files.
    If `strict` is True, raise if any table is missing; otherwise missing ones are ignored.
    Returns a dict with values as either DataFrame or str.
    """
    root = Path(cache_dir) / str(n34_id)
    manifest_path = root / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest for n34_id={n34_id} under {root}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    out = {}

    # 获取表信息，默认为字典，确保向后兼容旧格式
    tables_info = manifest.get('tables', {})

    for name, info in tables_info.items():
        # 处理向后兼容：如果info是字符串，则为旧格式（只有文件名）
        if isinstance(info, str):
            file_path = root / info
            file_type = 'parquet'  # 旧格式默认是parquet
        else:
            # 新格式：info是字典，包含 'file' 和 'type'
            file_path = root / info['file']
            file_type = info.get('type', 'parquet')  # 默认为parquet

        try:
            if file_type == 'parquet':
                # 读取Parquet文件（DataFrame）
                out[name] = _df_from_parquet(file_path)
            elif file_type == 'text':
                # 读取文本文件（字符串）
                with open(file_path, 'r', encoding='utf-8') as f:
                    out[name] = f.read()
            elif file_type == 'binary':  # 或 'pickle'，需与保存时定义的类型一致
                with open(file_path, 'rb') as f:  # 注意是 'rb' 二进制读取模式
                    out[name] = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

        except Exception as e:
            if strict:
                raise
            else:
                print(f"[load_tables_local] skip {name} ({file_type}): {e}")

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
    loguruer: Optional[Any] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    将 26/28（必需）及若干可选源表，转为 VRP 求解所需输入：df_cus, df_veh, D。

    参数
    ----
    df_26 : 仓库/线路或车辆配置相关表（必需）
    df_28 : 客户/需求相关表（必需）
    df_27, df_31, df_19, df_32, df_33 : 可选的辅助数据源
    loguruer : 可选日志器，需支持 .info/.warning/.error

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
    # add_n35_log_info(n34_id, "ALG_01",
    #                  f"范围规划算法-数据准备-开始构建 VRP 输入数据（客户/车辆/距离矩阵）")
    # ---- 简易 logger ----
    class _DummyLogger:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass

    logger = loguruer or _DummyLogger()

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

    # add_n35_log_info(n34_id, "ALG_02",
    #                  f"范围规划算法-数据准备-完成 VRP 输入数据构建，客户数={len(df_cus)}，发货点数={len(df_depot)}")
    return df_cus, df_depot, df_veh, D


# def plan_distribution_regions(
#     n34_id: str,
#     df_cus: pd.DataFrame,
#     df_depot: pd.DataFrame,
#     D_cd: Optional[pd.DataFrame] = None,  # 行=发货点(MD03_DIST_PLAN_DIST_SITE_ID), 列=客户(BB_RETAIL_CUSTOMER_CODE)
#     *,
#     return_unassigned: bool = True,
#     assign_unreachable_to_nearest: bool = False,  # 新增：是否把不可达/缺坐标客户也强行分配
# ) -> Tuple[Dict[str, set], Optional[Iterable[str]]]:
#     """基于距离将客户分配到最近的发货点。
#     返回:
#       dict_dist_plan: { depot_id: set(customer_codes) }
#       unassigned: 无法分配客户列表（当 return_unassigned=True）
#     """
#     # add_n35_log_info(n34_id, "ALG_03",
#     #                  f"范围规划算法-范围划分-开始执行客户到发货点的分配")
#     # ==== 列校验 ====
#     need_cus = {"BB_RETAIL_CUSTOMER_CODE", "MD03_RETAIL_CUST_LON", "MD03_RETAIL_CUST_LAT"}
#     miss = need_cus - set(df_cus.columns)
#     if miss:
#         raise KeyError(f"df_cus 缺少列: {sorted(miss)}")
#
#     need_dep = {"MD03_SHIPMENT_STATION_CODE", "MD03_SHIPMENT_STATION_LON", "MD03_SHIPMENT_STATION_LAT"}
#     miss = need_dep - set(df_depot.columns)
#     if miss:
#         raise KeyError(f"df_depot 缺少列: {sorted(miss)}")
#
#     # 清洗：去掉无发货点ID的行
#     df_depot = df_depot[df_depot["MD03_SHIPMENT_STATION_CODE"].notna()].copy()
#     if df_depot.empty:
#         raise ValueError("df_depot 中发货点ID皆为空。")
#
#     # 标准化为字符串 ID
#     cus_ids_all = df_cus["BB_RETAIL_CUSTOMER_CODE"].astype(str)
#     depot_ids = df_depot["MD03_SHIPMENT_STATION_CODE"].astype(str).tolist()
#     first_depot = depot_ids[0]  # 作为缺坐标时的回退仓（保持确定性）
#
#     # ==== 若未提供 D_cd，则用坐标计算 ====
#     if D_cd is None:
#         # 坐标齐全的客户/仓
#         cus_ok_mask = df_cus["MD03_RETAIL_CUST_LON"].notna() & df_cus["MD03_RETAIL_CUST_LAT"].notna()
#         depot_ok_mask = df_depot["MD03_SHIPMENT_STATION_LON"].notna() & df_depot["MD03_SHIPMENT_STATION_LAT"].notna()
#
#         missing_coord_customers = df_cus.loc[~cus_ok_mask, "BB_RETAIL_CUSTOMER_CODE"].astype(str).tolist()
#         df_cus2 = df_cus.loc[cus_ok_mask].copy()
#         df_depot2 = df_depot.loc[depot_ok_mask].copy()
#
#         # 若仓坐标全缺，无法计算任何距离
#         if df_depot2.empty:
#             dict_dist_plan = {d: set() for d in depot_ids}
#             if assign_unreachable_to_nearest:
#                 # 全部指派到第一个仓
#                 for c in cus_ids_all.tolist():
#                     dict_dist_plan[first_depot].add(c)
#                 return (dict_dist_plan, None) if return_unassigned else (dict_dist_plan, None)
#             else:
#                 if return_unassigned:
#                     return dict_dist_plan, cus_ids_all.tolist()
#                 return dict_dist_plan, None
#
#         cus_ids = df_cus2["BB_RETAIL_CUSTOMER_CODE"].astype(str).tolist()
#
#         # Haversine
#         def haversine(lon1, lat1, lon2, lat2):
#             R = 6371.0
#             lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
#             dlon, dlat = lon2 - lon1, lat2 - lat1
#             a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
#             return 2 * R * math.asin(math.sqrt(a))
#
#         cus_xy = df_cus2[["MD03_RETAIL_CUST_LON", "MD03_RETAIL_CUST_LAT"]].to_numpy(float)
#         depot_xy = df_depot2[["MD03_SHIPMENT_STATION_LON", "MD03_SHIPMENT_STATION_LAT"]].to_numpy(float)
#         depot_ids_ok = df_depot2["MD03_SHIPMENT_STATION_CODE"].astype(str).tolist()
#
#         dist = np.empty((len(depot_ids_ok), len(cus_ids)), dtype=float)
#         for i, (dlon, dlat) in enumerate(depot_xy):
#             for j, (clon, clat) in enumerate(cus_xy):
#                 dist[i, j] = haversine(dlon, dlat, clon, clat)
#
#         D_cd = pd.DataFrame(dist, index=depot_ids_ok, columns=cus_ids)
#
#         # 先对“坐标齐全”的客户完成常规分配；坐标缺失者后续按参数处理
#         post_assign_missing_coords = missing_coord_customers
#     else:
#         # 使用外部距离矩阵
#         D_cd = D_cd.copy()
#         D_cd.index = D_cd.index.astype(str)
#         D_cd.columns = D_cd.columns.astype(str)
#         post_assign_missing_coords = []  # 有 D_cd 时不区分是否缺坐标
#
#         # 补齐缺失仓行为 +inf
#         missing_rows = set(depot_ids) - set(D_cd.index)
#         if missing_rows:
#             D_cd = pd.concat(
#                 [D_cd, pd.DataFrame(np.inf, index=sorted(missing_rows), columns=D_cd.columns)],
#                 axis=0
#             )
#         # 仅保留本次仓顺序
#         D_cd = D_cd.loc[depot_ids]
#
#     # ==== 统一：NaN→+inf，避免 idxmin NaN ====
#     D_cd = D_cd.astype(float).replace([np.inf, -np.inf], np.inf).fillna(np.inf)
#
#     # 判定“不可达”（整列为 +inf）的客户
#     all_inf_mask = np.isinf(D_cd).all(axis=0)
#     unreachable_cols = D_cd.columns[all_inf_mask].tolist()
#
#     # 只对可达客户做最近仓分配
#     feasible_cols = D_cd.columns[~all_inf_mask]
#     # nearest_depot = D_cd.loc[:, feasible_cols].idxmin(axis=0) if len(feasible_cols) else pd.Series(dtype=str)
#
#     # assigned_depot: pd.Series
#     assign_strategy = "balanced"  # 可选 "nearest" | "random" | "balanced"
#     random_state = 42  # 用于随机分配的随机种子
#     if len(feasible_cols) == 0:
#         assigned_depot = pd.Series(dtype=str)
#     else:
#         # 每个客户的可达仓（距离非 inf）
#         D_feas = D_cd.loc[:, feasible_cols]
#         feas_mask = ~np.isinf(D_feas)
#         rng = np.random.default_rng(random_state)
#
#         if assign_strategy == "nearest":
#             # 与旧逻辑一致：各列取最小值的行索引
#             assigned_depot = D_feas.idxmin(axis=0)
#
#         elif assign_strategy == "random":
#             # 在每一列的可达仓中随机选择一个
#             choices = {}
#             for c in D_feas.columns:
#                 rows_ok = feas_mask.loc[:, c]
#                 depots_ok = D_feas.index[rows_ok].tolist()
#                 if depots_ok:
#                     choices[c] = rng.choice(depots_ok)
#             assigned_depot = pd.Series(choices, name="depot")
#
#         elif assign_strategy == "balanced":
#             # 数量均衡：贪心地把每个客户分配给当前负载最小的可达仓；并列时选距离最近
#             load = {d: 0 for d in D_feas.index.tolist()}
#             assign_list = []
#             # 为了稳定性，按照客户列名排序遍历
#             for c in sorted(D_feas.columns.tolist()):
#                 rows_ok = feas_mask.loc[:, c]
#                 depots_ok = D_feas.index[rows_ok].tolist()
#                 if not depots_ok:
#                     continue
#                 # 找到当前负载最小的可达仓集合
#                 min_load = min(load[d] for d in depots_ok)
#                 candidate_depots = [d for d in depots_ok if load[d] == min_load]
#                 if len(candidate_depots) == 1:
#                     d_star = candidate_depots[0]
#                 else:
#                     # 并列时选距离最近的仓
#                     d_star = D_feas.loc[candidate_depots, c].idxmin()
#                 load[d_star] += 1
#                 assign_list.append((c, d_star))
#             assigned_depot = pd.Series({c: d for c, d in assign_list}, name="depot")
#
#         else:
#             raise ValueError(f"未知 assign_strategy: {assign_strategy}. 允许: 'nearest'|'random'|'balanced'")
#
#
#     # 结果容器
#     dict_dist_plan: Dict[str, set] = {d: set() for d in depot_ids}
#     unassigned: list[str] = []
#
#     # 1) 正常分配
#     for c, d in assigned_depot.items():
#         dict_dist_plan[str(d)].add(str(c))
#
#     # 2) 对“不可达”客户的处理
#     if assign_unreachable_to_nearest:
#         # 有 D_cd 的情况下：对每个不可达客户列，直接选该列的 idxmin（全 inf 时取第一个仓）
#         for c in unreachable_cols:
#             # 全 inf → idxmin 返回第一个索引；保证确定性
#             d_star = D_cd.loc[:, c].idxmin()
#             dict_dist_plan[str(d_star)].add(str(c))
#     else:
#         unassigned.extend(unreachable_cols)
#
#     # 3) 对“缺坐标且没有 D_cd”客户的处理
#     if post_assign_missing_coords:
#         if assign_unreachable_to_nearest:
#             # 无距离可算，按约定回退到第一个仓（或在此处替换为你的业务逻辑）
#             for c in post_assign_missing_coords:
#                 dict_dist_plan[first_depot].add(str(c))
#         else:
#             unassigned.extend(map(str, post_assign_missing_coords))
#
#     # 去重 & 返回
#     unassigned = sorted(set(unassigned))
#
#     # add_n35_log_info(n34_id, "ALG_04",
#     #                  f"范围规划算法-范围划分-完成分配，共分配客户数={sum(len(v) for v in dict_dist_plan.values())}，未分配客户数={len(unassigned)}")
#
#     if return_unassigned:
#         return dict_dist_plan, (unassigned or None)
#
#     return dict_dist_plan, None


# 新增：单仓最近邻启发式线路规划
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



# === folium 可视化：线路规划（按 routes_df 逐条线路展示） ===

def _ensure_seq_list(x):
    """将 routes_df.sequence 的值规范为字符串列表。
    支持 list/tuple，或形如 '["C1","C2"]' / "['C1','C2']" 的字符串；
    若是逗号分隔无括号的字符串也能解析；其他情况返回空列表。
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(i) for i in x]
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or ("," in s):
            import ast
            try:
                val = ast.literal_eval(s)
                if isinstance(val, (list, tuple)):
                    return [str(i) for i in val]
            except Exception:
                return [i.strip().strip("'\"") for i in s.strip("[]").split(",") if i.strip()]
        return [s]
    return []


# --- 新增: visualize_route_plan_folium_with_osm ---
def visualize_route_plan_folium_with_osm(
    routes_df: pd.DataFrame,
    *,
    df_cus: pd.DataFrame,
    df_depot: pd.DataFrame,
    station_filter = None,
    tiles: str = "OpenStreetMap",
    show_all_station_customers: bool = False,
    save_path = None,
    zoom_start: int = 12,
    graph_pickle_path: str = "./notebook/graph/G_taiyuan.pkl",
    weight: str = "length",
):
    """与 visualize_route_plan_folium 功能相同，但**线路连线沿 OSM 路网**绘制。

    约定：
    - routes_df: 包含列 [route_id, station, sequence]，其中 sequence 为以逗号分隔的 BB_RETAIL_CUSTOMER_CODE 序列。
    - df_cus: 含客户坐标列 [BB_RETAIL_CUSTOMER_CODE, MD03_RETAIL_CUST_LON, MD03_RETAIL_CUST_LAT] 及所属站点列 MD03_DIST_STATION_CODE。
    - df_depot: 含站点坐标列 [MD03_SHIPMENT_STATION_CODE, MD03_SHIPMENT_STATION_LON, MD03_SHIPMENT_STATION_LAT]。
    - graph_pickle_path: 指向已持久化的 NetworkX 路网图（nodes 需包含 'x','y' 经度/纬度，edges 可包含 'length'）。
    - weight: 最短路权重键；若边上无该键，则自动回退为欧氏距离。

    返回：folium.Map
    """
    import pickle
    import networkx as nx
    from folium.plugins import MeasureControl
    import folium

    # ---- 读取路网 G ----
    with open(graph_pickle_path, "rb") as f:
        G = pickle.load(f)

    # ---- 基础数据准备 ----
    # 1) 客户坐标索引
    _cus = df_cus[[
        "BB_RETAIL_CUSTOMER_CODE",
        "MD03_RETAIL_CUST_LON",
        "MD03_RETAIL_CUST_LAT",
        "MD03_DIST_STATION_CODE",
    ]].drop_duplicates()
    _cus["MD03_RETAIL_CUST_LON"] = pd.to_numeric(_cus["MD03_RETAIL_CUST_LON"], errors="coerce")
    _cus["MD03_RETAIL_CUST_LAT"] = pd.to_numeric(_cus["MD03_RETAIL_CUST_LAT"], errors="coerce")
    cus_pos = {
        str(r.BB_RETAIL_CUSTOMER_CODE): (float(r.MD03_RETAIL_CUST_LON), float(r.MD03_RETAIL_CUST_LAT))
        for _, r in _cus.dropna(subset=["MD03_RETAIL_CUST_LON", "MD03_RETAIL_CUST_LAT"]).iterrows()
    }

    # 2) 站点坐标索引
    _dep = df_depot[[
        "MD03_SHIPMENT_STATION_CODE",
        "MD03_SHIPMENT_STATION_LON",
        "MD03_SHIPMENT_STATION_LAT",
    ]].drop_duplicates()
    _dep["MD03_SHIPMENT_STATION_LON"] = pd.to_numeric(_dep["MD03_SHIPMENT_STATION_LON"], errors="coerce")
    _dep["MD03_SHIPMENT_STATION_LAT"] = pd.to_numeric(_dep["MD03_SHIPMENT_STATION_LAT"], errors="coerce")
    depot_pos = {
        str(r.MD03_SHIPMENT_STATION_CODE): (float(r.MD03_SHIPMENT_STATION_LON), float(r.MD03_SHIPMENT_STATION_LAT))
        for _, r in _dep.dropna(subset=["MD03_SHIPMENT_STATION_LON", "MD03_SHIPMENT_STATION_LAT"]).iterrows()
    }

    # 3) 过滤并准备路线
    rdf = routes_df.copy()
    if station_filter is not None:
        rdf = rdf[rdf["station"].astype(str) == str(station_filter)]
    rdf = rdf.dropna(subset=["sequence", "station"]).copy()

    # 地图中心
    def _all_points_lonlat():
        pts = []
        for seq_raw, st in zip(rdf["sequence"].tolist(), rdf["station"].astype(str).tolist()):
            if str(st) in depot_pos:
                pts.append(depot_pos[str(st)])
            seq_codes = _ensure_seq_list(seq_raw)
            for cid in seq_codes:
                if cid in cus_pos:
                    pts.append(cus_pos[cid])
        return pts

    pts = _all_points_lonlat()
    if len(pts) == 0:
        # 空数据回退
        center = [31.2, 121.5]
    else:
        center = [sum(p[1] for p in pts) / len(pts), sum(p[0] for p in pts) / len(pts)]

    m = folium.Map(location=center, zoom_start=zoom_start, tiles=tiles, control_scale=True)
    m.add_child(MeasureControl(position='topleft', primary_length_unit='kilometers'))


    # ---- 可切换的路线图层（默认隐藏） ----
    route_lines_fg = folium.FeatureGroup(name="线路连线(OSM)", overlay=True, control=True, show=False)
    route_lines_fg.add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # ---- 建立最近节点查询 ----
    node_ids = np.array(list(G.nodes))
    node_xy = np.array([(G.nodes[n]['x'], G.nodes[n]['y']) for n in node_ids], dtype=float)

    def _nearest_node(lon: float, lat: float):
        # 直接向量化最近邻（数据量中等时够用）
        d2 = (node_xy[:, 0] - lon) ** 2 + (node_xy[:, 1] - lat) ** 2
        idx = int(np.argmin(d2))
        return int(node_ids[idx])

    def _shortest_path_coords(ptA, ptB):
        """从经纬度 ptA->ptB 生成沿路网的坐标序列[(lat, lon), ...]；失败时回退直线。"""
        try:
            a = _nearest_node(ptA[0], ptA[1])
            b = _nearest_node(ptB[0], ptB[1])

            def _w(u, v, data):
                if isinstance(weight, str) and weight in data:
                    return data[weight]
                # 回退：欧氏
                x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
                x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
                return math.hypot(x1 - x2, y1 - y2)

            path_nodes = nx.shortest_path(G, a, b, weight=_w)
            coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path_nodes]
            return coords
        except Exception:
            # 任一失败，退回直线
            return [(ptA[1], ptA[0]), (ptB[1], ptB[0])]

    # ---- 颜色表 ----
    palette = [
        'Purple','Blue','Green','Orange','Red','DarkRed','DarkBlue','DarkGreen','Gray',
        'CadetBlue','LightGray','Black','Pink','LightGreen','Beige','DarkPurple','LightBlue'
    ]

    # ---- 绘制每条线路（按 OSM 最短路拼接） ----
    for ridx, row in rdf.iterrows():
        route_id = str(row.get("route_id", ridx))
        station = str(row["station"]) if pd.notna(row["station"]) else None
        color = palette[ridx % len(palette)]

        if station not in depot_pos:
            # 无站点坐标：跳过该路线
            continue
        depot_lon, depot_lat = depot_pos[station]

        # 序列 → 点集合（使用健壮解析）
        seq_codes = _ensure_seq_list(row.get("sequence"))
        pts_ll = []
        # 起点仓
        pts_ll.append((depot_lon, depot_lat))
        # 客户
        for cid in seq_codes:
            if cid in cus_pos:
                pts_ll.append(cus_pos[cid])
        # 终点回仓
        pts_ll.append((depot_lon, depot_lat))

        # 为该线路创建一个独立图层，用于显示客户点与序号（默认显示）
        fg = folium.FeatureGroup(name=f"线路 {route_id}", overlay=True, control=True, show=True)
        fg.add_to(m)

        # 逐段拼接 OSM 路径
        osm_coords = []
        for i in range(len(pts_ll) - 1):
            seg = _shortest_path_coords(pts_ll[i], pts_ll[i + 1])
            if osm_coords and seg:
                # 避免重复拼接首节点
                osm_coords.extend(seg[1:])
            else:
                osm_coords.extend(seg)

        if len(osm_coords) >= 2:
            folium.PolyLine(
                locations=osm_coords,
                weight=4,
                opacity=0.9,
                color=color,
                tooltip=f"{route_id}",
            ).add_to(route_lines_fg)

        # 在线路图层上添加客户点与序号（确保可见）
        for idx, cid in enumerate(seq_codes, start=1):
            if cid not in cus_pos:
                continue
            lon_c, lat_c = cus_pos[cid]  # 注意：cus_pos 存的是 (lon, lat)
            folium.CircleMarker(
                location=(lat_c, lon_c),
                radius=4,
                color=color,
                weight=2,
                fill=True,
                fill_opacity=0.8,
                tooltip=f"#{idx} 客户 {cid}",
            ).add_to(fg)
            folium.map.Marker(
                location=(lat_c, lon_c),
                icon=folium.DivIcon(html=f"<div style='font-size:10px;color:{color};font-weight:bold;'>{idx}</div>"),
            ).add_to(fg)

    # ---- 可选：显示所有该站点客户点 ----
    if show_all_station_customers and station_filter is not None:
        st = str(station_filter)
        fg_pts = folium.FeatureGroup(name=f"{st} 客户点", overlay=True, control=True, show=True)
        fg_pts.add_to(m)
        _sub = _cus[_cus["MD03_DIST_STATION_CODE"].astype(str) == st]
        for _, r in _sub.iterrows():
            if pd.notna(r.MD03_RETAIL_CUST_LON) and pd.notna(r.MD03_RETAIL_CUST_LAT):
                folium.CircleMarker(
                    location=(float(r.MD03_RETAIL_CUST_LAT), float(r.MD03_RETAIL_CUST_LON)),
                    radius=3,
                    color='black',
                    weight=1,
                    fill=True,
                    fill_opacity=0.6,
                    popup=folium.Popup(
                        f"客户:{r.BB_RETAIL_CUSTOMER_CODE}", max_width=260
                    ),
                ).add_to(fg_pts)

    # 站点标记
    fg_depot = folium.FeatureGroup(name="发货点", overlay=True, control=True, show=True)
    fg_depot.add_to(m)
    for st, (lon, lat) in depot_pos.items():
        folium.RegularPolygonMarker(
            location=(lat, lon),
            number_of_sides=4,
            radius=5,
            weight=2,
            color='darkblue',
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup(f"发货点: {st}", max_width=260),
        ).add_to(fg_depot)

    if save_path:
        m.save(save_path)
    return m

# --- 可视化：按 routes_df 渲染线路（支持多条线路分图层开关） ---
def visualize_route_plan_folium03(
    routes_df: pd.DataFrame,
    *,
    df_cus: pd.DataFrame,
    df_depot: pd.DataFrame,
    station_filter: str | None = None,
    tiles: str = "CartoDB positron",
    show_all_station_customers: bool = True,
    save_path: str | None = None,
    zoom_start: int = 12,
    # --- 性能/显示开关 ---
    use_fast_cluster: bool = True,
    cluster_chunked: bool = True,
    prefer_canvas: bool = True,
    decimate_every_n: int | None = None,
    max_points_per_station: int | None = None,
    show_density_heatmap: bool = False,
    # --- 线路客户点显示策略（B做法） ---
    show_all_route_markers: bool = True,          # 为每条线路把所有客户画成点
    cluster_route_markers: bool = True,           # 这些点使用聚类（缩小自动合并）
    route_cluster_disable_at_zoom: int = 15,      # 该层在该缩放级别后不再聚类
    show_route_polylines: bool = False,           # 折线默认不显示（仍可在图层面板勾选）
    route_marker_radius: int = 3,                 # 非聚类时的点半径
):
    """
    可视化线路规划（B做法：按线路独立图层）：
    - 每条线路单独一个子图层，可独立开关查看/导出/对比；
    - 默认客户点聚类，缩放解散；折线图层默认关闭；
    - 背景客户（非本次路线客户）仍可用聚类/抽样/热力来减负。
    """
    import folium
    from folium import Map, FeatureGroup, LayerControl, CircleMarker, PolyLine
    from folium.plugins import (
        FastMarkerCluster, MarkerCluster, HeatMap, MiniMap, Fullscreen, FeatureGroupSubGroup
    )
    import numpy as np

    # --- 数据准备与过滤 ---
    routes = routes_df.copy()
    routes['station'] = routes['station'].astype(str)
    if station_filter:
        routes = routes[routes['station'].astype(str) == str(station_filter)].copy()
        if routes.empty:
            raise ValueError(f"没有找到站点 {station_filter} 的线路数据。")

    # 提取“本次路线涉及到的客户”集合
    def _split_codes(s: str) -> list[str]:
        if pd.isna(s) or not s:
            return []
        return [c.strip() for c in str(s).split(",") if c.strip()]

    route_cus_sets = [set(_split_codes(s)) for s in routes['sequence'].tolist()]
    route_cus_all = set().union(*route_cus_sets) if route_cus_sets else set()

    # 站点经纬度
    stations = df_depot.drop_duplicates(subset=["MD03_SHIPMENT_STATION_CODE"]).copy()
    stations["MD03_SHIPMENT_STATION_CODE"] = stations["MD03_SHIPMENT_STATION_CODE"].astype(str)
    st_coords = stations.set_index("MD03_SHIPMENT_STATION_CODE")[["MD03_SHIPMENT_STATION_LON","MD03_SHIPMENT_STATION_LAT"]].astype(float).to_dict("index")

    # 客户点
    cus = df_cus.copy()
    cus["MD03_DIST_STATION_CODE"] = cus["MD03_DIST_STATION_CODE"].astype(str)
    for col in ["MD03_RETAIL_CUST_LON","MD03_RETAIL_CUST_LAT"]:
        cus[col] = pd.to_numeric(cus[col], errors="coerce")
    cus = cus.dropna(subset=["MD03_RETAIL_CUST_LON","MD03_RETAIL_CUST_LAT"])
    cus["BB_RETAIL_CUSTOMER_CODE"] = cus["BB_RETAIL_CUSTOMER_CODE"].astype(str)

    # 地图初始中心
    if not routes.empty:
        st0 = routes.iloc[0]["station"]
        if str(st0) in st_coords:
            center = (st_coords[str(st0)]["MD03_SHIPMENT_STATION_LAT"], st_coords[str(st0)]["MD03_SHIPMENT_STATION_LON"])
        else:
            center = (cus["MD03_RETAIL_CUST_LAT"].median(), cus["MD03_RETAIL_CUST_LON"].median())
    else:
        center = (cus["MD03_RETAIL_CUST_LAT"].median(), cus["MD03_RETAIL_CUST_LON"].median())

    m = Map(location=center, zoom_start=zoom_start, tiles=tiles, prefer_canvas=prefer_canvas, no_touch=False, disable_3d=True)

    # 小地图和全屏
    MiniMap(toggle_display=True).add_to(m)
    Fullscreen().add_to(m)

    # ---------------------------
    # 背景客户：聚类 / 抽样 / 热力（三选一），不包含本次路线客户
    # ---------------------------
    if show_all_station_customers and not cus.empty:
        st_filter_vals = routes["station"].unique().tolist()
        cus_bg = cus[cus["MD03_DIST_STATION_CODE"].isin(st_filter_vals)].copy()

        if route_cus_all:
            cus_bg = cus_bg[~cus_bg["BB_RETAIL_CUSTOMER_CODE"].isin(route_cus_all)]

        if max_points_per_station is not None:
            tmp = []
            for st in st_filter_vals:
                sub = cus_bg[cus_bg["MD03_DIST_STATION_CODE"] == st]
                if len(sub) > max_points_per_station:
                    if str(st) in st_coords:
                        slon = float(st_coords[str(st)]["MD03_SHIPMENT_STATION_LON"])
                        slat = float(st_coords[str(st)]["MD03_SHIPMENT_STATION_LAT"])
                        d = ((sub["MD03_RETAIL_CUST_LON"] - slon)**2 + (sub["MD03_RETAIL_CUST_LAT"] - slat)**2).to_numpy()
                        keep_idx = np.argsort(d)[:max_points_per_station]
                        sub = sub.iloc[keep_idx]
                    else:
                        sub = sub.sample(n=max_points_per_station, random_state=0)
                tmp.append(sub)
            cus_bg = pd.concat(tmp, ignore_index=True)

        if decimate_every_n and decimate_every_n > 1:
            cus_bg = cus_bg.iloc[::decimate_every_n].copy()

        if show_density_heatmap:
            heat = cus_bg[["MD03_RETAIL_CUST_LAT","MD03_RETAIL_CUST_LON"]].to_numpy().tolist()
            if heat:
                HeatMap(heat, name="所有客户-热力", radius=8, blur=12, control=True).add_to(m)
        else:
            fg_bg = FeatureGroup(name="所有客户-聚类", show=False)
            if use_fast_cluster:
                points = cus_bg[["MD03_RETAIL_CUST_LAT","MD03_RETAIL_CUST_LON","BB_RETAIL_CUSTOMER_CODE"]].values.tolist()
                data = [[lat, lon, str(code)] for lat, lon, code in points]
                FastMarkerCluster(data, options={"chunkedLoading": bool(cluster_chunked)}).add_to(fg_bg)
            else:
                mc = MarkerCluster(options={"chunkedLoading": bool(cluster_chunked)}).add_to(fg_bg)
                for _, r in cus_bg.iterrows():
                    CircleMarker(
                        location=(float(r["MD03_RETAIL_CUST_LAT"]), float(r["MD03_RETAIL_CUST_LON"])),
                        radius=2,
                        weight=0,
                        fill=True,
                        fill_opacity=0.6,
                        popup=str(r["BB_RETAIL_CUSTOMER_CODE"]),
                    ).add_to(mc)
            fg_bg.add_to(m)

    # ---------------------------
    # 站点（少量） + 线路客户点（每条线路独立图层） + 折线（可选）
    # ---------------------------
    # 站点层
    fg_st = FeatureGroup(name="发货点", show=True)
    for st in routes["station"].unique():
        st = str(st)
        if st in st_coords:
            slon = float(st_coords[st]["MD03_SHIPMENT_STATION_LON"])
            slat = float(st_coords[st]["MD03_SHIPMENT_STATION_LAT"])
            CircleMarker(location=(slat, slon), radius=6, color="#2c7bb6", fill=True, fill_opacity=0.9,
                         popup=f"站点 {st}", weight=2).add_to(fg_st)
    fg_st.add_to(m)

    # 父容器：承载所有“线路客户点”的父层
    if show_all_route_markers:
        if cluster_route_markers:
            fg_route_pts_parent = FeatureGroup(name="线路客户-聚类（全部）", show=True)
            mc_route_parent = MarkerCluster(options={
                "chunkedLoading": bool(cluster_chunked),
                "disableClusteringAtZoom": int(route_cluster_disable_at_zoom),
                "spiderfyOnMaxZoom": True
            })
            mc_route_parent.add_to(fg_route_pts_parent)
            # 子组容器：route_id -> FeatureGroupSubGroup
            _route_subgroups: dict[str, FeatureGroupSubGroup] = {}
        else:
            fg_route_pts_parent = FeatureGroup(name="线路客户-点（全部）", show=True)
            _route_subgroups = {}  # route_id -> FeatureGroup

    # 折线父层（默认隐藏）
    fg_lines = FeatureGroup(name="线路折线", show=bool(show_route_polylines))

    # 为每条线路添加“所有客户点”，并（可选）折线
    for _, row in routes.iterrows():
        rid = str(row["route_id"])
        st = str(row["station"])
        seq_codes = _split_codes(row.get("sequence", ""))

        g = cus[cus["BB_RETAIL_CUSTOMER_CODE"].isin(seq_codes)][
            ["BB_RETAIL_CUSTOMER_CODE","MD03_RETAIL_CUST_LON","MD03_RETAIL_CUST_LAT"]
        ].drop_duplicates()
        if g.empty:
            continue
        g = g.set_index("BB_RETAIL_CUSTOMER_CODE").reindex(seq_codes).dropna()
        if g.empty:
            continue

        # --- 创建/获取 子层 ---
        if show_all_route_markers:
            if rid not in _route_subgroups:
                if cluster_route_markers:
                    sub = FeatureGroupSubGroup(mc_route_parent, name=f"线路 {rid}", show=False)
                    # 注意：子组要直接 add 到 map（不是加到 fg_route_pts_parent）
                    sub.add_to(m)
                else:
                    sub = FeatureGroup(name=f"线路 {rid}", show=False)
                    sub.add_to(m)
                _route_subgroups[rid] = sub
            sub_layer = _route_subgroups[rid]

            # --- 逐点添加到该线路的子层 ---
            for cid, (lon, lat) in g[["MD03_RETAIL_CUST_LON","MD03_RETAIL_CUST_LAT"]].iterrows():
                if cluster_route_markers:
                    folium.Marker(
                        location=(float(lat), float(lon)),
                        tooltip=f"{rid} | {cid}",
                        popup=folium.Popup(html=f"&lt;b&gt;线路&lt;/b&gt;：{rid}&lt;br/&gt;&lt;b&gt;客户&lt;/b&gt;：{cid}", max_width=280),
                        **{"title": rid}
                    ).add_to(sub_layer)
                else:
                    CircleMarker(
                        location=(float(lat), float(lon)),
                        radius=route_marker_radius,
                        weight=0.5,
                        fill=True,
                        fill_opacity=0.85,
                        tooltip=f"{rid} | {cid}"
                    ).add_to(sub_layer)

        # --- 可选：画折线（默认不显示；用户可在图层面板勾选） ---
        if show_route_polylines:
            latlons = [(float(lat), float(lon)) for lon, lat in g[["MD03_RETAIL_CUST_LON","MD03_RETAIL_CUST_LAT"]].to_numpy()]
            if st in st_coords:
                slon = float(st_coords[st]["MD03_SHIPMENT_STATION_LON"])
                slat = float(st_coords[st]["MD03_SHIPMENT_STATION_LAT"])
                latlons = [(slat, slon)] + latlons + [(slat, slon)]
            PolyLine(latlons, color="#f46d43", weight=2.0, opacity=0.9, smooth_factor=0.5,
                     tooltip=row.get("route_name", rid)).add_to(fg_lines)

    # 把父层加到地图
    if show_all_route_markers:
        fg_route_pts_parent.add_to(m)
    fg_lines.add_to(m)

    LayerControl(collapsed=False).add_to(m)

    if save_path:
        m.save(save_path)
    return m

def visualize_route_plan_folium02(
    routes_df: pd.DataFrame,
    *,
    df_cus: pd.DataFrame,
    df_depot: pd.DataFrame,
    station_filter: str | None = None,
    tiles: str = "CartoDB positron",
    show_all_station_customers: bool = True,
    save_path: str | None = None,
    zoom_start: int = 12,
    # 性能/背景层
    use_fast_cluster: bool = True,
    cluster_chunked: bool = True,
    prefer_canvas: bool = True,
    decimate_every_n: int | None = None,
    max_points_per_station: int | None = None,
    show_density_heatmap: bool = False,
    # —— 新增：路线客户点策略 ——
    show_all_route_markers: bool = True,     # 画所有路线客户点
    cluster_route_markers: bool = True,      # 路线客户点用聚类
    route_cluster_disable_at_zoom: int = 15, # 放大到该级别后不再聚类
    show_route_polylines: bool = False,      # 折线默认不显示
    route_marker_radius: int = 3,            # 非聚类时的点大小
):
    import folium
    from folium import Map, FeatureGroup, LayerControl, CircleMarker, PolyLine
    from folium.plugins import FastMarkerCluster, MarkerCluster, HeatMap, MiniMap, Fullscreen
    import numpy as np

    # --- 数据准备 ---
    routes = routes_df.copy()
    routes['station'] = routes['station'].astype(str)
    if station_filter:
        routes = routes[routes['station'] == str(station_filter)].copy()
        if routes.empty:
            raise ValueError(f"没有找到站点 {station_filter} 的线路数据。")

    def _split_codes(s: str) -> list[str]:
        if pd.isna(s) or not s:
            return []
        return [c.strip() for c in str(s).split(",") if c.strip()]

    route_cus_sets = [set(_split_codes(s)) for s in routes['sequence'].tolist()]
    route_cus_all = set().union(*route_cus_sets) if route_cus_sets else set()

    # 站点
    stations = df_depot.drop_duplicates(subset=["MD03_SHIPMENT_STATION_CODE"]).copy()
    stations["MD03_SHIPMENT_STATION_CODE"] = stations["MD03_SHIPMENT_STATION_CODE"].astype(str)
    st_coords = stations.set_index("MD03_SHIPMENT_STATION_CODE")[
        ["MD03_SHIPMENT_STATION_LON","MD03_SHIPMENT_STATION_LAT"]
    ].astype(float).to_dict("index")

    # 客户
    cus = df_cus.copy()
    cus["MD03_DIST_STATION_CODE"] = cus["MD03_DIST_STATION_CODE"].astype(str)
    for col in ["MD03_RETAIL_CUST_LON","MD03_RETAIL_CUST_LAT"]:
        cus[col] = pd.to_numeric(cus[col], errors="coerce")
    cus = cus.dropna(subset=["MD03_RETAIL_CUST_LON","MD03_RETAIL_CUST_LAT"])
    cus["BB_RETAIL_CUSTOMER_CODE"] = cus["BB_RETAIL_CUSTOMER_CODE"].astype(str)

    # 地图中心
    if not routes.empty and str(routes.iloc[0]["station"]) in st_coords:
        st0 = str(routes.iloc[0]["station"])
        center = (st_coords[st0]["MD03_SHIPMENT_STATION_LAT"], st_coords[st0]["MD03_SHIPMENT_STATION_LON"])
    else:
        center = (cus["MD03_RETAIL_CUST_LAT"].median(), cus["MD03_RETAIL_CUST_LON"].median())

    m = Map(location=center, zoom_start=zoom_start, tiles=tiles, prefer_canvas=prefer_canvas)
    MiniMap(toggle_display=True).add_to(m)
    Fullscreen().add_to(m)

    # 背景客户层：聚类/抽样/热力（排除本次线路客户）
    if show_all_station_customers and not cus.empty:
        st_filter_vals = routes["station"].unique().tolist()
        cus_bg = cus[cus["MD03_DIST_STATION_CODE"].isin(st_filter_vals)].copy()
        if route_cus_all:
            cus_bg = cus_bg[~cus_bg["BB_RETAIL_CUSTOMER_CODE"].isin(route_cus_all)]
        if max_points_per_station is not None:
            tmp = []
            for st in st_filter_vals:
                sub = cus_bg[cus_bg["MD03_DIST_STATION_CODE"] == st]
                if len(sub) > max_points_per_station and str(st) in st_coords:
                    slon = float(st_coords[str(st)]["MD03_SHIPMENT_STATION_LON"])
                    slat = float(st_coords[str(st)]["MD03_SHIPMENT_STATION_LAT"])
                    d = ((sub["MD03_RETAIL_CUST_LON"] - slon)**2 + (sub["MD03_RETAIL_CUST_LAT"] - slat)**2).to_numpy()
                    keep_idx = np.argsort(d)[:max_points_per_station]
                    sub = sub.iloc[keep_idx]
                tmp.append(sub)
            cus_bg = pd.concat(tmp, ignore_index=True)
        if decimate_every_n and decimate_every_n > 1:
            cus_bg = cus_bg.iloc[::decimate_every_n].copy()

        if show_density_heatmap:
            heat = cus_bg[["MD03_RETAIL_CUST_LAT","MD03_RETAIL_CUST_LON"]].to_numpy().tolist()
            if heat:
                HeatMap(heat, name="所有客户-热力", radius=8, blur=12, control=True).add_to(m)
        else:
            fg_bg = folium.FeatureGroup(name="所有客户-聚类", show=False)
            if use_fast_cluster:
                points = cus_bg[["MD03_RETAIL_CUST_LAT","MD03_RETAIL_CUST_LON","BB_RETAIL_CUSTOMER_CODE"]].values.tolist()
                data = [[lat, lon, str(code)] for lat, lon, code in points]
                FastMarkerCluster(data, options={"chunkedLoading": bool(cluster_chunked)}).add_to(fg_bg)
            else:
                mc = MarkerCluster(options={"chunkedLoading": bool(cluster_chunked)}).add_to(fg_bg)
                for _, r in cus_bg.iterrows():
                    CircleMarker((float(r["MD03_RETAIL_CUST_LAT"]), float(r["MD03_RETAIL_CUST_LON"])),
                                 radius=2, weight=0, fill=True, fill_opacity=0.6,
                                 popup=str(r["BB_RETAIL_CUSTOMER_CODE"])).add_to(mc)
            fg_bg.add_to(m)

    # 站点层
    fg_st = folium.FeatureGroup(name="发货点", show=True)
    for st in routes["station"].unique():
        st = str(st)
        if st in st_coords:
            slon = float(st_coords[st]["MD03_SHIPMENT_STATION_LON"])
            slat = float(st_coords[st]["MD03_SHIPMENT_STATION_LAT"])
            CircleMarker((slat, slon), radius=6, color="#2c7bb6", fill=True, fill_opacity=0.9,
                         popup=f"站点 {st}", weight=2).add_to(fg_st)
    fg_st.add_to(m)

    # 路线客户点层（全部画出；可聚类）
    if show_all_route_markers:
        if cluster_route_markers:
            fg_route_pts = folium.FeatureGroup(name="线路客户-聚类", show=True)
            mc_route = MarkerCluster(options={
                "chunkedLoading": bool(cluster_chunked),
                "disableClusteringAtZoom": int(route_cluster_disable_at_zoom),
                "spiderfyOnMaxZoom": True
            }).add_to(fg_route_pts)
        else:
            fg_route_pts = folium.FeatureGroup(name="线路客户-点", show=True)

    # 折线层（默认隐藏）
    fg_lines = folium.FeatureGroup(name="线路折线", show=bool(show_route_polylines))

    # 为每条线路添加“所有客户点”，并（可选）折线
    for _, row in routes.iterrows():
        rid = str(row["route_id"])
        st = str(row["station"])
        seq_codes = _split_codes(row.get("sequence", ""))

        g = cus[cus["BB_RETAIL_CUSTOMER_CODE"].isin(seq_codes)][
            ["BB_RETAIL_CUSTOMER_CODE","MD03_RETAIL_CUST_LON","MD03_RETAIL_CUST_LAT"]
        ].drop_duplicates()
        if g.empty:
            continue
        g = g.set_index("BB_RETAIL_CUSTOMER_CODE").reindex(seq_codes).dropna()
        if g.empty:
            continue

        # 1) 客户点：全部加到“路线客户点”层
        if show_all_route_markers:
            for cid, (lon, lat) in g[["MD03_RETAIL_CUST_LON","MD03_RETAIL_CUST_LAT"]].iterrows():
                if cluster_route_markers:
                    folium.Marker(
                        location=(float(lat), float(lon)),
                        tooltip=f"{rid} | {cid}"
                    ).add_to(mc_route)
                else:
                    CircleMarker(
                        location=(float(lat), float(lon)),
                        radius=route_marker_radius,
                        weight=0.5,
                        fill=True,
                        fill_opacity=0.85,
                        tooltip=f"{rid} | {cid}"
                    ).add_to(fg_route_pts)

        # 2) 可选折线（默认不显示）
        if show_route_polylines:
            latlons = [(float(lat), float(lon)) for lon, lat in g[["MD03_RETAIL_CUST_LON","MD03_RETAIL_CUST_LAT"]].to_numpy()]
            if st in st_coords:
                slon = float(st_coords[st]["MD03_SHIPMENT_STATION_LON"])
                slat = float(st_coords[st]["MD03_SHIPMENT_STATION_LAT"])
                latlons = [(slat, slon)] + latlons + [(slat, slon)]
            PolyLine(latlons, color="#f46d43", weight=2.5, opacity=0.9, smooth_factor=0.5,
                     tooltip=row.get("route_name", rid)).add_to(fg_lines)

    if show_all_route_markers:
        fg_route_pts.add_to(m)
    fg_lines.add_to(m)
    LayerControl(collapsed=False).add_to(m)

    if save_path:
        m.save(save_path)
    return m

def visualize_route_plan_folium(
    routes_df: pd.DataFrame,
    *,
    df_cus: pd.DataFrame,
    df_depot: pd.DataFrame,
    station_filter: str | None = None,
    tiles: str = "CartoDB positron",
    show_all_station_customers: bool = False,   # 默认关闭背景层，聚焦线路
    save_path: str | None = None,
    zoom_start: int = 12,
    # --- 背景/性能 ---
    use_fast_cluster: bool = True,
    cluster_chunked: bool = True,
    prefer_canvas: bool = True,
    decimate_every_n: int | None = None,
    max_points_per_station: int | None = None,
    show_density_heatmap: bool = False,
    # --- 线路展开显示策略（新增“中心点展开模式”） ---
    centroid_mode: bool = True,                 # True=每条线路初始以中心点代表；点击再展开该线路所有客户点
    route_cluster_disable_at_zoom: int = 15,    # 展开后的客户点在此级别以上不再聚类
    route_marker_radius: int = 3,               # 展开后单点半径（非聚类时）
):
    """
    可视化线路规划：
    - *中心点模式*（centroid_mode=True）：每条线路初始只显示一个中心点（经纬度均值），
      点击中心点 → 展开该线路的所有客户点（再次点击可收起）。其它线路不受影响，仍以单点呈现。
    - 背景客户层（非本次线路客户）可选显示。
    """
    import folium, numpy as np
    from folium import Map, FeatureGroup, LayerControl, CircleMarker, PolyLine
    from folium.plugins import (
        FastMarkerCluster, MarkerCluster, HeatMap, MiniMap, Fullscreen, FeatureGroupSubGroup
    )

    # --- 数据准备 ---
    routes = routes_df.copy()
    routes['station'] = routes['station'].astype(str)
    if station_filter:
        routes = routes[routes['station'].astype(str) == str(station_filter)].copy()
        if routes.empty:
            raise ValueError(f"没有找到站点 {station_filter} 的线路数据。")

    # 客户点表清洗
    cus = df_cus.copy()
    cus["MD03_DIST_STATION_CODE"] = cus["MD03_DIST_STATION_CODE"].astype(str)
    for col in ["MD03_RETAIL_CUST_LON","MD03_RETAIL_CUST_LAT"]:
        cus[col] = pd.to_numeric(cus[col], errors="coerce")
    cus = cus.dropna(subset=["MD03_RETAIL_CUST_LON","MD03_RETAIL_CUST_LAT"])
    cus["BB_RETAIL_CUSTOMER_CODE"] = cus["BB_RETAIL_CUSTOMER_CODE"].astype(str)

    # 站点坐标
    stations = df_depot.drop_duplicates(subset=["MD03_SHIPMENT_STATION_CODE"]).copy()
    stations["MD03_SHIPMENT_STATION_CODE"] = stations["MD03_SHIPMENT_STATION_CODE"].astype(str)
    st_coords = stations.set_index("MD03_SHIPMENT_STATION_CODE")[["MD03_SHIPMENT_STATION_LON","MD03_SHIPMENT_STATION_LAT"]].astype(float).to_dict("index")

    # 初始中心
    if not routes.empty:
        st0 = str(routes.iloc[0]["station"])
        if st0 in st_coords:
            center = (st_coords[st0]["MD03_SHIPMENT_STATION_LAT"], st_coords[st0]["MD03_SHIPMENT_STATION_LON"])
        else:
            center = (cus["MD03_RETAIL_CUST_LAT"].median(), cus["MD03_RETAIL_CUST_LON"].median())
    else:
        center = (cus["MD03_RETAIL_CUST_LAT"].median(), cus["MD03_RETAIL_CUST_LON"].median())

    m = Map(location=center, zoom_start=zoom_start, tiles=tiles, prefer_canvas=prefer_canvas, no_touch=False, disable_3d=True)
    MiniMap(toggle_display=True).add_to(m)
    Fullscreen().add_to(m)

    # ---------------- 背景客户（可选） ----------------
    if show_all_station_customers and not cus.empty:
        # 背景层仅展示非本次线路客户（避免与展开层重复）
        route_cus_all = set()
        def _split_codes(s: str) -> list[str]:
            if pd.isna(s) or not s:
                return []
            return [c.strip() for c in str(s).split(",") if c.strip()]
        for s in routes['sequence'].tolist():
            route_cus_all.update(_split_codes(s))

        st_filter_vals = routes["station"].unique().tolist()
        cus_bg = cus[cus["MD03_DIST_STATION_CODE"].isin(st_filter_vals)].copy()
        if route_cus_all:
            cus_bg = cus_bg[~cus_bg["BB_RETAIL_CUSTOMER_CODE"].isin(route_cus_all)]

        if max_points_per_station is not None and not cus_bg.empty:
            tmp = []
            for st in st_filter_vals:
                sub = cus_bg[cus_bg["MD03_DIST_STATION_CODE"] == st]
                if len(sub) > max_points_per_station:
                    if str(st) in st_coords:
                        slon = float(st_coords[str(st)]["MD03_SHIPMENT_STATION_LON"])
                        slat = float(st_coords[str(st)]["MD03_SHIPMENT_STATION_LAT"])
                        d = ((sub["MD03_RETAIL_CUST_LON"] - slon)**2 + (sub["MD03_RETAIL_CUST_LAT"] - slat)**2).to_numpy()
                        keep_idx = np.argsort(d)[:max_points_per_station]
                        sub = sub.iloc[keep_idx]
                    else:
                        sub = sub.sample(n=max_points_per_station, random_state=0)
                tmp.append(sub)
            cus_bg = pd.concat(tmp, ignore_index=True)

        if decimate_every_n and decimate_every_n > 1 and not cus_bg.empty:
            cus_bg = cus_bg.iloc[::decimate_every_n].copy()

        if show_density_heatmap:
            heat = cus_bg[["MD03_RETAIL_CUST_LAT","MD03_RETAIL_CUST_LON"]].to_numpy().tolist()
            if heat:
                HeatMap(heat, name="所有客户-热力", radius=8, blur=12, control=True).add_to(m)
        else:
            fg_bg = FeatureGroup(name="所有客户-聚类", show=False)
            if use_fast_cluster:
                points = cus_bg[["MD03_RETAIL_CUST_LAT","MD03_RETAIL_CUST_LON","BB_RETAIL_CUSTOMER_CODE"]].values.tolist()
                data = [[lat, lon, str(code)] for lat, lon, code in points]
                FastMarkerCluster(data, options={"chunkedLoading": bool(cluster_chunked)}).add_to(fg_bg)
            else:
                mc = MarkerCluster(options={"chunkedLoading": bool(cluster_chunked)}).add_to(fg_bg)
                for _, r in cus_bg.iterrows():
                    CircleMarker((float(r["MD03_RETAIL_CUST_LAT"]), float(r["MD03_RETAIL_CUST_LON"])),
                                 radius=2, weight=0, fill=True, fill_opacity=0.6,
                                 popup=str(r["BB_RETAIL_CUSTOMER_CODE"])).add_to(mc)
            fg_bg.add_to(m)

    # ---------------- 站点层 ----------------
    fg_st = FeatureGroup(name="发货点", show=True)
    for st in routes["station"].unique():
        st = str(st)
        if st in st_coords:
            slon = float(st_coords[st]["MD03_SHIPMENT_STATION_LON"])
            slat = float(st_coords[st]["MD03_SHIPMENT_STATION_LAT"])
            CircleMarker((slat, slon), radius=6, color="#2c7bb6", fill=True, fill_opacity=0.9,
                         popup=f"站点 {st}", weight=2).add_to(fg_st)
    fg_st.add_to(m)

    # ---------------- 线路中心点 & 展开层 ----------------
    fg_centroids = FeatureGroup(name="线路中心点", show=True)
    fg_routes_parent = FeatureGroup(name="线路客户（展开后）", show=True)
    fg_routes_parent.add_to(m)

    _route_layer_js_names = {}  # rid -> {'layer': jsvar, 'marker': jsvar}

    def _split_codes(s: str) -> list[str]:
        if pd.isna(s) or not s:
            return []
        return [c.strip() for c in str(s).split(",") if c.strip()]

    for _, row in routes.iterrows():
        rid = str(row["route_id"])
        seq_codes = _split_codes(row.get("sequence",""))
        if not seq_codes:
            continue

        g = cus[cus["BB_RETAIL_CUSTOMER_CODE"].isin(seq_codes)][
            ["BB_RETAIL_CUSTOMER_CODE","MD03_RETAIL_CUST_LON","MD03_RETAIL_CUST_LAT"]
        ].drop_duplicates()
        if g.empty:
            continue
        g = g.set_index("BB_RETAIL_CUSTOMER_CODE").reindex(seq_codes).dropna()
        if g.empty:
            continue

        # 中心点
        lat_c = float(np.mean(g["MD03_RETAIL_CUST_LAT"].astype(float)))
        lon_c = float(np.mean(g["MD03_RETAIL_CUST_LON"].astype(float)))

        # 统计/展示信息
        n_cus = len(g)
        veh_info = str(row.get("vehicle_id","")) if "vehicle_id" in row else ""
        veh_cap = str(row.get("vehicle_capacity","")) if "vehicle_capacity" in row else ""
        load_ratio = str(row.get("load_ratio","")) if "load_ratio" in row else ""
        info_lines = [f"&lt;b&gt;线路编号&lt;/b&gt;：{rid}", f"&lt;b&gt;客户数&lt;/b&gt;：{n_cus}"]
        if veh_info or veh_cap or load_ratio:
            info_lines.append(f"&lt;b&gt;车型/容量/装载率&lt;/b&gt;：{veh_info or '-'} / {veh_cap or '-'} / {load_ratio or '-'}")
        popup_html = "&lt;br/&gt;".join(info_lines)

        # 子层（展开后显示客户点）—— 默认不显示
        sub_layer = FeatureGroup(name=f"线路 {rid}（展开）", show=False)
        mc = MarkerCluster(options={
            "chunkedLoading": bool(cluster_chunked),
            "disableClusteringAtZoom": int(route_cluster_disable_at_zoom),
            "spiderfyOnMaxZoom": True
        }).add_to(sub_layer)
        for cid, (lon, lat) in g[["MD03_RETAIL_CUST_LON","MD03_RETAIL_CUST_LAT"]].iterrows():
            folium.Marker(
                location=(float(lat), float(lon)),
                tooltip=f"{rid} | {cid}",
                popup=folium.Popup(html=f"&lt;b&gt;线路&lt;/b&gt;：{rid}&lt;br/&gt;&lt;b&gt;客户&lt;/b&gt;：{cid}", max_width=280),
            ).add_to(mc)
        sub_layer.add_to(m)  # 先加到 map（便于 LayerControl 出现），但初始不勾选

        # 中心点 marker（点击切换该子层显隐）
        centroid_marker = CircleMarker(
            location=(lat_c, lon_c),
            radius=6, color="#1b9e77", fill=True, fill_opacity=0.95, weight=2,
        )
        folium.Popup(html=popup_html, max_width=300).add_to(centroid_marker)
        centroid_marker.add_to(fg_centroids)

        _route_layer_js_names[rid] = {
            "layer": sub_layer.get_name(),
            "marker": centroid_marker.get_name(),
        }

    fg_centroids.add_to(m)

    # JS：点击中心点 → 切换对应子层（展开/收起）
    try:
        from branca.element import MacroElement
    except Exception:
        # 兼容极老版本
        from folium.elements import MacroElement
    from jinja2 import Template
    class _ToggleRouteLayerJS(MacroElement):
        _template = Template(u"""
        {% macro script(this, kwargs) %}
        (function(){
          var map = {{this._parent.get_name()}};
          var registry = {{ registry|safe }};
          function isOnMap(layer){ try { return map.hasLayer(layer); } catch(e){ return false; } }
          Object.keys(registry).forEach(function(rid){
            var LAYER = registry[rid].layer;
            var MARKER = registry[rid].marker;
            if (!MARKER || !LAYER) return;
            MARKER.on('click', function(){
              if (isOnMap(LAYER)){ map.removeLayer(LAYER); } else { map.addLayer(LAYER); }
            });
          });
        })();
        {% endmacro %}
        """)
        def __init__(self, registry: dict):
            super().__init__()
            self._name = "ToggleRouteLayerJS"
            self.registry = registry
        def render(self, **kwargs):
            js_pairs = []
            for rid, obj in self.registry.items():
                js_pairs.append(f'"{rid}":{{"layer": {obj["layer"]}, "marker": {obj["marker"]}}}')
            js = "{" + ",".join(js_pairs) + "}"
            self._template.module.registry = js
            super().render(**kwargs)

    if centroid_mode and _route_layer_js_names:
        m.get_root().add_child(_ToggleRouteLayerJS(_route_layer_js_names))

    LayerControl(collapsed=False).add_to(m)

    if save_path:
        m.save(save_path)
    return m

def visualize_route_plan_folium01(
    routes_df: pd.DataFrame,
    df_cus: pd.DataFrame,
    df_depot: pd.DataFrame,
    *,
    station_filter: Optional[str] = None,
    tiles: str = "OpenStreetMap",
    show_stations: bool = True,
    show_customers: bool = True,
    show_return_to_depot: bool = True,
    show_all_station_customers: bool = True,
    save_path: Optional[str] = None,
    zoom_start: int = 12,
):
    """根据 routes_df 在 folium 上可视化线路。

    约定（与前文保持一致）：
      routes_df 列名：
        - route_id: 线路编码（字符串）
        - route_name: 线路名称（可选）
        - station: 发货点代码（=MD03_DIST_STATION_CODE）
        - vehicle_id: 车辆编码（字符串，可选）
        - load_ratio: 装载率（0-1 或 0-100，均可）
        - distance_km: 线路总里程（km）
        - duration_h: 线路总时长（h 或 min；仅展示，不做单位换算）
        - sequence: 零售户编码序列，支持 list / "C1,C2" / "['C1','C2']" 等
        - prev_distance_km: 上一段距离（与 sequence 对齐，逗号分隔或列表）
        - prev_duration_h: 上一段时长（与 sequence 对齐，逗号分隔或列表）

      df_cus 必含：BB_RETAIL_CUSTOMER_CODE, MD03_RETAIL_CUST_LON, MD03_RETAIL_CUST_LAT
      df_depot 必含：MD03_SHIPMENT_STATION_CODE, MD03_SHIPMENT_STATION_LON, MD03_SHIPMENT_STATION_LAT

    参数:
      ...
      show_all_station_customers: 是否在底图展示该发货点下的所有零售户（即使线路未经过）。
    """
    import folium
    from folium.plugins import MeasureControl

    # 固定色板（按顺序取色，避免每次随机）
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    ]

    # 过滤站点
    rdf = routes_df.copy()
    if station_filter is not None:
        rdf = rdf[rdf["station"].astype(str) == str(station_filter)]
    if rdf.empty:
        raise ValueError("routes_df 为空或与 station_filter 不匹配。")

    # 准备坐标字典
    cus_xy = (
        df_cus.dropna(subset=["MD03_RETAIL_CUST_LON", "MD03_RETAIL_CUST_LAT"])
              .assign(_lon=lambda d: d["MD03_RETAIL_CUST_LON"].astype(float),
                      _lat=lambda d: d["MD03_RETAIL_CUST_LAT"].astype(float))
    )
    cus_xy = dict(zip(cus_xy["BB_RETAIL_CUSTOMER_CODE"].astype(str),
                      zip(cus_xy["_lat"], cus_xy["_lon"])) )  # value=(lat, lon)

    dep_xy_df = (
        df_depot.dropna(subset=["MD03_SHIPMENT_STATION_LON", "MD03_SHIPMENT_STATION_LAT"])
                .assign(_lon=lambda d: d["MD03_SHIPMENT_STATION_LON"].astype(float),
                        _lat=lambda d: d["MD03_SHIPMENT_STATION_LAT"].astype(float))
    )
    dep_xy = dict(zip(dep_xy_df["MD03_SHIPMENT_STATION_CODE"].astype(str),
                      zip(dep_xy_df["_lat"], dep_xy_df["_lon"])) )  # value=(lat, lon)

    # 站点 -> 全部零售户（来自 df_cus）
    station_to_customers = {}
    if "MD03_DIST_STATION_CODE" in df_cus.columns and "BB_RETAIL_CUSTOMER_CODE" in df_cus.columns:
        tmp = df_cus.dropna(subset=["MD03_RETAIL_CUST_LON", "MD03_RETAIL_CUST_LAT"]).copy()
        tmp["_st"] = tmp["MD03_DIST_STATION_CODE"].astype(str)
        tmp["_code"] = tmp["BB_RETAIL_CUSTOMER_CODE"].astype(str)
        for st, grp in tmp.groupby("_st"):
            station_to_customers[st] = set(grp["_code"].tolist())

    # 地图中心：取本批次所有 depot 与客户的平均
    lats, lons = [], []
    for _, row in rdf.iterrows():
        st = str(row["station"]) if "station" in row else None
        if st and st in dep_xy:
            lat, lon = dep_xy[st]
            lats.append(lat); lons.append(lon)
        seq = _ensure_seq_list(row.get("sequence"))
        for c in seq:
            if c in cus_xy:
                lat, lon = cus_xy[c]
                lats.append(lat); lons.append(lon)
    if not lats:
        raise ValueError("无法确定地图中心：缺少坐标。")
    center = [sum(lats)/len(lats), sum(lons)/len(lons)]

    m = folium.Map(location=center, zoom_start=zoom_start, tiles=tiles, control_scale=True)
    m.add_child(MeasureControl(position='topleft', primary_length_unit='kilometers'))
    # ---- 线路连线图层（默认不显示） ----
    route_lines_fg = folium.FeatureGroup(name="线路连线", overlay=True, control=True, show=False)
    route_lines_fg.add_to(m)

    # 站点图层
    fg_station = folium.FeatureGroup(name="发货点", show=show_stations)
    if show_stations:
        for st_code, (lat, lon) in dep_xy.items():
            # 仅标当前可视 batch 涉及的站点
            if st_code not in set(rdf["station"].astype(str)):
                continue
            folium.RegularPolygonMarker(
                location=(lat, lon),
                number_of_sides=4, radius=10,
                weight=2, fill=True, fill_opacity=0.9,
                color="#222", fill_color="#222",
                tooltip=f"站点 {st_code}",
                popup=folium.Popup(f"<b>发货点</b>: {st_code}", max_width=260),
            ).add_to(fg_station)
    fg_station.add_to(m)

    # 底图：展示该站点下的全部零售户（即使不在任何线路中）
    if show_all_station_customers and station_to_customers:
        # 需要展示的站点集合：按过滤后的线路中的站点，或单个 station_filter
        if station_filter is not None:
            stations_in_view = {str(station_filter)}
        else:
            stations_in_view = set(rdf["station"].astype(str).unique())
        fg_allcus = folium.FeatureGroup(name="全部零售户（按站点）", show=True)
        for st_code in stations_in_view:
            cust_set = station_to_customers.get(st_code, set())
            for c in cust_set:
                if c in cus_xy:
                    lat, lon = cus_xy[c]
                    folium.CircleMarker(
                        location=(lat, lon), radius=2,
                        weight=1, color="#555", fill=True, fill_opacity=0.5,
                        tooltip=f"客户 {c}（站点 {st_code}）",
                    ).add_to(fg_allcus)
        fg_allcus.add_to(m)

    # 为每条线路创建独立图层，方便开关
    for i, (_, row) in enumerate(rdf.iterrows()):
        color = palette[i % len(palette)]
        rid = str(row.get("route_id", f"R{i+1}"))
        rname = str(row.get("route_name", rid))
        st_code = str(row.get("station", ""))
        load_ratio = row.get("load_ratio")
        dist_km = row.get("distance_km")
        dur = row.get("duration_h")

        # 统计信息：零售户数与装载量（多来源回退）
        seq_codes = _ensure_seq_list(row.get("sequence"))
        num_stops = len(seq_codes)

        load_amount = row.get("load")
        if load_amount in (None, "", float("nan")):
            load_amount = row.get("TOTAL_AVG_ORDER_QTY")
        # 若仍为空，尝试从 df_cus 的 AVG_ORDER_QTY 汇总估算
        if (load_amount in (None, "", float("nan"))) and ("AVG_ORDER_QTY" in df_cus.columns):
            try:
                tmp_sum = (
                    df_cus[df_cus["BB_RETAIL_CUSTOMER_CODE"].astype(str).isin(seq_codes)]
                    ["AVG_ORDER_QTY"].astype(float).sum()
                )
                load_amount = float(round(tmp_sum, 2))
            except Exception:
                load_amount = None

        fg = folium.FeatureGroup(name=f"线路 {rname}", show=True)

        # 构造经纬度序列：仓 -> 客户 -> 仓(可选)
        coords = []
        # 起点：仓
        if st_code in dep_xy:
            coords.append(dep_xy[st_code])
        # 中间：客户
        for c in seq_codes:
            if c in cus_xy:
                coords.append(cus_xy[c])
        # 终点：回仓（可选）
        if show_return_to_depot and st_code in dep_xy and coords:
            coords.append(dep_xy[st_code])

        # 画折线
        if len(coords) >= 2:
            folium.PolyLine(
                locations=coords,
                color=color, weight=4, opacity=0.9,
                tooltip=f"{rname}",
                popup=folium.Popup(
                    f"<b>线路</b>: {rname}<br>"
                    f"<b>站点</b>: {st_code}<br>"
                    f"<b>车辆</b>: {row.get('vehicle_id', '')}<br>"
                    f"<b>零售户数</b>: {num_stops}<br>"
                    f"<b>装载量</b>: {load_amount if load_amount is not None else '-'}<br>"
                    f"<b>装载率</b>: {load_ratio}<br>"
                    f"<b>里程(km)</b>: {dist_km}<br>"
                    f"<b>时长</b>: {dur}",
                    max_width=360,
                ),
            ).add_to(route_lines_fg)

            # 在线路中点位置叠加数字标注（N与装载量）
            try:
                if coords:
                    mid_idx = len(coords) // 2
                    lat_m, lon_m = coords[mid_idx]
                    label_html = (
                        f"<div style='background: rgba(255,255,255,0.75);"
                        f" padding:2px 4px; border-radius:3px;"
                        f" font-size:10px; line-height:1.2; border:1px solid #999;'>"
                        f"N:{num_stops} / Load:{load_amount if load_amount is not None else '-'}"
                        f"</div>"
                    )
                    folium.map.Marker(
                        location=(lat_m, lon_m),
                        icon=folium.DivIcon(html=label_html),
                    ).add_to(fg)
            except Exception:
                pass

        # 客户点与序号标注
        if show_customers:
            for idx, c in enumerate(seq_codes, start=1):
                if c not in cus_xy:
                    continue
                lat, lon = cus_xy[c]
                folium.CircleMarker(
                    location=(lat, lon), radius=4,
                    weight=2, color=color, fill=True, fill_opacity=0.8,
                    tooltip=f"#{idx} 客户 {c}",
                    popup=folium.Popup(f"<b>序号</b>: {idx}<br><b>客户</b>: {c}", max_width=260),
                ).add_to(fg)
                # 在点上叠加序号文字
                folium.map.Marker(
                    location=(lat, lon),
                    icon=folium.DivIcon(html=f"<div style='font-size:10px;color:{color};font-weight:bold;'>{idx}</div>"),
                ).add_to(fg)

        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    if save_path:
        m.save(save_path)
    return m


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

    def folium_map(info, solution_list, loguruer):

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

    def _build_model_add_priority(COORDS, DEMANDS, CITIES, NAMES, veh_count, veh_capacity, loguruer, D=[],
                                  group_indices=[], conflict_indices=[]):

        MAX_CUS_VISITED = 1000
        MAX_ROUTE_DURATION = 100000
        SERVICE_DURATION = 1
        IS_PRIORITY_CUS = False
        IS_CONFLICT_CUS = False
        VEHICLE_SPEED = 1

        loguruer.info(f"模型构建中: ")
        m = Model()

        # 1 添加profiles
        loguruer.info(f"  构建模型 add_vehicle_type")
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
        loguruer.info(f"  构建模型 add_client")
        depot = m.add_depot(x=COORDS[0][0], y=COORDS[0][1], name='DC')
        clients = [
            m.add_client(x=COORDS[idx][0], y=COORDS[idx][1],
                         delivery=[DEMANDS[idx], 1], service_duration=SERVICE_DURATION,
                         name=NAMES[idx])
            for idx in range(1, len(COORDS))
        ]

        loguruer.info(f"  构建模型 add_edge")
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
                #             loguruer.info(
                #                 f"添加边({frm}, {to})，距离为10^6，车型为{pro}，因为{to_city}在{pro}的hate_cities中")
                #             m.add_edge(frm, to, distance=10 ** 6, profile=profile)
                #         if to_city in like_cities:
                #             loguruer.info(f"添加边({frm}, {to})，距离为0，车型为{pro}，因为{to_city}在{pro}的like_cities中")
                #             m.add_edge(frm, to, distance=0, profile=profile)
                #
                #     # VEHICLE_CITY_HATE_MAP
                #     # if in_city(CITIES[j]):
                #     #     m.add_edge(frm, to, distance=10 ** 6,profile = list(profiles.values())[-1])
                #     # if in_city_like(CITIES[j]):
                #     #     m.add_edge(frm, to, distance=0, profile=list(profiles.values())[-1])

        return m


    def _build_model_route(COORDS, DEMANDS, CITIES, NAMES, veh_count, veh_capacity, loguruer, D=[]):

        MAX_CUS_VISITED = 1000
        MAX_ROUTE_DURATION = 100000
        SERVICE_DURATION = 1
        VEHICLE_SPEED = 1

        loguruer.info(f"模型构建中: ")
        m = Model()

        # 1 添加profiles
        for count, capacity in zip(veh_count, veh_capacity):
            m.add_vehicle_type(count,
                               capacity=[capacity, MAX_CUS_VISITED],
                               max_duration=MAX_ROUTE_DURATION,
                               profile=profiles[f"pro_{capacity}"],
                               name=f"{capacity}")  # ,initial_load=[0, 0]

        # 2 添加仓库和客户
        loguruer.info(f"  构建模型 add_client")
        depot = m.add_depot(x=COORDS[0][0], y=COORDS[0][1], name='DC')
        clients = [
            m.add_client(x=COORDS[idx][0], y=COORDS[idx][1],
                         delivery=[DEMANDS[idx], 1], service_duration=SERVICE_DURATION,
                         name=NAMES[idx])
            for idx in range(1, len(COORDS))
        ]

        loguruer.info(f"  构建模型 add_edge")
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


    def _pyvrp_solve_MultipleCriteria(info, loguruer, runtime=1, iters=100, is_display=False):

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

        m = _build_model_add_priority(COORDS, DEMANDS, CITIES, NAMES, veh_count, veh_capacity, loguruer,
                                      D=D, group_indices=group_indices, conflict_indices=conflict_indices)

        res = m.solve(stop=MultipleCriteria([NoImprovement(iters), MaxRuntime(runtime)]), display=is_display)

        loguruer.info(f"\n{res}")

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
            remaining_info, all_solutions, all_solution_lists, loguruer
    ):

        # 2.0 抽取final_indices
        final_indices = remaining_info["CUS"]["ID"]
        final_info, remaining_info = extract_sub_info_from_indices(remaining_info, final_indices)

        # 2.1 抽取suggest_vehicle_combinations
        final_res, final_m = _pyvrp_solve_MultipleCriteria(final_info, loguruer, runtime=PYVRP_MAX_RUNTIME,
                                                           iters=PYVRP_MAX_NO_IMPROVE_ITERS, is_display=True)
        if final_res.is_feasible():  # 检查是否为可行解
            if final_res.best.distance() > 10 ** 6:  # 如果距离过大, 认为是不可行解
                loguruer.info(f"<UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK>")

            remaining_info = update_remaining_veh(final_res, final_m, remaining_info)

            [sol, sol_list] = extract_solution(final_res, final_m, final_info)

            all_solutions = all_solutions[:-1] + sol  # 合并solution
            all_solution_lists.extend(sol_list)

            # # to del start
            from tabulate import tabulate
            df_sol = cov_sol_list(sol_list)
            loguruer.info(f"\ndf_sol:\n{tabulate(df_sol, headers='keys', tablefmt='grid')}")
            # # to del end
        else:
            # 提示错误 时间不够
            loguruer.error(f"求解失败，未找到可行解。请检查输入数据或调整参数。")


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

    loguruer.info(f"\n--------------- Part3 算法求解开始 ---------------")

    # 1 数据预处理 提取pyvrp计算必要数据
    info = extract_cus_veh_info(gdf_ty_grid, D, G, depot_node = 7839518188)

    # 3 初始化剩余信息[后续全面适用remaining]
    remaining_info = info.copy()

    all_solutions = []
    all_solution_lists = []

    if 0:
        loguruer.info(f"分区1求解开始： 求解无法拼车客户组")
        all_solutions, all_solution_lists, remaining_info = (
            solve_isolated_customers(remaining_info, all_solutions, all_solution_lists, loguruer))

    if 0:
        loguruer.info(f"分区2求解开始： 同city组 改为同发货点组 或 同小区组")
        for cities in splitCityArea:
            loguruer.info(f"当前分区 cities: {cities}")
            all_solutions, all_solution_lists, remaining_info = (
                solve_cityarea_customers(remaining_info, all_solutions, all_solution_lists, cities, loguruer)
            )

    if 1:
        loguruer.info(f"分区N求解开始： 求解最终剩余客户组")
        all_solutions, all_solution_lists, remaining_info = (
            solve_final_customers(remaining_info, all_solutions, all_solution_lists, loguruer))

    if 1:
        folium_map(info, all_solution_lists, loguruer)

    return all_solutions, all_solution_lists


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


def get_customer_avg_order_qty(db_n32, db_n28, db_n26):
    """
    1 从db_n32抽取零售户BB_RETAIL_CUSTOMER_CODE的平均订单需求量MD03_ORDER_TOTAL_QTY，
    并关联到db_n28，形成新的df_n28表，增加AVG_ORDER_QTY列。
    2 增加发货点的经纬度MD03_SHIPMENT_STATION_LON MD03_SHIPMENT_STATION_LAT
    """
    df_cus_demand = db_n32[["BB_RETAIL_CUSTOMER_CODE", "MD03_ORDER_TOTAL_QTY"]].groupby("BB_RETAIL_CUSTOMER_CODE").agg(
        AVG_ORDER_QTY=("MD03_ORDER_TOTAL_QTY", "mean")
    ).reset_index()
    print(df_cus_demand["AVG_ORDER_QTY"].dtype)
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

# def enrich_customer_vehicle_info(db_n32, df_n28, db_n27, db_n26):
#     """
#     从db_n32抽取客户车辆和线路信息，合并到df_n28，并将MD03_DIST_STATION_CODE移至最后一列。
#     """
#     # part1: 增加客户车辆和线路信息列
#     df_cus_vehicle = db_n32[
#         ["BB_RETAIL_CUSTOMER_CODE", "CD_LOGT_COM_DEY_VEHS_CODE", "MD03_DIST_LINE_CODE", "MD03_DELIVER_DELIVER_NO"]
#     ].groupby("BB_RETAIL_CUSTOMER_CODE").agg(
#         VEHICLE_CODES=("CD_LOGT_COM_DEY_VEHS_CODE", lambda x: list(x.unique())),
#         LINE_CODES=("MD03_DIST_LINE_CODE", lambda x: list(x.unique())),
#         LINE_SEQ=("MD03_DELIVER_DELIVER_NO", lambda x: list(x.unique())),
#     ).reset_index()
#
#     df_cus_vehicle["VEHICLE_CODES"] = df_cus_vehicle["VEHICLE_CODES"].apply(lambda x: ",".join(x) if isinstance(x, list) else "")
#     df_cus_vehicle["LINE_CODES"] = df_cus_vehicle["LINE_CODES"].apply(lambda x: ",".join(x) if isinstance(x, list) else "")
#     df_cus_vehicle["LINE_SEQ"] = df_cus_vehicle["LINE_SEQ"].apply(lambda x: ",".join(map(str, x)) if isinstance(x, list) else "")
#
#
#     # df_cus_vehicle从db_n27内抽取列MD03_MAX_LODING_CARTON_QTY到df_cus_vehicle内，形成新的df_cus_vehicle表
#     df_max_carton = db_n27[["CD_LOGT_COM_DEY_VEHS_CODE", "MD03_MAX_LODING_CARTON_QTY"]].drop_duplicates()
#     df_cus_vehicle = pd.merge(df_cus_vehicle, df_max_carton, how="left", left_on="VEHICLE_CODES", right_on="CD_LOGT_COM_DEY_VEHS_CODE")
#     df_cus_vehicle.rename(columns={"MD03_MAX_LODING_CARTON_QTY": "VEHICLE_MAX_CARTON_QTY"}, inplace=True)
#     df_cus_vehicle.drop(columns=["CD_LOGT_COM_DEY_VEHS_CODE"], inplace=True)
#
#
#     df_n28 = pd.merge(df_n28, df_cus_vehicle, how="left", left_on="BB_RETAIL_CUSTOMER_CODE", right_on="BB_RETAIL_CUSTOMER_CODE")
#     cols = list(df_n28.columns)
#     cols.append(cols.pop(cols.index("MD03_DIST_STATION_CODE")))
#     df_n28 = df_n28[cols]
#
#     # part2: 增加发货点经纬度, 精简列
#     # df_n28提取列MD03_DIST_STATION_CODE VEHICLE_CODES VEHICLE_MAX_CARTON_QTY BB_RETAIL_CUSTOMER_CODE MD03_RETAIL_CUST_LON MD03_RETAIL_CUST_LAT AVG_ORDER_QTY LINE_CODES LINE_SEQ
#     df_n28 = df_n28[["MD03_DIST_STATION_CODE", "VEHICLE_CODES", "VEHICLE_MAX_CARTON_QTY", "BB_RETAIL_CUSTOMER_CODE", "MD03_RETAIL_CUST_LON", "MD03_RETAIL_CUST_LAT", "AVG_ORDER_QTY", "LINE_CODES", "LINE_SEQ", "township", "district", "city"]].copy()
#     # df_n28增加发货点经纬度: 从df_n26表提取增加MD03_DIST_STATION_CODE对应的经纬度
#     station_coords = db_n26[["MD03_SHIPMENT_STATION_CODE", "MD03_SHIPMENT_STATION_LON", "MD03_SHIPMENT_STATION_LAT"]].drop_duplicates()
#     station_coords = station_coords.set_index("MD03_SHIPMENT_STATION_CODE").astype({"MD03_SHIPMENT_STATION_LON": float, "MD03_SHIPMENT_STATION_LAT": float})
#     df_n28 = pd.merge(df_n28, station_coords, how="left", left_on="MD03_DIST_STATION_CODE", right_on="MD03_SHIPMENT_STATION_CODE")
#
#     return df_n28


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

# === Balanced partition into n routes ===
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

def plan_distribution_routes(
    db_n19, db_n26, db_n27, db_n28, db_n31, db_n32, db_n33, db_n34, n34_id
):
    # 输入logger检查 (统计分析供应商 todo）
    from tabulate import tabulate
    loguruer.info(f" ALG_0: n34_id is {n34_id}")
    loguruer.info(f"\ndb_n34:\n{tabulate(db_n34, headers='keys', tablefmt='grid')}")
    loguruer.info(f"\ndb_n19:\n{tabulate(db_n19, headers='keys', tablefmt='grid')}")
    loguruer.info(f"\ndb_n31:\n{tabulate(db_n31, headers='keys', tablefmt='grid')}")
    loguruer.info(f"\ndb_n33:\n{tabulate(db_n33, headers='keys', tablefmt='grid')}")
    loguruer.info(f"\ndb_n28:\n{tabulate(db_n28.head(10), headers='keys', tablefmt='grid')}")
    add_n35_log_info(n34_id, "ALG_0", f" ALG_0: n34_id is {n34_id}")

    # 1.1 增加地址解析（行政区划）
    loguruer.info(f" ALG_0: 路线进入plan_distribution_routes函数")
    add_n35_log_info(n34_id, "ALG_0", f"路线进入plan_distribution_routes函数")
    db_n28 = parse_md03_address_component(db_n28)
    loguruer.info(f" ALG_0: 路线增加address解析成功")
    add_n35_log_info(n34_id, "ALG_0", f"路线增加address解析成功")
    # 1.2 增加人工线路
    db_n28 = parse_line_info(db_n32, db_n28)
    loguruer.info(f" ALG_0: 路线增加line解析成功")
    add_n35_log_info(n34_id, "ALG_0", f"路线增加line解析成功")
    # 1.3 增加客户需求和发货点经纬度,填充零售户经纬度
    db_n28 = get_customer_avg_order_qty(db_n32, db_n28, db_n26)
    loguruer.info(f" ALG_0: 路线增加增加客户需求和处理latlon解析成功")
    add_n35_log_info(n34_id, "ALG_0", f"路线增加latlon解析成功")
    db_n28 = db_n28[[
        "MD03_DIST_STATION_CODE", "MD03_SHIPMENT_STATION_LON", "MD03_SHIPMENT_STATION_LAT",
        "BB_RETAIL_CUSTOMER_CODE", "MD03_RETAIL_CUST_LON", "MD03_RETAIL_CUST_LAT",
        "AVG_ORDER_QTY", "LINE_CODES", "LINE_SEQ", "township", "district", "city"
    ]].copy()
    loguruer.info(f"\ndb_n28:\n{tabulate(db_n28.head(10), headers='keys', tablefmt='grid')}")

    # 2.1 发货点选择算法: 固定/动态/混合
    algo_code = db_n19["MD03_DIST_PLAN_ALGO_CODE"].iloc[0] if len(db_n19) > 0 else None
    # 如果algo_code为None, 则抛出异常 且logger记录
    if algo_code not in {"VRP1", "VRP2", "VRP3"}:
        loguruer.error(f" ALG_0: 未指定发货点选择算法")
        add_n35_log_info(n34_id, "ALG_1", f"未指定发货点选择算法")
        raise ValueError(f"未知的发货点选择算法: {algo_code}")
    else:
        loguruer.info(f" ALG_1: 发货点选择算法 is {algo_code}")
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
            loguruer.error(f" ALG_2: 未指定合理的线路数参数")
            add_n35_log_info(n34_id, "ALG_2", f"未指定合理的线路数参数")
            raise ValueError(f"未知的线路数参数: {param_nRoutes}")
        else:
            loguruer.info(f" ALG_2: 线路数参数 is {param_nRoutes}")
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
        loguruer.info(f" ALG_2: 平衡方式参数 is {balance_type}")

        # 动态平衡
        USE_BALANCED_PARTITION = 1
        try:
            routes_df_balanced = repartition_customers_to_n_routes(
                db_n28, db_n26, param_nRoutes, balance=balance_type,
                group_by_station=True, speed_kmh=50.0
            )
            loguruer.info(f"[Balanced] Built {len(routes_df_balanced)} routes with balance='{balance_type}', nRoutes={param_nRoutes}")
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
            loguruer.info(f" ALG_3: 动态路线规划算法执行成功")
            add_n35_log_info(n34_id, "ALG_3", f"动态路线规划算法执行成功")

        except Exception as e:
            loguruer.error(f"[Balanced] 生成均衡线路失败: {e}")
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

        loguruer.info(f" ALG_3: 固定路线规划算法执行成功")
        add_n35_log_info(n34_id, "ALG_3", f"固定路线规划算法执行成功")

    # 3. 输出结果整理
    # loguruer输出routes_df前10行（如有）
    loguruer.info(f"\nroutes_df:\n{tabulate(routes_df.head(10), headers='keys', tablefmt='grid') if routes_df is not None else 'No routes generated.'}")
    if routes_df is not None:
        # 删除车容量字段
        if "vehicle_capacity" in routes_df.columns:
            routes_df.drop(columns=["vehicle_capacity"], inplace=True)
        # 装载率为null时默认-1
        routes_df["load_ratio"] = pd.to_numeric(routes_df["load_ratio"], errors="coerce").fillna(-1.0)
        routes_dict = {row["route_id"]: row.to_dict() for _, row in routes_df.iterrows()}
    else:
        routes_dict = {}

    loguruer.info(f" ALG_4: 路线规划算法执行结束")
    add_n35_log_info(n34_id, "ALG_4", f"路线规划算法执行结束")
    return routes_df, routes_dict


# 新增个main函数，方便直接运行调试
if __name__ == "__main__":
    from db.line_data_service import pre_deal_line, out_n38_n39
    import datetime as _dt

    # === Cache options ===
    USE_LOCAL_CACHE = False   # True: load from local parquet; False: fetch from DB then save
    CACHE_DIR = 'cache_vrp'   # folder to store parquet files

    # === Input parameters 全局 动态 只能模式2 ===
    n34_id = "29374109947317925"  # 获取各种方案编码 md03DistPlanLogId
    plan_type = "1"  # 1 全局规划 2局部规划 md03PlanType 1PASS
    plan_mode = "2"  # 1 从线路规划结果来 2 从模式规划来 PASS md03PlanMode
    n28_station_codes_str = "1401000101"  # 全局规划时有效 md03ShipmentStationCode "1402000101,1402000102"
    n38_line_ids_str = ""  # 局部规划时有效 md03DistPlanLineId # n36_range_ids_str = "101510486268551704"  # n36_range_ids_str = "101510486268541377,29374109947317927" 101510486268541377 101510486268551704

    """ 前置方法 """
    if USE_LOCAL_CACHE:
        tables = load_tables_local(n34_id, CACHE_DIR)
        db_n34 = tables.get('db_n34')
        db_n19 = tables.get('db_n19')
        db_n26 = tables.get('db_n26')
        db_n27 = tables.get('db_n27')
        db_n28 = tables.get('db_n28')
        db_n31 = tables.get('db_n31')
        db_n32 = tables.get('db_n32')
        db_n32_group = tables.get('db_n32_group')
        db_n33 = tables.get('db_n33')
    else:
        status, message, db_tables = pre_deal_line(n34_id, plan_type, plan_mode, n28_station_codes_str,
                                                   n38_line_ids_str)
        print("status:", status)
        db_n34, db_n19, db_n20, db_n26, db_n27, db_n28, db_n31, db_n32, db_n32_group, db_n33 = db_tables.values()
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
                'db_n32_group': db_n32_group,
                'db_n33': db_n33,
            },
            CACHE_DIR,
        )

    # 调用函数
    routes_df, routes_dict = plan_distribution_routes(
        db_n19, db_n26, db_n27, db_n28, db_n31, db_n32, db_n33, db_n34, n34_id
    )

    # === 可视化 1402000102 的两条示例线路 ===
    try:
        district_filter = "小店区"  # 这里改成你要展示的行政区名称
        db_n28_filtered = db_n28[db_n28["district"] == district_filter].copy()
        _ = visualize_route_plan_folium(
            routes_df,
            df_cus=db_n28_filtered.rename(columns={"MD03_DIST_STATION_CODE": "MD03_DIST_STATION_CODE"}),
            df_depot=db_n26,
            station_filter=None,  # 也可以指定站点
            tiles="OpenStreetMap",
            show_all_station_customers=False,
            save_path=f"route_plan_{isFix}_{district_filter}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{n34_id}.html",
            zoom_start=12,
        )
        print("已生成可视化: route_plan_1402000102.html")

    except Exception as e:
        print(f"可视化失败: {e}")

    out_n38_n39(n34_id, plan_type, n38_line_ids_str, routes_dict, db_n32_group)
    print("Done route!")