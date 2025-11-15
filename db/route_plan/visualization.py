"""Utility functions for the route planning visualization module."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

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
