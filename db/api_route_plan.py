"""Public API surface for route planning utilities."""

from __future__ import annotations

from .route_plan.cache import (
    CacheManifest,
    load_tables_local,
    load_tables_or_str_local,
    save_tables_local,
    save_tables_or_str_local,
)
from .route_plan.geo import (
    attach_osm_coords,
    build_distance_matrix_from_graph,
    build_or_load_distance_matrix,
    load_and_prepare_taiyuan_grid,
    read_input,
)
from .route_plan.heuristics import (
    _allocate_counts_by_weight,
    _euclid,
    _haversine_km,
    _haversine_km_fast,
    _scale_to_uint,
    _to_int_coords,
    _zorder_key,
    add_route_distances,
    plan_distribution_routes01,
    repartition_customers_to_n_routes,
    reorder_lines_by_nearest,
    reorder_lines_by_pyvrp,
    solve_vrp,
)
from .route_plan.logger import LOGGER, configure_logger
from .route_plan.preparation import prepare_vrp_inputs
from .route_plan.service import (
    build_route_summary,
    build_route_veh_summary,
    get_customer_avg_order_qty,
    parse_line_info,
    parse_md03_address_component,
    parse_veh_info,
    plan_distribution_routes,
)
from .route_plan.visualization import (
    _ensure_seq_list,
    visualize_route_plan_folium,
    visualize_route_plan_folium01,
    visualize_route_plan_folium02,
    visualize_route_plan_folium03,
    visualize_route_plan_folium_with_osm,
)

__all__ = [
    "CacheManifest",
    "LOGGER",
    "configure_logger",
    "attach_osm_coords",
    "build_distance_matrix_from_graph",
    "build_or_load_distance_matrix",
    "load_and_prepare_taiyuan_grid",
    "read_input",
    "prepare_vrp_inputs",
    "plan_distribution_routes01",
    "plan_distribution_routes",
    "repartition_customers_to_n_routes",
    "reorder_lines_by_nearest",
    "reorder_lines_by_pyvrp",
    "add_route_distances",
    "solve_vrp",
    "_allocate_counts_by_weight",
    "_scale_to_uint",
    "_zorder_key",
    "_haversine_km_fast",
    "_haversine_km",
    "_to_int_coords",
    "_euclid",
    "save_tables_local",
    "save_tables_or_str_local",
    "load_tables_local",
    "load_tables_or_str_local",
    "_ensure_seq_list",
    "visualize_route_plan_folium",
    "visualize_route_plan_folium01",
    "visualize_route_plan_folium02",
    "visualize_route_plan_folium03",
    "visualize_route_plan_folium_with_osm",
    "parse_md03_address_component",
    "get_customer_avg_order_qty",
    "parse_line_info",
    "parse_veh_info",
    "build_route_summary",
    "build_route_veh_summary",
]
