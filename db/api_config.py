# ---线路约束参数配置 solver_pyvrp.py---
MAX_CUS_VISITED = 6      # 默认6 : 最大访问客户数（每辆车最多访问7个客户）[一般无需调整]
PYVRP_MAX_RUNTIME = 3
PYVRP_MAX_NO_IMPROVE_ITERS = 5000
DIST_MATRIX_TYPE = 'GaoDe'  # 用OSM的路网距离  Manhattan GaoDe

EXTRACT_INFO_TYPE = 'Node'  # 'Node' or 'Osm' or 'Grid' 抽取不同类型  大概率基于Node或Osm[缩小维度,]

VEH_CAPACITY = 5500