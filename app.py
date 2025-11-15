import asyncio
import time
import traceback

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.status import HTTP_504_GATEWAY_TIMEOUT

from db.api_dist_plan import parse_md03_address_component, prepare_vrp_inputs, plan_distribution_regions, calculate_depot_capacity
from db.api_route_plan import plan_distribution_routes, load_tables_local, save_tables_local
from db.line_data_service import pre_deal_line, out_n38_n39
from db.range_data_service import pre_deal_range, out_n36_n37
from db.log_service import add_n35_log_info, del_n35_log_info, set_n34_fault

# 配置日志
from industry_nb.util.logger import logger

# new
app = FastAPI()
# 允许所有来源的跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

expired_day = 3
REQUEST_TIMEOUT = 35


# 设置超时
@app.middleware('http')
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        return JSONResponse({'code': f'-1', 'msg': f'超时,当前超时值' + str(REQUEST_TIMEOUT) + '秒'},
                            status_code=HTTP_504_GATEWAY_TIMEOUT)


"""范围规划"""
@app.post("/callRange")
async def call_range(background_tasks: BackgroundTasks, request: Request):
    json_data = await request.json()
    n34_id = json_data["md03DistPlanLogId"]
    plan_type = json_data["md03PlanType"]  # 1全局规划 2局部规划
    n36_range_ids_str = json_data["md03DistPlanRangeId"]

    background_tasks.add_task(range_process_task, n34_id, plan_type, n36_range_ids_str)

    return {"status": "success"}

"""路线规划"""
@app.post("/callLine")
async def call_line(background_tasks: BackgroundTasks, request: Request):
    json_data = await request.json()
    n34_id = json_data["md03DistPlanLogId"]
    plan_type = json_data["md03PlanType"]  # 1全局规划 2局部规划
    plan_mode = json_data["md03PlanMode"]  # 规划模式 1基于范围规划方案进行线路规划 2基于现有配送站点方案【28表】进行线路规划
    n28_station_codes_str = json_data["md03ShipmentStationCode"]
    n38_line_ids_str = json_data["md03DistPlanLineId"]

    background_tasks.add_task(line_process_task, n34_id, plan_type, plan_mode, n28_station_codes_str, n38_line_ids_str)

    return {"status": "success"}


def range_process_task(n34_id, plan_type, n36_range_ids_str):
    try:
        add_n35_log_info(n34_id, "ST_0", f"范围{'全局' if plan_type == '1' else '局部'}规划开始执行")
        start_time = time.time()
        """ 前置方法 """
        status, message, db_tables = pre_deal_range(n34_id, plan_type, n36_range_ids_str)
        if not status:
            raise Exception(message)
        """算法调用"""
        db_n34, db_n19, db_n20, db_n26, db_n27, db_n28, db_n31, db_n32, db_n33 = db_tables.values()
        # 解析 MD03_ADDRESS_COMPONENT 字段，提取 city / district / township
        db_n28 = parse_md03_address_component(db_n28)
        df_cus, df_depot, df_veh, _ = prepare_vrp_inputs(n34_id, db_n26, db_n28, db_n27)

        # 运力计算：df_veh可能为空 无法元素你
        if df_veh.empty:
            print("警告：车辆信息表为空，跳过运力计算。")
            cap = None
        else: # 如不空仅当df_veh对每个应发货点均有数据时才计算，否则不传cap参数
            dep_codes_with_veh = set(df_veh['MD03_SHIPMENT_STATION_CODE'].astype(str).unique())
            dep_codes_in_depot = set(df_depot['MD03_SHIPMENT_STATION_CODE'].astype(str).unique())
            if not dep_codes_in_depot.issubset(dep_codes_with_veh):
                print(
                    f"警告：并非所有发货点均有车辆数据，跳过运力计算。发货点数={len(dep_codes_in_depot)}，有车辆数据的发货点数={len(dep_codes_with_veh)}")
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
                assign_strategy=assign_strategy,  # district_capacity  township_capacity
                depot_capacity=cap,
            )
            # html_map = visualize_dist_plan_folium(dict_dist_plan, df_cus, df_depot,
            #                                       html_path=f"dist_plan_{assign_strategy}_{n34_id}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            #                                       draw_lines=False)
        else:
            dict_dist_plan = {}
            add_n35_log_info(n34_id, "ALG_ERROR", f"车辆信息不足，无法进行CAP求解。")


        """ 后置方法输出 """
        out_n36_n37(n34_id, plan_type, dict_dist_plan, n36_range_ids_str)
        add_n35_log_info(n34_id, "ST_1", f"范围{'全局' if plan_type == '1' else '局部'}规划结束执行-总运行时间为{time.time() - start_time:.2f}秒")
        print("Done range!")

    except Exception as e:
        del_n35_log_info(n34_id)
        set_n34_fault(n34_id)
        add_n35_log_info(n34_id, "ERROR", f"{e}")
        logger.error(f"Unhandled exception in call_range: {e}")
        logger.error(traceback.format_exc())  # 这行会打印完整的错误堆栈
        raise


def line_process_task(n34_id, plan_type, plan_mode,n28_station_codes_str, n38_line_ids_str):
    try:
        add_n35_log_info(n34_id, "ST_0", f"路线{'全局' if plan_type == '1' else '局部'}规划开始执行")
        start_time = time.time()
        """ 前置方法 """
        status, message, db_tables = pre_deal_line(n34_id, plan_type, plan_mode,n28_station_codes_str, n38_line_ids_str)
        if not status:
            logger.error(f"Unhandled exception in call_range: {message}")
            add_n35_log_info(n34_id, "ERROR", f"status is false")
            raise Exception(message)
        """算法调用"""
        db_n34, db_n19, db_n20, db_n26, db_n27, db_n28, db_n31, db_n32, db_n32_group, db_n33 = db_tables.values()

        # # 运行 plan_distribution_regions
        # df_cus, df_depot, df_veh, _ = prepare_vrp_inputs(n34_id, db_n26, db_n28,
        #                                                  db_n27)  # 36和37呢？ 28可以 38或39  ？36和37应该和26和28类似，是输入表
        routes_df, routes_dict = plan_distribution_routes(
            db_n19, db_n26, db_n27, db_n28, db_n31, db_n32, db_n33, db_n34, n34_id
        )
        add_n35_log_info(n34_id, "ALG_INFO", f"路线规划结果，共生成线路数{len(routes_dict)}")

        """ 后置方法输出 """
        out_n38_n39(n34_id, plan_type, n38_line_ids_str, routes_dict, db_n32_group)
        add_n35_log_info(n34_id, "ST_1", f"路线{'全局' if plan_type == '1' else '局部'}规划结束执行-总运行时间为{time.time() - start_time:.2f}秒")
        print("Done route!")

    except Exception as e:
        del_n35_log_info(n34_id)
        logger.error(f"Unhandled exception in call_range: {e}")
        set_n34_fault(n34_id)
        add_n35_log_info(n34_id, "ERROR", f"{e}")
        logger.error(traceback.format_exc())  # 这行会打印完整的错误堆栈
        raise

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8140)

def test_line_app():
    # === Input parameters 全局 固定 ===
    n34_id = "29374109947317923"
    plan_type = '1'  # 1全局规划 2局部规划
    plan_mode = '2'  # 规划模式 1基于范围规划方案进行线路规划 2基于现有配送站点方案【28表】进行线路规划
    n28_station_codes_str = "1402000101,1402000102" #'S20230822001,S20230822002'
    n38_line_ids_str = '' #'L20230822001,L20230822002'

    # === Input parameters 全局 动态 只能模式2 ===
    n34_id = "29374109947317925"  # 获取各种方案编码 md03DistPlanLogId
    plan_type = "1"  # 1 全局规划 2局部规划 md03PlanType 1PASS
    plan_mode = "2"  # 1 从线路规划结果来 2 从模式规划来 PASS md03PlanMode
    n28_station_codes_str = "1401000101"  # 全局规划时有效 md03ShipmentStationCode "1402000101,1402000102"
    n38_line_ids_str = ""  # 局部规划时有效 md03DistPlanLineId # n36_range_ids_str = "101510486268551704"  # n36_range_ids_str = "101510486268541377,29374109947317927" 101510486268541377 101510486268551704

    line_process_task(n34_id, plan_type, plan_mode, n28_station_codes_str, n38_line_ids_str)

def test_line():
    # === Input parameters 全局 固定 ===
    n34_id = "29374109947317923"
    plan_type = '1'  # 1全局规划 2局部规划
    plan_mode = '2'  # 规划模式 1基于范围规划方案进行线路规划 2基于现有配送站点方案【28表】进行线路规划
    n28_station_codes_str = "1402000102" # "1402000101,1402000102"
    n38_line_ids_str = '' #'L20230822001,L20230822002'

    # === Input parameters 全局 动态 只能模式2 [大同]===
    n34_id = "29374109947317925"  # 获取各种方案编码 md03DistPlanLogId
    plan_type = "1"  # 1 全局规划 2局部规划 md03PlanType 1PASS
    plan_mode = "2"  # 1 从线路规划结果来 2 从模式规划来 PASS md03PlanMode
    n28_station_codes_str = "1401000101"  # 全局规划时有效 md03ShipmentStationCode "1402000101,1402000102"
    n38_line_ids_str = ""  # 局部规划时有效 md03DistPlanLineId # n36_range_ids_str = "101510486268551704"  # n36_range_ids_str = "101510486268541377,29374109947317927" 101510486268541377 101510486268551704

    USE_LOCAL_CACHE = False # True False
    CACHE_DIR = 'cache_vrp_2'

    if USE_LOCAL_CACHE:
        tables = load_tables_local(n34_id, CACHE_DIR)
        db_n34 = tables.get('db_n34')
        db_n20 = tables.get('db_n20')
        db_n19 = tables.get('db_n19')
        db_n26 = tables.get('db_n26')
        db_n27 = tables.get('db_n27')
        db_n28 = tables.get('db_n28')
        db_n31 = tables.get('db_n31')
        db_n32 = tables.get('db_n32')
        db_n33 = tables.get('db_n33')
        db_n32_group = tables.get('db_n32_group')
    else:
        status, message, db_tables = pre_deal_line(n34_id, plan_type, plan_mode, n28_station_codes_str,
                                                   n38_line_ids_str)
        print("status:", status)
        if status:
            db_n34, db_n19, db_n20, db_n26, db_n27, db_n28, db_n31, db_n32, db_n32_group, db_n33 = db_tables.values()
            save_tables_local(
                n34_id,
                {
                    'db_n34': db_n34,
                    'db_n19': db_n19,
                    'db_n20': db_n20,
                    'db_n26': db_n26,
                    'db_n27': db_n27,
                    'db_n28': db_n28,
                    'db_n31': db_n31,
                    'db_n32': db_n32,
                    'db_n33': db_n33,
                    'db_n32_group': db_n32_group,
                },
                CACHE_DIR,
            )
        else:
            print("Error in pre_deal_line:", message)
            return

    routes_df, routes_dict = plan_distribution_routes(
        db_n19, db_n26, db_n27, db_n28, db_n31, db_n32, db_n33, db_n34, n34_id
    )
    print("routes_df:", routes_df)
    print("Done route test ! ")


    # """ 后置方法输出 """
    out_n38_n39(n34_id, plan_type, n38_line_ids_str, routes_dict, db_n32_group)
    # add_n35_log_info(n34_id, "ST_1",
    #                  f"路线{'全局' if plan_type == '1' else '局部'}规划结束执行-总运行时间为{time.time() - start_time:.2f}秒")
    print("Done route test with output!")



if __name__ == "__main__":
    # main()
    test_line()
    # test_line_app()