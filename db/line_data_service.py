from datetime import datetime
from decimal import Decimal
from functools import lru_cache
from typing import Optional, Tuple

import pandas as pd

from db.db_conn import get_common_data, db_execute_batch, get_db, get_short_uuid, db_execute_callback, db_update
from db.log_service import add_n35_log_info

""" 线路规划 --plan_type 规划类型 1全局规划、2局部规划 
 --plan_mode 规划模式 1基于范围规方案进行线路规划 2基于现有配送站点方案【28表】进行线路规划  
 --station_code 规划发货点编码 全局规划有值
 --plan_range_id n38主键id，局部规划有值"""


def pre_deal_line(n34_id: str, plan_type: str, plan_mode: str, station_code: str, plan_range_id: str) -> Tuple[bool, str, Optional[dict]]:
    """ 插入日志 """
    add_n35_log_info(n34_id, "PDR_0", f"前置业务数据处理-开始-开始获取历史订单、区域规划结果、零售店坐标等相关数据..")

    """查询出所有需要的表数据"""

    """ n34 """
    db_n34 = get_n34_by_id(n34_id)
    n34_data = db_n34.iloc[0]

    if n34_data["MD03_DIST_PLAN_STATE"] == '2':
        return False, f"n34_id={n34_id}, 此数据路线规划状态已完成，请重新输入！", None

    dist_center_code = n34_data['MD03_DIST_CENTER_CODE']  # 配送中心编码
    dist_plan_scheme_code = n34_data['MD03_DIST_PLAN_SCHEME_CODE']  # 配送规划方案编码
    md03_dist_plan_type = n34_data['MD03_DIST_PLAN_TYPE']  # 配送规划类型 1、范围规划，2、线路规划

    """ n19 配送规划类型=(1范围规划、2线路规划)"""
    db_n19 = get_n19_by_play_type_and_center_code(md03_dist_plan_type, dist_center_code)

    db_n20 = None
    if db_n19 is not None and not db_n19.empty:
        alog_ids = []
        for item_n19 in db_n19.itertuples():
            alog_ids.append(item_n19.MD03_DIST_PLAN_ALGO_ID)
        db_n20 = get_n20_by_alog_ids(alog_ids)

    """ n28 """
    customer_codes = []
    station_codes = []
    if plan_type == '1':
        """全局规划"""
        station_codes = station_code.split(",")

        add_n35_log_info(n34_id, "PDR_1", "前置业务数据处理-路线全局规范数据正在处理中...")

        """参数3：规划模式 1基于范围规划方案进行线路规划 2基于现有配送站点方案【28表】进行线路规划"""
        if plan_mode == '1':
            """参数4：规划发货点编码(多个编码则以逗号隔开) 从n36、n37  + 配送中心编码、配送方案编码"""
            db_n36 = get_n36_by_center_code_and_scheme_code_and_station_codes(dist_center_code, dist_plan_scheme_code, station_codes)
            if db_n36 is not None and not db_n36.empty:
                n36_ids_str = ", ".join(f"'{item_n36.MD03_DIST_PLAN_RANGE_ID}'" for item_n36 in db_n36.itertuples())
                db_n37 = get_n37_by_ids(n36_ids_str)
                for item_n37 in db_n37.itertuples():
                    customer_codes.append(item_n37.BB_RETAIL_CUSTOMER_CODE)
            db_n28 = get_n28_by_center_code_and_scheme_code_and_customer_codes(dist_center_code, dist_plan_scheme_code, customer_codes)
        else:
            """参数4：规划发货点编码(多个编码则以逗号隔开) 从n28"""
            db_n28 = get_n28_by_center_code_and_scheme_code_and_station_codes(dist_center_code, dist_plan_scheme_code, station_codes)
            for item_n28 in db_n28.itertuples():
                customer_codes.append(item_n28.BB_RETAIL_CUSTOMER_CODE)
    else:
        """局部规划"""
        add_n35_log_info(n34_id, "PDR_1", "前置业务数据处理-路线局部规范正在处理中...")
        plan_range_ids = plan_range_id.split(",")
        # 取出n38 发货站点编码list
        db_n38 = get_n38_by_ids(plan_range_ids)
        for item_n38 in db_n38.itertuples():
            station_codes.append(item_n38.MD03_SHIPMENT_STATION_CODE)

        # 取出n39 零售户编码list
        db_n39 = get_n39_by_ids(plan_range_ids)
        for item_n39 in db_n39.itertuples():
            customer_codes.append(item_n39.BB_RETAIL_CUSTOMER_CODE)

        db_n28 = get_n28_by_center_code_and_scheme_code_and_customer_codes(dist_center_code, dist_plan_scheme_code, customer_codes)

    if db_n28 is None or db_n28.empty:
        return False, f"dist_center_code={dist_center_code},dist_plan_scheme_code={dist_plan_scheme_code},customer_codes={customer_codes}, 未获取到n28数据！", None

    """ n26 """
    db_n26 = get_n26_by_center_code_and_scheme_code_and_station_codes(dist_center_code, dist_plan_scheme_code, station_codes)
    if db_n26 is None or db_n26.empty:
        return False, f"dist_center_code={dist_center_code},dist_plan_scheme_code={dist_plan_scheme_code},station_codes={station_codes}, 未获取到n26数据！", None

    """ n27 """
    db_n27 = get_n27_by_center_code_and_scheme_code_and_station_codes(dist_center_code, dist_plan_scheme_code, station_codes)

    """ n31 """
    db_n31 = get_n31_play_type_and_by_play_type_and_center_code(md03_dist_plan_type, dist_center_code, dist_plan_scheme_code)

    """ n32 """
    db_n32 = get_n32_by_center_code_and_scheme_code_and_customer_code_and_station_codes(dist_center_code, dist_plan_scheme_code, customer_codes, station_codes)
    db_n32_group = get_n32_by_center_code_and_scheme_code_and_customer_code_and_station_codes_group(dist_center_code, dist_plan_scheme_code, customer_codes, station_codes)

    """ n33 """
    db_n33 = get_n33_by_center_code_and_scheme_code(dist_center_code, dist_plan_scheme_code, md03_dist_plan_type)

    data_dict = {
        'db_n34': db_n34,
        'db_n19': db_n19,
        'db_n20': db_n20,
        'db_n26': db_n26,
        'db_n27': db_n27,
        'db_n28': db_n28,
        'db_n31': db_n31,
        'db_n32': db_n32,
        'db_n32_group': db_n32_group,
        'db_n33': db_n33
    }
    add_n35_log_info(n34_id, "PDR_2", "前置业务数据处理-结束-历史订单、区域规划结果、零售店坐标等相关数据提取成功")
    return True, "数据取出成功", data_dict


""" 路线规划输出"""
def out_n38_n39(n34_id: str, plan_type: str, plan_range_id: str, data: {}, db_n32_group):
    add_n35_log_info(n34_id, "OR_0", "后置业务数据处理-开始-对算法数据集进行零售户等数据提取处理")
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

    """1.查出n34的数据"""
    db_n34 = get_n34_by_id(n34_id)
    n34_data = db_n34.iloc[0]
    md03_region_logi_center_code = n34_data["MD03_REGION_LOGI_CENTER_CODE"]  # MD03_REGION_LOGI_CENTER_CODE 区域物流中心编码
    md03_dist_center_code = n34_data["MD03_DIST_CENTER_CODE"]  # MD03_DIST_CENTER_CODE 配送中心编码
    md03_dist_plan_scheme_code = n34_data["MD03_DIST_PLAN_SCHEME_CODE"]  # MD03_DIST_PLAN_SCHEME_CODE 配送规划方案编码
    md03_dist_plan_scheme_name = n34_data["MD03_DIST_PLAN_SCHEME_NAME"]  # MD03_DIST_PLAN_SCHEME_NAME 配送规划方案名称

    # 1.1查出发货点-零售户 表名
    station_cust_table_name = get_distance_table_name(md03_dist_center_code,False)

    # 1.2查出零售户-零售户 表名
    cust_table_name = get_distance_table_name(md03_dist_center_code, True)

    """2. 1全局规划 2局部规划"""
    global_type = plan_type == "1"

    """3. 取出发货站点编码，生成主键，存储"""
    save_n38_data_list = []
    save_n40_data_list_from_n38 = []
    save_n39_data_list = []

    ###### n38数据
    for line_code in data:
        # MD03_DIST_PLAN_RANGE_ID 配送规划范围主键
        line_id = get_short_uuid()  # 默认22字符[1,2](@ref)

        # 取出n38数据,取出参数
        n38_param = data[line_code]
        line_name = n38_param["route_name"]  # 配送线路名称
        station = n38_param["station"]  # 发货站点编码
        vehicle_id = n38_param["vehicle_id"]
        load_ratio = n38_param["load_ratio"]
        distance_km = n38_param["distance_km"]
        duration_h = n38_param["duration_h"]

        # 取出n39参数数据
        sequence = str(n38_param["sequence"]).split(",")
        prev_distance_km = str(n38_param["prev_distance_km"]).split(",")
        prev_duration_h = str(n38_param["prev_duration_h"]).split(",")

        # md03_region_logi_center_code
        # md03_dist_center_code

        shipment_statio_name = ""
        db_n26 = get_n26_by_center_code_and_scheme_code_and_station_codes(md03_dist_center_code, md03_dist_plan_scheme_code, [station])
        if db_n26 is not None and not db_n26.empty:
            n26_data = db_n26.iloc[0]
            shipment_statio_name = n26_data["MD03_SHIPMENT_STATION_NAME"]

        cd_car_no = ""
        deliver_user_code = ""
        deliver_user_name = ""
        driving_user_code = ""
        driving_user_name = ""
        db_n27 = get_n27_by_center_code_and_scheme_code_and_station_codes(md03_dist_center_code, md03_dist_plan_scheme_code, [station])
        if db_n27 is not None and not db_n27.empty:
            n27_data = db_n27.iloc[0]
            cd_car_no = n27_data["CD_CAR_NO"]
            deliver_user_code = n27_data["MD03_DELIVER_USER_CODE"]
            deliver_user_name = n27_data["MD03_DELIVER_USER_NAME"]
            driving_user_code = n27_data["MD03_DRIVING_USER_CODE"]
            driving_user_name = n27_data["MD03_DRIVING_USER_NAME"]

        md03_issue_state = 0  # 0未发布,1已发布,2撤销发布,3待优化
        # md03_dist_plan_scheme_code
        # md03_dist_plan_scheme_name

        # save_n38_sql = f"""INSERT INTO t_p_ymd03n38_dist_plan_line (
        #                         `MD03_DIST_PLAN_LINE_ID`,
        #                         `MD03_REGION_LOGI_CENTER_CODE`,
        #                         `MD03_DIST_CENTER_CODE`,
        #                         `MD03_SHIPMENT_STATION_CODE`,
        #                         `MD03_SHIPMENT_STATION_NAME`,
        #                         `MD03_DIST_LINE_CODE`,
        #                         `MD03_DIST_LINE_NAME`,
        #                         `CD_LOGT_COM_DEY_VEHS_CODE`,
        #                         `CD_CAR_NO`,
        #                         `MD03_DELIVER_USER_CODE`,
        #                         `MD03_DELIVER_USER_NAME`,
        #                         `MD03_DRIVING_USER_CODE`,
        #                         `MD03_DRIVING_USER_NAME`,
        #                         `MD03_LODING_RATE`,
        #                         `MD03_ESTIMATED_DIST_DURATION`,
        #                         `MD03_ESTIMATED_DIST_MILEAGE`,
        #                         `MD03_ISSUE_STATE`,
        #                         `MD03_NEW_ORDER_WEEKDAY_CODE`,
        #                         `MD03_DIST_PLAN_SCHEME_CODE`,
        #                         `MD03_DIST_PLAN_SCHEME_NAME`
        #                     )
        #                     VALUES
        #                         (
        #                             '{line_id}',
        #                             '{md03_region_logi_center_code}',
        #                             '{md03_dist_center_code}',
        #                             '{station}',
        #                             '{shipment_statio_name}',
        #                             '{line_code}',
        #                             '{line_code}',
        #                             '{vehicle_id}',
        #                             '{cd_car_no}',
        #                             '{deliver_user_code}',
        #                             '{deliver_user_name}',
        #                             '{driving_user_code}',
        #                             '{driving_user_name}',
        #                             '{load_ratio}',
        #                             '{duration_h}',
        #                             '{distance_km}',
        #                             '{md03_issue_state}',
        #                             NULL,
        #                             '{md03_dist_plan_scheme_code}',
        #                             '{md03_dist_plan_scheme_name}'
        #                         )
        #                     """
        save_n38_data_list.append((
                                    line_id,
                                    md03_region_logi_center_code,
                                    md03_dist_center_code,
                                    station,
                                    shipment_statio_name,
                                    line_code,
                                    line_name,
                                    vehicle_id,
                                    cd_car_no,
                                    deliver_user_code,
                                    deliver_user_name,
                                    driving_user_code,
                                    driving_user_name,
                                    load_ratio,
                                    duration_h,
                                    distance_km,
                                    md03_issue_state,
                                    md03_dist_plan_scheme_code,
                                    md03_dist_plan_scheme_name
                                ))

        # 取n28数据
        db_n28_list = get_n28_by_center_code_and_scheme_code_and_customer_codes(md03_dist_center_code, md03_dist_plan_scheme_code, sequence)
        if db_n28_list.empty:
            return {False, "未获取到n28数据"}
        df_n28_indexed = db_n28_list.set_index('BB_RETAIL_CUSTOMER_CODE')  # 将 'id' 列设为索引

        # 服务时长 MD03_SERVICE_DURATION
        db_n23, n31_service_duration = get_service_duration(sequence, md03_dist_center_code)
        db_n23_indexed = None
        if not db_n23.empty:
            db_n23_indexed = db_n23.set_index('BB_RETAIL_CUSTOMER_CODE')

        # 平均订货量 MD03_AVG_ORDER_QTY 取 n32的历史订单
        db_n32_count = get_n32_count_by_center_code_and_scheme_code_and_customer_code(md03_dist_center_code, md03_dist_plan_scheme_code, sequence)
        df_n32_indexed = None
        if not db_n32_count.empty:
            df_n32_indexed = db_n32_count.set_index('BB_RETAIL_CUSTOMER_CODE')

        ### n39数据
        child_len = len(sequence)
        md03_deliver_total_qty = Decimal(0.0)
        for i in range(child_len):
            # 取出n39参数数据
            sequence_item = sequence[i]                 #客户零售户编码
            prev_distance_km_item = prev_distance_km[i]
            prev_duration_h_item = prev_duration_h[i]

            # line_detail_id = get_short_uuid()         # 默认22字符[1,2](@ref)
            dist_plan_line_id = line_id
            # md03_dist_center_code
            deliver_deliver_no = i

            bb_retail_customer_name = ""
            bb_rtl_cust_business_addr = ""
            md03_retail_cust_lon = "0.0"
            md03_retail_cust_lat = "0.0"
            bb_rtl_cust_prov_order_cycle_type = ""
            bb_rtl_cust_order_weekday = ""
            bb_rtl_cust_artificial_name = ""  # 取自N28表
            md03_service_duration = ""
            md03_avg_order_qty = ""
            md03_lock_state = 0  # 0未锁定、1已锁定
            n28_data = df_n28_indexed.loc[sequence_item]
            if n28_data is not None:
                bb_retail_customer_name = n28_data["BB_RETAIL_CUSTOMER_NAME"]
                bb_rtl_cust_business_addr = n28_data["BB_RTL_CUST_BUSINESS_ADDR"]
                md03_retail_cust_lon = n28_data["MD03_RETAIL_CUST_LON"] if n28_data["MD03_RETAIL_CUST_LON"] is not None else 0
                md03_retail_cust_lat = n28_data["MD03_RETAIL_CUST_LAT"] if n28_data["MD03_RETAIL_CUST_LAT"] is not None else 0
                bb_rtl_cust_prov_order_cycle_type = n28_data["BB_RTL_CUST_PROV_ORDER_CYCLE_TYPE"]
                bb_rtl_cust_order_weekday = n28_data["BB_RTL_CUST_ORDER_WEEKDAY"]
                bb_rtl_cust_artificial_name = n28_data["BB_RTL_CUST_ARTIFICIAL_NAME"]  # 取自N28表
                # MD03_SERVICE_DURATION
                if db_n23_indexed is not None and sequence_item in db_n23_indexed.index:
                    md03_service_duration = db_n23_indexed.loc[sequence_item]['MD03_SERVICE_DURATION']
                else:
                    md03_service_duration = n31_service_duration

                # 平均订货量 MD03_AVG_ORDER_QTY 取 n32的历史订单
                md03_avg_order_qty = 0
                if df_n32_indexed is not None and sequence_item in df_n32_indexed.index:
                    md03_avg_order_qty = df_n32_indexed.loc[sequence_item]['count'] if df_n32_indexed.loc[sequence_item]['count'] is not None else 0

            # 计算距离
            # data_distance = ''
            # data_duration = ''
            # if i==0:
            #     data_distance, data_duration = get_cust_station_for_distance_and_duration(station_cust_table_name, md03_dist_center_code, station, sequence_item, True)
            # elif i == (child_len-1):
            #     data_distance, data_duration = get_cust_station_for_distance_and_duration(station_cust_table_name, md03_dist_center_code, station, sequence_item,False)
            # else:
            #     data_distance, data_duration = get_cust_between_for_distance_and_duration(cust_table_name, md03_dist_center_code, sequence[i-1], sequence_item)

            # todo 251201时可删除，已用下面批量插入方式替换掉这种一条条插入
            # save_n39_sql = f"""INSERT INTO t_p_ymd03n39_dist_plan_line_detail (
            #                         `MD03_DIST_PLAN_LINE_DETAIL_ID`,
            #                         `MD03_DIST_PLAN_LINE_ID`,
            #                         `MD03_DIST_CENTER_CODE`,
            #                         `MD03_DELIVER_DELIVER_NO`,
            #                         `BB_RETAIL_CUSTOMER_CODE`,
            #                         `BB_RETAIL_CUSTOMER_NAME`,
            #                         `BB_RTL_CUST_BUSINESS_ADDR`,
            #                         `MD03_RETAIL_CUST_LON`,
            #                         `MD03_RETAIL_CUST_LAT`,
            #                         `BB_RTL_CUST_PROV_ORDER_CYCLE_TYPE`,
            #                         `BB_RTL_CUST_ORDER_WEEKDAY`,
            #                         `BB_RTL_CUST_ARTIFICIAL_NAME`,
            #                         `MD03_SERVICE_DURATION`,
            #                         `MD03_AVG_ORDER_QTY`,
            #                         `MD03_PREVIOUS_CUST_DRIVING_DISTANCE`,
            #                         `MD03_PREVIOUS_CUST_DRIVING_DURATION`,
            #                         `MD03_LOCK_STATE`
            #                     )
            #                     VALUES
            #                         (
            #                             UUID_SHORT(),
            #                             '{dist_plan_line_id}',
            #                             '{md03_dist_center_code}',
            #                             '{deliver_deliver_no}',
            #                             '{sequence_item}',
            #                             '{bb_retail_customer_name}',
            #                             '{bb_rtl_cust_business_addr}',
            #                             '{md03_retail_cust_lon}',
            #                             '{md03_retail_cust_lat}',
            #                             '{bb_rtl_cust_prov_order_cycle_type}',
            #                             '{bb_rtl_cust_order_weekday}',
            #                             '{bb_rtl_cust_artificial_name}',
            #                             '{md03_service_duration}',
            #                             '{md03_avg_order_qty}',
            #                             '{data_distance}',
            #                             '{data_duration}',
            #                             '{md03_lock_state}'
            #                         )
            #                             """
            save_n39_data_list.append((
                                        dist_plan_line_id,
                                        md03_dist_center_code,
                                        deliver_deliver_no,
                                        sequence_item,
                                        bb_retail_customer_name,
                                        bb_rtl_cust_business_addr,
                                        md03_retail_cust_lon,
                                        md03_retail_cust_lat,
                                        bb_rtl_cust_prov_order_cycle_type,
                                        bb_rtl_cust_order_weekday,
                                        bb_rtl_cust_artificial_name,
                                        md03_service_duration,
                                        md03_avg_order_qty,
                                        prev_distance_km_item,
                                        prev_duration_h_item,
                                        md03_lock_state
                                    ))

            md03_deliver_total_qty += md03_avg_order_qty


        ### save n40
        save_n40_data_list_from_n38.append({
            "MD03_DIST_PLAN_SCHEME_CODE": md03_dist_plan_scheme_code,
            "MD03_DIST_PLAN_SCHEME_NAME": md03_dist_plan_scheme_name,
            "MD03_SHIPMENT_STATION_CODE": station,
            "MD03_SHIPMENT_STATION_NAME": shipment_statio_name,
            "MD03_DIST_LINE_CODE": line_code,
            "MD03_DIST_LINE_NAME": line_name,
            "CD_LOGT_COM_DEY_VEHS_CODE": vehicle_id,
            "CD_CAR_NO": cd_car_no,
            "MD03_DELIVER_USER_CODE": deliver_user_code,
            "MD03_DELIVER_USER_NAME": deliver_user_name,
            "MD03_DRIVING_USER_CODE": driving_user_code,
            "MD03_DRIVING_USER_NAME": driving_user_name,
            "MD03_DELIVER_CUST_QTY": child_len,
            "MD03_DELIVER_TOTAL_QTY": md03_deliver_total_qty,
            "MD03_LODING_RATE": load_ratio,
            "MD03_ESTIMATED_DIST_DURATION": duration_h,
            "MD03_ESTIMATED_DIST_MILEAGE": distance_km
        })

    """4. 更新n34 数据"""
    """ MD03_DIST_PLAN_STATE  配送规划状态 1进行中，2已完成，3规划失败"""
    update_n34_sql = f"UPDATE t_p_ymd03n34_dist_plan_log SET `MD03_DIST_PLAN_STATE` = 2, `MD03_CALC_END_TIME` = '{formatted_time}' WHERE `MD03_DIST_PLAN_LOG_ID` = '{n34_id}'"

    """5. 获取全部需要修改的sql，执行"""
    save_sqls = []
    save_sqls.append(update_n34_sql)
    """todo 局部规划删除 n37 和 n38"""
    if not global_type:
        add_n35_log_info(n34_id, "ORD_0", "后置业务数据处理-历史局部规划数据删除中...")
        plan_range_ids = plan_range_id.split(",")
        for item_range_id in plan_range_ids:
            del_n38_sql = f"DELETE FROM t_p_ymd03n38_dist_plan_line WHERE MD03_DIST_PLAN_LINE_ID = '{item_range_id}'"
            del_n39_sql = f"DELETE FROM t_p_ymd03n39_dist_plan_line_detail WHERE MD03_DIST_PLAN_LINE_DETAIL_ID = '{item_range_id}'"
            save_sqls.append(del_n38_sql)
            save_sqls.append(del_n39_sql)
        add_n35_log_info(n34_id, "ORD_1", "后置业务数据处理-历史局部规划数据删除成功")

    # save_sqls.extend(save_n38_data_list)
    # save_sqls.extend(save_n39_data_list)
    # print(save_sqls)
    add_n35_log_info(n34_id, "OR_1", "后置业务数据处理-路线规划数据处理完毕，正在回写数据库中...")
    # db_execute_batch(save_sqls)
    """6. 添加n40 """
    add_n40(db_n32_group, save_n40_data_list_from_n38, md03_region_logi_center_code, md03_dist_center_code)
    # 执行后续sql
    db_execute_callback(lambda cursor: execute_sql_list(cursor, save_sqls, get_save_n38_sql(), save_n38_data_list, get_save_n39_sql(), save_n39_data_list, md03_dist_center_code, md03_dist_plan_scheme_code))
    add_n35_log_info(n34_id, "OR_2", "后置业务数据处理-结束-路线规划数据回写数据库成功")


# def in_n40():
#     MD03_DIST_PLAN_RESULT_STAT_ID = ""
#     MD03_REGION_LOGI_CENTER_CODE = ""
#     MD03_DIST_CENTER_CODE = ""
#     MD03_DIST_PLAN_BATCH_ID = ""
#     MD03_BATCH_CREATE_TIME = ""
#     MD03_DIST_PLAN_PERIOD = ""
#     MD03_DIST_PLAN_SCHEME_CODE = ""
#     MD03_DIST_PLAN_SCHEME_NAME = ""
#     MD03_SHIPMENT_STATION_CODE = ""
#     MD03_SHIPMENT_STATION_NAME = ""
#     MD03_DIST_LINE_CODE = ""
#     MD03_DIST_LINE_NAME = ""
#     CD_LOGT_COM_DEY_VEHS_CODE = ""
#     CD_CAR_NO = ""
#     MD03_DELIVER_USER_CODE = ""
#     MD03_DELIVER_USER_NAME = ""
#     MD03_DRIVING_USER_CODE = ""
#     MD03_DRIVING_USER_NAME = ""
#     MD03_DELIVER_CUST_QTY = ""
#     MD03_DELIVER_TOTAL_QTY = ""
#     MD03_LODING_RATE = ""
#     MD03_ESTIMATED_DIST_DURATION = ""
#     MD03_ESTIMATED_DIST_MILEAGE = ""
def execute_sql_list(cursor, sql_list, n38_add_sql, n38_add_data, n39_add_sql, n39_add_data, md03_dist_center_code, md03_dist_plan_scheme_code):
    #先清空历史数据
    set_n38_n39_status(md03_dist_center_code, md03_dist_plan_scheme_code)

    #在执行
    for sql in sql_list:
        cursor.execute(sql)
    cursor.executemany(n38_add_sql,n38_add_data)
    cursor.executemany(n39_add_sql,n39_add_data)


def set_n38_n39_status(dist_center_code: str, dist_plan_scheme_code: str):
    # 1.获取n38数据 通过同个方案
    n38_ids = []
    db_n38 = get_n38_by_center_code_and_scheme_code(dist_center_code,dist_plan_scheme_code)
    if db_n38 is not None and not db_n38.empty:
        for item_n38 in db_n38.itertuples():
            n38_ids.append(item_n38.MD03_DIST_PLAN_LINE_ID)
        # 1.1 根据ids 更新n38数据状态
        update_n38_state_by_ids(n38_ids, "1")

    # 2.获取n39数据 通过n38主键
    if not n38_ids:
        # 2.1 根据ids 更新n39数据状态
        update_n39_state_by_ids(n38_ids, "1")


""" ================================================ """


def get_service_duration(bb_retail_customer_codes: [], md03_dist_center_code: str):
    """
    获取零售户服务时长时，零售户有在停车点下时，最优先从N21表取值。否则如果N23表的MD03_SERVICE_DURATION有值时优先从N23表取，没值时取自配送规划参数设置N31表的MD03_DIST_PLAN_PARAM_VAL
    N23表过滤条件MD03_DIST_CENTER_CODE、BB_RETAIL_CUSTOMER_CODE
    N31表过滤条件MD03_DIST_CENTER_CODE、MD03_DIST_PLAN_TYPE=2、MD03_DIST_PLAN_PARAM_CODE= XLGH0002
    """

    customer_codes_str = ", ".join(f"'{code}'" for code in bb_retail_customer_codes)
    sql = f"""
                   SELECT
                       * 
                   FROM
                       `t_p_ymd03n23_retail_cust_param_set` 
                   WHERE
                       MD03_IS_DELETED = '0' 
                       AND MD03_DIST_CENTER_CODE = '{md03_dist_center_code}'
                       AND BB_RETAIL_CUSTOMER_CODE IN ({customer_codes_str})
               """
    db_n23 = get_db(sql)

    param_all2 = {"MD03_IS_DELETED": 0, "MD03_DIST_CENTER_CODE": md03_dist_center_code, 'MD03_DIST_PLAN_TYPE': 2, "MD03_DIST_PLAN_PARAM_CODE": "XLGH0002"}
    db_n31 = get_common_data("t_p_ymd03n31_dist_plan_param_set", param_all2)
    n31_data = db_n31.iloc[0]
    n31_service_duration = n31_data["MD03_DIST_PLAN_PARAM_VAL"]
    return db_n23, n31_service_duration


""" 通过 配送规划类型=(1范围规划、2线路规划) 和 配送中心编码 """
def get_n19_by_play_type_and_center_code(md03_dist_plan_type: int, md03_dist_center_code: str):
    param_all = {"MD03_IS_DELETED": 0, "MD03_DIST_PLAN_TYPE": md03_dist_plan_type,
                 "MD03_DIST_CENTER_CODE": md03_dist_center_code}
    res = get_common_data("t_p_ymd03n19_dist_plan_algo_info", param_all)
    return res

""" 通过 配送规划算法主键 查 n20 """
def get_n20_by_alog_ids( alog_ids: []):
    alog_ids_str = ", ".join(f"'{code}'" for code in alog_ids)
    sql = f"""
                           SELECT
                               * 
                           FROM
                               `t_p_ymd03n20_dist_plan_algo_detail` 
                           WHERE
                               MD03_IS_DELETED = '0' 
                               AND MD03_DIST_PLAN_ALGO_ID IN ({alog_ids_str})
                       """
    res = get_db(sql)
    return res

""" 根据 配送中心编码 和 配送规划方案编码 和 发货站点编码"""
def get_n26_by_center_code_and_scheme_code_and_station_codes(md03_dist_center_code: str, md03_dist_plan_scheme_code: str, station_codes: []):
    station_codes_str = ", ".join(f"'{code}'" for code in station_codes)
    sql = f"""
                       SELECT
                           * 
                       FROM
                           `t_p_ymd03n26_dist_plan_dist_site_info` 
                       WHERE
                           MD03_IS_DELETED = '0' 
                           AND MD03_DIST_CENTER_CODE = '{md03_dist_center_code}' 
                           AND MD03_DIST_PLAN_SCHEME_CODE = '{md03_dist_plan_scheme_code}' 
                           AND MD03_SHIPMENT_STATION_CODE IN ({station_codes_str})
                   """
    res = get_db(sql)
    return res


""" 根据 配送中心编码 和 配送规划方案编码 和 发货站点编码"""
def get_n27_by_center_code_and_scheme_code_and_station_codes(md03_dist_center_code: str, md03_dist_plan_scheme_code: str, station_codes: []):
    station_codes_str = ", ".join(f"'{code}'" for code in station_codes)
    sql = f"""
                       SELECT
                           * 
                       FROM
                           `t_p_ymd03n27_dist_plan_car_info` 
                       WHERE
                           MD03_IS_DELETED = '0' 
                           AND MD03_DIST_CENTER_CODE = '{md03_dist_center_code}' 
                           AND MD03_DIST_PLAN_SCHEME_CODE = '{md03_dist_plan_scheme_code}' 
                           AND MD03_SHIPMENT_STATION_CODE IN ({station_codes_str})
                   """
    res = get_db(sql)
    return res


def get_n28_by_center_code_and_scheme_code_and_customer_codes(md03_dist_center_code: str, md03_dist_plan_scheme_code: str, bb_retail_customer_codes: []):
    customer_codes_str = ", ".join(f"'{code}'" for code in bb_retail_customer_codes)
    sql = f"""
               SELECT
                   * 
               FROM
                   `t_p_ymd03n28_dist_plan_retail_cust` 
               WHERE
                   MD03_IS_DELETED = '0' 
                   AND MD03_DIST_CENTER_CODE = '{md03_dist_center_code}' 
                   AND MD03_DIST_PLAN_SCHEME_CODE = '{md03_dist_plan_scheme_code}' 
                   AND BB_RETAIL_CUSTOMER_CODE IN ({customer_codes_str})
           """
    res = get_db(sql)
    return res


def get_n28_by_center_code_and_scheme_code_and_station_codes(md03_dist_center_code: str, md03_dist_plan_scheme_code: str, station_codes: []):
    station_codes_str = ", ".join(f"'{code}'" for code in station_codes)
    sql = f"""
                   SELECT
                       * 
                   FROM
                       `t_p_ymd03n28_dist_plan_retail_cust` 
                   WHERE
                       MD03_IS_DELETED = '0' 
                       AND MD03_DIST_CENTER_CODE = '{md03_dist_center_code}' 
                       AND MD03_DIST_PLAN_SCHEME_CODE = '{md03_dist_plan_scheme_code}' 
                       AND MD03_DIST_STATION_CODE IN ({station_codes_str})
               """
    res = get_db(sql)
    return res


""" 通过 配送规划类型=(1范围规划、2线路规划) 和 配送中心编码 """
def get_n31_play_type_and_by_play_type_and_center_code(md03_dist_plan_type: int, md03_dist_center_code: str, md03_dist_plan_scheme_code: str):
    param_all = {"MD03_IS_DELETED": 0, "MD03_DIST_PLAN_TYPE": md03_dist_plan_type,
                 "MD03_DIST_CENTER_CODE": md03_dist_center_code, "MD03_DIST_PLAN_SCHEME_CODE": md03_dist_plan_scheme_code}
    res = get_common_data("t_p_ymd03n31_dist_plan_param_set", param_all)
    return res


def get_n32_count_by_center_code_and_scheme_code_and_customer_code(md03_dist_center_code: str, md03_dist_plan_scheme_code: str, bb_retail_customer_codes: []):
    customer_codes_str = ", ".join(f"'{code}'" for code in bb_retail_customer_codes)
    sql = f"""SELECT
                    ROUND(AVG(MD03_ORDER_TOTAL_QTY * 1.0), 2) AS count,
                    BB_RETAIL_CUSTOMER_CODE
                FROM
                    `t_p_ymd03n32_dist_plan_order` 
                where
                    MD03_IS_DELETED = '0' 
                    AND MD03_DIST_CENTER_CODE = '{md03_dist_center_code}' 
                    AND MD03_DIST_PLAN_SCHEME_CODE = '{md03_dist_plan_scheme_code}' 
                    AND BB_RETAIL_CUSTOMER_CODE IN ({customer_codes_str})
                GROUP BY
                    BB_RETAIL_CUSTOMER_CODE 
            """
    res = get_db(sql)
    return res


def get_n32_by_center_code_and_scheme_code_and_customer_code_and_station_codes(md03_dist_center_code: str, md03_dist_plan_scheme_code: str,
                                                                               bb_retail_customer_codes: [], station_codes: []):
    customer_codes_str = ", ".join(f"'{code}'" for code in bb_retail_customer_codes)
    station_codes_str = ", ".join(f"'{code}'" for code in station_codes)
    sql = f"""
                   SELECT
                       *
                   FROM
                       `t_p_ymd03n32_dist_plan_order`
                   WHERE
                       MD03_IS_DELETED = '0'
                       AND MD03_DIST_CENTER_CODE = '{md03_dist_center_code}'
                       AND MD03_DIST_PLAN_SCHEME_CODE = '{md03_dist_plan_scheme_code}'
                       AND BB_RETAIL_CUSTOMER_CODE IN ({customer_codes_str})
                       AND MD03_DIST_STATION_CODE IN ({station_codes_str})
               """
    res = get_db(sql)
    return res

def get_n32_by_center_code_and_scheme_code_and_customer_code_and_station_codes_group(md03_dist_center_code: str, md03_dist_plan_scheme_code: str,
                                                                               bb_retail_customer_codes: [], station_codes: []):
    customer_codes_str = ", ".join(f"'{code}'" for code in bb_retail_customer_codes)
    station_codes_str = ", ".join(f"'{code}'" for code in station_codes)
    sql = f"""
                   SELECT
                       *, s.MD03_SHIPMENT_STATION_NAME,
	                    GROUP_CONCAT(BB_RETAIL_CUSTOMER_CODE separator ',') AS CUSTOMER_CODES,
	                    count(*) AS MD03_DELIVER_CUST_QTY,
	                    SUM(MD03_ORDER_TOTAL_QTY) as MD03_DELIVER_TOTAL_QTY
                   FROM
                       `t_p_ymd03n32_dist_plan_order` a LEFT JOIN t_p_ymd03n26_dist_plan_dist_site_info s ON MD03_DIST_STATION_CODE = MD03_SHIPMENT_STATION_CODE 
                   WHERE
                       a.MD03_IS_DELETED = '0'
                       AND a.MD03_DIST_CENTER_CODE = '{md03_dist_center_code}'
                       AND a.MD03_DIST_PLAN_SCHEME_CODE = '{md03_dist_plan_scheme_code}'
                       AND a.BB_RETAIL_CUSTOMER_CODE IN ({customer_codes_str})
                       AND MD03_DIST_STATION_CODE IN ({station_codes_str})
                       GROUP BY
                            MD03_DELIVER_DATE,
                            MD03_BIG_DIST_LINE_CODE,
                            CD_CAR_NO,
                            MD03_DRIVING_USER_CODE
                        ORDER BY MD03_DELIVER_DATE asc
               """
    res = get_db(sql)
    return res


""" 根据 配送中心编码 和 配送规划方案编码 """
def get_n33_by_center_code_and_scheme_code(md03_dist_center_code: str, md03_dist_plan_scheme_code: str, md03_dist_plan_type: int):
    param_all = {"MD03_IS_DELETED": 0, "MD03_DIST_CENTER_CODE": md03_dist_center_code, "MD03_DIST_PLAN_SCHEME_CODE": md03_dist_plan_scheme_code,
                 "MD03_DIST_PLAN_TYPE": md03_dist_plan_type}
    res = get_common_data("t_p_ymd03n33_dist_plan_balance_mode", param_all)
    return res


def get_n34_by_id(n34_id: str):
    param_all = {"MD03_IS_DELETED": 0, "MD03_DIST_PLAN_LOG_ID": n34_id}
    res = get_common_data("t_p_ymd03n34_dist_plan_log", param_all)
    return res


def get_n36_by_center_code_and_scheme_code_and_station_codes(md03_dist_center_code: str, md03_dist_plan_scheme_code: str, md03_shipment_station_codes: []):
    md03_shipment_station_codes_str = ", ".join(f"'{code}'" for code in md03_shipment_station_codes)
    sql = f"""
                SELECT
                    * 
                FROM
                    `t_p_ymd03n36_dist_plan_range` 
                WHERE
                    MD03_IS_DELETED = '0' 
                    AND MD03_DIST_CENTER_CODE = '{md03_dist_center_code}' 
                    AND MD03_DIST_PLAN_SCHEME_CODE = '{md03_dist_plan_scheme_code}' 
                    AND MD03_SHIPMENT_STATION_CODE IN ({md03_shipment_station_codes_str})
            """
    res = get_db(sql)
    return res


def get_n37_by_ids(n36_ids: str):
    sql = f"""
                    SELECT
                        * 
                    FROM
                        `t_p_ymd03n37_dist_plan_range_detail` 
                    WHERE
                        MD03_IS_DELETED = '0' 
                        AND MD03_DIST_PLAN_RANGE_ID IN ({n36_ids})
                """
    res = get_db(sql)
    return res


def get_n38_by_ids(n38_ids: []):
    n38_ids_str = ", ".join(f"'{code}'" for code in n38_ids)
    sql = f"""
                    SELECT
                        * 
                    FROM
                        `t_p_ymd03n38_dist_plan_line` 
                    WHERE
                        MD03_IS_DELETED = '0' 
                        AND MD03_DIST_PLAN_LINE_ID IN ({n38_ids_str})
                """
    res = get_db(sql)
    return res

def get_n38_by_center_code_and_scheme_code(md03_dist_center_code: str, md03_dist_plan_scheme_code: str):
    sql = f"""
                    SELECT
                        * 
                    FROM
                        `t_p_ymd03n38_dist_plan_line` 
                    WHERE
                        MD03_IS_DELETED = '0' 
                        AND MD03_DIST_CENTER_CODE IN ({md03_dist_center_code})
                        AND MD03_DIST_PLAN_SCHEME_CODE IN ({md03_dist_plan_scheme_code})
                """
    res = get_db(sql)
    return res


def get_n39_by_ids(n38_ids: []):
    n38_ids_str = ", ".join(f"'{code}'" for code in n38_ids)
    sql = f"""
                    SELECT
                        * 
                    FROM
                        `t_p_ymd03n39_dist_plan_line_detail` 
                    WHERE
                        MD03_IS_DELETED = '0' 
                        AND MD03_DIST_PLAN_LINE_ID IN ({n38_ids_str})
                """
    res = get_db(sql)
    return res

@lru_cache(maxsize=None)
def get_distance_table_name(center_code: str, cust_distance=False):
    type_code = ''
    if cust_distance:
        type_code = 'CUST_DISTANCE'
    else:
        type_code = 'STATION_DISTANCE'

    sql = f"""
                    SELECT MD03_PARAM_VALUE FROM CM_PARAM_CONFIG
                         WHERE MD03_TYPE_CODE = '{type_code}'
                           AND MD03_ENABLE_FLAG = 1
                             AND MD03_PARAM_CODE = '{center_code}'
                """
    res = get_db(sql)

    table_name = None
    if res is not None and not res.empty:
        cust_station_data = res.iloc[0]
        table_name = cust_station_data["MD03_PARAM_VALUE"]
    return table_name


def get_cust_station_for_distance_and_duration(table_name: Optional[str], md03_dist_center_code: str, station: str, cust_code: str, station_to_cust=True):
    if table_name is None:
        return '', ''

    md03_distance_type = '1' #1代表发货点至零售户、2代表零售户至发货点
    if station_to_cust:
        md03_distance_type = '1'
    else:
        md03_distance_type = '2'

    sql = f"""
                        SELECT MD03_DRIVING_DISTANCE,MD03_DRIVING_DURATION 
                          FROM {table_name}
                         WHERE MD03_DIST_CENTER_CODE = '{md03_dist_center_code}'
                           AND MD03_SHIPMENT_STATION_CODE = '{station}'
                             AND BB_RETAIL_CUSTOMER_CODE = '{cust_code}'
                             AND MD03_DISTANCE_TYPE = '{md03_distance_type}'
                    """
    res = get_db(sql)

    md03_driving_distance = ''
    md03_driving_duration = ''
    if res is not None and not res.empty:
        res_data = res.iloc[0]
        md03_driving_distance = res_data["MD03_DRIVING_DISTANCE"]
        md03_driving_duration = res_data["MD03_DRIVING_DURATION"]

    return md03_driving_distance, md03_driving_duration

def get_cust_between_for_distance_and_duration(table_name: Optional[str], md03_dist_center_code: str, cust_code_one: str, cust_code_two: str):
    if table_name is None:
        return '', ''

    sql = f"""
                        SELECT MD03_DRIVING_DISTANCE,MD03_DRIVING_DURATION 
                          FROM {table_name}
                         WHERE MD03_DIST_CENTER_CODE = '{md03_dist_center_code}'
                           AND MD03_RETAIL_CUST_CODE_ONE = '{cust_code_one}'
                             AND MD03_RETAIL_CUST_CODE_TWO = '{cust_code_two}'
                    """
    res = get_db(sql)

    md03_driving_distance = ''
    md03_driving_duration = ''
    if res is not None and not res.empty:
        res_data = res.iloc[0]
        md03_driving_distance = res_data["MD03_DRIVING_DISTANCE"]
        md03_driving_duration = res_data["MD03_DRIVING_DURATION"]

    return md03_driving_distance, md03_driving_duration

def get_save_n38_sql():
    sql = """
    INSERT INTO t_p_ymd03n38_dist_plan_line (
                                `MD03_DIST_PLAN_LINE_ID`,
                                `MD03_REGION_LOGI_CENTER_CODE`,
                                `MD03_DIST_CENTER_CODE`,
                                `MD03_SHIPMENT_STATION_CODE`,
                                `MD03_SHIPMENT_STATION_NAME`,
                                `MD03_DIST_LINE_CODE`,
                                `MD03_DIST_LINE_NAME`,
                                `CD_LOGT_COM_DEY_VEHS_CODE`,
                                `CD_CAR_NO`,
                                `MD03_DELIVER_USER_CODE`,
                                `MD03_DELIVER_USER_NAME`,
                                `MD03_DRIVING_USER_CODE`,
                                `MD03_DRIVING_USER_NAME`,
                                `MD03_LODING_RATE`,
                                `MD03_ESTIMATED_DIST_DURATION`,
                                `MD03_ESTIMATED_DIST_MILEAGE`,
                                `MD03_ISSUE_STATE`,
                                `MD03_NEW_ORDER_WEEKDAY_CODE`,
                                `MD03_DIST_PLAN_SCHEME_CODE`,
                                `MD03_DIST_PLAN_SCHEME_NAME` 
                            )
                            VALUES
                                (
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    NULL,
                                    %s,
                                    %s
                                )
    """
    return sql

def get_save_n39_sql():
    sql = """INSERT INTO t_p_ymd03n39_dist_plan_line_detail (
                                    `MD03_DIST_PLAN_LINE_DETAIL_ID`,
                                    `MD03_DIST_PLAN_LINE_ID`,
                                    `MD03_DIST_CENTER_CODE`,
                                    `MD03_DELIVER_DELIVER_NO`,
                                    `BB_RETAIL_CUSTOMER_CODE`,
                                    `BB_RETAIL_CUSTOMER_NAME`,
                                    `BB_RTL_CUST_BUSINESS_ADDR`,
                                    `MD03_RETAIL_CUST_LON`,
                                    `MD03_RETAIL_CUST_LAT`,
                                    `BB_RTL_CUST_PROV_ORDER_CYCLE_TYPE`,
                                    `BB_RTL_CUST_ORDER_WEEKDAY`,
                                    `BB_RTL_CUST_ARTIFICIAL_NAME`,
                                    `MD03_SERVICE_DURATION`,
                                    `MD03_AVG_ORDER_QTY`,
                                    `MD03_PREVIOUS_CUST_DRIVING_DISTANCE`,
                                    `MD03_PREVIOUS_CUST_DRIVING_DURATION`,
                                    `MD03_LOCK_STATE` 
                                )
                                VALUES
                                    (
                                        UUID_SHORT(),
                                        %s,
                                        %s,
                                        %s,
                                        %s,
                                        %s,
                                        %s,
                                        %s,
                                        %s,
                                        %s,
                                        %s,
                                        %s,
                                        %s,
                                        %s,
                                        %s,
                                        %s,
                                        %s
                                    )
                                        """
    return sql

def update_n38_state_by_ids(n38_ids:[], status: str):
    n38_ids_str = ", ".join(f"'{code}'" for code in n38_ids)
    sql = f"""
                        UPDATE
                            `t_p_ymd03n38_dist_plan_line` 
                        SET
                            MD03_IS_DELETED = '{status}'
                        WHERE
                            MD03_IS_DELETED = '0' 
                            AND MD03_DIST_PLAN_LINE_ID IN ({n38_ids_str})
                    """
    res = db_update(sql)

def update_n39_state_by_ids(n38_ids:[], status: str):
    n38_ids_str = ", ".join(f"'{code}'" for code in n38_ids)
    sql = f"""
                        UPDATE
                            `t_p_ymd03n39_dist_plan_line_detail` 
                        SET
                            MD03_IS_DELETED = '{status}'
                        WHERE
                            MD03_IS_DELETED = '0' 
                            AND MD03_DIST_PLAN_LINE_ID IN ({n38_ids_str})
                    """
    res = db_update(sql)

def get_save_n40_sql():
    sql = """
    INSERT INTO t_p_ymd03n40_dist_plan_result_stat (
                                `MD03_DIST_PLAN_RESULT_STAT_ID`,
                                `MD03_REGION_LOGI_CENTER_CODE`,
                                `MD03_DIST_CENTER_CODE`,
                                `MD03_DIST_PLAN_BATCH_ID`,
                                `MD03_BATCH_CREATE_TIME`,
                                `MD03_DIST_PLAN_PERIOD`,
                                `MD03_DIST_PLAN_SCHEME_CODE`,
                                `MD03_DIST_PLAN_SCHEME_NAME`,
                                `MD03_SHIPMENT_STATION_CODE`,
                                `MD03_SHIPMENT_STATION_NAME`,
                                `MD03_DIST_LINE_CODE`,
                                `MD03_DIST_LINE_NAME`,
                                `CD_LOGT_COM_DEY_VEHS_CODE`,
                                `CD_CAR_NO`,
                                `MD03_DELIVER_USER_CODE`,
                                `MD03_DELIVER_USER_NAME`,
                                `MD03_DRIVING_USER_CODE`,
                                `MD03_DRIVING_USER_NAME`,
                                `MD03_DELIVER_CUST_QTY`,
                                `MD03_DELIVER_TOTAL_QTY`,
                                `MD03_LODING_RATE`,
                                `MD03_ESTIMATED_DIST_DURATION`,
                                `MD03_ESTIMATED_DIST_MILEAGE`
                            )
                            VALUES
                                (
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s
                                )
    """
    return sql

def add_n40(db_n32, db_n38_list, md03_region_logi_center_code, md03_dist_center_code):
    '''添加n40表数据
        1.db_n32_group是数据库分组查询的数据，从上层方法传入
        2.save_n40_data_list_from_n38 提取于算法处理后的n38表数据
        3.md03_region_logi_center_code 区域物流中心编码
        4.md03_dist_center_code 中心编码
    '''
    save_n40_data_list = []
    batch_id = get_short_uuid()
    batch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    date_groups = None
    if not db_n32 is None and not db_n32.empty:
        date_groups = get_n40_period_list(db_n32)

        for index, item_n32 in db_n32.iterrows():
            # 计算总距离、时间
            total_data_distance, total_data_duration = get_distance_and_duration(item_n32["CUSTOMER_CODES"].split(","), item_n32["MD03_DIST_STATION_CODE"], md03_dist_center_code)

            save_n40_data_list.append((
                get_short_uuid(),
                md03_region_logi_center_code,
                md03_dist_center_code,
                batch_id,
                batch_time,
                get_n40_period(date_groups, item_n32['MD03_DELIVER_DATE']),
                'YFA0001',
                '原有方案',
                item_n32['MD03_DIST_STATION_CODE'],
                item_n32['MD03_SHIPMENT_STATION_NAME'],
                f"{item_n32['MD03_DELIVER_DATE']}_{item_n32['MD03_DIST_LINE_NAME']}",
                item_n32['MD03_DIST_LINE_NAME'],
                item_n32['CD_LOGT_COM_DEY_VEHS_CODE'],
                item_n32['CD_CAR_NO'],
                item_n32['MD03_DELIVER_USER_CODE'],
                item_n32['MD03_DELIVER_USER_NAME'],
                item_n32['MD03_DRIVING_USER_CODE'],
                item_n32['MD03_DRIVING_USER_NAME'],
                item_n32['MD03_DELIVER_CUST_QTY'],
                item_n32['MD03_DELIVER_TOTAL_QTY'],
                0.0,
                total_data_distance,
                total_data_duration
            ))

    if db_n38_list is not None and db_n38_list:
        last_date_period = ''
        if date_groups is not None:
            last_date = date_groups.iloc[-1]
            last_date_period = f'{last_date["start_date"]}-{last_date["end_date"]}'
        for index, item_n38 in enumerate(db_n38_list):
            save_n40_data_list.append((
                get_short_uuid(),
                md03_region_logi_center_code,
                md03_dist_center_code,
                batch_id,
                batch_time,
                last_date_period,
                item_n38['MD03_DIST_PLAN_SCHEME_CODE'],
                item_n38['MD03_DIST_PLAN_SCHEME_NAME'],
                item_n38['MD03_SHIPMENT_STATION_CODE'],
                item_n38['MD03_SHIPMENT_STATION_NAME'],
                item_n38['MD03_DIST_LINE_CODE'],
                item_n38['MD03_DIST_LINE_NAME'],
                item_n38['CD_LOGT_COM_DEY_VEHS_CODE'],
                item_n38['CD_CAR_NO'],
                item_n38['MD03_DELIVER_USER_CODE'],
                item_n38['MD03_DELIVER_USER_NAME'],
                item_n38['MD03_DRIVING_USER_CODE'],
                item_n38['MD03_DRIVING_USER_NAME'],
                item_n38['MD03_DELIVER_CUST_QTY'],
                item_n38['MD03_DELIVER_TOTAL_QTY'],
                item_n38['MD03_LODING_RATE'],
                item_n38['MD03_ESTIMATED_DIST_DURATION'],
                item_n38['MD03_ESTIMATED_DIST_MILEAGE']
            ))
    db_execute_callback(lambda cursor: cursor.executemany(get_save_n40_sql(), save_n40_data_list))


def get_distance_and_duration(customer_code_list, station, md03_dist_center_code):
    """获取用户和站点间的总距离和总时间
        customer_code_list：     用户编码列表：['140109122550', '140109114395', '140109122783', '140109121588' ...]
        station：                站点编码:   '1401000101'
        md03_dist_center_code：  中心编码    '1401000100'
    """
    # 查出发货点-零售户 表名
    station_cust_table_name = get_distance_table_name(md03_dist_center_code, False)
    # 查出零售户-零售户 表名
    cust_table_name = get_distance_table_name(md03_dist_center_code, True)

    # 计算总距离、时间 start
    total_data_distance = 0.0
    total_data_duration = 0.0

    codes_length = len(customer_code_list)
    for i, customer_code in enumerate(customer_code_list):
        data_distance = ''
        data_duration = ''
        if i == 0:
            data_distance, data_duration = get_cust_station_for_distance_and_duration(station_cust_table_name, md03_dist_center_code, station, customer_code,
                                                                                      True)
        elif i == (codes_length - 1):
            data_distance, data_duration = get_cust_station_for_distance_and_duration(station_cust_table_name, md03_dist_center_code, station, customer_code,
                                                                                      False)
        else:
            data_distance, data_duration = get_cust_between_for_distance_and_duration(cust_table_name, md03_dist_center_code, customer_code_list[i - 1],
                                                                                      customer_code)
        total_data_distance += float(data_distance) if data_distance != "" else 0.0
        total_data_duration += float(data_duration) if data_duration != "" else 0.0
    # 计算总距离、时间 end
    return total_data_distance, total_data_duration


def get_n40_period_list(db_n32):
    # 2. 数据预处理：转换日期格式并排序
    db_n32['MD03_DELIVER_DATE'] = pd.to_datetime(db_n32['MD03_DELIVER_DATE']).dt.date  # 转换为日期（不含时间部分）
    df = db_n32.sort_values('MD03_DELIVER_DATE').reset_index(drop=True)  # 确保按日期升序排列

    # 3. 核心：识别连续日期并分组
    # 计算当前行与前一行的日期差（天数）
    df['day_diff'] = df['MD03_DELIVER_DATE'].diff().dt.days
    # 当日期差不为1时，视为一个新分组的开始，利用累积求和为连续日期段生成相同的组ID
    df['group_id'] = (df['day_diff'] != 1).cumsum()

    # 4. 按组ID聚合，计算每组的统计信息
    continuous_groups = df.groupby('group_id').agg(
        start_date=('MD03_DELIVER_DATE', 'min'),
        end_date=('MD03_DELIVER_DATE', 'max'),
        days_count=('MD03_DELIVER_DATE', 'count'),
        group_members=('MD03_DELIVER_DATE', lambda x: list(x))  # 可选：查看每组具体包含哪些日期
    ).reset_index(drop=True)  # 丢弃group_id列，使结果更整洁

    # print("连续日期分组结果：")
    # print(continuous_groups.to_string(index=False))
    return continuous_groups

def get_n40_period(date_groups, date):
    # target_date_str = '2025-05-09'
    target_date_str = date
    target_date = pd.to_datetime(target_date_str).date()  # 转换为 datetime.date 对象，便于比较

    # 初始化一个变量来记录找到的组，None 表示未找到
    found_group = None
    # 遍历 continuous_groups 的每一行
    for index, row in date_groups.iterrows():
        # 检查目标日期是否在当前行的开始日期和结束日期之间（包含起止日期）
        if row['start_date'] <= target_date <= row['end_date']:
            found_group = row  # 如果找到，记录这一行的信息
            break  # 找到后可以跳出循环

    # 根据查找结果输出信息
    if found_group is not None:
        # print(f"日期 '{target_date_str}' 属于以下分组：")
        # print(f"开始日期: {found_group['start_date']}")
        # print(f"结束日期: {found_group['end_date']}")
        # print(f"持续天数: {found_group['days_count']}")
        # 如果您在分组时计算了 group_id，也可以在这里打印
        # print(f"组ID: {found_group['group_id']}")
        return f'{found_group["start_date"]}-{found_group["end_date"]}'
    else:
        # print(f"日期 '{target_date_str}' 不在任何已知的分组中。")
        return ''

if __name__ == '__main__':
    print()
    # pre_deal_line("29374109947317923", '2', "", "", "29374109947317928,29374109947317929")
    # out_n38_n39("29374109947317921", "1", {'1402000101': {'140105100059', '140105100076'}}, "")

    # CACHE_DIR = 'cache_vrp'
    # tables = load_tables_or_str_local("29374109947317923_out", CACHE_DIR)
    # n34_id = tables.get('n34_id')
    # plan_type = tables.get('plan_type')
    # n38_line_ids_str = tables.get('n38_line_ids_str')
    # routes_dict = tables.get('routes_dict')
    # db_n32_group = tables.get('db_n32_group')
    # out_n38_n39(n34_id, plan_type, n38_line_ids_str, routes_dict, db_n32_group)

    ## 测试add n40
    # if False:
    #     tables = load_tables_or_str_local("add_n40_params")
    #     db_n32_group = tables.get('db_n32_group')
    #     save_n40_data_list_from_n38 = tables.get('save_n40_data_list_from_n38')
    #     md03_region_logi_center_code = tables.get('md03_region_logi_center_code')
    #     md03_dist_center_code = tables.get('md03_dist_center_code')
    # else:
    #     save_tables_or_str_local(
    #         "add_n40_params",
    #         {
    #             'db_n32_group': db_n32_group,
    #             'save_n40_data_list_from_n38': save_n40_data_list_from_n38,
    #             'md03_region_logi_center_code': md03_region_logi_center_code,
    #             'md03_dist_center_code': md03_dist_center_code,
    #         })

    # tables = load_tables_or_str_local("add_n40_params")
    # db_n32_group = tables.get('db_n32_group')
    # save_n40_data_list_from_n38 = tables.get('save_n40_data_list_from_n38')
    # md03_region_logi_center_code = tables.get('md03_region_logi_center_code')
    # md03_dist_center_code = tables.get('md03_dist_center_code')
    # add_n40(db_n32_group, save_n40_data_list_from_n38, md03_region_logi_center_code, md03_dist_center_code)

    customer_codes = ['140109122550', '140109114395', '140109122783', '140109121588', '140108119624', '140105223183', '140109105065', '140109121349', '140109120017', '140109117685', '140109122663', '140109213834', '140122101487', '140109122990', '140108120191', '140121100141', '140108119744', '140109121607', '140108118616', '140109122157', '140108115593', '140105122800', '140108120713', '140109122986', '140109123079', '140109118888', '140108119049', '140109122799', '140105222340', '140109122774', '140121102613', '140109123084', '140108119557', '140109107561', '140121101992', '140108120686', '140109221924', '140109121962', '140121102263', '140108120719', '140109122648', '140109122759', '140105124160', '140109123073', '140109122552', '140108119625', '140106118935', '140109122326', '140106121092', '140108120219', '140109100632', '140109119022', '140109123037', '140181101844', '140105123832', '140108119873', '140109121264', '140105224498', '140105121400', '140105123005', '140121101529', '140121103129', '140122101005', '140105122514', '140109121110', '140105122274', '140105122410', '140121102251', '140109122946', '140123101176', '140108120621', '140108107526', '140181101656', '140109121696', '140109119847', '140105122213', '140109122495', '140109121068', '1401092223']
    station = '1401000101'
    md03_dist_center_code = '1401000100'
    distance, duration = get_distance_and_duration(customer_codes, station, md03_dist_center_code)
    print(f"总距离：{distance}")
    print(f"总时间：{duration}")