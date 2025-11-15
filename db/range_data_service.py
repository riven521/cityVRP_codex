from datetime import datetime
from typing import Optional, Tuple

from db.db_conn import get_common_data, db_execute_batch, get_db, get_short_uuid, db_execute_batch2, db_execute_callback, db_update
from db.log_service import add_n35_log_info

""" 范围规划 plan_type 规划类型 1全局规划、2局部规划 """
def pre_deal_range(n34_id: str, plan_type: str, plan_range_id: str) -> Tuple[bool, str, Optional[dict]]:
    """ 插入日志 """
    add_n35_log_info(n34_id, "PDR_0", f"前置业务数据处理-开始-开始获取历史订单、区域规划结果、零售店坐标等相关数据..")

    """查询出所有需要的表数据"""

    """ n34 """
    db_n34 = get_n34_by_id(n34_id)
    n34_data = db_n34.iloc[0]

    if n34_data["MD03_DIST_PLAN_STATE"] == '2':
        return False, f"n34_id={n34_id}, 此数据范围规划状态已完成，请重新输入！", None

    dist_center_code = n34_data['MD03_DIST_CENTER_CODE']  # 配送中心编码
    dist_plan_scheme_code = n34_data['MD03_DIST_PLAN_SCHEME_CODE']  # 配送规划方案编码
    md03_dist_plan_type = n34_data['MD03_DIST_PLAN_TYPE']  # 配送规划规划类型 1全局规划 2局部规划

    """ n19 配送规划类型=(1范围规划、2线路规划)"""
    db_n19 = get_n19_by_play_type_and_center_code(md03_dist_plan_type, dist_center_code)

    db_n20 = None
    if db_n19 is not None and not db_n19.empty:
        alog_ids = []
        for item_n19 in db_n19.itertuples():
            alog_ids.append(item_n19.MD03_DIST_PLAN_ALGO_ID)
        db_n20 = get_n20_by_alog_ids(alog_ids)

    """ n26 """
    db_n26 = get_n26_by_center_code_and_scheme_code(dist_center_code, dist_plan_scheme_code)

    """ n27 """
    db_n27 = get_n27_by_center_code_and_scheme_code(dist_center_code, dist_plan_scheme_code)

    """ n28 """
    if plan_type == '1':
        """全局规划"""
        add_n35_log_info(n34_id, "PDR_1", "前置业务数据处理-范围全局规范数据正在处理中...")
        db_n28 = get_n28_by_center_code_and_scheme_code(dist_center_code, dist_plan_scheme_code)
        customer_codes = []
        for item_n28 in db_n28.itertuples():
            customer_codes.append(item_n28.BB_RETAIL_CUSTOMER_CODE)
    else:
        """局部规划"""
        add_n35_log_info(n34_id, "PDR_1", "前置业务数据处理-范围局部规范正在处理中...")
        plan_range_ids = plan_range_id.split(",")
        customer_codes = []
        for item_range_id in plan_range_ids:
            db_n37 = get_n37_by_id(item_range_id)
            for item_n37 in db_n37.itertuples():
                customer_codes.append(item_n37.BB_RETAIL_CUSTOMER_CODE)
        db_n28 = get_n28_by_center_code_and_scheme_code_and_customer_codes(dist_center_code, dist_plan_scheme_code, customer_codes)

    """ n31 """
    db_n31 = get_n31_play_type_and_by_play_type_and_center_code(md03_dist_plan_type, dist_center_code, dist_plan_scheme_code)

    """ n32 """
    db_n32 = get_n32_by_center_code_and_scheme_code_and_customer_code(dist_center_code, dist_plan_scheme_code, customer_codes)

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
        'db_n33': db_n33
    }
    add_n35_log_info(n34_id, "PDR_2", "前置业务数据处理-结束-历史订单、区域规划结果、零售店坐标等相关数据提取成功")
    return True, "数据取出成功", data_dict


""" 范围规划输出"""
def out_n36_n37(n34_id: str, plan_type: str, data: {}, plan_range_id: str):
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

    """2. 1全局规划 2局部规划"""
    global_type = plan_type == "1"

    """3. 取出发货站点编码，生成主键，存储"""
    save_n36_data_list = []
    save_n37_data_list = []
    # print(f"n36_out_length:{len(data)}" )
    for index1, station_code in enumerate(data):
        # MD03_DIST_PLAN_RANGE_ID 配送规划范围主键
        range_id = get_short_uuid()  # 默认22字符[1,2](@ref)
        print(f"生成的主键UUID: {range_id}")

        md03_shipment_station_code = station_code
        db_n26 = get_n26_by_shipment_station_code(md03_shipment_station_code)  # MD03_SHIPMENT_STATION_CODE 发货站点编码 算法给到
        md03_shipment_station_name = ""  # MD03_SHIPMENT_STATION_NAME 发货站点名称
        if db_n26 is not None:
            n26_data = db_n26.iloc[0]
            md03_shipment_station_name = n26_data["MD03_SHIPMENT_STATION_NAME"]
        # MD03_ISSUE_STATE 发布状态 统一写入0
        # 存 n36
        # save_n36_sql = f"INSERT INTO t_p_ymd03n36_dist_plan_range (`MD03_DIST_PLAN_RANGE_ID`, `MD03_REGION_LOGI_CENTER_CODE`, `MD03_DIST_CENTER_CODE`, `MD03_SHIPMENT_STATION_CODE`, `MD03_SHIPMENT_STATION_NAME`, `MD03_ISSUE_STATE`, `MD03_DIST_PLAN_SCHEME_CODE`, `MD03_DIST_PLAN_SCHEME_NAME`) VALUES ('{range_id}', '{md03_region_logi_center_code}', '{md03_dist_center_code}', '{md03_shipment_station_code}', '{md03_shipment_station_name}', 0, '{md03_dist_plan_scheme_code}', '{md03_dist_plan_scheme_name}')"
        save_n36_data_list.append((range_id, md03_region_logi_center_code, md03_dist_center_code, md03_shipment_station_code, md03_shipment_station_name, md03_dist_plan_scheme_code, md03_dist_plan_scheme_name))

        cust_code_list = data[station_code]
        # 取n28数据
        db_n28_list = get_n28_by_center_code_and_scheme_code_and_customer_codes(md03_dist_center_code, md03_dist_plan_scheme_code, cust_code_list)
        df_n28_indexed = db_n28_list.set_index('BB_RETAIL_CUSTOMER_CODE')  # 将 'id' 列设为索引

        # 服务时长 MD03_SERVICE_DURATION
        db_n23, n31_service_duration = get_service_duration(cust_code_list, md03_dist_center_code)
        db_n23_indexed = None
        if not db_n23.empty:
            db_n23_indexed = db_n23.set_index('BB_RETAIL_CUSTOMER_CODE')

        # 平均订货量 MD03_AVG_ORDER_QTY 取 n32的历史订单
        db_n32_count = get_n32_count_by_center_code_and_scheme_code_and_customer_code(md03_dist_center_code, md03_dist_plan_scheme_code, cust_code_list)
        df_n32_indexed = db_n32_count.set_index('BB_RETAIL_CUSTOMER_CODE')

        """3.1 取出多个零售户编码，生成主键存储"""
        print(f"n37_out_length: {len(data[station_code])}")
        for index2, cust_code in enumerate(data[station_code]):
            # range_detail_id = get_short_uuid()  # 默认22字符[1,2](@ref)

            bb_retail_customer_name = ""
            bb_rtl_cust_business_addr = ""
            md03_retail_cust_lon = ""
            md03_retail_cust_lat = ""
            bb_rtl_cust_artificial_name = ""
            bb_rtl_cust_prov_order_cycle_type = ""
            bb_rtl_cust_order_weekday = ""
            # MD03_SERVICE_DURATION
            if db_n23_indexed is not None and cust_code in db_n23_indexed.index:
                md03_service_duration = db_n23_indexed.loc[cust_code]['MD03_SERVICE_DURATION']
            else:
                md03_service_duration = n31_service_duration

            # 平均订货量 MD03_AVG_ORDER_QTY 取 n32的历史订单
            md03_avg_order_qty = 0
            if df_n32_indexed is not None and cust_code in df_n32_indexed.index:
                md03_avg_order_qty = df_n32_indexed.loc[cust_code]['count'] if df_n32_indexed.loc[cust_code]['count'] is not None else 0

            # 取n28数据
            n28_data = df_n28_indexed.loc[cust_code]
            if n28_data is not None:
                bb_retail_customer_name = n28_data["BB_RETAIL_CUSTOMER_NAME"]
                bb_rtl_cust_business_addr = n28_data["BB_RTL_CUST_BUSINESS_ADDR"]
                md03_retail_cust_lon = n28_data["MD03_RETAIL_CUST_LON"] if n28_data["MD03_RETAIL_CUST_LON"] is not None else 0
                md03_retail_cust_lat = n28_data["MD03_RETAIL_CUST_LAT"] if n28_data["MD03_RETAIL_CUST_LAT"] is not None else 0
                bb_rtl_cust_artificial_name = n28_data["BB_RTL_CUST_ARTIFICIAL_NAME"]
                bb_rtl_cust_prov_order_cycle_type = n28_data["BB_RTL_CUST_PROV_ORDER_CYCLE_TYPE"]
                bb_rtl_cust_order_weekday = n28_data["BB_RTL_CUST_ORDER_WEEKDAY"]

            # 存 n37
            # save_n37_sql = f"INSERT INTO t_p_ymd03n37_dist_plan_range_detail (`MD03_DIST_PLAN_RANGE_DETAIL_ID`, `MD03_DIST_PLAN_RANGE_ID`, `MD03_DIST_CENTER_CODE`, `BB_RETAIL_CUSTOMER_CODE`, `BB_RETAIL_CUSTOMER_NAME`, `BB_RTL_CUST_BUSINESS_ADDR`, `MD03_RETAIL_CUST_LON`, `MD03_RETAIL_CUST_LAT`, `BB_RTL_CUST_ARTIFICIAL_NAME`, `BB_RTL_CUST_PROV_ORDER_CYCLE_TYPE`, `BB_RTL_CUST_ORDER_WEEKDAY`, `MD03_SERVICE_DURATION`, `MD03_AVG_ORDER_QTY`) VALUES (UUID_SHORT(), '{range_id}', '{md03_dist_center_code}', '{cust_code}', '{bb_retail_customer_name}', '{bb_rtl_cust_business_addr}', '{md03_retail_cust_lon}', '{md03_retail_cust_lat}', '{bb_rtl_cust_artificial_name}', '{bb_rtl_cust_prov_order_cycle_type}', '{bb_rtl_cust_order_weekday}', '{md03_service_duration}', '{md03_avg_order_qty}')"
            save_n37_data_list.append((range_id, md03_dist_center_code, cust_code, bb_retail_customer_name, bb_rtl_cust_business_addr, md03_retail_cust_lon, md03_retail_cust_lat, bb_rtl_cust_artificial_name, bb_rtl_cust_prov_order_cycle_type, bb_rtl_cust_order_weekday, md03_service_duration, md03_avg_order_qty))

    """4. 更新n34 数据"""
    """ MD03_DIST_PLAN_STATE  配送规划状态 1进行中，2已完成，3规划失败"""
    update_n34_sql = f"UPDATE t_p_ymd03n34_dist_plan_log SET `MD03_DIST_PLAN_STATE` = 2, `MD03_CALC_END_TIME` = '{formatted_time}' WHERE `MD03_DIST_PLAN_LOG_ID` = '{n34_id}'"

    """5. 获取全部需要修改的sql，执行"""
    save_sqls = []
    save_sqls.append(update_n34_sql)
    """局部规划删除 n36 和 n37"""
    if not global_type:
        add_n35_log_info(n34_id, "ORD_0", "后置业务数据处理-历史局部规划数据删除中...")
        plan_range_ids = plan_range_id.split(",")
        for item_range_id in plan_range_ids:
            del_n36_sql = f"DELETE FROM t_p_ymd03n36_dist_plan_range WHERE MD03_DIST_PLAN_RANGE_ID = '{item_range_id}'"
            del_n37_sql = f"DELETE FROM t_p_ymd03n37_dist_plan_range_detail WHERE MD03_DIST_PLAN_RANGE_ID = '{item_range_id}'"
            save_sqls.append(del_n36_sql)
            save_sqls.append(del_n37_sql)
        add_n35_log_info(n34_id, "ORD_1", "后置业务数据处理-历史局部规划数据删除成功")

    # save_sqls.extend(save_n36_data_list)
    # save_sqls.extend(save_n37_data_list)
    # print(save_sqls)
    add_n35_log_info(n34_id, "OR_1", "后置业务数据处理-范围规划数据处理完毕，正在回写数据库中...")
    # db_execute_batch(save_sqls)
    db_execute_callback(lambda cursor: execute_sql_list(cursor,save_sqls,get_save_n36_sql(),save_n36_data_list, get_save_n37_sql(),save_n37_data_list, md03_dist_center_code, md03_dist_plan_scheme_code))

    add_n35_log_info(n34_id, "OR_2", "后置业务数据处理-结束-范围规划数据回写数据库成功")

def execute_sql_list(cursor, sql_list, n36_add_sql, n36_add_data, n37_add_sql, n37_add_data, md03_dist_center_code, md03_dist_plan_scheme_code):
    # 先清空历史数据
    set_n38_n39_status(md03_dist_center_code, md03_dist_plan_scheme_code)

    # 在执行
    for sql in sql_list:
        cursor.execute(sql)
    cursor.executemany(n36_add_sql,n36_add_data)
    cursor.executemany(n37_add_sql,n37_add_data)


def set_n38_n39_status(dist_center_code: str, dist_plan_scheme_code: str):
    # 1.获取n38数据 通过同个方案
    n36_ids = []
    db_n36 = get_n36_by_center_code_and_scheme_code(dist_center_code,dist_plan_scheme_code)
    if db_n36 is not None and not db_n36.empty:
        for item_n36 in db_n36.itertuples():
            n36_ids.append(item_n36.MD03_DIST_PLAN_RANGE_ID)
        # 1.1 根据ids 更新n38数据状态
        update_n36_state_by_ids(n36_ids, "1")

    # 2.获取n39数据 通过n38主键
    if not n36_ids:
        # 2.1 根据ids 更新n39数据状态
        update_n37_state_by_ids(n36_ids, "1")

""" ======================================================= """


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
def get_n20_by_alog_ids(alog_ids: []):
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

""" 根据 配送中心编码 和 配送规划方案编码 """
def get_n26_by_center_code_and_scheme_code(md03_dist_center_code: str, md03_dist_plan_scheme_code: str):
    param_all = {"MD03_IS_DELETED": 0, "MD03_DIST_CENTER_CODE": md03_dist_center_code, "MD03_DIST_PLAN_SCHEME_CODE": md03_dist_plan_scheme_code}
    res = get_common_data("t_p_ymd03n26_dist_plan_dist_site_info", param_all)
    return res


""" 根据发货站点编码 """
def get_n26_by_shipment_station_code(md03_shipment_station_code: str):
    param_all = {"MD03_IS_DELETED": 0, "MD03_SHIPMENT_STATION_CODE": md03_shipment_station_code}
    res = get_common_data("t_p_ymd03n26_dist_plan_dist_site_info", param_all)
    return res


""" 根据 配送中心编码 和 配送规划方案编码 """
def get_n27_by_center_code_and_scheme_code(md03_dist_center_code: str, md03_dist_plan_scheme_code: str):
    param_all = {"MD03_IS_DELETED": 0, "MD03_DIST_CENTER_CODE": md03_dist_center_code, "MD03_DIST_PLAN_SCHEME_CODE": md03_dist_plan_scheme_code}
    res = get_common_data("t_p_ymd03n27_dist_plan_car_info", param_all)
    return res


""" 根据 配送中心编码 和 配送规划方案编码 """
def get_n28_by_center_code_and_scheme_code(md03_dist_center_code: str, md03_dist_plan_scheme_code: str):
    param_all = {"MD03_IS_DELETED": 0, "MD03_DIST_CENTER_CODE": md03_dist_center_code, "MD03_DIST_PLAN_SCHEME_CODE": md03_dist_plan_scheme_code}
    res = get_common_data("t_p_ymd03n28_dist_plan_retail_cust", param_all)
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


def get_n28_by_bb_retail_customer_code(bb_retail_customer_code: str):
    param_all = {"MD03_IS_DELETED": 0, "BB_RETAIL_CUSTOMER_CODE": bb_retail_customer_code}
    res = get_common_data("t_p_ymd03n28_dist_plan_retail_cust", param_all)
    return res


# def get_n31_by_play_type_and_by_play_type_and_center_code_and_station_codes(d03_dist_plan_type: int, md03_dist_center_code: str, md03_dist_plan_scheme_code: str, station_codes: []):
#     station_codes_str = ", ".join(f"'{code}'" for code in station_codes)
#     sql = f"""
#                    SELECT
#                        *
#                    FROM
#                        `t_p_ymd03n31_dist_plan_param_set`
#                    WHERE
#                        MD03_IS_DELETED = '0'
#                        AND MD03_DIST_PLAN_TYPE = {d03_dist_plan_type}
#                        AND MD03_DIST_CENTER_CODE = '{md03_dist_center_code}'
#                        AND MD03_DIST_PLAN_SCHEME_CODE = '{md03_dist_plan_scheme_code}'
#                        AND MD03_DIST_STATION_CODE IN ({station_codes_str})
#                """
#     res = get_db(sql)
#     return res

""" 通过 配送规划类型=(1范围规划、2线路规划) 和 配送中心编码 """
def get_n31_play_type_and_by_play_type_and_center_code(md03_dist_plan_type: int, md03_dist_center_code: str, md03_dist_plan_scheme_code: str):
    param_all = {"MD03_IS_DELETED": 0, "MD03_DIST_PLAN_TYPE": md03_dist_plan_type,
                 "MD03_DIST_CENTER_CODE": md03_dist_center_code, "MD03_DIST_PLAN_SCHEME_CODE": md03_dist_plan_scheme_code}
    res = get_common_data("t_p_ymd03n31_dist_plan_param_set", param_all)
    return res


""" 根据 配送中心编码 和 配送规划方案编码 """
def get_n32_by_center_code_and_scheme_code_and_customer_code(md03_dist_center_code: str, md03_dist_plan_scheme_code: str, bb_retail_customer_codes: []):
    customer_codes_str = ", ".join(f"'{code}'" for code in bb_retail_customer_codes)
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
        """
    res = get_db(sql)
    return res


def get_n32_count_by_center_code_and_scheme_code_and_customer_code(md03_dist_center_code: str, md03_dist_plan_scheme_code: str, bb_retail_customer_codes: str):
    customer_codes_str = ", ".join(f"'{code}'" for code in bb_retail_customer_codes)
    sql = f"""SELECT
                    ROUND(AVG(MD03_ORDER_TOTAL_QTY * 1.0), 2) AS count,
                    BB_RETAIL_CUSTOMER_CODE
                FROM
                    `t_p_ymd03n32_dist_plan_order` 
                WHERE
                    MD03_IS_DELETED = '0' 
                    AND MD03_DIST_CENTER_CODE = '{md03_dist_center_code}' 
                    AND MD03_DIST_PLAN_SCHEME_CODE = '{md03_dist_plan_scheme_code}' 
                    AND BB_RETAIL_CUSTOMER_CODE IN ({customer_codes_str})
                GROUP BY
                    BB_RETAIL_CUSTOMER_CODE 
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


def get_n36_by_id(n36_id: str):
    param_all = {"MD03_IS_DELETED": 0, "MD03_DIST_PLAN_RANGE_ID": n36_id}
    res = get_common_data("t_p_ymd03n36_dist_plan_range", param_all)
    return res

def get_n36_by_center_code_and_scheme_code(md03_dist_center_code: str, md03_dist_plan_scheme_code: str):
    param_all = {"MD03_IS_DELETED": 0, "MD03_DIST_CENTER_CODE": md03_dist_center_code, "MD03_DIST_PLAN_SCHEME_CODE": md03_dist_plan_scheme_code}
    res = get_common_data("t_p_ymd03n36_dist_plan_range", param_all)
    return res

def get_n37_by_id(n36_id: str):
    param_all = {"MD03_IS_DELETED": 0, "MD03_DIST_PLAN_RANGE_ID": n36_id}
    res = get_common_data("t_p_ymd03n37_dist_plan_range_detail", param_all)
    return res


def get_save_n36_sql():
    # data_list = [('{range_id}', '{md03_region_logi_center_code}', '{md03_dist_center_code}', '{md03_shipment_station_code}', '{md03_shipment_station_name}', 0, '{md03_dist_plan_scheme_code}', '{md03_dist_plan_scheme_name}')]
    return "INSERT INTO t_p_ymd03n36_dist_plan_range (`MD03_DIST_PLAN_RANGE_ID`, `MD03_REGION_LOGI_CENTER_CODE`, `MD03_DIST_CENTER_CODE`, `MD03_SHIPMENT_STATION_CODE`, `MD03_SHIPMENT_STATION_NAME`, `MD03_ISSUE_STATE`, `MD03_DIST_PLAN_SCHEME_CODE`, `MD03_DIST_PLAN_SCHEME_NAME`) VALUES (%s, %s, %s, %s, %s, 0, %s, %s)"

def get_save_n37_sql():
    return "INSERT INTO t_p_ymd03n37_dist_plan_range_detail (`MD03_DIST_PLAN_RANGE_DETAIL_ID`, `MD03_DIST_PLAN_RANGE_ID`, `MD03_DIST_CENTER_CODE`, `BB_RETAIL_CUSTOMER_CODE`, `BB_RETAIL_CUSTOMER_NAME`, `BB_RTL_CUST_BUSINESS_ADDR`, `MD03_RETAIL_CUST_LON`, `MD03_RETAIL_CUST_LAT`, `BB_RTL_CUST_ARTIFICIAL_NAME`, `BB_RTL_CUST_PROV_ORDER_CYCLE_TYPE`, `BB_RTL_CUST_ORDER_WEEKDAY`, `MD03_SERVICE_DURATION`, `MD03_AVG_ORDER_QTY`) VALUES (UUID_SHORT(), %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s)"

def update_n36_state_by_ids(n36_ids:[], status: str):
    ids_str = ", ".join(f"'{code}'" for code in n36_ids)
    sql = f"""
                        UPDATE
                            `t_p_ymd03n36_dist_plan_range` 
                        SET
                            MD03_IS_DELETED = '{status}'
                        WHERE
                            MD03_IS_DELETED = '0' 
                            AND MD03_DIST_PLAN_RANGE_ID IN ({ids_str})
                    """
    res = db_update(sql)

def update_n37_state_by_ids(n36_ids:[], status: str):
    ids_str = ", ".join(f"'{code}'" for code in n36_ids)
    sql = f"""
                        UPDATE
                            `t_p_ymd03n37_dist_plan_range_detail` 
                        SET
                            MD03_IS_DELETED = '{status}'
                        WHERE
                            MD03_IS_DELETED = '0' 
                            AND MD03_DIST_PLAN_RANGE_ID IN ({ids_str})
                    """
    res = db_update(sql)


if __name__ == '__main__':
    # pre_deal_range(29374109947317921,'1',"1")
    # print(pre_deal_range)
    # out_n36_n37("29374109947317921", "1", {'1402000101': {'140105100059', '140105100076'}}, "")

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    try:
        add_n35_log_info("29374109947317923", "1211", "测测测测测是是是是")
    except Exception as e:
        logger.error(f"Unhandled exception in call_range: {e}")

