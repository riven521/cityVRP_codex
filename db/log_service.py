from db.db_conn import  get_common_data, db_execute_batch, get_short_uuid
from datetime import datetime

""" 添加记录 """
def add_n35_log_info(n34_id, node_code, node_name):
    db_n34 = get_n34_by_id(n34_id)
    if db_n34 is not None:
        n34_data = db_n34.iloc[0]
        center_code = n34_data["MD03_DIST_CENTER_CODE"]
        log_detail_id = get_short_uuid()  # 默认22字符[1,2](@ref)
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        del_n37_sql = f"""
                            INSERT INTO t_p_ymd03n35_dist_plan_log_detail ( `MD03_DIST_PLAN_LOG_DETAIL_ID`, `MD03_DIST_PLAN_LOG_ID`, `MD03_DIST_CENTER_CODE`, `MD03_NODE_CODE`, `MD03_NODE_NAME`, `MD03_EXEC_TIME` )
                                VALUES
                                    ( '{log_detail_id}', '{n34_id}', '{center_code}', '{node_code}', '{node_name}', '{formatted_time}' )
                        """
        db_execute_batch([del_n37_sql])


""" 删除记录 """
def del_n35_log_info(n34_id):
    if n34_id is not None:
        del_n37_sql = f"""
                                    DELETE FROM `t_p_ymd03n35_dist_plan_log_detail` WHERE MD03_DIST_PLAN_LOG_ID = '{n34_id}'
                                """
        db_execute_batch([del_n37_sql])


def get_n34_by_id(n34_id: str):
    param_all = {"MD03_IS_DELETED": 0, "MD03_DIST_PLAN_LOG_ID": n34_id}
    res = get_common_data("t_p_ymd03n34_dist_plan_log", param_all)
    return res

def set_n34_fault(n34_id: str):
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    sql = f"UPDATE t_p_ymd03n34_dist_plan_log SET `MD03_DIST_PLAN_STATE` = 3, `MD03_CALC_END_TIME` = '{formatted_time}' WHERE `MD03_DIST_PLAN_LOG_ID` = '{n34_id}'"
    db_execute_batch([sql])