from openpyxl.pivot.fields import Boolean
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np

from db_conn import db_insert, get_common_data


def get_dist_plan_dist_site_info(param=None) -> DataFrame:
    res = get_common_data("t_p_ymd03n26_dist_plan_dist_site_info", param)
    return res


def get_dist_plan_car_info(param=None) -> DataFrame:
    res = get_common_data("t_p_ymd03n27_dist_plan_car_info", param)
    return res


def get_dist_plan_retail_cust(param=None):
    res = get_common_data("t_p_ymd03n28_dist_plan_retail_cust", param)
    return res


def get_dist_plan_param_set(param=None) -> DataFrame:
    res = get_common_data("t_p_ymd03n31_dist_plan_param_set", param)
    return res


def get_dist_plan_order(param=None) -> DataFrame:
    res = get_common_data("t_p_ymd03n32_dist_plan_order", param)
    return res


def get_dist_plan_balance_mode(param=None) -> DataFrame:
    res = get_common_data("t_p_ymd03n33_dist_plan_balance_mode", param)
    return res


def add_dist_plan_range(df: DataFrame) -> Boolean:
    columns = df.columns.tolist()
    data_to_insert = [tuple(row) for row in df.values]
    table_name = "t_p_ymd03n36_dist_plan_range"
    append = False
    result = db_insert(data_to_insert, table_name, append, columns)
    return result


def add_dist_plan_range_detail(df: DataFrame) -> Boolean:
    columns = df.columns.tolist()
    data_to_insert = [tuple(row) for row in df.values]
    table_name = "t_p_ymd03n37_dist_plan_range_detail"
    append = False
    result = db_insert(data_to_insert, table_name, append, columns)
    return result

def add_dist_38_plan_line(df: DataFrame) -> Boolean:
    columns = df.columns.tolist()
    data_to_insert = [tuple(row) for row in df.values]
    table_name = "t_p_ymd03n38_dist_plan_line"
    append = False
    result = db_insert(data_to_insert, table_name, append, columns)
    return result

def add_dist_39_plan_line_detail(df: DataFrame) -> Boolean:
    columns = df.columns.tolist()
    data_to_insert = [tuple(row) for row in df.values]
    table_name = "t_p_ymd03n39_dist_plan_line_detail"
    append = False
    result = db_insert(data_to_insert, table_name, append, columns)
    return result


def add_dist_40_plan_result_stat(df: DataFrame) -> Boolean:
    columns = df.columns.tolist()
    data_to_insert = [tuple(row) for row in df.values]
    table_name = "t_p_ymd03n40_dist_plan_result_stat"
    append = False
    result = db_insert(data_to_insert, table_name, append, columns)
    return result



if __name__ == '__main__':
    # 定义要查询的表名
    table_name = "t_p_ymd03n33_dist_plan_balance_mode"

    # 编写SQL查询语句，包含条件MD03_IS_DELETED='0'
    sql = f"""
        SELECT *
        FROM {table_name} 
        WHERE MD03_IS_DELETED = %s
    """

    # 1. 查询数据
    ### param = {"age",20,"name":"张三"} ； 搜索条件为: age=20岁 and name="张三" 的所有数据
    param = None
    # result_df = get_dist_plan_dist_site_info(param)
    # result_df = get_dist_plan_car_info(param)
    # result_df = get_dist_plan_retail_cust(param)
    # result_df = get_dist_plan_param_set(param)
    # result_df = get_dist_plan_order(param)
    # result_df = get_dist_plan_balance_mode(param)
    # result_df = add_dist_plan_line_detail(param)

    # 2.插入数据
    # data = {"MD03_DIST_PLAN_LINE_DETAIL_ID":[122,223],"MD03_DIST_CENTER_CODE":[1402000100,1402000100],"MD03_DIST_PLAN_LINE_ID":[0, np.nan]}
    # df = pd.DataFrame(data)
    #  result_df = add_dist_plan_line_detail(df) #result_df=true 插入成功 否则失败
