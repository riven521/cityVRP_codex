import time

import numpy as np
import pandas as pd
import pymysql
from pymysql.cursors import DictCursor
from typing import List, Dict, Optional, Union

# 定义数据库连接参数（MySQL）
account = "root"
password = "P@ssw0rd12#$"
host = "146.56.245.143"
port = 43306
db_name = "shanxi"
charset = "utf8mb4"


# account = "root"
# password = "zx54258ms"
# host = "192.168.50.195"
# port = 3306
# db_name = "shanxi"
# charset = "utf8mb4"

# account = "tms"  # JDBC配置中的用户名
# password = "Hsrc@2025"  # JDBC配置中的密码
# host = "rm-v9m4527855ot9zr17.mysql.rds.ops.cloud.sx.yc"  # JDBC配置中的主机
# port = 3306
# db_name = "sxyc_dms"  # JDBC配置中的数据库名
# charset = "utf8"  # 根据JDBC配置调整字符集

# 全局数据库连接对象
db_conn = None


def get_connection():
    """获取数据库连接，自动处理连接断开重连"""
    global db_conn
    try:
        # 检查连接是否有效
        if db_conn and db_conn.ping():
            raise ConnectionError("无法获取数据库连接")

        # 连接无效时重新创建连接
        db_conn = pymysql.connect(
            host=host,
            port=port,
            user=account,
            password=password,
            database=db_name,
            charset=charset,
            cursorclass=DictCursor,
            autocommit=False  # 手动控制事务
        )
        return db_conn
    except Exception as e:
        print(f"数据库连接失败: {str(e)}")
        db_conn = None
        raise


def get_db(sql, params=None):
    print(f"执行sql: {str(sql)}")
    conn = get_connection()
    if not conn:
        raise ConnectionError("无法获取数据库连接")
    try:
        start_time = time.time()
        cursor = conn.cursor()
        cursor.execute(sql, params or ())
        raw_result = cursor.fetchall()  # 获取原始数据
        # print(f"查询执行时间: {time.time() - start_time:.2f}秒")
        # print("原始结果示例:", raw_result[:2])  # 打印前两行
        cursor.close()
        # 再通过 pandas 转换
        data = pd.DataFrame(raw_result)
        return data
    except Exception as e:
        print(f"查询失败: {str(e)}")
        raise


def get_db2(sql, params=[], arraysize=100000):
    """执行查询并返回numpy数组结果"""
    conn = get_connection()
    if not conn:
        raise ConnectionError("无法获取数据库连接")

    try:
        start_time = time.time()
        cursor = conn.cursor()
        cursor.arraysize = arraysize
        cursor.execute(sql, params)
        result = cursor.fetchall()
        data = np.array(result)
        cursor.close()
        print(f"查询执行时间: {time.time() - start_time:.2f}秒")
        return data
    except Exception as e:
        print(f"查询执行失败: {str(e)}")
        raise


def db_insert(data, table_name, append=False, columns=[]):
    """批量插入数据到指定表"""
    conn = get_connection()
    if not conn or not data:
        raise ConnectionError("无法获取数据库连接")

    try:
        start_time = time.time()
        # 转换数据格式
        if isinstance(data, np.ndarray):
            data = data.tolist()

        cursor = conn.cursor()

        # 获取表字段名
        if not columns:
            cursor.execute(f"DESCRIBE {table_name}")
            columns_info = cursor.fetchall()
            columns = [col['Field'] for col in columns_info]

            # 移除UPDATE_TIME字段（如果存在）
            if "UPDATE_TIME" in columns:
                columns.remove("UPDATE_TIME")

        # 构造插入SQL
        fields = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))  # MySQL使用%s作为占位符
        sql_inst = f"INSERT INTO {table_name} ({fields}) VALUES ({placeholders})"

        # 如果不需要追加，则先清空表(TW: 删除表数据，太危险了。注释掉)
        # if not append:
        #     cursor.execute(f"TRUNCATE TABLE {table_name}")

        # 批量插入
        cursor.executemany(sql_inst, data)
        conn.commit()
        cursor.close()

        print(f"插入成功，共 {len(data)} 条记录，耗时: {time.time() - start_time:.2f}秒")
        return True
    except Exception as e:
        conn.rollback()
        print(f"插入失败: {str(e)}")
        raise


def db_del(tablename, condition='1=2'):
    """删除指定表中符合条件的记录"""
    conn = get_connection()
    if not conn:
        raise ConnectionError("无法获取数据库连接")

    try:
        start_time = time.time()
        cursor = conn.cursor()
        sql_del = f"DELETE FROM {tablename} WHERE {condition}"
        cursor.execute(sql_del)
        conn.commit()
        cursor.close()
        print(f"删除执行成功，耗时: {time.time() - start_time:.2f}秒")
        return True
    except Exception as e:
        conn.rollback()
        print(f"删除执行失败: {str(e)}")
        raise


def db_update(sql, params=None):
    """执行更新操作"""
    conn = get_connection()
    if not conn:
        raise ConnectionError("无法获取数据库连接")

    try:
        start_time = time.time()
        cursor = conn.cursor()
        from typing import List, Dict, Optional, Union
        cursor.execute(sql, params or ())
        conn.commit()
        cursor.close()
        # print(f"更新执行成功，耗时: {time.time() - start_time:.2f}秒")
        return True
    except Exception as e:
        conn.rollback()
        print(f"更新执行失败: {str(e)}， \n报错sql:{str(sql)}， \n 执行参数：{params}")
        raise


def db_execute_batch(sqls : List[str]):
    """批量执行更新操作"""
    conn = get_connection()
    if not conn:
        raise ConnectionError("无法获取数据库连接")

    err_sql = ""
    try:
        start_time = time.time()
        cursor = conn.cursor()
        for sql in sqls:
            err_sql = sql
            cursor.execute(sql)
        conn.commit()
        cursor.close()
        # print(f"更新执行成功，耗时: {time.time() - start_time:.2f}秒")
        return True
    except Exception as e:
        conn.rollback()
        print(f"更新执行失败: {str(e)}， \n报错sql:{err_sql}")
        raise

def db_execute_batch2(sql : str, data_list: []):
    """批量执行更新操作  适合同一个sql的多数据插入"""
    conn = get_connection()
    if not conn:
        raise ConnectionError("无法获取数据库连接")

    try:
        start_time = time.time()
        cursor = conn.cursor()
        cursor.executemany(sql, data_list)
        conn.commit()
        cursor.close()
        # print(f"更新执行成功，耗时: {time.time() - start_time:.2f}秒")
        return True
    except Exception as e:
        conn.rollback()
        print(f"更新执行失败: {str(e)}， \n报错sql:{sql}")
        raise

def db_execute_callback(function):
    """批量执行更新操作"""
    conn = get_connection()
    if not conn:
        raise ConnectionError("无法获取数据库连接")

    try:
        start_time = time.time()
        cursor = conn.cursor()
        function(cursor)
        conn.commit()
        cursor.close()
        print(f"更新执行成功，耗时: {time.time() - start_time:.2f}秒")
        return True
    except Exception as e:
        conn.rollback()
        print(f"更新执行失败-function: {str(e)}")
        raise


### param = {"age",20,"name":"张三"} ； 搜索条件为: 20岁 名为 张三的数据
def get_common_data(table_name, param=None):  # 初始化SQL和参数
    conditions = []
    values = []
    sql = f"SELECT * FROM {table_name}"

    # 动态拼接WHERE条件
    if param:
        for key, value in param.items():
            conditions.append(f"{key} = %s")  # 占位符
            values.append(str(value))
        sql += " WHERE " + " AND ".join(conditions)
    result_df = get_db(sql, values)
    return result_df

def get_short_uuid():
    conn = get_connection()
    if not conn:
        raise ConnectionError("无法获取数据库连接")
    try:
        start_time = time.time()
        cursor = conn.cursor()
        cursor.execute('SELECT UUID_SHORT() AS uuid')
        raw_result = cursor.fetchone()  # 获取原始数据
        cursor.close()
        # 再通过 pandas 转换
        data = raw_result['uuid']
        return data
    except Exception as e:
        print(f"查询失败: {str(e)}")
        raise


# 程序结束时关闭连接
def close_connection():
    global db_conn
    if db_conn:
        try:
            db_conn.close()
            print("数据库连接已关闭")
        except Exception as e:
            print(f"关闭连接失败: {str(e)}")
            raise
