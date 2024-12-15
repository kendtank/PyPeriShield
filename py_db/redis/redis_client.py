# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/6 下午8:42
@Author  : Kend
@FileName: redis.py
@Software: PyCharm
@modifier:
"""

import redis
from typing import Any, Optional



class RedisClient:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, password: Optional[str] = None):
        """
        初始化Redis客户端。

        :param host: Redis服务器地址，默认为'localhost'
        :param port: Redis服务器端口，默认为6379
        :param db: Redis数据库编号，默认为0
        :param password: Redis密码，若无则为None
        """
        self.client = redis.StrictRedis(host=host, port=port, db=db, password=password, decode_responses=True)

    def set_data(self, key: str, value: Any) -> bool:
        """
        向Redis中插入或更新键值对。

        :param key: 键名
        :param value: 值
        :return: 操作成功返回True，失败返回False
        """
        try:
            return self.client.set(key, value) is not None
        except Exception as e:
            print(f"Error setting data: {e}")
            return False

    def get_data(self, key: str) -> Optional[Any]:
        """
        从Redis中获取指定键的值。

        :param key: 键名
        :return: 返回键对应的值，若键不存在则返回None
        """
        try:
            return self.client.get(key)
        except Exception as e:
            print(f"Error getting data: {e}")
            return None

    def delete_data(self, key: str) -> int:
        """
        从Redis中删除指定键。

        :param key: 键名
        :return: 删除成功的键数量
        """
        try:
            return self.client.delete(key)
        except Exception as e:
            print(f"Error deleting data: {e}")
            return 0

    def update_data(self, key: str, value: Any) -> bool:
        """
        更新Redis中已存在的键的值。如果键不存在，则相当于插入新键值对。

        :param key: 键名
        :param value: 新的值
        :return: 操作成功返回True，失败返回False
        """
        # Note: In Redis, the SET command can be used to insert or update a key.
        # If the key exists, it will be updated; if not, it will be created.
        return self.set_data(key, value)



# 使用示例
if __name__ == "__main__":
    # 创建RedisClient实例
    rc = RedisClient(host='127.0.0.1', port=6379, db=0, password=None)
    # 插入数据
    print("Setting data:", rc.set_data('test_key', 'Hello, Redis!'))
    # 获取数据
    print("Getting data:", rc.get_data('test_key'))
    # 更新数据
    print("Updating data:", rc.update_data('test_key', 'Updated Value'))
    # 再次获取数据以验证更新
    print("Getting updated data:", rc.get_data('test_key'))
    # 删除数据
    print("Deleting data:", rc.delete_data('test_key'))
    # 验证删除
    print("Checking deleted data:", rc.get_data('test_key'))