import mysql.connector
from mysql.connector import Error



class MySQLClient:
    def __init__(self, host, database, user, password):
        self.connection = None
        try:
            self.connection = mysql.connector.connect(
                host=host,
                database=database,
                user=user,
                password=password
            )
            if self.connection.is_connected():
                print("Successfully connected to MySQL database")
        except Error as e:
            print(f"Error: {e}")


    def create_table(self, table_name, columns):
        cursor = self.connection.cursor()
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
        cursor.execute(query)
        self.connection.commit()
        print(f"Table '{table_name}' created or already exists")


    def insert_data(self, table_name, data):
        cursor = self.connection.cursor()
        placeholders = ', '.join(['%s'] * len(data))
        columns = ', '.join(data.keys())
        values = tuple(data.values())
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        cursor.execute(query, values)
        self.connection.commit()
        print(f"Inserted into '{table_name}': {data}")


    def select_data(self, table_name, where_clause=None):
        cursor = self.connection.cursor(dictionary=True)
        query = f"SELECT * FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        cursor.execute(query)
        rows = cursor.fetchall()
        return rows


    def update_data(self, table_name, set_clause, where_clause):
        cursor = self.connection.cursor()
        query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
        cursor.execute(query)
        self.connection.commit()
        print(f"Updated in '{table_name}' with set: {set_clause}, where: {where_clause}")


    def delete_data(self, table_name, where_clause):
        cursor = self.connection.cursor()
        query = f"DELETE FROM {table_name} WHERE {where_clause}"
        cursor.execute(query)
        self.connection.commit()
        print(f"Deleted from '{table_name}' where: {where_clause}")

    def close(self):
        if self.connection.is_connected():
            self.connection.close()
            print("MySQL connection is closed")

# 使用示例
if __name__ == "__main__":
    db = MySQLClient(host='localhost', database='test_db', user='root', password='password')

    # 创建表
    db.create_table('users', 'id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), age INT')

    # 插入数据
    db.insert_data('users', {'name': 'Alice', 'age': 30})
    db.insert_data('users', {'name': 'Bob', 'age': 25})

    # 查询数据
    users = db.select_data('users')
    for user in users:
        print(user)

    # 更新数据
    db.update_data('users', 'age = 31', 'name = "Alice"')

    # 删除数据
    db.delete_data('users', 'name = "Bob"')

    # 关闭连接
    db.close()