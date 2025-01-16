import pymysql
def get_db_connection():
    return pymysql.connect(
        host='3.39.222.135',
        port=3306,
        user='shcoding',
        password='1234',
        database='my_db',
        charset='utf8mb4'
    )

# def get_db_connection():
#     return pymysql.connect(
#         host='127.0.0.1',
#         port=3306,
#         user='root',
#         password='whtjdgh2367@',
#         database='my_db',
#         charset='utf8mb4'
#     )

# def get_db_connection():
#     return pymysql.connect(
#         host='3.36.28.140',
#         port=3306,
#         user='jmcoding',
#         password='123qwe!',
#         database='my_db',
#     )