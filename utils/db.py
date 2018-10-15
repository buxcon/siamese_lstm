import psycopg2
from config import DbConfig


class Db:
    def __init__(self):
        self.conn = psycopg2.connect(host=DbConfig.HOST,
                                     port=DbConfig.PORT,
                                     user=DbConfig.USER,
                                     password=DbConfig.PASS,
                                     database=DbConfig.DB)
        self.cursor = self.conn.cursor()

    def __del__(self):
        self.cursor.close()
        self.conn.close()

    def select(self, cols):
        sql = "SELECT "
        for col in cols:
            sql = "%s%s," % (sql, col)
        sql = "%s FROM %s" % (sql[:-1], DbConfig.TABLE)

        self.cursor.execute(sql)
        return self.cursor.fetchall()
