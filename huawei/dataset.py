# coding=utf-8

import pymysql as MySQLdb  # 这里是python3  如果你是python2.x的话，import MySQLdb

host = 'localhost'
user = 'root'
passwd = 'root'
port = 3306
db = 'huawei'


class SelectMySQL(object):
    def select_data(self, sql):
        result = []
        try:
            conn = MySQLdb.connect(host=host,
                                   port=port,
                                   user=user,
                                   passwd=passwd,
                                   db=db,
                                   charset='utf8', )
            cur = conn.cursor()
            cur.execute(sql)
            alldata = cur.fetchall()
            # print(alldata)
            for rec in alldata:
                result.append(rec[0])  # 注意，我这里只是把查询出来的第一列数据保存到结果中了,如果是多列的话，稍微修改下就ok了
        except Exception as e:
            print('Error msg: ' + e)
        finally:
            cur.close()
            conn.close()

        return result

    def get_result(self, sql, filename):
        print(sql)
        results = self.select_data(sql)
        print('The amount of datas: %d' % (len(results)))
        with open(filename, 'w',encoding='utf-8') as f:
            for result in results:
                f.write(str(result) + '\n')
        print('Data write is over!')
        return results


if __name__ == '__main__':
    sql = "select post_content from weibo_post"
    select = SelectMySQL()
    result1 = select.get_result(sql, 'namemsg.txt')
    # print(result1)