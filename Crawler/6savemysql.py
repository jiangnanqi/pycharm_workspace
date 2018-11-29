import MySQLdb
from bs4 import BeautifulSoup
import requests


conn = MySQLdb.connect(host='localhost',user='root',passwd='root',db='scraping',charset='utf8')
cur = conn.cursor()

link = "http://www.santostang.com/"
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
r = requests.get(link,headers=headers)
soup = BeautifulSoup(r.text,"lxml")
title_list = soup.findAll("h1",class_="post-title")
for eachone in title_list:
    url = eachone.a['href']
    title = eachone.a.text.strip()
    print(url+"     "+title)
    cur.execute("insert into urls(url,content) values (%s,%s)",(url,title))

cur.close()
conn.commit()
conn.close()