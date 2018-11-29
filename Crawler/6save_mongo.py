from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
import datetime

client = MongoClient('localhost',27017)
db = client.blog_database

collection = db.blog

link = "http://www.santostang.com/"
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
r = requests.get(link,headers=headers)
soup = BeautifulSoup(r.text,"lxml")
title_list = soup.findAll("h1",class_="post-title")
for eachone in title_list:
    url = eachone.a['href']
    title = eachone.a.text.strip()
    post = {"url":url,
            "title":title,
            "date":datetime.datetime.utcnow()}
    collection.insert_one(post)