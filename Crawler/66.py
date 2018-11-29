import requests
from bs4 import BeautifulSoup
import datetime

def get_page(link):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
    r = requests.get(link,headers=headers)
    html = r.content
    html = html.decode('utf-8')
    soup = BeautifulSoup(html,"lxml")
    return soup

def get_data(post_list):
    # data_list = []
    for post in post_list:
        title_td = post.find("div",class_="titlelink.box")
        # title = title_td.a.text.strip()
        # print(title)

link = "https://bbs.hupu.com/bxj"
soup = get_page(link)
post_list = soup.findAll("li")
get_page(post_list)