import requests
import urllib.request
import urllib.parse
import time
import os

import json

class Img:
    def __init__(self):#初始化函数
        #我们将要访问的url        {}是用于接受参数的     当前一次 json 数据 有30 条  ,
        # self.temp_url="http://image.so.com/zj?ch=beauty&sn={}&listtype=new&temp=1"
        self.temp_url="http://image.so.com/j?q=%E9%92%B3%E5%AD%90&src=srp&correct=%E9%92%B3%E5%AD%90&pn=60&ch=&sn=1"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.62 Safari/537.36"
            , "Connection": "keep-alive"}
        self.num=0
    def get_img_list(self,url): # 获取 存放 图片url 的集合
        response = requests.get(url, headers=self.headers)

        html_str = response.content.decode()
        print(html_str)
        json_str = json.loads(html_str)
        img_str_list = json_str["list"]
        img_list = []
        for img_object in img_str_list:
            img_list.append({'url':img_object['img'],'index':img_object['index']})
        return  img_list

    def save_img_list(self,img_list):
        for img in img_list:
            self.save_img(img)

    # @retry(stop_max_attempt_number=3)#当保存图片出现异常的时候  就需要用retry   进行回滚  , 再次 保存当前图片 stop_max_attempt_number   重试的次数
    def save_img(self,img):#j对获取的 图片url进行下载 保存到本地
        #print(img['index'])
        #print(img['url'])
        f = open("/home/qjn/Desktop/picture/" + str(img['index']) + ".jpg", "wb")
        try:
            f.write((urllib.request.urlopen(img['url'],timeout=0.5)).read())
            print(str(img['index']) + "保存成功")
        except BaseException:
            print(str(img['index']) + "保存失败")
            os.remove("/home/qjn/Desktop/picture/" + str(img['index']) + ".jpg")
        # time.sleep(10)


    def run(self):#实现主要逻辑
        total=1500
        while self.num<=total:
            #1获取链接
            # url=self.temp_url.format(self.num)
            stringname = "扳手"
            stringnameofurl = urllib.parse.quote(stringname)

            start  = self.num*60
            # stop = (self.num+1)*60
            url = 'http://image.so.com/j?q='+stringnameofurl+'&src=srp&correct=%E9%92%B3%E5%AD%90&pn=60&ch=&sn='+str(start)
            #获取数据
            img_list = self.get_img_list(url)
            #保存数据
            self.save_img_list(img_list)
            #不要获取数据过于频繁
            # time.sleep(10)
            print("先休息一会")
            self.num +=1


if __name__ == '__main__':
    img=Img()
    img.run()

