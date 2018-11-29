import nltk

with open('news.txt','r',encoding='utf-8') as f:
    text = f.read()  #读取文件
    tokens = nltk.word_tokenize(text)  #分词
    tagged = nltk.pos_tag(tokens)  #词性标注
    entities = nltk.chunk.ne_chunk(tagged)  #命名实体识别
    a1=str(entities) #将文件转换为字符串
    with open('out.txt','w',encoding='utf-8') as w:
        w.write(a1)   #写入到文件中