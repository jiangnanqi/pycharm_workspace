import jieba


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r',encoding='utf-8').readlines()]
    return stopwords

def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('stopword.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr.strip()



with open('namemsg.txt','r',encoding='utf-8') as file:
    with open('result_fenci.txt','w',encoding='utf-8') as writefile:
        for line in file.readlines():
            line_seg = seg_sentence(line)
            writefile.write(line_seg+'\n')
