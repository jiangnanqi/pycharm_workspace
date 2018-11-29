import nltk
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB #多项式分布、伯努利分布、高斯分布
#一般来说，如果样本特征的分布大部分是连续值，使用GaussianNB会比较好。如果如果样本特征的分大部分是多元离散值，使用MultinomialNB比较合适。而如果样本特征是二元离散值或者很稀疏的多元离散值，应该使用BernoulliNB
import matplotlib.pyplot as plt


def TextProcessing(test_size=0.2):
    data_list = []
    class_list = []

    with open('label.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            class_list.append(line.split())
    with open('result_fenci.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data_list.append(list(line.split(' ')))
    data_list = data_list[0:500]

    ## 划分训练集和测试集
    index = int(len(data_list) * test_size) + 1
    print(index)
    # train_data_list = data_list[index:]
    # train_class_list = class_list[index:]
    # test_data_list = data_list[:index]
    # test_class_list = class_list[:index]

    test_data_list = data_list[index:]
    test_class_list = class_list[index:]
    train_data_list = data_list[:index]
    train_class_list = class_list[:index]

    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # key函数利用词频进行降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)  # 内建函数sorted参数需为list
    with open('all_word_list1.txt','w',encoding='utf-8') as f:
        for words in all_words_tuple_list:
            if not words[0].isdigit() and words[0] and 1 < len(words[0]) < 10 and words[0] != '\u200b\n':
                f.write(str(words[0])+' '+str(words[1])+'\n')

    all_words_list = []
    print(all_words_tuple_list)

    for item in all_words_tuple_list:
        all_words_list.append(item[0])

    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def words_dict(all_words_list, deleteN):
    # 选取特征词
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:  # feature_words的维度1000
            break
        # print all_words_list[t]
        if not all_words_list[t].isdigit() and all_words_list[t] and 1 < len(all_words_list[t]) < 10 and all_words_list[t]!='\u200b\n':
            feature_words.append(all_words_list[t])
            n += 1
    # print(feature_words)
    return feature_words


def TextFeatures(train_data_list, test_data_list, feature_words, flag='nltk'):
    def text_features(text, feature_words):
        text_words = set(text)

        if flag == 'nltk':
            ## nltk特征 dict
            features = {word: 1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            ## sklearn特征 list
            features = [1 if word in text_words else 0 for word in feature_words]
        else:
            features = []

        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag='nltk'):
    ## -----------------------------------------------------------------------------------
    if flag == 'nltk':
        ## nltk分类器
        train_flist = zip(train_feature_list, train_class_list)
        test_flist = zip(test_feature_list, test_class_list)
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        # print classifier.classify_many(test_feature_list)
        # for test_feature in test_feature_list:
        #     print classifier.classify(test_feature),
        # print ''
        test_accuracy = nltk.classify.accuracy(classifier, test_flist)
    elif flag == 'sklearn':
        ## sklearn分类器
        classifier = BernoulliNB().fit(train_feature_list, train_class_list)
        # print classifier.predict(test_feature_list)
        # for test_feature in test_feature_list:
        #     print classifier.predict(test_feature)[0],
        # print ''
        test_accuracy = classifier.score(test_feature_list, test_class_list)
    else:
        test_accuracy = []
    return test_accuracy


if __name__ == '__main__':
    print("start")
    ## 文本预处理
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(test_size=0.8)
    # print(all_words_list)
    ## 文本特征提取和分类
    # flag = 'nltk'
    flag = 'sklearn'
    deleteNs = range(0, 1000, 1)
    test_accuracy_list = []
    for deleteN in deleteNs:
        # feature_words = words_dict(all_words_list, deleteN)
        feature_words = words_dict(all_words_list, deleteN)
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words, flag)
        test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag)
        test_accuracy_list.append(test_accuracy)
    print(test_accuracy_list)
    print(len(test_accuracy_list))

    # 结果评价
    plt.figure()
    plt.plot(deleteNs, test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.savefig('result2.png')
    plt.show()
    print("finished")
