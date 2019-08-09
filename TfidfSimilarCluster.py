from gensim import corpora, models, similarities
from jieba_fast import cut_for_search
'''
通过gensim的similarities 包，实现基于tfidf 的短文本聚类
'''

def readData():
    data = []
    filename = "/data/stargazer/t_brand_info.csv";
    with open(filename,encoding='utf-8',mode='r') as f:
        for line in f:
            data.append(line.split(",")[1].strip())

    return data

if __name__ == '__main__':

    data = readData()
    texts = [list(cut_for_search(document)) for document in data]

    # # 2.计算词频
    # frequency = defaultdict(int)  # 构建一个字典对象
    # # 遍历分词后的结果集，计算每个词出现的频率
    # for text in texts:
    #     for token in text:
    #         frequency[token] += 1

    # 3.创建字典（单词与编号之间的映射）
    dictionary = corpora.Dictionary(texts)
    # 4。建立语料库
    corpus = [dictionary.doc2bow(text) for text in texts]
    # 6.初始化模型
    # 初始化一个tfidf模型,可以用它来转换向量（词袋整数计数）表示方法为新的表示方法（Tfidf 实数权重）
    tfidf = models.TfidfModel(corpus)

    # 将整个语料库转为tfidf表示方法
    corpus_tfidf = tfidf[corpus]

    # 7.创建索引，
    # 此处可以将模型保存起来。这样下次就不需要再训练了
    index = similarities.MatrixSimilarity(corpus_tfidf)

    # 8.相似度计算
    # new_doc = contents[0][0]  # 假定用contents的第1篇文章对比，由于contents每个元素由id和content组成，所以是contents[0][0]
    # new_vec = dic.doc2bow(tokenization(new_doc))

    saveFilename = "/data/stargazer/simRes_matrix.csv"
    record = [False for i in range(len(data))]
    with open(saveFilename, encoding='utf-8', mode='a') as f:
        no = 1
        for i,vec in enumerate(corpus):
            if record[i] is True: continue
            idf = tfidf[vec]
            sims = list(enumerate(index[idf]))
            sorted(sims,key=lambda x:x[1],reverse=True)
            res = list(filter(lambda x:x[0] != i and x[1]>0.3 ,sims))
            f.write(str.format("{0},{1},{2}\n", no, data[i], 1 if i > 24977 else 0))
            record[i] = True
            for id,simValue in res:
                if record[id] is True: continue
                record[id] = True
                f.write(str.format("{0},{1},{2}\n", no, data[id], 1 if id > 24977 else 0))
            no = no + 1



    # new_vec_tfidf_ls = [tfidf[new_vec] for new_vec in new_vecs]  # 将要比较文档转换为tfidf表示方法
    #
    # # 计算要比较的文档与语料库中每篇文档的相似度
    # sims = [index[new_vec_tfidf] for new_vec_tfidf in new_vec_tfidf_ls]
    # print(sims)




