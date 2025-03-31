#词向量
import pandas as pd
import jieba
from gensim.models.word2vec import Word2Vec

# 读入训练集文件
data = pd.read_csv('train.csv')
# 转字符串数组
corpus = data['comment'].values.astype(str)
# 分词，再重组为字符串数组
corpus = [jieba.lcut(corpus[index]
                          .replace("，", "")
                          .replace("!", "")
                          .replace("！", "")
                          .replace("。", "")
                          .replace("~", "")
                          .replace("；", "")
                          .replace("？", "")
                          .replace("?", "")
                          .replace("【", "")
                          .replace("】", "")
                          .replace("#", "")
                        ) for index in range(len(corpus))]
# 词向量模型训练
model = Word2Vec(corpus, sg=0, vector_size=300, window=5, min_count=3, workers=4)
#模型显示
print('模型参数：',model,'\n')
#最匹配
print('最匹配的词是：',model.wv.most_similar(positive=['点赞', '不错'], negative=['难吃']),'\n')
#最不匹配
#print('最不匹配的词是：',model.wv.doesnt_match("点赞 好吃 支持 难吃".split()),'\n')
#语义相似度
print('相似度为=',model.wv.similarity('推荐','好吃'),'\n')
#坐标返回
print(model.wv.__getitem__('地道'))

# 使用Skip-Gram训练Word2Vec模型 (sg=1表示Skip-Gram)
sg_model = Word2Vec(corpus, sg=1, vector_size=300, window=5, min_count=3, workers=4)
print('Skip-Gram模型参数：', sg_model)

# 获取"环境"的词向量
env_vector = sg_model.wv['环境']
print('"环境"的词向量：\n', env_vector)
print('词向量形状：', env_vector.shape)

# 找出与"好吃"最相似的3个词
similar_words = sg_model.wv.most_similar('好吃', topn=3)
print('与"好吃"最相似的3个词：')
for word, similarity in similar_words:
    print(f'{word}: {similarity:.4f}')

# 计算词语相似度
print('"好吃"和"美味"的相似度:', sg_model.wv.similarity('好吃', '美味'))
print('"好吃"和"蟑螂"的相似度:', sg_model.wv.similarity('好吃', '蟑螂'))

# 完成向量运算"餐厅+聚会-安静=？"
result = sg_model.wv.most_similar(positive=['餐厅', '聚会'], negative=['安静'], topn=1)
print('\n向量运算"餐厅+聚会-安静="最相关结果:', result[0][0])

