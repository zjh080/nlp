# 算法基础

## 多项式朴素贝叶斯分类器

### 特征条件独立性假设
假设所有特征(词项)在给定类别条件下相互独立   

### 多项式分布假设
假设特征服从多项式分布，适用于词频统计

### 贝叶斯定理在邮件分类中的具体应用形式
在邮件分类中，贝叶斯定理表示为：   

$P(类别|邮件) = P(邮件|类别) * P(类别) / P(邮件)$

由于P(邮件)对所有类别相同，实际计算简化为：        

$argmax P(类别) * ∏ P(词项|类别)$  

其中：P(类别) 是类别的先验概率，P(词项|类别) 是词项在给定类别下的条件概率

多项式朴素贝叶斯的数学表达：  
$$
P(c|d) ∝ P(c) * ∏ P(w|c)^count(w,d)
$$ 

其中：$count(w,d)$是词$w$在文档d中的出现次数

$P(w|c)$ 使用拉普拉斯平滑估计：  

$$
P(w|c) = (count(w,c) + α) / (∑ count(w',c) + α|V|)
$$  

$\alpha$是平滑参数  
通常$\alpha=1$，$|V|$是词汇表大小  
通常$\alpha$

## 数据处理流程

### 预处理步骤

#### 分词处理

使用正则表达式或分词工具将邮件文本分割为单词序列   
示例代码可能包含类似```re.findall(r'\w+', text.lower())```的操作

#### 停用词过滤
移除常见无意义词(the, a, is等)   
通过预定义的停用词列表实现
示例：```[word for word in words if word not in stopwords]```

#### 其他处理
大小写归一化(转为小写)    
词干提取或词形还原(如使用Porter Stemmer)   
特殊字符和数字的移除

## 特征构建过程

### 高频词特征选择
数学表达：   
选择语料中出现频率最高的N个词作为特征   
文档表示为词频向量：
$$
X = [count(w1,d), count(w2,d), ..., count(wN,d)]
$$

特点：
实现简单，计算效率高   
可能包含许多无区分能力的常用词   
忽略低频但有意义的词

### TF-IDF特征加权
数学表达：
词频-逆文档频率：

$$
tfidf(w,d) = tf(w,d) * log(N/(df(w)+1))
$$      

$tf(w,d)$：词w在文档d中的频率   
$df(w)$：包含词w的文档数  
$N$：总文档数   
文档表示为TF-IDF向量：

$$
X = [tfidf(w1,d), tfidf(w2,d), ..., tfidf(wN,d)]
$$

特点：
降低常见词的权重，提升有区分能力词的权重   
能更好捕捉文档特征   
计算复杂度较高

### 实现差异对比                

| 方面  | 高频词特征 |  TF-IDF特征   |
| :---: | :---:    |:-----------:|
| 特征选择 | 简单计数排列 |  需要计算文档频率   |
| 特征值  | 原始词频  | 加权后的TF-IDF值 |
| 内存需求 | 较低    |     较高      |
| 分类效果 | 一般    |    通常更好     |

# 高频词/TF-IDF两种特征模式的切换方法

在该项目中，通常可以通过配置参数或修改少量代码切换特征模式：

高频词模式：   
```from sklearn.feature_extraction.text import CountVectorizer```   
```vectorizer = CountVectorizer(max_features=1000)  # 选择前1000高频词``` 
  
TF-IDF模式：   
```from sklearn.feature_extraction.text import TfidfVectorizer```   
```vectorizer = TfidfVectorizer(max_features=1000)  # 选择TF-IDF最高的1000词``` 