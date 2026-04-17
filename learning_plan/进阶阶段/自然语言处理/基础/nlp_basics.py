import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

# 下载NLTK资源
def download_nltk_resources():
    """下载NLTK资源"""
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("NLTK资源下载完成")

# 1. 文本预处理
def text_preprocessing(text):
    """文本预处理"""
    # 转换为小写
    text = text.lower()
    
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 去除特殊字符和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 分词
    tokens = nltk.word_tokenize(text)
    
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 词干提取
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # 重新组合为文本
    processed_text = ' '.join(tokens)
    return processed_text

# 2. 词袋模型和TF-IDF
def vectorization_example(texts):
    """文本向量化示例"""
    # 词袋模型
    count_vectorizer = CountVectorizer()
    count_vectors = count_vectorizer.fit_transform(texts)
    
    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectors = tfidf_vectorizer.fit_transform(texts)
    
    print(f"词袋模型特征数量: {count_vectorizer.vocabulary_.__len__()}")
    print(f"TF-IDF特征数量: {tfidf_vectorizer.vocabulary_.__len__()}")
    
    return count_vectors, tfidf_vectors, count_vectorizer, tfidf_vectorizer

# 3. Word2Vec词嵌入
def word2vec_example(sentences):
    """Word2Vec词嵌入示例"""
    # 分词
    tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]
    
    # 训练Word2Vec模型
    model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    # 获取词向量
    word_vectors = model.wv
    
    # 示例：获取词向量
    if 'hello' in word_vectors:
        print(f"'hello'的词向量: {word_vectors['hello']}")
    
    # 示例：查找相似词
    if 'hello' in word_vectors:
        similar_words = word_vectors.most_similar('hello', topn=5)
        print(f"与'hello'相似的词: {similar_words}")
    
    print("Word2Vec模型训练完成")
    return model

# 4. 循环神经网络（RNN）
def rnn_model():
    """RNN模型示例"""
    model = models.Sequential([
        layers.Embedding(input_dim=10000, output_dim=64, input_length=100),
        layers.SimpleRNN(64, return_sequences=True),
        layers.SimpleRNN(32),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    print("RNN模型创建完成")
    return model

# 5. LSTM模型
def lstm_model():
    """LSTM模型示例"""
    model = models.Sequential([
        layers.Embedding(input_dim=10000, output_dim=64, input_length=100),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    print("LSTM模型创建完成")
    return model

# 6. 文本分类示例
def text_classification_example():
    """文本分类示例"""
    # 创建示例数据
    texts = [
        "I love this movie, it's fantastic!",
        "This movie is terrible, I hate it.",
        "The acting was great, but the plot was boring.",
        "I would recommend this movie to everyone.",
        "Worst movie I've ever seen.",
        "This is one of the best movies I've watched recently.",
        "The movie was okay, not great but not bad.",
        "I didn't like the movie at all."
    ]
    labels = [1, 0, 0, 1, 0, 1, 0, 0]  # 1: positive, 0: negative
    
    # 文本预处理
    processed_texts = [text_preprocessing(text) for text in texts]
    
    # 向量化
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(processed_texts)
    y = np.array(labels)
    
    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # 训练朴素贝叶斯分类器
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    
    # 预测
    y_pred = clf.predict(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"分类准确率: {accuracy}")
    print("分类报告:")
    print(report)
    
    return clf, tfidf_vectorizer

# 7. 深度学习文本分类
def deep_learning_text_classification():
    """深度学习文本分类示例"""
    # 创建示例数据
    texts = [
        "I love this movie, it's fantastic!",
        "This movie is terrible, I hate it.",
        "The acting was great, but the plot was boring.",
        "I would recommend this movie to everyone.",
        "Worst movie I've ever seen.",
        "This is one of the best movies I've watched recently.",
        "The movie was okay, not great but not bad.",
        "I didn't like the movie at all."
    ]
    labels = [1, 0, 0, 1, 0, 1, 0, 0]  # 1: positive, 0: negative
    
    # 文本预处理
    processed_texts = [text_preprocessing(text) for text in texts]
    
    #  tokenization
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(processed_texts)
    word_index = tokenizer.word_index
    
    # 序列转换和填充
    sequences = tokenizer.texts_to_sequences(processed_texts)
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    
    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, np.array(labels), test_size=0.25, random_state=42)
    
    # 创建模型
    model = models.Sequential([
        layers.Embedding(input_dim=10000, output_dim=64, input_length=100),
        layers.LSTM(64),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # 编译模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 训练模型
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    
    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"深度学习分类准确率: {accuracy}")
    
    return model, tokenizer

# 8. 情感分析示例
def sentiment_analysis_example():
    """情感分析示例"""
    # 使用上面的文本分类示例作为情感分析
    print("情感分析示例:")
    clf, vectorizer = text_classification_example()
    
    # 测试新文本
    test_texts = [
        "This movie was amazing! I loved every minute of it.",
        "I couldn't stand this movie, it was awful."
    ]
    
    # 预处理和向量化
    processed_test_texts = [text_preprocessing(text) for text in test_texts]
    test_vectors = vectorizer.transform(processed_test_texts)
    
    # 预测
    predictions = clf.predict(test_vectors)
    
    for text, pred in zip(test_texts, predictions):
        sentiment = "正面" if pred == 1 else "负面"
        print(f"文本: {text}")
        print(f"情感: {sentiment}")
        print()
    
    return clf, vectorizer

# 9. 命名实体识别示例
def named_entity_recognition():
    """命名实体识别示例"""
    # 下载NLTK命名实体识别资源
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    
    # 示例文本
    text = "Barack Obama was born in Hawaii. He was the 44th President of the United States."
    
    # 分词和词性标注
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    
    # 命名实体识别
    named_entities = nltk.ne_chunk(tagged, binary=False)
    
    print("命名实体识别结果:")
    for entity in named_entities:
        if hasattr(entity, 'label'):
            print(f"实体类型: {entity.label()}, 实体: {' '.join([word for word, tag in entity])}")
    
    return named_entities

# 10. 文本生成示例
def text_generation_example():
    """文本生成示例"""
    # 创建示例文本
    text = "The quick brown fox jumps over the lazy dog. The dog barks at the fox. The fox runs away."
    
    # 预处理
    tokens = nltk.word_tokenize(text.lower())
    
    # 创建n-gram模型
    n = 2
    n_grams = list(nltk.ngrams(tokens, n))
    
    # 创建词到下一个词的映射
    word_dict = {}
    for i in range(len(tokens) - n):
        key = tuple(tokens[i:i+n])
        next_word = tokens[i+n]
        if key not in word_dict:
            word_dict[key] = []
        word_dict[key].append(next_word)
    
    # 生成文本
    seed = tuple(tokens[:n])
    generated_text = list(seed)
    
    for _ in range(10):
        if seed in word_dict:
            next_word = np.random.choice(word_dict[seed])
            generated_text.append(next_word)
            seed = tuple(generated_text[-n:])
        else:
            break
    
    generated_text = ' '.join(generated_text)
    print(f"生成的文本: {generated_text}")
    
    return generated_text

# 主函数
if __name__ == "__main__":
    print("=== 自然语言处理基础示例 ===")
    
    # 下载NLTK资源
    download_nltk_resources()
    
    # 1. 文本预处理
    print("\n1. 文本预处理")
    test_text = "Hello World! This is a test sentence. I love NLP."
    processed_text = text_preprocessing(test_text)
    print(f"原始文本: {test_text}")
    print(f"预处理后: {processed_text}")
    
    # 2. 文本向量化
    print("\n2. 文本向量化")
    sample_texts = ["I love NLP", "NLP is interesting", "I am learning NLP"]
    count_vectors, tfidf_vectors, count_vectorizer, tfidf_vectorizer = vectorization_example(sample_texts)
    
    # 3. Word2Vec
    print("\n3. Word2Vec词嵌入")
    word2vec_model = word2vec_example(sample_texts)
    
    # 4. RNN模型
    print("\n4. RNN模型")
    rnn = rnn_model()
    
    # 5. LSTM模型
    print("\n5. LSTM模型")
    lstm = lstm_model()
    
    # 6. 文本分类
    print("\n6. 文本分类")
    clf, vectorizer = text_classification_example()
    
    # 7. 深度学习文本分类
    print("\n7. 深度学习文本分类")
    try:
        dl_model, tokenizer = deep_learning_text_classification()
        print("深度学习文本分类完成")
    except Exception as e:
        print(f"深度学习文本分类失败: {e}")
    
    # 8. 情感分析
    print("\n8. 情感分析")
    sentiment_clf, sentiment_vectorizer = sentiment_analysis_example()
    
    # 9. 命名实体识别
    print("\n9. 命名实体识别")
    try:
        entities = named_entity_recognition()
        print("命名实体识别完成")
    except Exception as e:
        print(f"命名实体识别失败: {e}")
    
    # 10. 文本生成
    print("\n10. 文本生成")
    generated_text = text_generation_example()
    
    print("\n=== 自然语言处理基础示例完成 ===")
