import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import BertTokenizer, TFBertForSequenceClassification, GPT2Tokenizer, TFGPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.corpus import stopwords
import re

# 1. 预训练模型 - BERT文本分类
def bert_text_classification():
    """使用BERT进行文本分类"""
    # 加载预训练模型和分词器
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 示例文本
    texts = ["I love this movie!", "This movie is terrible."]
    labels = [1, 0]  # 1: positive, 0: negative
    
    # 分词
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]
    
    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    # 训练模型
    model.fit({'input_ids': input_ids, 'attention_mask': attention_mask}, 
              np.array(labels), epochs=3, batch_size=2)
    
    # 预测
    test_texts = ["This film was amazing!", "I hated this movie."]
    test_encoded = tokenizer(test_texts, padding=True, truncation=True, return_tensors="tf")
    test_input_ids = test_encoded["input_ids"]
    test_attention_mask = test_encoded["attention_mask"]
    
    predictions = model({'input_ids': test_input_ids, 'attention_mask': test_attention_mask})
    logits = predictions.logits
    predicted_labels = np.argmax(logits, axis=1)
    
    print("BERT文本分类结果:")
    for text, label in zip(test_texts, predicted_labels):
        sentiment = "正面" if label == 1 else "负面"
        print(f"文本: {text}")
        print(f"预测情感: {sentiment}")
        print()
    
    return model, tokenizer

# 2. GPT-2文本生成
def gpt2_text_generation():
    """使用GPT-2进行文本生成"""
    # 加载预训练模型和分词器
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = TFGPT2LMHeadModel.from_pretrained(model_name)
    
    # 设置padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # 输入文本
    prompt = "Once upon a time,"
    
    # 分词
    input_ids = tokenizer.encode(prompt, return_tensors="tf")
    
    # 生成文本
    output = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7
    )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"GPT-2生成的文本:")
    print(generated_text)
    print()
    
    return model, tokenizer

# 3. T5文本摘要
def t5_text_summarization():
    """使用T5进行文本摘要"""
    # 加载预训练模型和分词器
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 输入文本
    text = "The quick brown fox jumps over the lazy dog. The dog barks at the fox, and the fox runs away. The dog chases the fox, but the fox is too fast."
    
    # 预处理文本
    input_text = "summarize: " + text
    
    # 分词
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # 生成摘要
    outputs = model.generate(
        inputs["input_ids"],
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    
    # 解码生成的摘要
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"原始文本:")
    print(text)
    print(f"\nT5生成的摘要:")
    print(summary)
    print()
    
    return model, tokenizer

# 4. 自定义Transformer模型
def custom_transformer_model():
    """自定义Transformer模型"""
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
        # 多头自注意力
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs
        
        # 前馈网络
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res
    
    # 构建模型
    inputs = layers.Input(shape=(100, 64))
    x = inputs
    for i in range(3):
        x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    
    print("自定义Transformer模型创建完成")
    return model

# 5. 多标签文本分类
def multi_label_classification():
    """多标签文本分类"""
    # 创建示例数据
    texts = [
        "I love this movie, it's fantastic and exciting!",
        "This movie is terrible, I hate it and it's boring.",
        "The acting was great, but the plot was boring.",
        "I would recommend this movie to everyone, it's amazing!"
    ]
    # 多标签：[正面, 负面, 无聊, 推荐]
    labels = [[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0], [1, 0, 0, 1]]
    
    # 分词
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100, padding='post')
    
    # 构建模型
    model = models.Sequential([
        layers.Embedding(input_dim=10000, output_dim=64, input_length=100),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(4, activation='sigmoid')  # 多标签使用sigmoid
    ])
    
    # 编译模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 训练模型
    model.fit(padded_sequences, np.array(labels), epochs=10, batch_size=2)
    
    # 预测
    test_texts = ["This movie was amazing and I would recommend it to everyone."]
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=100, padding='post')
    
    predictions = model.predict(test_padded)
    predicted_labels = (predictions > 0.5).astype(int)
    
    print("多标签分类结果:")
    print(f"文本: {test_texts[0]}")
    print(f"预测标签: {predicted_labels[0]}")
    print("标签含义: [正面, 负面, 无聊, 推荐]")
    print()
    
    return model, tokenizer

# 6. 问答系统 - 抽取式
def extractive_qa():
    """抽取式问答系统"""
    # 加载预训练模型和分词器
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 示例数据
    context = "Albert Einstein was born in Ulm, Germany in 1879. He is best known for his theory of relativity."
    question = "Where was Albert Einstein born?"
    
    # 分词
    inputs = tokenizer(question, context, return_tensors="tf")
    
    # 简单的问答模型（实际应用中使用专门的问答模型）
    print("抽取式问答示例:")
    print(f"问题: {question}")
    print(f"上下文: {context}")
    print("答案: Ulm, Germany")
    print()
    
    return model, tokenizer

# 7. 对话系统
def dialogue_system():
    """对话系统"""
    # 加载GPT-2模型
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = TFGPT2LMHeadModel.from_pretrained(model_name)
    
    # 设置padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # 对话历史
    dialogue_history = ""
    
    # 示例对话
    user_inputs = ["Hello, how are you?", "What's the weather like today?", "Tell me a joke."]
    
    print("对话系统示例:")
    for user_input in user_inputs:
        # 更新对话历史
        dialogue_history += f"User: {user_input}\nAssistant: "
        
        # 分词
        input_ids = tokenizer.encode(dialogue_history, return_tensors="tf")
        
        # 生成回复
        output = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7
        )
        
        # 解码生成的回复
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 提取助手回复
        assistant_response = response.split("Assistant: ")[-1]
        
        # 更新对话历史
        dialogue_history += f"{assistant_response}\n"
        
        print(f"User: {user_input}")
        print(f"Assistant: {assistant_response}")
        print()
    
    return model, tokenizer

# 8. 知识图谱与NLP
def knowledge_graph_nlp():
    """知识图谱与NLP"""
    # 示例知识图谱数据
    knowledge_graph = {
        "Albert Einstein": {
            "born_in": "Ulm, Germany",
            "born_year": "1879",
            "known_for": "Theory of Relativity"
        },
        "Stephen Hawking": {
            "born_in": "Oxford, England",
            "born_year": "1942",
            "known_for": "Black hole theory"
        }
    }
    
    # 简单的实体识别和关系提取
    def extract_entities(text):
        entities = []
        for person in knowledge_graph:
            if person in text:
                entities.append(person)
        return entities
    
    def answer_question(question, entities):
        for entity in entities:
            if "born" in question and "where" in question:
                return knowledge_graph[entity].get("born_in", "Unknown")
            elif "born" in question and "year" in question:
                return knowledge_graph[entity].get("born_year", "Unknown")
            elif "known" in question or "famous" in question:
                return knowledge_graph[entity].get("known_for", "Unknown")
        return "I don't know"
    
    # 测试
    test_questions = [
        "Where was Albert Einstein born?",
        "What is Stephen Hawking known for?",
        "When was Albert Einstein born?"
    ]
    
    print("知识图谱与NLP示例:")
    for question in test_questions:
        entities = extract_entities(question)
        answer = answer_question(question, entities)
        print(f"问题: {question}")
        print(f"答案: {answer}")
        print()
    
    return knowledge_graph

# 9. 多模态学习 - 图像描述
def image_captioning():
    """图像描述生成"""
    # 加载预训练模型和分词器
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 示例图像描述任务
    # 注意：实际应用中需要加载图像特征提取模型
    print("多模态学习 - 图像描述示例:")
    print("图像: [一只猫在沙发上]")
    print("生成的描述: A cat is sitting on a sofa.")
    print()
    
    return model, tokenizer

# 10. 低资源语言处理
def low_resource_language():
    """低资源语言处理"""
    # 示例低资源语言（假设为虚构语言）
    # 训练数据
    train_data = [
        ("bonjour", "hello"),
        ("merci", "thank you"),
        ("au revoir", "goodbye"),
        ("oui", "yes"),
        ("non", "no")
    ]
    
    # 简单的翻译模型
    # 实际应用中使用多语言预训练模型
    def translate(word):
        translations = dict(train_data)
        return translations.get(word, "Unknown")
    
    # 测试
    test_words = ["bonjour", "merci", "au revoir"]
    
    print("低资源语言处理示例:")
    for word in test_words:
        translation = translate(word)
        print(f"输入: {word}")
        print(f"翻译: {translation}")
        print()
    
    return train_data

# 主函数
if __name__ == "__main__":
    print("=== 自然语言处理进阶示例 ===")
    
    # 1. BERT文本分类
    print("\n1. BERT文本分类")
    try:
        bert_model, bert_tokenizer = bert_text_classification()
        print("BERT文本分类完成")
    except Exception as e:
        print(f"BERT文本分类失败: {e}")
    
    # 2. GPT-2文本生成
    print("\n2. GPT-2文本生成")
    try:
        gpt2_model, gpt2_tokenizer = gpt2_text_generation()
        print("GPT-2文本生成完成")
    except Exception as e:
        print(f"GPT-2文本生成失败: {e}")
    
    # 3. T5文本摘要
    print("\n3. T5文本摘要")
    try:
        t5_model, t5_tokenizer = t5_text_summarization()
        print("T5文本摘要完成")
    except Exception as e:
        print(f"T5文本摘要失败: {e}")
    
    # 4. 自定义Transformer模型
    print("\n4. 自定义Transformer模型")
    try:
        transformer_model = custom_transformer_model()
        print("自定义Transformer模型创建完成")
    except Exception as e:
        print(f"自定义Transformer模型创建失败: {e}")
    
    # 5. 多标签文本分类
    print("\n5. 多标签文本分类")
    try:
        multi_label_model, multi_label_tokenizer = multi_label_classification()
        print("多标签文本分类完成")
    except Exception as e:
        print(f"多标签文本分类失败: {e}")
    
    # 6. 抽取式问答系统
    print("\n6. 抽取式问答系统")
    try:
        qa_model, qa_tokenizer = extractive_qa()
        print("抽取式问答系统完成")
    except Exception as e:
        print(f"抽取式问答系统失败: {e}")
    
    # 7. 对话系统
    print("\n7. 对话系统")
    try:
        dialogue_model, dialogue_tokenizer = dialogue_system()
        print("对话系统完成")
    except Exception as e:
        print(f"对话系统失败: {e}")
    
    # 8. 知识图谱与NLP
    print("\n8. 知识图谱与NLP")
    try:
        knowledge_graph = knowledge_graph_nlp()
        print("知识图谱与NLP完成")
    except Exception as e:
        print(f"知识图谱与NLP失败: {e}")
    
    # 9. 多模态学习 - 图像描述
    print("\n9. 多模态学习 - 图像描述")
    try:
        image_model, image_tokenizer = image_captioning()
        print("多模态学习 - 图像描述完成")
    except Exception as e:
        print(f"多模态学习 - 图像描述失败: {e}")
    
    # 10. 低资源语言处理
    print("\n10. 低资源语言处理")
    try:
        train_data = low_resource_language()
        print("低资源语言处理完成")
    except Exception as e:
        print(f"低资源语言处理失败: {e}")
    
    print("\n=== 自然语言处理进阶示例完成 ===")
