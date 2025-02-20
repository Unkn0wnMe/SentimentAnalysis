import json
import glob
import os
import re
import csv
import urllib.request
import langdetect
import pandas as pd
import torch
import numpy as np
from scipy.special import softmax
from datetime import datetime
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Функция для определения, что текст на английском
def is_english(text):
    try:
        return langdetect.detect(text) == "en"
    except Exception:
        return False

# Функция очистки текста от ссылок, упоминаний, спецсимволов и лишних пробелов
def clean_text(text):
    text = re.sub(r"http\S+", "", text)      # удаляем ссылки
    text = re.sub(r"@\w+", "", text)           # удаляем упоминания
    text = re.sub(r"[^\w\s]", "", text)        # удаляем спецсимволы
    return re.sub(r"\s+", " ", text).strip()   # удаляем лишние пробелы

# Анализ тональности с помощью TextBlob
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

# Анализ тональности с помощью модели Hugging Face
def get_model_sentiment(text, tokenizer, model, labels):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoded_input)
    # Преобразуем логиты в вероятности
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)[::-1]
    # Возвращаем метку с наивысшей вероятностью
    return labels[ranking[0]]

if __name__ == "__main__":
    
    # 1. Слияние всех JSON-файлов из папки "data/search_results"
    folder_path = "data/search_results"
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    merged_data = []
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            merged_data.append(data)
    
    # 2. Извлечение данных о твитах
    tweets = []
    for outer_list in merged_data:
        for tweet_entry in outer_list:
            try:
                tweet = tweet_entry['content']['itemContent']['tweet_results']['result']['legacy']
                tweets.append({
                    'date': tweet['created_at'], 
                    'text': tweet['full_text']
                })
            except KeyError:
                continue
            
    # 3. Создание DataFrame и предобработка
    df = pd.DataFrame(tweets)
    df = df[df["text"].apply(is_english)]       # оставляем только твиты на английском
    df["text"] = df["text"].apply(clean_text)     # чистим текст
    df["date"] = pd.to_datetime(df["date"], format="%a %b %d %H:%M:%S %z %Y")  # преобразуем дату
    
    # 4. Анализ тональности с помощью TextBlob
    df["score"] = df["text"].apply(get_sentiment)
    
    # 5. Анализ тональности с помощью модели Hugging Face
    task = "sentiment"
    model_name = f"cardiffnlp/twitter-roberta-base-{task}"
    
    # Загрузка маппинга меток
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    df["model_sentiment"] = df["text"].apply(lambda x: get_model_sentiment(x, tokenizer, model, labels))
    
    # 6. Сохранение результата в единый CSV-файл
    output_file = "final_output1.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Данные успешно сохранены в {output_file}")
