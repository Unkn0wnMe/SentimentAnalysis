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

def is_english(text):
    try:
        return langdetect.detect(text) == "en"
    except Exception:
        return False

def clean_text(text):
    text = re.sub(r"http\S+", "", text)    
    text = re.sub(r"@\w+", "", text)         
    text = re.sub(r"[^\w\s]", "", text)     
    return re.sub(r"\s+", " ", text).strip()  

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"


def get_model_sentiment(text, tokenizer, model, labels):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoded_input)

    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)[::-1]

    return labels[ranking[0]]

if __name__ == "__main__":
    

    folder_path = "data/search_results"
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    merged_data = []
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            merged_data.append(data)
    

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
            
 
    df = pd.DataFrame(tweets)
    df = df[df["text"].apply(is_english)]       
    df["text"] = df["text"].apply(clean_text)     
    df["date"] = pd.to_datetime(df["date"], format="%a %b %d %H:%M:%S %z %Y")  
    
  
    df["score"] = df["text"].apply(get_sentiment)
    

    task = "sentiment"
    model_name = f"cardiffnlp/twitter-roberta-base-{task}"
    
   
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    df["model_sentiment"] = df["text"].apply(lambda x: get_model_sentiment(x, tokenizer, model, labels))
    

    output_file = "final_output1.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Данные успешно сохранены в {output_file}")
