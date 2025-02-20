import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv("final_output1.csv", parse_dates=["date"])

df["model_sentiment"] = df["model_sentiment"].astype(str).str.lower()

plt.figure(figsize=(8, 5))
df["model_sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0).plot(
    kind="bar", color=["green", "gray", "red"]
)
plt.xlabel("Отношение")
plt.ylabel("Количество сообщений")
plt.title("Столбчатая диаграмма")
plt.xticks(rotation=0)
plt.show()

def generate_wordcloud(text, title, color):
    if not text:
        return 

    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap=color).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


for sentiment, color in zip(["positive", "neutral", "negative"], ["Greens", "gray", "Reds"]):
    text = " ".join(df[df["model_sentiment"] == sentiment]["text"].astype(str))  # Собираем все тексты
    generate_wordcloud(text, f"Облако слов ({sentiment})", color)
