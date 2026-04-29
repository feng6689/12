from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def analyze_sentiment(text):
    if not text or not text.strip():
        return {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "sentiment": "中性",
            "has_text": False
        }
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if polarity > 0:
        sentiment = "正面"
    elif polarity < 0:
        sentiment = "负面"
    else:
        sentiment = "中性"
    
    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sentiment": sentiment,
        "has_text": True
    }

def generate_wordcloud(text, output_path="wordcloud.png"):
    if not text or not text.strip():
        return False
    
    try:
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        clean_text = ' '.join(filtered_words)
        
        if not clean_text.strip():
            clean_text = text
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            min_font_size=10,
            collocations=False
        ).generate(clean_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"生成词云时出错: {e}")
        return False

if __name__ == "__main__":
    test_text = "I love Python programming. It's amazing and wonderful. OpenCV is great for image processing."
    
    result = analyze_sentiment(test_text)
    print(f"情感分析结果: {result}")
    
    generate_wordcloud(test_text, "test_wordcloud.png")
    print("词云已生成")
