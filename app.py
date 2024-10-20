import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import re

import altair as alt
from transformers import pipeline  # HuggingFace Transformers

# Memuat model IndoBERT dari HuggingFace
nlp = pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")

# Fungsi untuk analisis sentimen menggunakan IndoBERT
def analyze_indonesian_sentiment(text):
    result = nlp(text)
    return result

# Fungsi untuk menganalisis sentimen per token
def analyze_token_sentiment(docx):
    tokens = docx.split()
    token_sentiments = []

    # Analisis setiap token menggunakan IndoBERT
    for token in tokens:
        sentiment = analyze_indonesian_sentiment(token)
        token_sentiments.append({token: sentiment[0]['label']})
    
    return token_sentiments

# Set up page configuration
st.set_page_config(page_title="Futuristic Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a futuristic look
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: #dcdcdc;
        }
        h1, h2, h3 {
            color: #7ec8e3;
        }
        .css-1v3fvcr {
            background-color: #1e1e1e;
        }
        .stDataFrame {
            background-color: #333333;
            color: #dcdcdc;
        }
        .stSelectbox {
            color: #7ec8e3;
            background-color: #1e1e1e;
        }
        .sidebar-title {
            font-size: 25px;
            color: #7ec8e3;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation using selectbox
st.sidebar.title("Navigation")
options = st.sidebar.selectbox(
    "Choose a section:",
    ["Project Overview", "Sentiment Analysis", "Raw Data & Visualization", "Processed Data", "Coming Soon"]
)

# Page 1: Project Overview
if options == 'Project Overview':
    st.title("Welcome to the Futuristic Dashboard")
    st.markdown("""
    This dashboard showcases the following sections:
    
    - **Sentiment Analysis**: Analyze sentiments from your dataset.
    - **Raw Data & Visualization**: Displays raw data from YouTube and visualizes it.
    - **Processed Data**: Shows the cleaned and labeled data, along with further visualizations.
    - **Coming Soon**: Stay tuned for future updates!

    Feel free to explore each page using the sidebar.
    """)

# Page 2: Sentiment Analysis
elif options == 'Sentiment Analysis':
    st.title("Sentiment Analysis")
    st.markdown("""
    This section allows you to analyze sentiments from your dataset.
    
    You can input text data or load preprocessed data for sentiment analysis.
    """)
    # Membuat form input 
    with st.form("nlpForm"):
        raw_text = st.text_area("Enter Text to Analyze Sentiment")
        submit_button = st.form_submit_button(label='Analyze')

    # Layout
    col1, col2 = st.columns(2)

    if submit_button:
        with col1:
            st.info("Results")

            # Analisis sentimen menggunakan IndoBERT
            sentiment_result = analyze_indonesian_sentiment(raw_text)
            
            # Menampilkan hasil analisis
            st.write("Sentiment Result:", sentiment_result)

            # Ambil label dan skor sentimen dari hasil prediksi
            label = sentiment_result[0]['label']
            score = sentiment_result[0]['score']

            # Emoji berdasarkan hasil IndoBERT
            if label == "positive":
                st.markdown(f"**Sentiment: Positive 😊 (Confidence: {score:.2f})**")
            elif label == "negative":
                st.markdown(f"**Sentiment: Negative 😡 (Confidence: {score:.2f})**")
            else:
                st.markdown(f"**Sentiment: Neutral 😐 (Confidence: {score:.2f})**")

            # Konversi hasil ke DataFrame untuk visualisasi
            result_df = pd.DataFrame(sentiment_result)
            st.dataframe(result_df)

            # Visualisasi menggunakan Altair
            c = alt.Chart(result_df).mark_bar().encode(
                x=alt.X('label', title='Sentiment Label'),
                y=alt.Y('score', title='Confidence Score'),
                color='label'
            )
            st.altair_chart(c, use_container_width=True)

        with col2:
            st.info("Token Sentiment")

            # Analisis sentimen per kata/token
            token_sentiments = analyze_token_sentiment(raw_text)
            st.write(token_sentiments)

# Page 3: Raw Data & Visualization
elif options == 'Raw Data & Visualization':
    st.title("YouTube Data (Raw)")
    st.markdown("Here is the raw YouTube data along with some visualizations.")
    
    # Load dataset
    data = pd.read_csv('youtube-comments.csv')

    # Display first 10 rows of the dataset
    st.subheader("First 10 Rows of Data")
    st.dataframe(data.head(10))

    
    # Visualisasi Data WordCloud
    st.subheader("WordCloud of YouTube Comments")

    # Concatenate all comments in 'textDisplay' column
    df = ' '.join(data['textDisplay'].dropna().tolist())

    # Define stopwords
    stopwords = set(STOPWORDS)
    stopwords.update([
        "https", "youtube", "com", "watch", "v", "www", "http", "href", 
        "youtu", "be", "channel", "user", "video", "videos", "watching", "watched"
    ])

    # Generate WordCloud
    WC = WordCloud(stopwords=stopwords, background_color="white", max_words=2000, width=800, height=400)
    WC.generate(df)

    # Plotting WordCloud
    plt.figure(figsize=(10, 6))
    plt.imshow(WC, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # Word Frequency Count
    st.subheader("Top 10 Most Frequent Words")

    # Tokenization and word frequency
    tokens = df.split()
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)
    word, count = zip(*top_words)

    # Plotting top 10 words
    colors = plt.cm.Paired(range(len(word)))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(word, count, color=colors)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Words in YouTube Comments')
    plt.xticks(rotation=45)

    # Adding text labels above bars
    for bar, num in zip(bars, count):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, num + 1, str(num), 
                 fontsize=12, color='black', ha='center')

    # Display bar chart
    st.pyplot(plt)
    
    
    

# Page 4: Processed Data & Visualization
# Page 4: Processed Data & Labeling
elif options == 'Processed Data':
    st.title("Processed Data & Labeling")
    st.markdown("This section shows the processed and labeled data after sentiment analysis.")

    # Load dataset
    data = pd.read_csv('preprocessed_youtube_comments.csv')

    # Display data info
    st.subheader("Dataset Information")
    st.write(data.info())

    # Show first 10 rows of the dataset
    st.subheader("First 10 Rows of Data")
    st.dataframe(data.head(10))

    # Word Cloud Visualization
    st.subheader("Word Cloud Visualization")
    
    import pandas as pd
    import numpy as np
    from PIL import Image
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    import matplotlib.pyplot as plt

    # Ensure there are no missing values in 'stemming_data' and concatenate into a single string
    df_text = ' '.join(data['stemming_data'].dropna().tolist())

    # Re-run with correct stopwords
    stopwords = set(STOPWORDS)
    stopwords.update([
        "https", "yg", "youtube", "com", "watch", "v", "www", "http", "href", 
        "youtu", "be", "channel", "user", "video", "videos", "watching", "watched"
    ])

    # Generate the word cloud
    WC = WordCloud(stopwords=stopwords, background_color="white", max_words=2000, width=800, height=400)
    WC.generate(df_text)

    # Visualize the word cloud
    plt.figure(figsize=[10,10])
    plt.imshow(WC, interpolation='bilinear')
    plt.axis('off')

    # Display in Streamlit
    st.pyplot(plt)

    # Top 10 Words in YouTube Comments
    st.subheader("Top 10 Words in YouTube Comments")

    from collections import Counter

    # Combine all text from 'stemming_data' column into one string
    text = ' '.join(data['stemming_data'].dropna().tolist())
    tokens = text.split()
    word_counts = Counter(tokens)

    top_words = word_counts.most_common(10)
    word, count = zip(*top_words)

    # Visualize top words
    colors = plt.cm.Paired(range(len(word)))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(word, count, color=colors)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Words in YouTube Comments')
    plt.xticks(rotation=45)

    for bar, num in zip(bars, count):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, num + 1, str(num), 
                 fontsize=12, color='black', ha='center')

    # Display in Streamlit
    st.pyplot(plt)

    # Sentiment Analysis
    st.subheader("Sentiment Analysis and Labeling")

    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # Initialize VADER SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    # List for storing sentiment labels and scores
    labels = []
    scores = []

    # Memastikan bahwa tidak ada nilai NaN di kolom 'stemming_data'
    data['stemming_data'] = data['stemming_data'].fillna('')  # Mengganti NaN dengan string kosong

    # Pastikan semua data di kolom 'stemming_data' adalah string
    data['stemming_data'] = data['stemming_data'].astype(str)

    
    # Analyze sentiment for each text in 'stemming_data' column
    for text in data['stemming_data']:
        sentiment_scores = sid.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        scores.append(compound_score)

        if compound_score > 0:
            labels.append('Positif')
        elif compound_score < 0:
            labels.append('Negatif')
        else:
            labels.append('Netral')

    # Add sentiment results to DataFrame
    data['sentiment'] = labels
    data['sentiment_score'] = scores

    # Select desired columns to display
    selected_columns = ['stemming_data', 'sentiment_score', 'sentiment']
    st.dataframe(data[selected_columns].head(1000))

    # Sentiment Count Visualization
    st.subheader("Sentiment Distribution")

    import seaborn as sns

    sentiment_count = data['sentiment'].value_counts()

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.barplot(x=sentiment_count.index, y=sentiment_count.values, palette='pastel')
    plt.title('Sentiment Analysis Distribution', fontsize=14, pad=20)
    plt.xlabel('Sentiment Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    for i, count in enumerate(sentiment_count.values):
        ax.text(i, count + 0.10, str(count), ha='center', va='bottom')

    st.pyplot(fig)
    
    st.subheader("Kmeans Clustering")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    # Pastikan kolom 'cleaned_text' di dataset sudah ada dan bersih
    if 'stemming_data' not in data.columns:
        st.error("Kolom 'stemming_data' tidak ditemukan dalam dataset. Pastikan Anda telah melakukan pembersihan data terlebih dahulu.")
    else:
        # Vectorization
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(data['stemming_data'])

        # KMeans Clustering
        true_k = 8  # Tentukan jumlah cluster
        model = KMeans(n_clusters=true_k)
        model.fit(X)

        # Menampilkan hasil clustering berdasarkan kata
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        for i in range(true_k):
            st.write(f"Cluster {i}: {', '.join(terms[ind] for ind in order_centroids[i, :10])}")

        # Menghitung Silhouette Score
        score = silhouette_score(X, model.labels_)
        st.write(f"Silhouette Score: {score:.2f}")

        # PCA for visualization
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(X.toarray())
        reduced_cluster_centers = pca.transform(model.cluster_centers_)

        # Visualisasi Clustering
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=model.labels_, cmap='viridis', alpha=0.6)
        plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='red', label='Centroids')
        plt.title('KMeans Clustering Visualization')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(scatter)
        plt.legend()
        st.pyplot(plt)


# Page 5: Coming Soon
elif options == 'Coming Soon':
    st.title("Coming Soon")
    st.markdown("Stay tuned for exciting updates in this section!")
