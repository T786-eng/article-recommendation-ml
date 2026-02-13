import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load Data with Pandas
def run_recommender():
    try:
        # Loading dataset (Ensure the CSV has 'Title' and 'Article' columns)
        df = pd.read_csv('articles.csv', encoding='latin1')
        df = df.dropna().reset_index(drop=True)
        
        # Limit to 15 articles for a clear, readable graph
        plot_df = df.head(15)

        # 2. Vectorization using Scikit-learn (TF-IDF)
        # We convert text content into a NumPy-compatible matrix
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(plot_df['Article'])

        # 3. Calculate Similarity using Cosine Similarity
        # This returns a NumPy ndarray
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # 4. Generate and Auto-Save the Similarity Graph
        save_visual_analysis(cosine_sim_matrix, plot_df)

        # 5. Recommendation Logic
        recommend_article(plot_df, cosine_sim_matrix, article_title=plot_df['Title'][0])

    except FileNotFoundError:
        print("Error: articles.csv not found. Please place the dataset in the folder.")

def save_visual_analysis(matrix, dataframe):
    plt.figure(figsize=(12, 10))
    
    # Using Seaborn for the heatmap visualization
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='YlGnBu',
                xticklabels=dataframe['Title'].values, 
                yticklabels=dataframe['Title'].values)
    
    plt.title('Article Similarity Analysis (TF-IDF + Cosine Similarity)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Auto-save the file
    file_path = "article_similarity_heatmap.png"
    plt.savefig(file_path)
    print(f"--- Graph saved successfully: {file_path} ---")
    plt.show()

def recommend_article(df, sim_matrix, article_title):
    # Find the index using Pandas
    idx = df[df['Title'] == article_title].index[0]
    
    # Use NumPy to handle the similarity scores
    scores = list(enumerate(sim_matrix[idx]))
    
    # Sort scores based on similarity (index 1 of the tuple)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]
    
    print(f"\nRecommendations for: '{article_title}'")
    print("-" * 30)
    for i, score in sorted_scores:
        print(f"Recommended: {df['Title'][i]} (Match Score: {score:.2f})")

if __name__ == "__main__":
    run_recommender()