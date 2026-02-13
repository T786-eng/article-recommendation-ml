# Article Recommendation System 📰

A machine learning project that recommends articles based on content similarity using Natural Language Processing (NLP) techniques. Built as part of my 30 Days of Project Building challenge.

## 🚀 Overview
This system analyzes the text content of articles to find similarities. It uses **TF-IDF Vectorization** to convert text into numerical data and **Cosine Similarity** to calculate the relationship between different articles.

## 🛠️ Tech Stack
- **Python**: Core logic
- **Pandas**: Data manipulation and cleaning
- **NumPy**: Numerical operations and array handling
- **Scikit-learn**: Machine Learning (TF-IDF & Cosine Similarity)
- **Matplotlib/Seaborn**: Data visualization

## 📊 Features
- Cleans and processes raw article data.
- Automatically generates and saves a **Similarity Heatmap** (`similarity_analysis.png`).
- Provides top 3 article recommendations for any given title.

## 📁 Project Structure
- `main.py`: The primary script.
- `articles.csv`: Dataset containing titles and content.
- `requirements.txt`: List of dependencies.
- `similarity_analysis.png`: Auto-generated visualization.

## ⚙️ How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the program:
    ``` bash
    python main.py
    ```

    