import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load datasets
videos_df = pd.read_csv(r"C:\Users\hi\Downloads\videos-stats.csv")
comments_df = pd.read_csv(r"C:\Users\hi\Downloads\comments.csv\comments.csv")

# Drop unnecessary columns
videos_df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
comments_df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

# Fill missing values
videos_df.fillna({"Likes": 0, "Comments": 0, "Views": 0}, inplace=True)
comments_df.fillna({"Comment": "", "Likes": 0}, inplace=True)

# Convert 'Published At' to datetime format
videos_df["Published At"] = pd.to_datetime(videos_df["Published At"], errors="coerce")

# Merge datasets on 'Video ID'
merged_df = pd.merge(videos_df, comments_df, on="Video ID", how="left")

# Ensure Likes_x column exists
if "Likes_x" in merged_df.columns:
    merged_df["Engagement Score"] = (merged_df["Views"] * 0.5) + (merged_df["Likes_x"] * 0.3) + (merged_df["Comments"] * 0.2)
else:
    merged_df["Engagement Score"] = (merged_df["Views"] * 0.5) + (merged_df["Likes"] * 0.3) + (merged_df["Comments"] * 0.2)

# Calculate average sentiment per video
if "Sentiment" in merged_df.columns:
    sentiment_df = merged_df.groupby("Video ID")["Sentiment"].mean().reset_index()
    videos_df = videos_df.merge(sentiment_df, on="Video ID", how="left")
    videos_df["Sentiment"].fillna(0, inplace=True)

# Combine text features for recommendation
videos_df["text"] = videos_df["Title"].fillna("") + " " + videos_df["Keyword"].fillna("")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(videos_df["text"])

# Compute Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation Function
def get_recommendations(video_title, num_recommendations=5):
    video_title = video_title.lower().strip()
    
    # Find matching video title (case-insensitive)
    matching_videos = videos_df[videos_df["Title"].str.lower().str.strip() == video_title]
    
    if matching_videos.empty:
        return ["Error: Video title not found! Please enter a valid title."]

    idx = matching_videos.index[0]

    # Compute similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]

    # Get recommended video titles
    video_indices = [i[0] for i in sim_scores]
    return videos_df["Title"].iloc[video_indices].tolist()

# Streamlit Web App
st.title("🎥 YouTube Video Recommendation System")
video_name = st.text_input("Enter a video title:")
if st.button("Recommend"):
    recommendations = get_recommendations(video_name)
    st.write(recommendations)
