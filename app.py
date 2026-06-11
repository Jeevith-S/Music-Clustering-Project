import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(
    page_title="Amazon Music Clustering",
    layout="wide"
)

st.title("🎵 Amazon Music Clustering Dashboard")

df = pd.read_csv("final_clustered_songs.csv")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("kmeans_model.pkl", "rb") as f:
    model = pickle.load(f)

page = st.sidebar.selectbox(
    "Select Page",
    [
        "Overview",
        "Cluster Analysis",
        "Predict Cluster"
    ]
)

if page == "Overview":

    st.subheader("Dataset")

    st.write(df.head())

    st.metric("Songs", len(df))
    st.metric("Clusters", df["cluster"].nunique())

    st.subheader("Cluster Distribution")

    st.bar_chart(
        df["cluster"].value_counts()
    )

elif page == "Cluster Analysis":

    st.subheader("Cluster Mean Features")

    features = [
        'danceability',
        'energy',
        'loudness',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo',
        'duration_ms'
    ]

    cluster_mean = (
        df.groupby("cluster")[features]
        .mean()
    )

    st.dataframe(cluster_mean)

    fig, ax = plt.subplots(figsize=(12,6))

    sns.heatmap(
        cluster_mean,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

elif page == "Predict Cluster":

    st.subheader("Predict New Song Cluster")

    danceability = st.slider(
        "danceability",0.0,1.0,0.5
    )

    energy = st.slider(
        "energy",0.0,1.0,0.5
    )

    loudness = st.number_input(
        "loudness",-60.0,5.0,-10.0
    )

    speechiness = st.slider(
        "speechiness",0.0,1.0,0.1
    )

    acousticness = st.slider(
        "acousticness",0.0,1.0,0.5
    )

    instrumentalness = st.slider(
        "instrumentalness",0.0,1.0,0.0
    )

    liveness = st.slider(
        "liveness",0.0,1.0,0.2
    )

    valence = st.slider(
        "valence",0.0,1.0,0.5
    )

    tempo = st.number_input(
        "tempo",0.0,250.0,120.0
    )

    duration_ms = st.number_input(
        "duration_ms",10000,500000,200000
    )

    if st.button("Predict"):

        sample = np.array([[
            danceability,
            energy,
            loudness,
            speechiness,
            acousticness,
            instrumentalness,
            liveness,
            valence,
            tempo,
            duration_ms
        ]])

        sample_scaled = scaler.transform(sample)

        cluster = model.predict(sample_scaled)[0]

        cluster_names = {
            0: "Rap Songs",
            1: "Chill Acoustic",
            2: "Instrumental Music",
            3: "Happy Dance Songs",
            4: "High Energy Party Songs"
        }

        st.success(
            f"Cluster {cluster} : {cluster_names[cluster]}"
        )