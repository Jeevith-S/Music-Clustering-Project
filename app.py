import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.decomposition import PCA

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Amazon Music Clustering",
    page_icon="🎵",
    layout="wide"
)

# ─── Load Saved Models ───────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("pca_model.pkl", "rb") as f:
        pca = pickle.load(f)
    return kmeans, scaler, pca

kmeans, scaler, pca = load_models()

# ─── Features & Cluster Names ────────────────────────────────────────────────
FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_ms'
]

CLUSTER_NAMES = {
    0: "🎤 Rap Songs",
    1: "🎸 Chill Acoustic",
    2: "🎻 Instrumental Music",
    3: "💃 Happy Dance Songs",
    4: "🔥 High Energy Party Songs"
}

CLUSTER_COLORS = {
    0: "#FF6B6B",
    1: "#4ECDC4",
    2: "#45B7D1",
    3: "#96CEB4",
    4: "#FFEAA7"
}

# ─── Sidebar Navigation ──────────────────────────────────────────────────────
st.sidebar.title("🎵 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Cluster Visualizations", "🔮 Predict Your Song", "📁 Upload & Analyze Dataset"]
)

# ============================================================
# PAGE 1 — HOME
# ============================================================
if page == "🏠 Home":
    st.title("🎵 Amazon Music Song Clustering")
    st.markdown("### Automatically group songs by their audio characteristics using K-Means Clustering")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Algorithm Used", "K-Means")
    with col2:
        st.metric("🗂️ Number of Clusters", "5")
    with col3:
        st.metric("🎼 Features Used", "10")

    st.markdown("---")
    st.subheader("📌 What Does Each Cluster Represent?")

    for cluster_id, name in CLUSTER_NAMES.items():
        color = CLUSTER_COLORS[cluster_id]
        st.markdown(
            f"""<div style='background-color:{color}30; border-left: 5px solid {color};
            padding:12px; border-radius:8px; margin-bottom:10px;'>
            <b>Cluster {cluster_id} — {name}</b>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("🎯 Why K-Means?")
    st.markdown("""
    | Model | Reason |
    |---|---|
    | ✅ **K-Means** | Fast, interpretable, works well for audio feature data |
    | ❌ DBSCAN | Hard to tune, produces inconsistent cluster counts |
    | ❌ Hierarchical | Too slow for large datasets |

    **K-Means** was selected as the final model based on:
    - Best **Silhouette Score**
    - Clear cluster separation in PCA visualization
    - Easy to interpret cluster profiles
    """)


# ============================================================
# PAGE 2 — CLUSTER VISUALIZATIONS
# ============================================================
elif page == "📊 Cluster Visualizations":
    st.title("📊 Cluster Visualizations")
    st.markdown("Upload your **final_clustered_songs.csv** to see visualizations.")

    uploaded = st.file_uploader("Upload final_clustered_songs.csv", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        # Check cluster column exists
        if 'cluster' not in df.columns:
            st.error("❌ CSV must have a 'cluster' column. Run the notebook first.")
        else:
            st.success(f"✅ Loaded {len(df):,} songs with {df['cluster'].nunique()} clusters")

            # ── Cluster Distribution ──────────────────────────
            st.subheader("🥧 Cluster Distribution")
            fig, ax = plt.subplots(figsize=(7, 4))
            cluster_counts = df['cluster'].value_counts().sort_index()
            colors = [CLUSTER_COLORS[i] for i in cluster_counts.index]
            ax.bar(
                [CLUSTER_NAMES.get(i, f"Cluster {i}") for i in cluster_counts.index],
                cluster_counts.values,
                color=colors,
                edgecolor='black'
            )
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Number of Songs")
            ax.set_title("Songs per Cluster")
            plt.xticks(rotation=15)
            st.pyplot(fig)

            # ── PCA Scatter ────────────────────────────────────
            st.subheader("🔵 PCA Cluster Scatter Plot")
            if 'PC1' not in df.columns or 'PC2' not in df.columns:
                X = scaler.transform(df[FEATURES])
                X_pca = pca.transform(X)
                df['PC1'] = X_pca[:, 0]
                df['PC2'] = X_pca[:, 1]

            fig2, ax2 = plt.subplots(figsize=(9, 6))
            for cid in sorted(df['cluster'].unique()):
                mask = df['cluster'] == cid
                ax2.scatter(
                    df.loc[mask, 'PC1'], df.loc[mask, 'PC2'],
                    label=CLUSTER_NAMES.get(cid, f"Cluster {cid}"),
                    alpha=0.6, s=15, color=CLUSTER_COLORS.get(cid, 'grey')
                )
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            ax2.set_title("PCA Visualization of Clusters")
            ax2.legend(loc='best', fontsize=8)
            st.pyplot(fig2)

            # ── Heatmap ────────────────────────────────────────
            st.subheader("🌡️ Feature Heatmap per Cluster")
            cluster_mean = df.groupby('cluster')[FEATURES].mean()
            fig3, ax3 = plt.subplots(figsize=(12, 4))
            sns.heatmap(cluster_mean, annot=True, fmt=".2f", cmap='coolwarm', ax=ax3)
            ax3.set_title("Average Feature Values per Cluster")
            ax3.set_yticklabels(
                [CLUSTER_NAMES.get(i, f"Cluster {i}") for i in cluster_mean.index],
                rotation=0
            )
            st.pyplot(fig3)

            # ── Bar Chart ──────────────────────────────────────
            st.subheader("📊 Average Feature Values per Cluster")
            fig4, ax4 = plt.subplots(figsize=(13, 5))
            cluster_mean.T.plot(kind='bar', ax=ax4, color=list(CLUSTER_COLORS.values()))
            ax4.set_title("Feature Comparison Across Clusters")
            ax4.set_xlabel("Features")
            ax4.set_ylabel("Mean Value")
            ax4.legend(
                [CLUSTER_NAMES.get(i, f"Cluster {i}") for i in cluster_mean.index],
                loc='upper right', fontsize=7
            )
            plt.xticks(rotation=30)
            st.pyplot(fig4)

            # ── Distribution Plot ──────────────────────────────
            st.subheader("📈 Feature Distribution by Cluster")
            selected_feature = st.selectbox("Select a feature to explore:", FEATURES)
            fig5, ax5 = plt.subplots(figsize=(9, 4))
            for cid in sorted(df['cluster'].unique()):
                mask = df['cluster'] == cid
                sns.kdeplot(
                    df.loc[mask, selected_feature],
                    label=CLUSTER_NAMES.get(cid, f"Cluster {cid}"),
                    fill=True, alpha=0.3,
                    color=CLUSTER_COLORS.get(cid, 'grey'),
                    ax=ax5
                )
            ax5.set_title(f"Distribution of '{selected_feature}' per Cluster")
            ax5.legend(fontsize=8)
            st.pyplot(fig5)

            # ── Top Songs per Cluster ──────────────────────────
            st.subheader("🎵 Sample Songs per Cluster")
            for cid in sorted(df['cluster'].unique()):
                with st.expander(f"  {CLUSTER_NAMES.get(cid, f'Cluster {cid}')}"):
                    subset = df[df['cluster'] == cid]
                    # Show available text columns
                    show_cols = [c for c in ['name_song', 'track_name', 'name_artists', 'artist_name', 'genres', 'cluster'] if c in df.columns]
                    if show_cols:
                        st.dataframe(subset[show_cols].head(10), use_container_width=True)
                    else:
                        st.dataframe(subset[FEATURES + ['cluster']].head(10), use_container_width=True)

    else:
        st.info("👆 Please upload your **final_clustered_songs.csv** file to see visualizations.")


# ============================================================
# PAGE 3 — PREDICT YOUR SONG
# ============================================================
elif page == "🔮 Predict Your Song":
    st.title("🔮 Predict Your Song's Cluster")
    st.markdown("Enter the audio features of a song and find out which cluster it belongs to!")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        danceability   = st.slider("💃 Danceability",     0.0, 1.0, 0.5, 0.01)
        energy         = st.slider("⚡ Energy",            0.0, 1.0, 0.5, 0.01)
        speechiness    = st.slider("🗣️ Speechiness",      0.0, 1.0, 0.1, 0.01)
        acousticness   = st.slider("🎸 Acousticness",     0.0, 1.0, 0.3, 0.01)
        instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.0, 0.01)

    with col2:
        liveness       = st.slider("🎤 Liveness",         0.0, 1.0, 0.1, 0.01)
        valence        = st.slider("😊 Valence (Mood)",   0.0, 1.0, 0.5, 0.01)
        loudness       = st.slider("🔊 Loudness (dB)",   -60.0, 0.0, -10.0, 0.1)
        tempo          = st.slider("🥁 Tempo (BPM)",      50.0, 250.0, 120.0, 1.0)
        duration_ms    = st.number_input("⏱️ Duration (ms)", min_value=30000, max_value=600000, value=210000, step=1000)

    st.markdown("---")

    if st.button("🔮 Predict Cluster", use_container_width=True):
        input_data = np.array([[
            danceability, energy, loudness, speechiness,
            acousticness, instrumentalness, liveness,
            valence, tempo, duration_ms
        ]])

        input_scaled = scaler.transform(input_data)
        cluster_id   = kmeans.predict(input_scaled)[0]
        cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        color        = CLUSTER_COLORS.get(cluster_id, "#DDDDDD")

        st.markdown(
            f"""<div style='background-color:{color}40; border: 2px solid {color};
            padding:20px; border-radius:12px; text-align:center; margin-top:10px;'>
            <h2>🎵 This song belongs to:</h2>
            <h1 style='color:{color};'>{cluster_name}</h1>
            <p style='font-size:16px;'>Cluster ID: {cluster_id}</p>
            </div>""",
            unsafe_allow_html=True
        )

        # PCA position
        input_pca = pca.transform(input_scaled)
        st.markdown("#### 📍 Your Song's Position on PCA Map")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([0], [0], s=200, color=color, zorder=5, label=f"Your Song → {cluster_name}", marker='★')
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Song Position in PCA Space")
        ax.legend()
        st.pyplot(fig)

        # Feature radar-style bar
        st.markdown("#### 🎛️ Your Song's Audio Profile")
        feature_labels = ['danceability', 'energy', 'speechiness', 'acousticness',
                          'instrumentalness', 'liveness', 'valence']
        feature_values = [danceability, energy, speechiness, acousticness,
                          instrumentalness, liveness, valence]

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        bars = ax2.barh(feature_labels, feature_values, color=color, edgecolor='black')
        ax2.set_xlim(0, 1)
        ax2.set_title("Audio Feature Profile")
        for bar, val in zip(bars, feature_values):
            ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.2f}", va='center', fontsize=9)
        st.pyplot(fig2)


# ============================================================
# PAGE 4 — UPLOAD & ANALYZE DATASET
# ============================================================
elif page == "📁 Upload & Analyze Dataset":
    st.title("📁 Upload & Cluster a New Dataset")
    st.markdown("Upload any CSV with the required audio features and get cluster predictions!")

    uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df):,} songs | {df.shape[1]} columns")
        st.dataframe(df.head(5), use_container_width=True)

        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing}")
        else:
            if st.button("🚀 Run Clustering", use_container_width=True):
                with st.spinner("Clustering songs..."):
                    X = df[FEATURES]
                    X_scaled = scaler.transform(X)
                    df['cluster'] = kmeans.predict(X_scaled)
                    df['cluster_name'] = df['cluster'].map(CLUSTER_NAMES)

                    X_pca = pca.transform(X_scaled)
                    df['PC1'] = X_pca[:, 0]
                    df['PC2'] = X_pca[:, 1]

                st.success("✅ Clustering complete!")

                # Summary
                st.subheader("📊 Cluster Summary")
                summary = df.groupby(['cluster', 'cluster_name']).size().reset_index(name='song_count')
                st.dataframe(summary, use_container_width=True)

                # Scatter plot
                st.subheader("🔵 PCA Scatter")
                fig, ax = plt.subplots(figsize=(9, 5))
                for cid in sorted(df['cluster'].unique()):
                    mask = df['cluster'] == cid
                    ax.scatter(
                        df.loc[mask, 'PC1'], df.loc[mask, 'PC2'],
                        label=CLUSTER_NAMES.get(cid, f"Cluster {cid}"),
                        alpha=0.5, s=15, color=CLUSTER_COLORS.get(cid, 'grey')
                    )
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.legend(fontsize=8)
                st.pyplot(fig)

                # Download
                st.subheader("📥 Download Results")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Clustered CSV",
                    data=csv,
                    file_name="clustered_songs_output.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        st.info("👆 Upload a CSV file to get started.")
        st.markdown("**Required columns:** " + ", ".join(FEATURES))

# ─── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:grey;'>🎵 Amazon Music Clustering | Unsupervised ML Project</div>",
    unsafe_allow_html=True
)
