import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🎵 Music Clustering", layout="wide")

# ── Load Models ──────────────────────────────────────────────
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

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("🎵 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Dashboard", "🔮 Predict Song"])

# ══════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🎵 Amazon Music Song Clustering")
    st.markdown("Group songs automatically by their audio features using **K-Means Clustering**.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Algorithm", "K-Means")
    col2.metric("Clusters", "5")
    col3.metric("Features Used", "10")

    st.markdown("---")
    st.subheader("🗂️ Cluster Groups")
    for cid, name in CLUSTER_NAMES.items():
        st.markdown(f"**Cluster {cid}** → {name}")

# ══════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 Cluster Dashboard")
    uploaded = st.file_uploader("Upload final_clustered_songs.csv", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        if 'cluster' not in df.columns:
            st.error("❌ No 'cluster' column found. Please upload the final clustered CSV.")
        else:
            df['cluster_name'] = df['cluster'].map(CLUSTER_NAMES)
            st.success(f"✅ {len(df):,} songs loaded")

            # Cluster count
            st.subheader("Songs per Cluster")
            counts = df['cluster_name'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 3))
            counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Count")
            plt.xticks(rotation=20)
            st.pyplot(fig)

            # Heatmap
            st.subheader("Feature Heatmap per Cluster")
            cluster_mean = df.groupby('cluster')[FEATURES].mean()
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            sns.heatmap(cluster_mean, annot=True, fmt=".2f", cmap='coolwarm', ax=ax2)
            ax2.set_yticklabels([CLUSTER_NAMES.get(i, i) for i in cluster_mean.index], rotation=0)
            st.pyplot(fig2)

            # PCA scatter
            st.subheader("PCA Scatter Plot")
            if 'PC1' not in df.columns:
                X_scaled = scaler.transform(df[FEATURES])
                X_pca = pca.transform(X_scaled)
                df['PC1'] = X_pca[:, 0]
                df['PC2'] = X_pca[:, 1]

            fig3, ax3 = plt.subplots(figsize=(8, 5))
            for cid in sorted(df['cluster'].unique()):
                mask = df['cluster'] == cid
                ax3.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'],
                            label=CLUSTER_NAMES.get(cid, cid), alpha=0.5, s=10)
            ax3.set_xlabel("PC1")
            ax3.set_ylabel("PC2")
            ax3.legend(fontsize=8)
            st.pyplot(fig3)

    else:
        st.info("👆 Upload your final_clustered_songs.csv to see charts.")

# ══════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Predict Song":
    st.title("🔮 Predict Your Song's Cluster")
    st.markdown("Adjust the sliders and click Predict.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        danceability     = st.slider("💃 Danceability",       0.0, 1.0, 0.5, 0.01)
        energy           = st.slider("⚡ Energy",              0.0, 1.0, 0.5, 0.01)
        speechiness      = st.slider("🗣️ Speechiness",        0.0, 1.0, 0.1, 0.01)
        acousticness     = st.slider("🎸 Acousticness",       0.0, 1.0, 0.3, 0.01)
        instrumentalness = st.slider("🎹 Instrumentalness",   0.0, 1.0, 0.0, 0.01)
    with col2:
        liveness     = st.slider("🎤 Liveness",           0.0, 1.0, 0.1, 0.01)
        valence      = st.slider("😊 Valence",            0.0, 1.0, 0.5, 0.01)
        loudness     = st.slider("🔊 Loudness (dB)",     -60.0, 0.0, -10.0, 0.1)
        tempo        = st.slider("🥁 Tempo (BPM)",        50.0, 250.0, 120.0, 1.0)
        duration_ms  = st.number_input("⏱️ Duration (ms)", 30000, 600000, 210000, 1000)

    if st.button("🔮 Predict", use_container_width=True):
        inp = np.array([[danceability, energy, loudness, speechiness,
                         acousticness, instrumentalness, liveness,
                         valence, tempo, duration_ms]])
        inp_scaled = scaler.transform(inp)
        cluster_id = kmeans.predict(inp_scaled)[0]
        st.success(f"🎵 Cluster {cluster_id} → **{CLUSTER_NAMES[cluster_id]}**")

st.markdown("---")
st.markdown("<center style='color:grey'>🎵 Amazon Music Clustering Project</center>", unsafe_allow_html=True)
