# Music-Clustering-Project
Music clustering project using PCA and K-Means to group songs based on audio features and visualize music similarity patterns.



#  Music Clustering Project

This project focuses on grouping similar songs using Machine Learning clustering techniques.

The main idea was to understand how songs can be grouped based on their audio features like energy, danceability, tempo, loudness, acousticness, and instrumentalness.

Instead of using predefined labels or genres, the model automatically discovered hidden patterns between songs using unsupervised learning.



##  What I Did

- Cleaned and prepared the dataset
- Removed duplicate values
- Selected important audio features
- Scaled the data using StandardScaler
- Reduced dimensions using PCA for visualization
- Applied K-Means clustering
- Evaluated clusters using Silhouette Score and Davies-Bouldin Index
- Visualized clusters using scatter plots and heatmaps


## 🛠️ Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn



##  Features Used

The clustering was done using these audio features:

- danceability
- energy
- loudness
- speechiness
- acousticness
- instrumentalness
- liveness
- valence
- tempo
- duration_ms

---

##  Results

The clustering model was able to identify different types of songs such as:

- Rap / spoken songs
- Chill acoustic songs
- Instrumental music
- Happy dance songs
- High-energy party tracks



##  Evaluation Scores

 Metric  
 Silhouette Score | 0.186 
 Davies-Bouldin Index | 1.687 
Inertia | 547629 

The scores show moderate clustering performance, which is expected because music features often overlap between songs.



##  Visualizations

The project includes:
- PCA cluster visualization
- Heatmaps
- Feature comparison charts
- Distribution plots

These visualizations helped understand how different clusters behave.

