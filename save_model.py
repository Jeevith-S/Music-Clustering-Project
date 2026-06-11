# ============================================================
# STEP: SAVE MODEL — Run this at the END of your notebook
# Add this as a new cell in Amazon_Music_Clustering__.ipynb
# ============================================================

import pickle

# Save the trained KMeans model
with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# Save the scaler (MUST save this too — to scale new inputs)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save the PCA
with open("pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)

print("✅ Models saved successfully!")
print("Files created: kmeans_model.pkl, scaler.pkl, pca_model.pkl")
