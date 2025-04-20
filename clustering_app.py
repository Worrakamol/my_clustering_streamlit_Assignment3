# -- coding: utf-8 --
"""
Created on Sat Apr 19 21:19:26 2025
@author: Nongnuch
"""

# app.py
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Set Streamlit page config
st.set_page_config(page_title="K-Means Clustering App", layout="wide")

# Sidebar - number of clusters
st.sidebar.header("Configure Clustering")
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 3)

# Title
st.markdown("<h1 style='text-align: center;'>🔍 K-Means Clustering App with Iris Dataset</h1>", unsafe_allow_html=True)

# Load Iris dataset
iris = load_iris()
X = iris.data

# PCA for 2D projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Fit KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Define fixed colors to match the image
custom_colors = ['orange', 'green', 'deepskyblue', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
colors = [custom_colors[label] for label in y_kmeans]

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=50)

# Labels and title
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title("Clusters (2D PCA Projection)")

# Legend
unique_labels = list(set(y_kmeans))
handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
                      markerfacecolor=custom_colors[i], markersize=10)
           for i in unique_labels]
ax.legend(handles=handles, title="Clusters")

# Show the plot in Streamlit
st.pyplot(fig)
