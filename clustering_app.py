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
import numpy as np

# Set the page config
st.set_page_config(page_title="K-Means Clustering App", layout="centered")

# Title
st.title("🔍 K-Means Clustering App with Iris Dataset")

# Load Iris dataset
iris = load_iris()
X = iris.data

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Sidebar - number of clusters
st.sidebar.header("Configure Clustering")
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 3)

# Fit KMeans model
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X)

# Define custom color list (match image: orange, green, blue, red)
custom_colors = ['orange', 'green', 'deepskyblue', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
color_values = [custom_colors[label] for label in clusters]

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color_values, s=50)

ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title("Clusters (2D PCA Projection)")

# Create legend manually
handles = []
for i in range(k):
    handles.append(plt.Line2D([0], [0], marker='o', color='w',
                              label=f'Cluster {i}', markerfacecolor=custom_colors[i], markersize=10))
ax.legend(handles=handles, title="Clusters")

# Show the plot
st.pyplot(fig)
