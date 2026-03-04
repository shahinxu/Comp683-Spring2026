import pandas as pd
import numpy as np
import networkx as nx
import community.community_louvain as community_louvain # pip install python-louvain
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ==========================================
# Data Loading
# ==========================================
print("Loading data...")
# 1. Load Matrix X
df = pd.read_csv('Levine_matrix.csv')
X = df.drop(columns=['label']).values
y_true = df['label'].values
feature_names = df.drop(columns=['label']).columns

# 2. Identify labeled cells (ignore NaNs)
labeled_mask = ~np.isnan(y_true)

# 3. Load Graph G (assuming nodes correspond to row indices 0 to N-1)
G = nx.read_edgelist('cell_graph.edgelist', nodetype=int)

# ==========================================
# Part 1: Clustering on cell x protein data (X)
# ==========================================
print("\n--- Part 1: KMeans on Matrix X ---")
# Count unique valid labels to set K for KMeans
n_clusters = len(np.unique(y_true[labeled_mask])) 
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_pred_X = kmeans.fit_predict(X)

nmi_X = normalized_mutual_info_score(y_true[labeled_mask], y_pred_X[labeled_mask])
print(f"NMI for Matrix Clustering (KMeans): {nmi_X:.4f}")

# ==========================================
# Part 2: Graph Partitioning (G)
# ==========================================
print("\n--- Part 2: Louvain on Graph G ---")
# Louvain automatically finds the optimal number of communities
partition = community_louvain.best_partition(G, random_state=42)
# Map partition dictionary to a list that matches row indices of X
y_pred_G = np.array([partition.get(i, -1) for i in range(len(X))])

nmi_G = normalized_mutual_info_score(y_true[labeled_mask], y_pred_G[labeled_mask])
print(f"NMI for Graph Partitioning (Louvain): {nmi_G:.4f}")

# ==========================================
# Part 4: Rare Cell-types (pDCs, label 21)
# ==========================================
print("\n--- Part 4: Rare Cell-types ---")
pdc_mask = (y_true == 21)
n_clusters_pdc_X = len(np.unique(y_pred_X[pdc_mask]))
n_clusters_pdc_G = len(np.unique(y_pred_G[pdc_mask]))

print(f"Number of distinct clusters containing pDCs (Matrix clustering): {n_clusters_pdc_X}")
print(f"Number of distinct clusters containing pDCs (Graph clustering): {n_clusters_pdc_G}")

# ==========================================
# Part 5: Cell Classification
# ==========================================
print("\n--- Part 5: Cell Classification ---")
t_cell_labels = [11, 12, 17, 18]
monocyte_labels = [1, 2, 3]

# Create subset for classification
mask_clf = np.isin(y_true, t_cell_labels) | np.isin(y_true, monocyte_labels)
X_clf = X[mask_clf]
# Label T-cells as 0, Monocytes as 1
y_clf = np.where(np.isin(y_true[mask_clf], monocyte_labels), 1, 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

# Train Classifier (Random Forest is robust and provides feature importances)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and calculate ROC
y_score = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC Score: {roc_auc:.4f}")

# Feature importances
importances = clf.feature_importances_
top_features_idx = np.argsort(importances)[::-1][:3] # Top 3 features
print("Top 3 helpful features for prediction:")
for idx in top_features_idx:
    print(f"- {feature_names[idx]} (Importance: {importances[idx]:.4f})")

# Plot ROC Curve
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (T-cells vs Monocytes)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
print("ROC curve saved as 'roc_curve.png'")