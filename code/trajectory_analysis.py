import os
import numpy as np
import pandas as pd
import logging
import argparse
import umap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from file_paths import file_paths

def load_cell_distance_data(cell_group_file, pathway_distance_files):
    print("Loading cell-cell distance data for each pathway...")
    cell_group_df = pd.read_csv(cell_group_file, sep=',', header=0, index_col=0)
    combined_df = None
    for i, file_path in enumerate(pathway_distance_files):
        pathway_id = pathways[i]  # Use pathway identifier from the pathways list
        print(f"\tReading distance file {i+1}/{len(pathway_distance_files)}: {file_path}")
        df = pd.read_csv(file_path, header=None, names=['Cell1', 'Cell2', f'distance_{pathway_id}'])
        combined_df = df if combined_df is None else pd.merge(combined_df, df, on=['Cell1', 'Cell2'], how='outer')
    combined_df = combined_df.dropna()
    return combined_df, cell_group_df

def prepare_combined_df(combined_df, cell_group_df):
    print("Preparing combined dataframe by merging cell group information...")
    cell_group_df['Cell'] = cell_group_df['Cell'].astype(str).str.strip()
    
    combined_df['Cell1_number'] = combined_df['Cell1'].str.split('_').str[1]
    combined_df['Cell2_number'] = combined_df['Cell2'].str.split('_').str[1]
    
    cell_group_map = cell_group_df.set_index('Cell')['Group'].to_dict()
    
    combined_df['Cell1_group'] = combined_df['Cell1_number'].map(cell_group_map)
    combined_df['Cell2_group'] = combined_df['Cell2_number'].map(cell_group_map)
    
    columns_to_merge = ['Cell', 'Group'] + [f'hsa{pathway}' for pathway in pathways]
    
    combined_df = combined_df.merge(cell_group_df[columns_to_merge], left_on='Cell1_number', right_on='Cell', suffixes=('', '_group'))
    combined_df = combined_df.drop(columns=['Cell'])
    return combined_df

def load_cluster_summary(cluster_file):
    # Load cluster summary and filter based on genes
    cluster_data = pd.read_csv(cluster_file, header=None)
    cluster_genes = cluster_data.iloc[:, 0]
    cluster_values = cluster_data.iloc[:, 1:].values
    return cluster_genes, cluster_values

def plot_cell_distance_kmeans_elbow(combined_df, pathways, output_dir):
    """
    Generate K-means elbow plots for each pathway based on cell-cell distances.
    
    Args:
        combined_df (pd.DataFrame): DataFrame with cell-cell distance data for each pathway.
        pathways (list of str): List of pathway identifiers (e.g., pathway names or IDs).
        output_dir (str): Directory to save the elbow plot images.
        max_clusters (int): Maximum number of clusters to test for the elbow plot (default is 10).
    """
    print("Generating K-means elbow plots for each pathway...")
    
    # Directory setup for saving plots
    elbow_plots_dir = f"{output_dir}/kmeans_elbow_plots"
    if not os.path.exists(elbow_plots_dir):
        os.makedirs(elbow_plots_dir)
        
    max_clusters = 10
    
    # Loop over each pathway and generate an elbow plot
    for pathway in pathways:
        # Extract the specific pathway distance column
        distance_column = f'distance_{pathway}'
        pathway_distances = combined_df[[distance_column]].values
        
        # Calculate inertia for different numbers of clusters
        inertia = []
        cluster_range = range(1, max_clusters + 1)
        
        for k in cluster_range:
            # Initialize and fit K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(pathway_distances)
            inertia.append(kmeans.inertia_)
        
        # Plot the elbow curve for the current pathway
        plt.figure(figsize=(8, 5))
        plt.plot(cluster_range, inertia, marker='o')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
        plt.title(f"Elbow Plot for Pathway {pathway}")
        plt.grid(True)
        
        # Save the plot
        plt.savefig(f"{elbow_plots_dir}/elbow_plot_{pathway}.png", dpi=300)
        plt.close()
        
def plot_cell_distance_umap(embedding, combined_df, group_color_map, cell_distance_output_dir):
    print("Plotting UMAP embedding...")
    
    colors = combined_df['Cell1_group'].map(group_color_map)
    plt.figure(figsize=(10, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.7, s=1)
    for group, color in group_color_map.items():
        plt.scatter([], [], color=color, label=group, s=50)
    plt.legend(title="Group")
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('UMAP Projection of Cells Based on Combined Pathway Cell-Cell Distances, Colored by Group')
    plt.grid(False)
    plt.savefig(f'{cell_distance_output_dir}/umap_cell_distance.png', dpi=300)

def plot_cell_distance_umap_by_pathway(embedding, combined_df, pathway, cell_distance_output_dir):
    """
    Plot UMAP embedding for each pathway with cells colored by the dominant group in each cluster.
    
    Args:
        embedding (np.ndarray): UMAP embedding for the cells.
        combined_df (pd.DataFrame): DataFrame containing cell distance and cluster information.
        pathway (str): The pathway identifier for the specific pathway column.
        cell_distance_output_dir (str): Directory to save the UMAP plot images.
    """
    print(f"\tPlotting UMAP embedding for pathway {pathway}...")
    
    # Setup output directory
    cell_distance_umap_pathway_dir = f'{cell_distance_output_dir}/umap_pathways_by_cluster'
    if not os.path.exists(cell_distance_umap_pathway_dir):
        os.makedirs(cell_distance_umap_pathway_dir)

    # Calculate the dominant group for each cluster
    cluster_groups = combined_df.groupby(pathway)['Group'].value_counts().unstack().fillna(0)
    cluster_colors = {}
    for cluster, counts in cluster_groups.iterrows():
        if counts.get('HIV', 0) > counts.get('Healthy', 0):
            cluster_colors[cluster] = '#ff7f0e'  # Orange for HIV-dominant clusters
        else:
            cluster_colors[cluster] = '#1f77b4'  # Blue for Healthy-dominant clusters

    # Map colors to clusters
    pathway_clusters = combined_df[pathway]
    colors = pathway_clusters.map(cluster_colors)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.5, s=1)
    for cluster, color in cluster_colors.items():
        plt.scatter([], [], color=color, label=f'Cluster {cluster}', s=50)
    plt.legend(title="Cluster (Dominant Group)")
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title(f'UMAP Projection Colored by Dominant Group in {pathway} Clusters')
    plt.grid(False)
    plt.savefig(f'{cell_distance_umap_pathway_dir}/umap_{pathway}_color_by_cluster.png', dpi=300)
    plt.close()

def plot_cell_distance_tsne(embedding, combined_df, group_color_map, cell_distance_output_dir):
    print("Plotting t-SNE embedding...")
    colors = combined_df['Cell1_group'].map(group_color_map)
    plt.figure(figsize=(10, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.7, s=1)
    for group, color in group_color_map.items():
        plt.scatter([], [], color=color, label=group, s=50)
    plt.legend(title="Group")
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Projection of Cells Colored by Group')
    plt.grid(False)
    plt.savefig(f'{cell_distance_output_dir}/tsne_cell_distance_color_by_group.png', dpi=300)
    
def plot_cell_distance_tsne_by_pathway(embedding, combined_df, pathway, cell_distance_output_dir):
    print(f"\tPlotting t-SNE embedding for pathway {pathway}...")
    
    cell_distance_tsne_pathway_dir = f'{cell_distance_output_dir}/tsne_pathways_by_cluster'
    
    if not os.path.exists(cell_distance_tsne_pathway_dir):
        os.makedirs(cell_distance_tsne_pathway_dir)

    # Calculate the dominant group for each cluster
    cluster_groups = combined_df.groupby(pathway)['Group'].value_counts().unstack().fillna(0)
    cluster_colors = {}
    for cluster, counts in cluster_groups.iterrows():
        if counts.get('HIV', 0) > counts.get('Healthy', 0):
            cluster_colors[cluster] = '#ff7f0e'  # Orange for HIV-dominant clusters
        else:
            cluster_colors[cluster] = '#1f77b4'  # Blue for Healthy-dominant clusters

    # Map colors to clusters
    pathway_clusters = combined_df[pathway]
    colors = pathway_clusters.map(cluster_colors)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.7, s=1)
    for cluster, color in cluster_colors.items():
        plt.scatter([], [], color=color, label=f'Cluster {cluster}', s=50)
    plt.legend(title="Cluster (Dominant Group)")
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f't-SNE Projection Colored by Dominant Group in {pathway} Clusters')
    plt.grid(False)
    plt.savefig(f'{cell_distance_tsne_pathway_dir}/tsne_{pathway}_color_by_cluster', dpi=300)
    plt.close()
    
def plot_cell_distance_roc_curve(y_test, y_proba, output_dir):
    """Plots and saves the ROC curve."""
    print("Generating ROC curve...")
        
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for random chance
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(False)
    plt.savefig(f'{output_dir}/roc_cell_distance_random_forest.png', dpi=300)
    print(f"ROC AUC: {roc_auc:.2f}\n")

def plot_cell_distance_pr_curve(y_test, y_proba, output_dir):
    """Plots and saves the Precision-Recall curve."""
    print("Generating Precision-Recall (PR) curve...")
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    average_precision = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc="upper right")
    plt.grid(False)
    plt.savefig(f'{output_dir}/pr_cell_distance_random_forest.png', dpi=300)
    print(f"Average Precision (AP): {average_precision:.2f}\n")

def calculate_closest_cluster(cell_file, cluster_1_genes, cluster_2_genes, cluster_1_values, cluster_2_values):
    """
    For each cell, the euclidean distance between the cell’s simulation trajectory
    and the average cluster trajectory for each pathway is calculated. The cell is
    assigned to the cluster it is closest to, and the cell’s distance to each pathway
    cluster is stored in a dataframe.
    """
    # Load cell trajectory file
    cell_data = pd.read_csv(cell_file, header=None)
    cell_genes = cell_data.iloc[:, 0]
    cell_values = cell_data.iloc[:, 1:].values
    
    # Find common genes
    common_genes = sorted(set(cell_genes) & set(cluster_1_genes) & set(cluster_2_genes))
    
    # Filter values by common genes
    cell_filtered = cell_data[cell_data[0].isin(common_genes)].set_index(0).reindex(common_genes).values
    cluster_1_filtered = pd.DataFrame(cluster_1_values, index=cluster_1_genes).reindex(common_genes).values
    cluster_2_filtered = pd.DataFrame(cluster_2_values, index=cluster_2_genes).reindex(common_genes).values
    
    # Calculate Euclidean distances
    distance_to_cluster_1 = np.linalg.norm(cell_filtered - cluster_1_filtered)
    distance_to_cluster_2 = np.linalg.norm(cell_filtered - cluster_2_filtered)
    
    # Assign the closest cluster
    closest_cluster = "Cluster 1" if distance_to_cluster_1 < distance_to_cluster_2 else "Cluster 2"
    return closest_cluster, distance_to_cluster_1, distance_to_cluster_2

def plot_group_colored_scatter(results_df, pathway, organism, output_dir):
    print(f"\tPlotting group-colored scatter for pathway {pathway}...")
    
    single_path_cluster_dist_dir = f'{output_dir}/single_pathway_distance_to_clusters/colored_by_group'
    
    if not os.path.exists(single_path_cluster_dist_dir):
        os.makedirs(single_path_cluster_dist_dir)
        
    default_orange = '#ff7f0e'
    default_blue = '#1f77b4'
    
    # Define the color map for groups
    group_color_map = {'HIV': default_orange, 'Healthy': default_blue}  # Orange for HIV, Blue for Healthy
    
    plt.figure(figsize=(10, 6))
    for group in results_df['Group'].unique():
        group_data = results_df[results_df['Group'] == group]
        plt.scatter(
            group_data[f'Distance to Cluster 1 ({pathway})'],
            group_data[f'Distance to Cluster 2 ({pathway})'],
            label=group,
            color=group_color_map.get(group, 'grey'),
            marker='o',
            alpha=0.7,
            s=15
        )
    
    # Plot settings
    plt.xlabel(f'Distance to Cluster 1 ({pathway})')
    plt.ylabel(f'Distance to Cluster 2 ({pathway})')
    plt.title(f'Cell Clustering Based on Similarity to Clusters for Pathway {pathway}')
    plt.legend(title='Group')
    plt.grid(False)
    plt.savefig(f"{single_path_cluster_dist_dir}/{organism}{pathway}_cell_trajectory_clustering.png", dpi=300)
    plt.close()

def plot_cluster_colored_scatter(results_df, pathway, organism, output_dir):
    print(f"\tPlotting cluster-colored scatter for pathway {pathway}...")
    
    single_path_cluster_dist_dir = f'{output_dir}/single_pathway_distance_to_clusters/colored_by_cluster'
    
    if not os.path.exists(single_path_cluster_dist_dir):
        os.makedirs(single_path_cluster_dist_dir)
    
    # Determine the clusters with the highest number of cells in each group
    hiv_cluster = results_df[results_df['Group'] == 'HIV'][f'Closest Cluster ({pathway})'].value_counts().idxmax()
    healthy_cluster = results_df[results_df['Group'] == 'Healthy'][f'Closest Cluster ({pathway})'].value_counts().idxmax()
    
    default_orange = '#ff7f0e'
    default_blue = '#1f77b4'
    
    # Define the specific color map for identified clusters
    specific_color_map = {hiv_cluster: default_orange, healthy_cluster: default_blue}  # Orange for HIV cluster, Blue for Healthy cluster
    
    plt.figure(figsize=(10, 6))
    for cluster in ['Cluster 1', 'Cluster 2']:
        cluster_data = results_df[results_df[f'Closest Cluster ({pathway})'] == cluster]
        plt.scatter(
            cluster_data[f'Distance to Cluster 1 ({pathway})'],
            cluster_data[f'Distance to Cluster 2 ({pathway})'],
            label=cluster,
            color=specific_color_map.get(cluster, default_orange),
            marker='o',
            alpha=0.7,
            s=15
        )
    
    # Plot settings
    plt.xlabel(f'Distance to Cluster 1 ({pathway})')
    plt.ylabel(f'Distance to Cluster 2 ({pathway})')
    plt.title(f'Cell Clustering Based on Similarity to Clusters for Pathway {pathway}')
    plt.legend(title='Closest Cluster')
    plt.grid(False)
    plt.savefig(f"{single_path_cluster_dist_dir}/{organism}{pathway}_cell_trajectory_clustering.png", dpi=300)
    plt.close()

def plot_cluster_elbow_method(results_df, pathway, organism, output_dir):
    print(f"\tCalculating the optimal number of clusters for pathway {pathway} using the Elbow Method...")
    
    elbow_plot_dir = f'{output_dir}/pathway_elbow_plots'
    
    if not os.path.exists(elbow_plot_dir):
        os.makedirs(elbow_plot_dir)
    
    # Extract distances for the specific pathway
    data = results_df[[f'Distance to Cluster 1 ({pathway})', f'Distance to Cluster 2 ({pathway})']]

    # Determine the range of clusters to test
    wcss = []
    cluster_range = range(1, 10)  # Testing 1 to 10 clusters
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    # Plot the WCSS values to find the "elbow"
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, wcss, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.title(f'Elbow Method for Optimal k (Pathway {pathway})')
    plt.grid(False)
    plt.savefig(f"{elbow_plot_dir}/{organism}{pathway}_num_clusters_kmeans_elbow_plot.png", dpi=300)
    plt.close()

def plot_cluster_roc_curve(y_test, y_proba, output_dir):
    """Plots and saves the ROC curve."""
    print("\tGenerating ROC curve...")
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for random chance
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(False)
    plt.savefig(f'{output_dir}/random_forest_classifier_roc_curve.png', dpi=300)
    print(f"\t\tROC AUC: {roc_auc:.2f}\n")

def plot_cluster_pr_curve(y_test, y_proba, output_dir):
    """Plots and saves the Precision-Recall curve."""
    print("\tGenerating Precision-Recall (PR) curve...")
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    average_precision = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc="upper right")
    plt.grid(False)
    plt.savefig(f'{output_dir}/random_forest_classifier_pr_curve.png', dpi=300)
    print(f"\t\tAverage Precision (AP): {average_precision:.2f}\n")

def plot_cluster_feature_importances(rf_model, X, output_dir):
    print("\tPlotting feature importances from Random Forest model...")
    feature_importances = rf_model.feature_importances_ * 100  # Convert to percentage
    sorted_indices = np.argsort(feature_importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances for Pathway Distances")
    plt.bar(range(X.shape[1]), feature_importances[sorted_indices], align='center')
    plt.xticks(range(X.shape[1]), X.columns[sorted_indices], rotation=45, ha="right")
    plt.xlabel("Pathway Distance")
    plt.ylabel("Importance (%)")  # Label y-axis as percentage
    plt.tight_layout()
    plt.savefig(f'{output_dir}/random_forest_pathway_cluster_importance.png', dpi=300)

def train_and_evaluate_rf(combined_df, distance_columns, output_dir):
    
    random_forest_cluster_distance_dir = f'{output_dir}/random_forest_cluster_distance'
    
    if not os.path.exists(random_forest_cluster_distance_dir):
        os.makedirs(random_forest_cluster_distance_dir)
    
    # Define the feature matrix (X) and target vector (y)
    X = combined_df[distance_columns]
    y = combined_df['Group'].map({'HIV': 1, 'Healthy': 0})
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the Random Forest model
    print("Training Random Forest classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model performance
    print("Evaluating model performance...")
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]  # Probability for the positive class (HIV)
    
    # Plot ROC and PR curves
    plot_cluster_roc_curve(y_test, y_proba, random_forest_cluster_distance_dir)
    plot_cluster_pr_curve(y_test, y_proba, random_forest_cluster_distance_dir)
    
    # Generate and print classification report
    report = classification_report(y_test, y_pred, target_names=['Healthy', 'HIV'], output_dict=True)
    print("Classification Report for Random Forest Model:\n")
    print_classification_report(report, y_test, y_pred, random_forest_cluster_distance_dir)
    
    # Plot feature importances
    plot_cluster_feature_importances(rf_model, X, random_forest_cluster_distance_dir)
    
    # Identify and summarize pathways with the best predictive power
    feature_importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'Pathway': X.columns,
        'Importance': feature_importances * 100  # Convert to percentage
    }).sort_values(by='Importance', ascending=False)
    
    return rf_model, importance_df

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_cluster_tsne_from_features(combined_df, distance_columns, output_dir, group_column="Group", random_state=42):
    """
    For cluster-distance analysis: embed CELLS using per-cell feature vectors
    (distances to cluster 1/2 for each pathway). This is NOT precomputed pairwise distances.
    """
    print("Generating t-SNE plot from per-cell cluster-distance features...")

    os.makedirs(output_dir, exist_ok=True)

    X = combined_df[distance_columns].to_numpy(dtype=float)
    n_samples, n_features = X.shape

    if n_samples < 2:
        raise ValueError(f"Need >=2 cells to embed; got {n_samples}")

    # If only 1 feature, TSNE with init='pca' can fail; use init='random'
    init = "random" if n_features < 2 else "pca"

    # t-SNE perplexity must be < n_samples
    perplexity = min(30, max(2, (n_samples - 1) // 3))

    tsne = TSNE(
        n_components=2,
        init=init,
        random_state=random_state,
        perplexity=perplexity,
    )
    emb = tsne.fit_transform(X)

    group_colors = {"HIV": "#ff7f0e", "Healthy": "#1f77b4"}
    colors = combined_df[group_column].map(group_colors).fillna("grey").to_list()

    plt.figure(figsize=(10, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=colors, alpha=0.8, s=12)

    for g, col in group_colors.items():
        if g in set(combined_df[group_column].astype(str)):
            plt.scatter([], [], color=col, label=g, s=60)

    plt.legend(title="Group")
    plt.title("t-SNE of Cells from Cluster-Distance Features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(False)

    outpath = os.path.join(output_dir, "tsne_cluster_distance_features_by_group.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved: {outpath}")


def print_classification_report(report, y_test, y_pred, output_dir):
    
    # Calculate the TP, TN, FP, FN scores from the predictions
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    with open(f'{output_dir}/random_forest_classification_report.txt', 'w') as file:
    
        file.write(f"Confusion matrix:\n")
        file.write(f"  True Positives: {tp} of HIV cases were correctly predicted to be HIV\n")
        file.write(f"  False Positives: {fp} of HIV cases were incorrectly predicted to be Healthy\n")
        file.write(f"  True Negatives: {tn} of Healthy cases are correctly predicted to be Healthy\n")
        file.write(f"  False Negatives: {fn} of Healthy cases were incorrectly predicted to be HIV\n")
        
        # Class 0 (Healthy)
        file.write("Class 0 (Healthy):\n")
        file.write(f"  Precision: {report['Healthy']['precision']:.2%} — When the model predicts a cell as Healthy, it is correct {report['Healthy']['precision']:.2%} of the time.\n")
        file.write(f"  Recall: {report['Healthy']['recall']:.2%} — Out of all actual Healthy cells, the model correctly identifies {report['Healthy']['recall']:.2%}.\n")
        file.write(f"  F1-Score: {report['Healthy']['f1-score']:.2%} — The harmonic mean of precision and recall, indicating overall effectiveness for the Healthy class.\n\n")

        # Class 1 (HIV)
        file.write("Class 1 (HIV):\n")
        file.write(f"  Precision: {report['HIV']['precision']:.2%} — When the model predicts a cell as HIV, it is correct {report['HIV']['precision']:.2%} of the time.\n")
        file.write(f"  Recall: {report['HIV']['recall']:.2%} — Out of all actual HIV cells, the model correctly identifies {report['HIV']['recall']:.2%}.\n")
        file.write(f"  F1-Score: {report['HIV']['f1-score']:.2%} — Indicates overall effectiveness for the HIV class.\n\n")

        # Overall metrics
        file.write("Overall (Accuracy, Macro, and Weighted Averages):\n")
        file.write(f"  Accuracy: {report['accuracy']:.2%} — The proportion of all correctly classified cells.\n")
        file.write(f"  Macro Average Precision: {report['macro avg']['precision']:.2%}\n")
        file.write(f"  Macro Average Recall: {report['macro avg']['recall']:.2%}\n")
        file.write(f"  Macro Average F1-Score: {report['macro avg']['f1-score']:.2%}\n")
        file.write(f"  Weighted Average Precision: {report['weighted avg']['precision']:.2%}\n")
        file.write(f"  Weighted Average Recall: {report['weighted avg']['recall']:.2%}\n")
        file.write(f"  Weighted Average F1-Score: {report['weighted avg']['f1-score']:.2%}")


from sklearn.cluster import KMeans

def plot_embedding_by_cell_cluster_dominant_group(
    embedding,
    cells,
    groups,                # list aligned with `cells`, values "HIV"/"Healthy"/"Unknown"
    D,                     # cell×cell distance matrix (NxN)
    pathway_label,         # e.g. "hsa05417"
    output_dir,
    method_name="umap",    # "umap" or "tsne"
    n_clusters=2,
    random_state=42,
):
    """
    Cluster cells from the distance matrix (using rows of D as features),
    then color the embedding by dominant group per cluster.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use each cell's distances-to-all-cells as its feature vector
    X = D.astype(np.float64)

    # KMeans on distance-vectors
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_ids = km.fit_predict(X)

    # Determine dominant group per cluster
    cluster_to_color = {}
    for cid in range(n_clusters):
        members = [groups[i] for i in range(len(groups)) if cluster_ids[i] == cid]
        hiv = sum(g == "HIV" for g in members)
        healthy = sum(g == "Healthy" for g in members)

        if hiv > healthy:
            cluster_to_color[cid] = "#ff7f0e"  # HIV-dominant
        else:
            cluster_to_color[cid] = "#1f77b4"  # Healthy-dominant

    colors = [cluster_to_color[cid] for cid in cluster_ids]

    plt.figure(figsize=(10, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.6, s=10)

    # Legend
    plt.scatter([], [], color="#ff7f0e", label="HIV-dominant cluster", s=60)
    plt.scatter([], [], color="#1f77b4", label="Healthy-dominant cluster", s=60)
    plt.legend(title=f"{pathway_label} clusters")

    plt.xlabel(f"{method_name.upper()} 1")
    plt.ylabel(f"{method_name.upper()} 2")
    plt.title(f"{method_name.upper()} colored by dominant group in {pathway_label} clusters")
    plt.grid(False)

    outpath = os.path.join(output_dir, f"{method_name}_{pathway_label}_color_by_cluster_dominant_group.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved: {outpath}")


def run_cell_distance_analysis():
    # ---------------------------------------------------------
    # Output dir
    # ---------------------------------------------------------
    cell_distance_output_dir = f"{output_dir}/cell_distance_analysis"
    os.makedirs(cell_distance_output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Load + merge inputs (pairwise distances + group info)
    # ---------------------------------------------------------
    cell_distance_df, cell_group_df = load_cell_distance_data(cell_group_file, pathway_distance_files)
    cell_distance_df = prepare_combined_df(cell_distance_df, cell_group_df)

    # Distance columns (for CLI you used, this will be 1 column)
    cell_distance_columns = [c for c in cell_distance_df.columns if c.startswith("distance_")]
    if len(cell_distance_columns) == 0:
        raise ValueError("No distance columns found (expected columns like 'distance_05417').")

    if len(cell_distance_columns) != len(pathways):
        print(f"[WARN] Found {len(cell_distance_columns)} distance column(s) but pathways has {len(pathways)} item(s). "
              f"Continuing with detected distance columns: {cell_distance_columns}")

    # ---------------------------------------------------------
    # Build cell×cell distance matrix (Fix 3)
    # If multiple pathways exist, we aggregate distances per pair via mean.
    # For ONE pathway this is just that single distance column.
    # ---------------------------------------------------------
    print("Building cell×cell distance matrix for embedding (precomputed distances)...")

    # unique cell IDs are the strings in Cell1/Cell2
    cells = pd.Index(
        pd.concat([cell_distance_df["Cell1"].astype(str), cell_distance_df["Cell2"].astype(str)], ignore_index=True).unique()
    ).tolist()
    n_cells = len(cells)
    if n_cells < 2:
        raise ValueError(f"Need >=2 cells to embed; found {n_cells}")

    idx = {c: i for i, c in enumerate(cells)}

    # Aggregate distance across pathways for each (Cell1, Cell2) row.
    # For 1 pathway, this is identical to that column.
    pair_dist = np.nanmean(cell_distance_df[cell_distance_columns].to_numpy(dtype=float), axis=1)

    D = np.full((n_cells, n_cells), np.nan, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    c1 = cell_distance_df["Cell1"].astype(str).to_numpy()
    c2 = cell_distance_df["Cell2"].astype(str).to_numpy()

    for a, b, d in zip(c1, c2, pair_dist):
        if np.isnan(d):
            continue
        i, j = idx[a], idx[b]
        D[i, j] = d
        D[j, i] = d

    # Fill missing distances (if any) with max observed distance
    max_d = np.nanmax(D)
    if not np.isfinite(max_d):
        raise ValueError("All distances are NaN; check your distance files and columns.")

    missing = np.isnan(D).sum()
    if missing > 0:
        print(f"[WARN] Distance matrix has {missing} missing entries; filling with max distance={max_d:.4g}")
        D = np.where(np.isnan(D), max_d, D)

    # ---------------------------------------------------------
    # Make a per-cell group vector for coloring
    # Prefer group labels derived from the mapping in prepare_combined_df
    # ---------------------------------------------------------
    cell_to_group = {}

    # If prepare_combined_df created Cell1_group / Cell2_group, use them.
    if "Cell1_group" in cell_distance_df.columns:
        for a, g in zip(cell_distance_df["Cell1"].astype(str), cell_distance_df["Cell1_group"]):
            if pd.notna(g) and a not in cell_to_group:
                cell_to_group[a] = str(g)
    if "Cell2_group" in cell_distance_df.columns:
        for b, g in zip(cell_distance_df["Cell2"].astype(str), cell_distance_df["Cell2_group"]):
            if pd.notna(g) and b not in cell_to_group:
                cell_to_group[b] = str(g)

    groups = [cell_to_group.get(c, "Unknown") for c in cells]

    # ---------------------------------------------------------
    # Embeddings: UMAP + t-SNE on PRECOMPUTED distances
    # ---------------------------------------------------------
    print("Generating UMAP and t-SNE embeddings from precomputed cell×cell distances...")

    umap_embedding = umap.UMAP(metric="precomputed", n_jobs=-1).fit_transform(D)

    # TSNE with precomputed distances must use init="random"
    perplexity = min(30, max(2, (n_cells - 1) // 3))  # safe choice
    tsne_embedding = TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        random_state=42,
        perplexity=perplexity,
    ).fit_transform(D)

    group_color_map = {"HIV": "#ff7f0e", "Healthy": "#1f77b4", "Unknown": "grey"}

    # Plot UMAP
    print("Plotting UMAP embedding (per cell)...")
    plt.figure(figsize=(10, 6))
    plt.scatter(
        umap_embedding[:, 0],
        umap_embedding[:, 1],
        c=[group_color_map.get(g, "grey") for g in groups],
        alpha=0.7,
        s=10,
    )
    for g, col in group_color_map.items():
        if g in set(groups):
            plt.scatter([], [], color=col, label=g, s=60)
    plt.legend(title="Group")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("UMAP of Cells from Precomputed Cell–Cell Distances")
    plt.grid(False)
    plt.savefig(f"{cell_distance_output_dir}/umap_cells_precomputed_distance_by_group.png", dpi=300)
    plt.close()

    # Plot t-SNE
    print("Plotting t-SNE embedding (per cell)...")
    plt.figure(figsize=(10, 6))
    plt.scatter(
        tsne_embedding[:, 0],
        tsne_embedding[:, 1],
        c=[group_color_map.get(g, "grey") for g in groups],
        alpha=0.7,
        s=10,
    )
    for g, col in group_color_map.items():
        if g in set(groups):
            plt.scatter([], [], color=col, label=g, s=60)
    plt.legend(title="Group")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE of Cells from Precomputed Cell–Cell Distances")
    plt.grid(False)
    plt.savefig(f"{cell_distance_output_dir}/tsne_cells_precomputed_distance_by_group.png", dpi=300)
    plt.close()
    
    # For each pathway (works fine for one)
    for pathway in pathways:
        pathway_label = f"{organism}{pathway}"  # e.g. hsa05417

        plot_embedding_by_cell_cluster_dominant_group(
            embedding=umap_embedding,
            cells=cells,
            groups=groups,
            D=D,
            pathway_label=pathway_label,
            output_dir=f"{cell_distance_output_dir}/umap_pathways_by_cluster",
            method_name="umap",
            n_clusters=2,
        )

        plot_embedding_by_cell_cluster_dominant_group(
            embedding=tsne_embedding,
            cells=cells,
            groups=groups,
            D=D,
            pathway_label=pathway_label,
            output_dir=f"{cell_distance_output_dir}/tsne_pathways_by_cluster",
            method_name="tsne",
            n_clusters=2,
        )


    # ---------------------------------------------------------
    # Random Forest classification (IMPORTANT NOTE)
    # Your existing RF uses pairwise rows as samples.
    # That is a "pair classifier", not a "cell classifier".
    # Keeping it, but making it robust to a single pathway.
    # ---------------------------------------------------------
    print("Training Random Forest classifier (pairwise rows)...")
    X = cell_distance_df[cell_distance_columns]
    if "Group" not in cell_distance_df.columns:
        raise ValueError("Expected a 'Group' column in cell_distance_df for RF. "
                         "If you intended to classify cells, we should rebuild an N×P per-cell feature table instead.")
    y = cell_distance_df["Group"].map({"HIV": 1, "Healthy": 0})

    # Drop any rows with unknown labels
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)

    print("Evaluating model performance...")
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]

    cell_distance_random_forest_dir = f"{cell_distance_output_dir}/random_forest_cell_distance"
    os.makedirs(cell_distance_random_forest_dir, exist_ok=True)

    plot_cell_distance_roc_curve(y_test, y_proba, cell_distance_random_forest_dir)
    plot_cell_distance_pr_curve(y_test, y_proba, cell_distance_random_forest_dir)

    report = classification_report(y_test, y_pred, target_names=["Healthy", "HIV"], output_dict=True)
    print_classification_report(report, y_test, y_pred, cell_distance_random_forest_dir)

    print("Cell distance analysis complete.")


def run_cell_cluster_analysis():
    # Create a subdirectory to store cluster distance analysis results
    cluster_distance_output_dir = f'{output_dir}/cluster_distance_analysis'
    if not os.path.exists(cluster_distance_output_dir):
        os.makedirs(cluster_distance_output_dir)
    
    combined_results = None
    
    # ----- Cluster distance analysis results -----
    for pathway in pathways:
        print(f'Clustering for pathway {pathway}')
        
        # Path to the trajectory files for the current pathway
        pathway_dir = f'{file_paths["trajectories"]}/{dataset_name}_{organism}{pathway}/text_files'
        
        cell_trajectory_dir = f'{pathway_dir}/cell_trajectories/'

        # Set paths for cluster summaries
        cluster_1_file = f"{pathway_dir}/cluster_summaries/cluster_1_summary.csv"
        cluster_2_file = f"{pathway_dir}/cluster_summaries/cluster_2_summary.csv"

        # Load cluster summaries
        cluster_1_genes, cluster_1_values = load_cluster_summary(cluster_1_file)
        cluster_2_genes, cluster_2_values = load_cluster_summary(cluster_2_file)

        # Load the cell groups
        cell_group_df = pd.read_csv(cell_group_file, sep=',', header=0, index_col=0)

        # Collect distances for each cell in this pathway
        results = []
        for cell_file in os.listdir(cell_trajectory_dir):
            if cell_file.endswith(".csv"):
                cell_path = os.path.join(cell_trajectory_dir, cell_file)
                cell_num = int(cell_file.split('_')[1:2][0])
                group = str(cell_group_df.loc[cell_group_df['Cell'] == cell_num, 'Group'].values[0])
                
                # Calculate distances to each cluster
                closest_cluster, dist_cluster_1, dist_cluster_2 = calculate_closest_cluster(cell_path, cluster_1_genes, cluster_2_genes, cluster_1_values, cluster_2_values)
                
                # Append pathway-specific distances
                results.append({
                    "Cell": cell_num,
                    "Group": group,
                    f"Closest Cluster ({pathway})": closest_cluster,
                    f"Distance to Cluster 1 ({pathway})": dist_cluster_1,
                    f"Distance to Cluster 2 ({pathway})": dist_cluster_2
                })

        # Convert pathway-specific results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Plot group-colored scatter plot
        plot_group_colored_scatter(results_df, pathway, organism, output_dir=cluster_distance_output_dir)
        
        # Plot cluster-colored scatter plot
        plot_cluster_colored_scatter(results_df, pathway, organism, output_dir=cluster_distance_output_dir)
        
        # Plot the Elbow Method for determining optimal k
        plot_cluster_elbow_method(results_df, pathway, organism, output_dir=cluster_distance_output_dir)
        
        # Merge results with combined_results on Cell and Group, handling the initial None case
        if combined_results is None:
            combined_results = results_df  # First assignment
        else:
            combined_results = pd.merge(combined_results, results_df, on=["Cell", "Group"], how="outer")

    # Final combined DataFrame with all pathways' distances
    combined_df = pd.DataFrame(combined_results)
    combined_df = combined_df.dropna()

    # Identify the distance columns for further analysis (if needed)
    distance_columns = [col for col in combined_df.columns if 'Distance' in col]

    # Save combined DataFrame to CSV
    combined_df.to_csv(f"{cluster_distance_output_dir}/combined_pathway_distances.csv", index=False)
    print("Combined results saved to 'combined_pathway_distances.csv'")

    plot_cluster_tsne_from_features(combined_df, distance_columns, cluster_distance_output_dir, group_column="Group")

    # Run the function for training and evaluating the Random Forest model
    rf_model, importance_df = train_and_evaluate_rf(combined_df, distance_columns, cluster_distance_output_dir)

if __name__ == "__main__":
    # Set the logging level for output
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # Allow the user to either add in the dataset name and network name from the command line or as a prompt
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=False,
        default="",
        help='Name of the dataset for loading the correct ruleset pickle files'
    )
    parser.add_argument(
        "--list_of_kegg_pathways",
        nargs="+",
        type=str,
        help="Which KEGG pathways should scBonita download? Specify the five letter pathway IDs.",
        required=False
    )
    parser.add_argument(
        "--organism",
        type=str,
        help="Three-letter organism code. Which organism is the dataset derived from?",
        default="hsa",
        required=False,
        metavar='organism code'
    )
    
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    pathways = args.list_of_kegg_pathways
    organism = args.organism

    output_dir = file_paths['trajectory_analysis']
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the cell-cell distances for each patthway
    pathway_distance_files = [f'{file_paths["trajectories"]}/{dataset_name}_{organism}{pathway}/text_files/distances.csv'for pathway in pathways]
    
    # Load the cell clustering file, which contains the cell number, its group, and the cluster it belongs to for each pathway
    cell_group_file = f'{file_paths["trajectories"]}/{dataset_name}_cell_groups.csv'
    
    run_cell_distance_analysis()
    
    run_cell_cluster_analysis()

