import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

# Step 1: Load the .npz file
def load_embeddings(file_path):
    data = np.load(file_path)
    print("Available arrays in the .npz file:", data.files)
    embeddings = data['arr_0']  # Replace 'embeddings' with the correct key if needed
    return embeddings

# Step 2: Reduce dimensionality
def reduce_dimensionality(embeddings, method='pca', n_components=2):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components)
    else:
        raise ValueError("Unsupported method. Use 'pca', 'tsne', or 'umap'.")
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

# Step 3: Visualize embeddings
def visualize_embeddings(embeddings_2d, labels=None, title="2D Visualization", save_path=None):
    plt.figure(figsize=(10, 8))
    if labels is not None:
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Labels')
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Main function
def main():
    # Load embeddings
    embeddings = load_embeddings('embeddings/embeddings.npz')  # Replace with your .npz file path

    # Optional: Generate dummy labels if none are available
    labels = np.random.randint(0, 5, size=len(embeddings))  # Replace with actual labels if available

    # Reduce dimensionality using different methods
    methods = ['pca', 'tsne', 'umap']
    for method in methods:
        embeddings_2d = reduce_dimensionality(embeddings, method=method)
        visualize_embeddings(
            embeddings_2d,
            labels=labels,
            title=f'2D Visualization using {method.upper()}',
            save_path=f'embeddings_{method}.png'
        )

if __name__ == "__main__":
    main()