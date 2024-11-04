import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
import os

file_paths = [
    "../results/code_embeddings/code_embeddings_base_model.json",
    "../results/code_embeddings/code_embeddings_base_model_pool.json",
    "../results/code_embeddings/code_embeddings_clone_detection_model.json",
    "../results/code_embeddings/code_embeddings_clone_detection_model_pool.json"
]

output_dir = "../results"
os.makedirs(output_dir, exist_ok=True)

def create_tsne_visualization(file_path, output_image_name):
    with open(file_path, "r") as file:
        embeddings_dict = json.load(file)

    labels = list(embeddings_dict.keys())
    embeddings = np.array([np.array(embedding) for embedding in embeddings_dict.values()])

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    tsne = TSNE(n_components=3, random_state=42, perplexity=5, n_iter_without_progress=1000, metric="cosine", method="exact")
    reduced_embeddings = tsne.fit_transform(embeddings)

    # File for saving distances
    distances_file_path = os.path.join(output_dir+"/text_files", output_image_name + "_distances.txt")
    with open(distances_file_path, "w") as dist_file:

        # Calculate pairwise distances and find closest pairs
        pair_distances = []
        for i in range(len(reduced_embeddings)):
            for j in range(i + 1, len(reduced_embeddings)):
                distance = euclidean(reduced_embeddings[i], reduced_embeddings[j])
                pair_distances.append((labels[i], labels[j], distance))

        pair_distances.sort(key=lambda x: x[2])

        # Write closest pairs with distances to file
        dist_file.write(f"Closest pairs with distances for {output_image_name}:\n")
        closest_pairs = pair_distances[:5]
        for label1, label2, dist in closest_pairs:
            dist_file.write(f"Pair: ({label1}, {label2}) - Distance: {dist:.2f}\n")
            print(f"Pair: ({label1}, {label2}) - Distance: {dist:.2f}")

        # Calculate and write specific pair distances
        dist_file.write(f"\nSpecific pair distances for {output_image_name}:\n")
        print(f"\nSpecific pair distances for {output_image_name}:")
        def calculate_specific_distance(label_a, label_b):
            idx_a, idx_b = labels.index(label_a), labels.index(label_b)
            distance = euclidean(reduced_embeddings[idx_a], reduced_embeddings[idx_b])
            dist_file.write(f"Pair: ({label_a}, {label_b}) - Distance: {distance:.2f}\n")
            print(f"Pair: ({label_a}, {label_b}) - Distance: {distance:.2f}")

        for i in range(118, 123):
            calculate_specific_distance(f"{i}_1", f"{i}_2")

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, label in enumerate(labels):
        x, y, z = reduced_embeddings[i, 0], reduced_embeddings[i, 1], reduced_embeddings[i, 2]
        ax.scatter(x, y, z)
        offset_x, offset_y, offset_z = (i % 3) * 0.02, (i % 3) * 0.02, (i % 3) * 0.02
        ax.text(x + offset_x, y + offset_y, z + offset_z, label, fontsize=8, alpha=0.75)

    ax.set_title(f"3D t-SNE Visualization of {output_image_name}")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_zlabel("t-SNE Dimension 3")

    # Save the plot
    output_path = os.path.join(output_dir+"/images", output_image_name + ".png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved 3D t-SNE plot for {output_image_name} to {output_path}")

for file_path in file_paths:
    output_image_name = os.path.basename(file_path).replace(".json", "")
    create_tsne_visualization(file_path, output_image_name)
