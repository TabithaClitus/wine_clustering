🍷 Wine Clustering Project

This project applies unsupervised machine learning techniques to cluster wine data and visualize the results through a user-friendly Gradio web interface with login functionality.

🔗 Live Demo: Wine Clustering App on Hugging Face Spaces

https://huggingface.co/spaces/tabithaclitus/wine-clustering-app

🔍 Features

Data Preprocessing: Standardization and dimensionality reduction (PCA).

Clustering Algorithms:

K-Means

Hierarchical Clustering (Agglomerative)

DBSCAN (tuned with eps & min_samples)

Evaluation Metrics:

Silhouette Score

Davies–Bouldin Index

Calinski–Harabasz Index

Visualization:

PCA-based 2D plots of cluster results

Comparison charts of clustering metrics

<img width="1489" height="495" alt="image" src="https://github.com/user-attachments/assets/1829b486-67ef-4c19-9117-f3c616be70f7" />


Frontend:

Built with Gradio

Login functionality for secure access

Interactive clustering parameter selection

📊 Results Summary

K-Means: Stable clusters, good CH score

Hierarchical: Similar to K-Means, slightly less separation

DBSCAN: Best Silhouette Score (≈ 0.84 with tuned parameters), but many small clusters

🚀 Getting Started
1️⃣ Clone the Repository
git clone https://github.com/yourusername/wine-clustering.git
cd wine-clustering

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App
python app.py


The Gradio interface will launch in your browser.

🛠 Tech Stack

Python 3.x

Scikit-learn (clustering, PCA, metrics)

Matplotlib / Seaborn (visualizations)

Gradio (frontend interface)

📸 Sample Screenshots

<img width="1888" height="796" alt="Screenshot 2025-09-08 062108" src="https://github.com/user-attachments/assets/c320f30e-cb60-4c0c-b22d-8ecd0ec15dc4" />

<img width="1874" height="836" alt="Screenshot 2025-09-08 062124" src="https://github.com/user-attachments/assets/f18859d5-a734-41ee-ab10-e0c9ba011b60" />

✅ Conclusion

This project demonstrates the power of unsupervised machine learning in discovering hidden patterns within wine data. By applying K-Means, Hierarchical Clustering, and DBSCAN, we explored different clustering strategies and compared their effectiveness using metrics like Silhouette Score and Davies–Bouldin Index.

Among the algorithms, DBSCAN with tuned parameters achieved the highest Silhouette Score (~0.84), highlighting its strength in detecting well-separated clusters without needing to predefine the number of clusters.

The addition of a Gradio web interface makes the project interactive and user-friendly, allowing anyone to experiment with clustering parameters and visualize results in real time. This approach bridges the gap between technical analysis and practical usability, making the project both educational and accessible.

In summary, the Wine Clustering App is a valuable tool for understanding unsupervised learning, dimensionality reduction, and cluster evaluation — all presented through a simple and engaging web interface.
