# gradio_clustering_app_complete.py
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import gradio as gr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# -------------------------
# Demo users
# -------------------------
USERS = {"student": "password123", "admin": "adminpass"}

# -------------------------
# Load & scale data
# -------------------------
SCALED_PATH = "wine_clustering_expanded_scaled.csv"
RAW_PATH = "wine_clustering_expanded_raw.csv"
ORIG_RAW = "wine-clustering.csv"

def load_scaled_data():
    if os.path.exists(SCALED_PATH):
        df_scaled = pd.read_csv(SCALED_PATH)
    else:
        if os.path.exists(RAW_PATH):
            df_raw = pd.read_csv(RAW_PATH)
        elif os.path.exists(ORIG_RAW):
            df_small = pd.read_csv(ORIG_RAW)
            df_raw = df_small.sample(n=2000, replace=True, random_state=42).reset_index(drop=True)
            df_raw = df_raw + np.random.normal(0, 0.02, df_raw.shape)
        else:
            raise FileNotFoundError("No dataset found. Place a dataset CSV in the working directory.")
        scaler = StandardScaler()
        scaled_arr = scaler.fit_transform(df_raw)
        df_scaled = pd.DataFrame(scaled_arr, columns=df_raw.columns)
        df_scaled.to_csv(SCALED_PATH, index=False)
    return df_scaled

X_df = load_scaled_data()
X = X_df.values

# Pre-fit PCA(2)
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X)

# -------------------------
# Helper functions
# -------------------------
def compute_metrics(X_arr, labels):
    unique = set(labels)
    if len(unique) <= 1 or (len(unique) == 2 and -1 in unique and list(unique)[0] == -1):
        return None, None, None
    sil = silhouette_score(X_arr, labels)
    dbi = davies_bouldin_score(X_arr, labels)
    chi = calinski_harabasz_score(X_arr, labels)
    return float(sil), float(dbi), float(chi)

def plot_pca_clusters(labels, title="Clusters (PCA 2D)", show_legend=False):
    fig, ax = plt.subplots(figsize=(6,5))
    unique_labels = sorted(set(labels))
    cmap = plt.cm.tab20
    color_vals = cmap(np.linspace(0, 1, max(2, len(unique_labels))))
    for idx, lab in enumerate(unique_labels):
        if lab == -1:
            color = 'k'
            alpha = 0.6
            size = 8
        else:
            color = color_vals[idx % len(color_vals)]
            alpha = 0.8
            size = 20
        mask = (labels == lab)
        ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1], c=[color], s=size, alpha=alpha, edgecolors='none')
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(title)
    if show_legend:
        handles = []
        for lab in unique_labels[:10]:
            lbl = "Noise" if lab == -1 else f"Cluster {lab}"
            handles.append(ax.scatter([], [], c=('k' if lab == -1 else color_vals[unique_labels.index(lab) % len(color_vals)]), label=lbl))
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf

def run_clustering(algo, n_clusters, eps, min_samples):
    algo = algo or "K-Means"
    if algo == "K-Means":
        model = KMeans(n_clusters=int(n_clusters), random_state=42)
        labels = model.fit_predict(X)
    elif algo == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=int(n_clusters), linkage="ward")
        labels = model.fit_predict(X)
    elif algo == "DBSCAN":
        model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
        labels = model.fit_predict(X)
    else:
        return "Invalid algorithm selected.", None

    sil, dbi, chi = compute_metrics(X, labels)
    if sil is None:
        metrics_text = "⚠️ Invalid clustering result (only 1 cluster or all noise). Metrics unavailable."
    else:
        metrics_text = f"**Silhouette:** {sil:.3f}  \n**Davies–Bouldin:** {dbi:.3f}  \n**Calinski–Harabasz:** {chi:.1f}"
    plot_buf = plot_pca_clusters(labels, title=f"{algo} (k={n_clusters}, eps={eps}, min_samples={min_samples})")
    return metrics_text, plot_buf

# -------------------------
# Login helpers
# -------------------------
def do_login(username, password):
    if username in USERS and USERS[username] == password:
        return True, f"✅ Welcome, {username}!", True
    else:
        return False, "❌ Invalid username or password", False

def _on_login(u, p):
    ok, msg, show = do_login(u, p)
    return (
        msg,
        gr.update(visible=show),  # algo_dropdown
        gr.update(visible=show),  # k_slider
        gr.update(visible=show),  # eps_slider
        gr.update(visible=show),  # min_samples_slider
        gr.update(visible=show),  # run_btn
        gr.update(visible=show),  # metrics_md
        gr.update(visible=show),  # img_out
    )

def _on_run(algo, k, eps, min_samples):
    metrics_text, plot_buf = run_clustering(algo, k, eps, min_samples)
    img = Image.open(plot_buf).convert("RGB")
    return metrics_text, img

# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks(css=".gradio-container { max-width: 1100px; margin: auto; }") as demo:
    gr.Markdown("# 🍷 Wine Clustering Explorer")
    gr.Markdown("Login to access clustering tools (demo auth).")

    with gr.Row():
        with gr.Column(scale=1):
            username = gr.Textbox(label="Username", placeholder="Enter username")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_message = gr.Markdown("")
        with gr.Column(scale=2):
            algo_dropdown = gr.Dropdown(choices=["K-Means", "Hierarchical", "DBSCAN"], label="Algorithm", value="K-Means", visible=False)
            k_slider = gr.Slider(2, 10, value=3, step=1, label="Number of clusters (k)", visible=False)
            eps_slider = gr.Slider(0.1, 3.0, value=1.5, step=0.1, label="DBSCAN eps", visible=False)
            min_samples_slider = gr.Slider(1, 50, value=5, step=1, label="DBSCAN min_samples", visible=False)
            run_btn = gr.Button("Run Clustering", visible=False)
            metrics_md = gr.Markdown("", visible=False)
            img_out = gr.Image(type="pil", label="PCA visualization", visible=False)

    # Events
    login_btn.click(
        _on_login,
        inputs=[username, password],
        outputs=[login_message, algo_dropdown, k_slider, eps_slider, min_samples_slider, run_btn, metrics_md, img_out]
    )
    run_btn.click(
        _on_run,
        inputs=[algo_dropdown, k_slider, eps_slider, min_samples_slider],
        outputs=[metrics_md, img_out]
    )

# Launch
if __name__ == "__main__":
    demo.launch(share=True)

