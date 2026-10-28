# Saves PNGs + CSVs with feature names, loadings, cluster means, and 2-D coords
# Command to run: python plotting.py --pickle .\valid_experiments\experiment_X\iterX_dict_cluster_df.pickle --out .\output_plots\experimentX_iterX --title "Experiment X — Iter X"

import matplotlib
matplotlib.use("Agg")  

import argparse, os, json, pickle, traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def build_dataframe(obj):
    """
    Accepts:
      - dict: {cluster_id: DataFrame}
      - DataFrame with a 'cluster' column
    Returns a combined DataFrame with a 'cluster' column.
    """
    if isinstance(obj, dict):
        frames = []
        for cid, df in obj.items():
            d = df.copy()
            d["cluster"] = int(cid)
            frames.append(d)
        return pd.concat(frames, ignore_index=True)
    elif isinstance(obj, pd.DataFrame):
        if "cluster" not in obj.columns:
            raise ValueError("DataFrame lacks required 'cluster' column.")
        return obj.copy()
    else:
        raise TypeError(f"Unsupported pickle content type: {type(obj)}")

def main():
    ap = argparse.ArgumentParser(description="Plot clusters and export feature info from iterX_dict_cluster_df.pickle")
    ap.add_argument("--pickle", required=True, help="Path to iterX_dict_cluster_df.pickle")
    ap.add_argument("--out", required=True, help="Output folder for PNGs/CSVs") # Geändert zu required
    ap.add_argument("--title", default=None, help='Optional title (e.g., "Experiment 4 — Iter 3")')
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    pickle_path = os.path.abspath(args.pickle)
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle not found: {pickle_path}")

    obj = load_pickle(pickle_path)
    full = build_dataframe(obj)

    # Choose numeric features (exclude the cluster label)
    num_cols = [c for c in full.select_dtypes(include=[np.number]).columns if c != "cluster"]
    if not num_cols:
        raise ValueError("No numeric feature columns found (after excluding 'cluster').")

    # Print feature names with indices
    print("\nFeatures used (numeric):")
    for i, name in enumerate(num_cols):
        print(f"{i:>2}: {name}")

    # Save the feature list
    pd.Series(num_cols, name="feature").to_csv(os.path.join(out_dir, "features_used.csv"), index=False)

    # Build X/y
    X = full[num_cols].to_numpy()
    y = full["cluster"].to_numpy()

    # Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ---- PCA 2D ----
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)

    # Save PCA scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=y, s=10, alpha=0.7)
    plt.colorbar(label="Cluster")
    plt.title(f"{args.title} Clusters (PCA)" if args.title else "Clusters (PCA)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.grid(True, linestyle="--", alpha=0.4)
    pca_png = os.path.join(out_dir, "clusters_pca.png")
    plt.tight_layout(); plt.savefig(pca_png, dpi=300, bbox_inches="tight"); plt.close()
    print("Saved:", pca_png)

    # Save PCA 2D coordinates
    pca_2d = pd.DataFrame(Z, columns=["PC1", "PC2"])
    pca_2d["cluster"] = y
    pca_2d.to_csv(os.path.join(out_dir, "pca_2d_coords.csv"), index=False)

    # PCA loadings (feature contributions to PC1/PC2)
    loadings = pd.DataFrame(pca.components_, columns=num_cols, index=["PC1", "PC2"]).T
    loadings.to_csv(os.path.join(out_dir, "pca_loadings.csv"))
    print("\nTop contributors to PC1:")
    print(loadings["PC1"].abs().sort_values(ascending=False).head(10))
    print("\nTop contributors to PC2:")
    print(loadings["PC2"].abs().sort_values(ascending=False).head(10))

    # ---- Optional UMAP ----
    umap_png = None
    try:
        import umap
        Zu = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42).fit_transform(Xs)
        plt.figure(figsize=(8, 6))
        plt.scatter(Zu[:, 0], Zu[:, 1], c=y, s=10, alpha=0.7)
        plt.colorbar(label="Cluster")
        plt.title(f"{args.title} Clusters (UMAP)" if args.title else "Clusters (UMAP)")
        plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
        plt.grid(True, linestyle="--", alpha=0.4)
        umap_png = os.path.join(out_dir, "clusters_umap.png")
        plt.tight_layout(); plt.savefig(umap_png, dpi=300, bbox_inches="tight"); plt.close()
        print("Saved:", umap_png)

        # Save UMAP 2D coordinates
        umap_2d = pd.DataFrame(Zu, columns=["UMAP1", "UMAP2"])
        umap_2d["cluster"] = y
        umap_2d.to_csv(os.path.join(out_dir, "umap_2d_coords.csv"), index=False)
    except Exception as e:
        print("[warn] UMAP skipped:", e)

    # ---- Cluster sizes ----
    counts = Counter(y)
    ks = sorted(counts.keys())
    vs = [counts[k] for k in ks]

    plt.figure(figsize=(6, 4))
    plt.bar(ks, vs)
    plt.xlabel("Cluster"); plt.ylabel("#samples")
    plt.title(f"{args.title} Cluster Sizes" if args.title else "Cluster Sizes")
    sizes_png = os.path.join(out_dir, "cluster_sizes.png")
    plt.tight_layout(); plt.savefig(sizes_png, dpi=300, bbox_inches="tight"); plt.close()
    print("Saved:", sizes_png)

    # ---- Cluster-wise feature means (interpretation) ----
    means = full.groupby("cluster")[num_cols].mean()
    means.to_csv(os.path.join(out_dir, "cluster_feature_means.csv"))
    print("Saved:", os.path.join(out_dir, "cluster_feature_means.csv"))

    # ---- Tiny report ----
    report = {
        "pickle": pickle_path,
        "out_dir": out_dir,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_clusters": int(len(counts)),
        "cluster_sizes": {int(k): int(v) for k, v in counts.items()},
        "features_used_csv": os.path.join(out_dir, "features_used.csv"),
        "pca_loadings_csv": os.path.join(out_dir, "pca_loadings.csv"),
        "cluster_feature_means_csv": os.path.join(out_dir, "cluster_feature_means.csv"),
        "pca_2d_coords_csv": os.path.join(out_dir, "pca_2d_coords.csv"),
        "umap_2d_coords_csv": os.path.join(out_dir, "umap_2d_coords.csv") if umap_png else None,
        "outputs": {
            "pca_scatter_png": pca_png,
            "umap_scatter_png": umap_png,
            "cluster_sizes_png": sizes_png
        }
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\nAll done. See:", out_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Leave a breadcrumb if anything fails
        try:
            os.makedirs("plots", exist_ok=True)
            with open(os.path.join("plots", "error.log"), "w") as f:
                f.write(traceback.format_exc())
            print("ERROR. See:", os.path.abspath(os.path.join("plots", "error.log")))
        except Exception:
            pass
        raise
