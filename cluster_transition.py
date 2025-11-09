import os
import glob
import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
import dask.array as da
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from matplotlib.colors import ListedColormap, to_rgb

# ------------------------------
# Helper Functions
# ------------------------------

def _get_extent_from_dataset(ds: xr.Dataset) -> tuple:
    """
    Get spatial extent (xmin, xmax, ymin, ymax) from an xarray.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 'x' and 'y' coordinates.

    Returns
    -------
    tuple
        (xmin, xmax, ymin, ymax)
    """
    x = ds['x'].values
    y = ds['y'].values
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    return xmin, xmax, ymin, ymax


# ------------------------------
# Core Functions
# ------------------------------

def load_dataset(base_dir: str, indices: list, start_year: int, end_year: int, 
                 band: int = 4, chunks: dict = None) -> xr.Dataset:
    """
    Load multiple GeoTIFF time series into a single xarray.Dataset.

    Parameters
    ----------
    base_dir : str
        Base directory containing subfolders for each index.
    indices : list
        List of index folder names to load.
    start_year : int
        First year to include.
    end_year : int
        Last year to include.
    band : int, optional
        Band to extract if multi-band raster, by default 4
    chunks : dict, optional
        Chunking for dask arrays, by default None

    Returns
    -------
    xr.Dataset
        Combined dataset containing all indices and years.
    """
    data_vars = {}

    for idx in indices:
        index_path = os.path.join(base_dir, idx)
        tif_files = sorted(glob.glob(os.path.join(index_path, '*.tif')))

        # Extract years from filenames
        try:
            years = [int(os.path.splitext(os.path.basename(f))[0]) for f in tif_files]
        except ValueError:
            raise ValueError(f"Failed to extract years from filenames in {index_path}. "
                             "Ensure filenames are formatted as '<year>.tif'.")

        # Filter files by year range
        selected = [(f, y) for f, y in zip(tif_files, years) if start_year <= y <= end_year]
        if not selected:
            raise ValueError(f"No files found in range {start_year}-{end_year} for index '{idx}'.")

        selected_files, selected_years = zip(*selected)

        # Load rasters
        data_arrays = []
        for f in selected_files:
            da_ = rioxarray.open_rasterio(f, chunks=chunks)
            if da_.rio.count > 1:
                da_ = da_.isel(band=band - 1)
                da_.attrs['long_name'] = da_.attrs['long_name'][band-1]
            da_ = da_.squeeze(drop=True)
            data_arrays.append(da_)

        combined = xr.concat(data_arrays, dim=pd.Index(selected_years, name='time'))
        data_vars[idx] = combined

    ds = xr.Dataset(data_vars)
    ds = ds.drop_vars('band', errors='ignore')

    # Ensure consistent CRS
    reference_var = list(ds.data_vars)[0]
    crs = ds[reference_var].rio.crs
    for v in ds.data_vars:
        ds[v] = ds[v].rio.write_crs(crs)

    return ds


def prepare_data(ds: xr.Dataset, variables: list):
    """
    Prepare data for clustering: stack spatial dimensions and remove NaNs.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to prepare.
    variables : list
        List of variable names to include.

    Returns
    -------
    X_valid : np.ndarray
        2D array of valid (non-NaN) samples.
    mask : np.ndarray
        Boolean mask of invalid samples.
    n_samples : int
        Total number of samples before masking.
    """
    data = ds[variables]
    if 'time' in data.dims:
        data = data.isel(time=0)

    arr = data.to_array().squeeze()
    arr = arr.stack(z=("y", "x"))

    dims = list(arr.dims)
    if 'z' in dims and 'variable' in dims:
        arr = arr.transpose('z', 'variable')
    else:
        raise ValueError(f"Unexpected dims after stacking: {dims}")

    stacked = da.compute(arr.values)[0]
    mask = np.any(np.isnan(stacked), axis=1)
    X_valid = stacked[~mask]

    return X_valid, mask, stacked.shape[0]


def run_kmeans(X_valid: np.ndarray, mask: np.ndarray, n_clusters: int = 5, 
               batch_size: int = 10000, random_state: int = 42):
    """
    Perform MiniBatchKMeans clustering.

    Parameters
    ----------
    X_valid : np.ndarray
        Data array without NaNs.
    mask : np.ndarray
        Boolean mask indicating invalid samples.
    n_clusters : int
        Number of clusters.
    batch_size : int
        Batch size for MiniBatchKMeans.
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    labels_full : np.ndarray
        Array of cluster labels including NaNs.
    kmeans_model : MiniBatchKMeans
        Fitted KMeans model.
    """
    kmeans_model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state)
    labels_full = np.full(mask.shape, np.nan)
    labels_full[~mask] = kmeans_model.fit_predict(X_valid)
    return labels_full, kmeans_model


def align_clusters(ref_kmeans, target_kmeans, target_labels: np.ndarray) -> np.ndarray:
    """
    Align clusters of a target model to a reference model using the Hungarian algorithm.

    Parameters
    ----------
    ref_kmeans : MiniBatchKMeans
        Reference KMeans model.
    target_kmeans : MiniBatchKMeans
        Target KMeans model to align.
    target_labels : np.ndarray
        Target cluster labels to realign.

    Returns
    -------
    np.ndarray
        Aligned cluster labels.
    """
    ref_centroids = ref_kmeans.cluster_centers_
    tgt_centroids = target_kmeans.cluster_centers_
    cost_matrix = np.linalg.norm(ref_centroids[:, None, :] - tgt_centroids[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    aligned_labels = target_labels.copy()
    for i, j in zip(row_ind, col_ind):
        aligned_labels[target_labels == j] = i
    return aligned_labels


def compute_transition_matrix(labels_t1: np.ndarray, labels_t2: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Compute cluster transition matrix between two time periods.

    Parameters
    ----------
    labels_t1 : np.ndarray
        Cluster labels at time 1.
    labels_t2 : np.ndarray
        Cluster labels at time 2.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Transition matrix of shape (n_clusters, n_clusters)
    """
    transition_matrix = np.zeros((n_clusters, n_clusters), dtype=int)
    for i in range(n_clusters):
        for j in range(n_clusters):
            transition_matrix[i,j] = np.sum((labels_t1 == i) & (labels_t2 == j))
    return transition_matrix


# ------------------------------
# Visualization Functions
# ------------------------------

def visualize_transition_map(labels_t1: np.ndarray, labels_t2: np.ndarray, n_clusters: int, 
                             ds_ref: xr.Dataset, title: str = "Cluster Transition Map") -> np.ndarray:
    """
    Visualize cluster transitions between two time periods.

    Parameters
    ----------
    labels_t1 : np.ndarray
        Cluster labels at time 1 (2D array).
    labels_t2 : np.ndarray
        Cluster labels at time 2 (2D array).
    n_clusters : int
        Number of clusters.
    ds_ref : xr.Dataset
        Reference dataset for spatial extent.
    title : str
        Plot title.

    Returns
    -------
    np.ndarray
        Encoded transition map (t1 * n_clusters + t2)
    """
    transition_map = labels_t1 * n_clusters + labels_t2
    unique_transitions = np.unique(transition_map[~np.isnan(transition_map)]).astype(int)
    transition_labels = [f"{t // n_clusters}→{t % n_clusters}" for t in unique_transitions]

    base_cmap = plt.get_cmap('tab20', len(unique_transitions))
    colors = [base_cmap(i) for i in range(len(unique_transitions))]
    listed_cmap = ListedColormap(colors)

    transition_to_index = {t: idx for idx, t in enumerate(unique_transitions)}
    transition_index_map = np.full(transition_map.shape, np.nan)
    for t, idx in transition_to_index.items():
        transition_index_map[transition_map == t] = idx

    extent = _get_extent_from_dataset(ds_ref)

    plt.figure(figsize=(14, 8))
    im = plt.imshow(transition_index_map, cmap=listed_cmap, interpolation='nearest',
                    extent=extent, origin='upper')
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    cbar = plt.colorbar(im, ticks=np.arange(len(unique_transitions)))
    cbar.ax.set_yticklabels(transition_labels)
    cbar.set_label('Cluster Transitions (t1 → t2)')
    plt.show()

    return transition_map


def visualize_cluster_maps(labels_list: list, n_clusters: int, ds_ref: xr.Dataset, titles: list = None):
    """
    Visualize cluster maps for multiple periods.

    Parameters
    ----------
    labels_list : list
        List of 2D arrays of cluster labels.
    n_clusters : int
        Number of clusters.
    ds_ref : xr.Dataset
        Reference dataset for spatial extent.
    titles : list, optional
        Titles for each subplot, by default None
    """
    n_periods = len(labels_list)
    fig, axes = plt.subplots(1, n_periods, figsize=(6 * n_periods, 6))
    cmap = plt.get_cmap('tab10', n_clusters)
    extent = _get_extent_from_dataset(ds_ref)

    if n_periods == 1:
        axes = [axes]

    for i, labels in enumerate(labels_list):
        labels_int = np.full_like(labels, np.nan)
        mask = ~np.isnan(labels)
        labels_int[mask] = labels[mask].astype(int)

        im = axes[i].imshow(labels_int, cmap=cmap, interpolation='nearest',
                            extent=extent, origin='upper')
        axes[i].set_title(titles[i] if titles else f"Period {i+1}")
        axes[i].set_xlabel("Longitude")
        axes[i].set_ylabel("Latitude")

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, ticks=np.arange(n_clusters))
    cbar.set_label("Cluster ID")
    cbar.ax.set_yticklabels([str(i) for i in range(n_clusters)])
    plt.show()


def visualize_change_mask(labels_t1: np.ndarray, labels_t2: np.ndarray, ds_ref: xr.Dataset, 
                          title: str = "Change Mask (1 = Changed Cluster)"):
    """
    Visualize areas where cluster assignment changed between two periods.

    Parameters
    ----------
    labels_t1 : np.ndarray
        Cluster labels at time 1.
    labels_t2 : np.ndarray
        Cluster labels at time 2.
    ds_ref : xr.Dataset
        Reference dataset for spatial extent.
    title : str
        Plot title.
    """
    valid_mask = ~np.isnan(labels_t1) & ~np.isnan(labels_t2)
    change_mask = np.zeros_like(labels_t1, dtype=bool)
    change_mask[valid_mask] = labels_t1[valid_mask] != labels_t2[valid_mask]

    extent = _get_extent_from_dataset(ds_ref)

    plt.figure(figsize=(10, 6))
    im = plt.imshow(change_mask, cmap="Reds", interpolation='nearest', extent=extent, origin='upper')
    plt.title(title)
    plt.colorbar(im, label="Change (1=True, 0=False)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


# ------------------------------
# High-Level Analysis
# ------------------------------

def cluster_transition_analysis(datasets: list, variables: list = None, n_clusters: int = 5, 
                                scale_together: bool = True):
    """
    Perform cluster transition analysis across multiple datasets.

    Parameters
    ----------
    datasets : list
        List of xarray.Dataset objects.
    variables : list, optional
        Variables to include, by default all variables in first dataset.
    n_clusters : int
        Number of clusters.
    scale_together : bool
        Whether to standardize all datasets together.

    Returns
    -------
    labels_2d_list : list
        List of 2D arrays of cluster labels.
    transition_matrices : list
        List of transition matrices for each period.
    transition_maps : list
        List of transition maps (encoded).
    """
    n_periods = len(datasets)
    if n_periods < 2:
        raise ValueError("At least 2 datasets are required.")
    if variables is None:
        variables = list(datasets[0].data_vars.keys())

    # Prepare data
    X_list, masks_list, shapes_list = [], [], []
    for ds in datasets:
        data = ds[variables].to_array().squeeze()
        if 'time' in data.dims:
            data = data.isel(time=0)
        arr = data.stack(sample=("y", "x")).transpose('sample','variable')
        stacked = da.compute(arr.values)[0]
        mask = np.any(np.isnan(stacked), axis=1)
        X_list.append(stacked[~mask])
        masks_list.append(mask)
        shapes_list.append(stacked.shape[0])

    # Standardize
    if scale_together:
        scaler = StandardScaler()
        scaler.fit(np.vstack(X_list))
        X_scaled_list = [scaler.transform(X) for X in X_list]
    else:
        X_scaled_list = [StandardScaler().fit_transform(X) for X in X_list]

    # Run KMeans
    kmeans_models, labels_flat = [], []
    for X_valid, mask, n_samples in zip(X_scaled_list, masks_list, shapes_list):
        labels, kmeans_model = run_kmeans(X_valid, mask, n_clusters)
        kmeans_models.append(kmeans_model)
        labels_flat.append(labels)

    # Align clusters
    ref_model = kmeans_models[0]
    aligned_labels = [labels_flat[0]]
    for i in range(1, n_periods):
        aligned_labels.append(align_clusters(ref_model, kmeans_models[i], labels_flat[i]))

    ydim, xdim = datasets[0].dims['y'], datasets[0].dims['x']
    labels_2d_list = [lab.reshape(ydim, xdim) for lab in aligned_labels]

    # Compute transition matrices and maps
    transition_matrices, transition_maps = [], []
    for t in range(n_periods - 1):
        tm = compute_transition_matrix(labels_2d_list[t], labels_2d_list[t+1], n_clusters)
        transition_matrices.append(tm)
        transition_maps.append(
            visualize_transition_map(
                labels_2d_list[t], labels_2d_list[t+1], n_clusters,
                ds_ref=datasets[0],
                title=f"Transition Map: Period {t+1} → Period {t+2}"
            )
        )

    return labels_2d_list, transition_matrices, transition_maps


# ------------------------------
# Cluster Description
# ------------------------------
def describe_clusters(ds: xr.Dataset, labels_2d: np.ndarray, variables: list) -> pd.DataFrame:
    """
    Compute summary statistics for each cluster.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing variables.
    labels_2d : np.ndarray
        2D array of cluster labels.
    variables : list
        List of variables to summarize.

    Returns
    -------
    pd.DataFrame
        DataFrame with cluster, variable, mean, median, and std.
    """
    data = ds[variables].to_array().squeeze().stack(z=("y","x")).transpose("z","variable")
    stacked = data.values
    labels_flat = labels_2d.ravel()
    n_clusters = int(np.nanmax(labels_flat)) + 1

    summary_list = []
    for c in range(n_clusters):
        mask = labels_flat == c
        cluster_values = stacked[mask, :]
        summary_list.append(pd.DataFrame({
            "cluster": c,
            "variable": variables,
            "mean": np.nanmean(cluster_values, axis=0),
            "median": np.nanmedian(cluster_values, axis=0),
            "std": np.nanstd(cluster_values, axis=0)
        }))

    return pd.concat(summary_list, ignore_index=True)


def describe_cluster_transitions(ds_t1: xr.Dataset, ds_t2: xr.Dataset,
                                 labels_t1: np.ndarray, labels_t2: np.ndarray,
                                 variables: list, transition_matrix: np.ndarray) -> pd.DataFrame:
    """
    Provide a detailed description of how clusters changed between two periods.

    Parameters
    ----------
    ds_t1, ds_t2 : xr.Dataset
        Datasets for time 1 and time 2.
    labels_t1, labels_t2 : np.ndarray
        Cluster label maps for the two periods.
    variables : list
        Variables to summarize.
    transition_matrix : np.ndarray
        Precomputed transition matrix (from compute_transition_matrix()).

    Returns
    -------
    pd.DataFrame
        Detailed comparison between cluster states across time.
    """
    # --- Basic summaries for both periods ---
    desc_t1 = describe_clusters(ds_t1, labels_t1, variables)
    desc_t1 = desc_t1.rename(columns={"mean": "mean_t1", "median": "median_t1", "std": "std_t1"})
    desc_t2 = describe_clusters(ds_t2, labels_t2, variables)
    desc_t2 = desc_t2.rename(columns={"mean": "mean_t2", "median": "median_t2", "std": "std_t2"})

    merged = pd.merge(desc_t1, desc_t2, on=["cluster", "variable"], how="outer")

    # --- Cluster area (counts) ---
    n_clusters = transition_matrix.shape[0]
    total_pixels = np.sum(~np.isnan(labels_t1))
    area_t1 = np.array([np.sum(labels_t1 == i) for i in range(n_clusters)])
    area_t2 = np.array([np.sum(labels_t2 == i) for i in range(n_clusters)])

    # --- Transition stats ---
    inflow = transition_matrix.sum(axis=0)
    outflow = transition_matrix.sum(axis=1)
    retention = np.diag(transition_matrix)

    area_df = pd.DataFrame({
        "cluster": np.arange(n_clusters),
        "area_t1": area_t1,
        "area_t2": area_t2,
        "area_change": area_t2 - area_t1,
        "pct_change": (area_t2 - area_t1) / np.maximum(area_t1, 1) * 100,
        "retention_rate": retention / np.maximum(area_t1, 1),
        "inflow": inflow - retention,
        "outflow": outflow - retention,
        "inflow_ratio": (inflow - retention) / np.maximum(inflow, 1),
        "outflow_ratio": (outflow - retention) / np.maximum(outflow, 1)
    })

    # --- Combine all info ---
    summary = merged.merge(area_df, on="cluster", how="left")

    # --- Compute variable change metrics ---
    summary["mean_change"] = summary["mean_t2"] - summary["mean_t1"]
    summary["median_change"] = summary["median_t2"] - summary["median_t1"]

    return summary
