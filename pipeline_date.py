
import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from google.cloud import storage

DATA_DIR = "/Users/chitzu/Desktop/ml-work/datasets/floods/v1.1/data/flood_events/HandLabeled"
S1_DIR = os.path.join(DATA_DIR, "S1Hand")
LABEL_DIR = os.path.join(DATA_DIR, "LabelHand")
DEM_DIR = os.path.join(DATA_DIR, "DEM_Real")

os.makedirs(S1_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(DEM_DIR, exist_ok=True)

# =====================================================================
# 1. DESCĂRCARE IMAGINI S1 FĂRĂ MARGINI NEGRE (NODATA < 5%)
# =====================================================================
def download_clean_sen1floods11_samples(num_samples=2):
    print("[1/4] Căutare imagini complete în gs://sen1floods11...")
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket("sen1floods11")
    prefix_s1 = "v1.1/data/flood_events/HandLabeled/S1Hand/"
    blobs = list(bucket.list_blobs(prefix=prefix_s1, max_results=25))

    downloaded_pairs = []
    for blob in blobs:
        if not blob.name.endswith("_S1Hand.tif"):
            continue
        s1_filename = os.path.basename(blob.name)
        label_filename = s1_filename.replace("_S1Hand.tif", "_LabelHand.tif")
        label_blob = bucket.blob(f"v1.1/data/flood_events/HandLabeled/LabelHand/{label_filename}")

        if not label_blob.exists():
            continue

        local_s1 = os.path.join(S1_DIR, s1_filename)
        local_lbl = os.path.join(LABEL_DIR, label_filename)

        if not os.path.exists(local_s1):
            blob.download_to_filename(local_s1)
        if not os.path.exists(local_lbl):
            label_blob.download_to_filename(local_lbl)

        # Verificăm să nu aibă jumătatea neagră (NoData) și să conțină apă
        with rasterio.open(local_s1) as src_s1, rasterio.open(local_lbl) as src_lbl:
            s1_arr = src_s1.read(1)
            lbl_arr = src_lbl.read(1)
            nodata_ratio = np.isnan(s1_arr).mean()
            water_ratio = (lbl_arr == 1).mean()

        if nodata_ratio < 0.05 and water_ratio > 0.05:
            print(f" -> Selectat: {s1_filename} (Apă: {water_ratio*100:.1f}%, NoData: {nodata_ratio*100:.1f}%)")
            downloaded_pairs.append((local_s1, local_lbl))

        if len(downloaded_pairs) >= num_samples:
            break

    return downloaded_pairs

# =====================================================================
# 2. EXTRAGERE AUTOMATĂ DEM REAL (30m) ALINIAT PE COORDONATELE S1
# =====================================================================
def get_real_dem_for_s1(s1_path):
    """Extrage automat DEM-ul real (30m) exact pe bounding-box-ul imaginii Sentinel-1"""
    base_name = os.path.basename(s1_path).replace("_S1Hand.tif", "_DEM.tif")
    local_dem_path = os.path.join(DEM_DIR, base_name)

    if os.path.exists(local_dem_path):
        return local_dem_path

    with rasterio.open(s1_path) as src_s1:
        bounds = src_s1.bounds
        target_h, target_w = src_s1.height, src_s1.width
        profile = src_s1.profile.copy()

    lat_floor = int(np.floor(bounds.bottom))
    lon_floor = int(np.floor(bounds.left))
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    tile_id = f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM"
    cog_url = f"https://copernicus-dem-30m.s3.amazonaws.com/{tile_id}/{tile_id}.tif"

    print(f"[2/4] Descărcare DEM 30m real pentru {base_name} din {tile_id}...")
    with rasterio.open(cog_url) as src_dem:
        win = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, src_dem.transform)
        dem_data = src_dem.read(
            1,
            window=win,
            out_shape=(target_h, target_w),
            resampling=Resampling.bilinear
        ).astype(np.float32)

    profile.update(count=1, dtype='float32')
    with rasterio.open(local_dem_path, 'w', **profile) as dst:
        dst.write(dem_data, 1)

    return local_dem_path

# =====================================================================
# 3. DATASET PYTORCH: SENTINEL-1 (VV, VH) + DEM REAL -> MASCĂ APĂ
# =====================================================================
class HazardTwinDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s1_path, label_path = self.pairs[idx]
        dem_path = get_real_dem_for_s1(s1_path)

        with rasterio.open(s1_path) as src_s1:
            s1_data = src_s1.read().astype(np.float32)
        s1_data = np.nan_to_num(s1_data, nan=-30.0)
        s1_norm = np.clip((s1_data + 30.0) / 30.0, 0.0, 1.0)

        with rasterio.open(dem_path) as src_dem:
            dem_raw = src_dem.read(1).astype(np.float32)
        dem_raw = np.nan_to_num(dem_raw, nan=0.0)
        dem_norm = (dem_raw - dem_raw.min()) / (dem_raw.max() - dem_raw.min() + 1e-6)

        x_tensor = torch.from_numpy(np.concatenate([s1_norm, np.expand_dims(dem_norm, 0)], axis=0))

        with rasterio.open(label_path) as src_lbl:
            label_data = src_lbl.read(1).astype(np.int64)

        return x_tensor, torch.from_numpy(label_data), dem_raw, os.path.basename(s1_path)

# =====================================================================
# 4. RULARE ȘI SALVARE IMAGINE CORECTATĂ
# =====================================================================
if __name__ == "__main__":
    pairs = download_clean_sen1floods11_samples(num_samples=2)
    dataset = HazardTwinDataset(pairs)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    batch_x, batch_y, batch_dem_raw, filenames = next(iter(loader))
    print(f"[3/4] Dimensiune Tensor Intrare (VV, VH, DEM): {batch_x.shape}")

    sample_x = batch_x[0].numpy()
    sample_y = batch_y[0].numpy()
    dem_meters = batch_dem_raw[0].numpy()

    valid_pixels = (sample_y != -1).sum()
    water_ratio = ((sample_y == 1).sum() / max(valid_pixels, 1)) * 100

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(sample_x[0], cmap='gray')
    axes[0].set_title(f"Sentinel-1 VV (SAR)\n{filenames[0]}", fontsize=11, pad=8)

    axes[1].imshow(sample_x[1], cmap='gray')
    axes[1].set_title("Sentinel-1 VH (SAR)\nBackscatter normalizat", fontsize=11, pad=8)

    im_dem = axes[2].imshow(dem_meters, cmap='terrain')
    axes[2].set_title(f"DEM 30m Re-eșantionat la 10m\nAltitudine: [{dem_meters.min():.0f}m - {dem_meters.max():.0f}m]", fontsize=11, pad=8)
    plt.colorbar(im_dem, ax=axes[2], fraction=0.046, pad=0.04, label="Metri")

    mask_vis = np.where(sample_y == -1, 0, sample_y)
    axes[3].imshow(mask_vis, cmap='Blues', vmin=0, vmax=1)
    axes[3].set_title(f"Ground Truth Mască Apă\n({water_ratio:.1f}% Apă / {100-water_ratio:.1f}% Uscat)", fontsize=11, pad=8)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout(pad=1.5)
    plt.savefig("dovada_pipeline_cp1.png", dpi=150, bbox_inches='tight')
    print("[4/4] Imaginea 'dovada_pipeline_cp1.png' cu DEM real a fost salvată!")