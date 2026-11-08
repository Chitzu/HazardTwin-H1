import os
import warnings
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from google.cloud import storage

warnings.filterwarnings("ignore")

DATA_DIR = "/Users/chitzu/Desktop/ml-work/datasets/floods/v1.1/data/flood_events/HandLabeled"
S1_DIR = os.path.join(DATA_DIR, "S1Hand")
LABEL_DIR = os.path.join(DATA_DIR, "LabelHand")
DEM_DIR = os.path.join(DATA_DIR, "DEM_Real")

os.makedirs(S1_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(DEM_DIR, exist_ok=True)

# =====================================================================
# 1. DESCARCARE PATCH-URI DIN BAZINE DIFERITE (SPLIT SPATIAL)
# =====================================================================
def fetch_spatial_split_data(n_train=6, n_test=3):
    """
    Construiește un Split Spațial (Leave-One-Basin-Out):
    - Train: patch-uri din bazine non-Bolivia (ex. Ghana, Paraguay, India, Mekong, USA)
    - Test (Hold-out spațial): patch-uri din bazinul Bolivia (inclusiv Bolivia_129334)
    """
    print("[1/5] Pregătire date pentru Split Spațial (Leave-One-Basin-Out)...")
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket("sen1floods11")
    prefix_s1 = "v1.1/data/flood_events/HandLabeled/S1Hand/"
    blobs = list(bucket.list_blobs(prefix=prefix_s1, max_results=220))

    train_pairs, test_pairs = [], []

    # Sortăm astfel încât Bolivia_129334_S1Hand.tif (folosit la CP1) să fie primul în Test
    blobs.sort(key=lambda b: 0 if "Bolivia_129334" in b.name else 1)

    for blob in blobs:
        if not blob.name.endswith("_S1Hand.tif"):
            continue
        s1_filename = os.path.basename(blob.name)
        region = s1_filename.split("_")[0]
        is_test_basin = (region == "Bolivia")

        if is_test_basin and len(test_pairs) >= n_test:
            continue
        if (not is_test_basin) and len(train_pairs) >= n_train:
            continue

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

        with rasterio.open(local_s1) as src_s1, rasterio.open(local_lbl) as src_lbl:
            s1_arr = src_s1.read(1)
            lbl_arr = src_lbl.read(1)
            nodata_ratio = np.isnan(s1_arr).mean()
            water_ratio = (lbl_arr == 1).mean()

        if nodata_ratio < 0.05 and 0.08 < water_ratio < 0.80:
            if is_test_basin:
                test_pairs.append((local_s1, local_lbl))
                print(f" -> [TEST - Bazin Hold-out] {s1_filename} (Apă: {water_ratio*100:.1f}%)")
            else:
                train_pairs.append((local_s1, local_lbl))
                print(f" -> [TRAIN - Bazin {region}] {s1_filename} (Apă: {water_ratio*100:.1f}%)")

        if len(train_pairs) >= n_train and len(test_pairs) >= n_test:
            break

    return train_pairs, test_pairs

# =====================================================================
# 2. EXTRAGERE AUTOMATĂ DEM REAL (30m -> 10m) PENTRU FIECARE PATCH
# =====================================================================
def get_real_dem_for_s1(s1_path):
    base_name = os.path.basename(s1_path).replace("_S1Hand.tif", "_DEM.tif")
    local_dem_path = os.path.join(DEM_DIR, base_name)
    if os.path.exists(local_dem_path):
        with rasterio.open(local_dem_path) as src:
            return src.read(1).astype(np.float32)

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

    try:
        with rasterio.open(cog_url) as src_dem:
            win = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, src_dem.transform)
            dem_data = src_dem.read(
                1, window=win, out_shape=(target_h, target_w), resampling=Resampling.bilinear
            ).astype(np.float32)
    except Exception:
        # Fallback sigur dacă un tile extern are conexiunea întreruptă
        dem_data = np.zeros((target_h, target_w), dtype=np.float32)

    profile.update(count=1, dtype='float32')
    with rasterio.open(local_dem_path, 'w', **profile) as dst:
        dst.write(dem_data, 1)
    return dem_data

# =====================================================================
# 3. CONSTRUIREA CELOR DOUĂ REPREZENTĂRI (ALTERNATIVA VS. ALEASĂ)
# =====================================================================
def extract_representations(s1_path, label_path):
    """
    Returnează:
    - X_alt (H, W, 2): Reprezentarea Alternativă [VV, VH]
    - X_chosen (H, W, 8): Reprezentarea Aleasă [VV, VH, VV-VH, VV_mean5, VH_mean5, VV_std5, DEM_rel, Slope_deg]
    - y (H, W): Masca Ground Truth (-1 = invalid, 0 = uscat, 1 = apă)
    - meta: dicționar cu straturi pentru vizualizare
    """
    with rasterio.open(s1_path) as src_s1:
        s1 = src_s1.read().astype(np.float32) # (2, H, W)
    with rasterio.open(label_path) as src_lbl:
        y = src_lbl.read(1).astype(np.int64)

    vv = np.nan_to_num(s1[0], nan=-25.0)
    vh = np.nan_to_num(s1[1], nan=-30.0)
    vv = np.clip(vv, -35.0, 5.0)
    vh = np.clip(vh, -35.0, 5.0)

    # --- 1. Reprezentarea Alternativă (Doar SAR brut: 2 canale) ---
    X_alt = np.stack([vv, vh], axis=-1)

    # --- 2. Reprezentarea Aleasă (SAR Filtrat Speckle + Topografie DEM: 8 canale) ---
    diff_pol = vv - vh
    vv_mean = uniform_filter(vv, size=5)
    vh_mean = uniform_filter(vh, size=5)
    vv_var = uniform_filter(vv**2, size=5) - vv_mean**2
    vv_std = np.sqrt(np.maximum(vv_var, 0.0))

    dem_raw = np.nan_to_num(get_real_dem_for_s1(s1_path), nan=0.0)
    # Înălțimea relativă față de lunca râului (percentila 10 din bazin) - invariantă între bazine!
    p10, p90 = np.percentile(dem_raw, 10), np.percentile(dem_raw, 90)
    dem_rel = np.clip((dem_raw - p10) / (p90 - p10 + 1e-4), -0.5, 2.0)

    # Panta terenului (Slope în grade) la rezoluția de 10m
    dy, dx = np.gradient(uniform_filter(dem_raw, size=3), 10.0)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    X_chosen = np.stack([
        vv, vh, diff_pol, vv_mean, vh_mean, vv_std, dem_rel, slope_deg
    ], axis=-1)

    meta = {"vv": vv, "vh": vh, "dem": dem_raw, "slope": slope_deg}
    return X_alt, X_chosen, y, meta

def sample_pixels_from_pairs(pairs, samples_per_patch=15000, seed=42):
    rng = np.random.default_rng(seed)
    X_alt_list, X_chosen_list, y_list = [], [], []

    for s1_path, lbl_path in pairs:
        X_alt, X_chosen, y, _ = extract_representations(s1_path, lbl_path)
        valid_mask = (y == 0) | (y == 1)
        X_a_val = X_alt[valid_mask]
        X_c_val = X_chosen[valid_mask]
        y_val = y[valid_mask]

        n_avail = len(y_val)
        n_pick = min(samples_per_patch, n_avail)
        idx = rng.choice(n_avail, size=n_pick, replace=False)

        X_alt_list.append(X_a_val[idx])
        X_chosen_list.append(X_c_val[idx])
        y_list.append(y_val[idx])

    return (
        np.vstack(X_alt_list),
        np.vstack(X_chosen_list),
        np.concatenate(y_list)
    )

# =====================================================================
# 4. ANTRENARE BASELINE-URI & EVALUARE PE SPLIT-UL SPAȚIAL
# =====================================================================
if __name__ == "__main__":
    train_pairs, test_pairs = fetch_spatial_split_data(n_train=6, n_test=3)

    print("\n[2/5] Extragere trăsături pentru antrenare și testare...")
    X_train_alt, X_train_chosen, y_train = sample_pixels_from_pairs(train_pairs, samples_per_patch=15000)
    X_test_alt, X_test_chosen, y_test = sample_pixels_from_pairs(test_pairs, samples_per_patch=30000)
    print(f" -> Pixeli eșantionați Train: {len(y_train):,} | Pixeli Test (Hold-out): {len(y_test):,}")

    print("\n[3/5] Rulare cele 3 Baseline-uri...")
    # Baseline 1: Prag fizic Otsu pe banda SAR VH (fără ML)
    vh_train = X_train_alt[:, 1]
    otsu_thresh = threshold_otsu(vh_train)
    # În SAR, suprafața apei reflectă specular -> valori dB mai mici decât pragul
    y_pred_otsu = (X_test_alt[:, 1] < otsu_thresh).astype(int)

    # Baseline 2: Random Forest pe Reprezentarea Alternativă (Doar SAR [VV, VH])
    rf_alt = RandomForestClassifier(
        n_estimators=100, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42
    )
    rf_alt.fit(X_train_alt, y_train)
    y_pred_rf_alt = rf_alt.predict(X_test_alt)

    # Baseline 3: Random Forest pe Reprezentarea Aleasă (SAR + Filtru Speckle + DEM relativ + Pantă)
    rf_chosen = RandomForestClassifier(
        n_estimators=100, max_depth=14, class_weight="balanced", n_jobs=-1, random_state=42
    )
    rf_chosen.fit(X_train_chosen, y_train)
    y_pred_rf_chosen = rf_chosen.predict(X_test_chosen)

    # Calculăm metricile pe setul de Test Spațial
    results = {}
    for name, preds in [
        ("1. Prag Otsu (SAR VH)", y_pred_otsu),
        ("2. RF - Alternativa (Doar SAR 2ch)", y_pred_rf_alt),
        ("3. RF - Reprezentarea Aleasă (SAR+DEM 8ch)", y_pred_rf_chosen),
    ]:
        results[name] = {
            "IoU": jaccard_score(y_test, preds, pos_label=1),
            "F1": f1_score(y_test, preds, pos_label=1),
            "Precision": precision_score(y_test, preds, pos_label=1),
            "Recall": recall_score(y_test, preds, pos_label=1),
        }

    print("\n================= REZULTATE CP2 (SPLIT SPAȚIAL HOLD-OUT) =================")
    print(f"{'Model / Reprezentare':<44} | {'IoU':<6} | {'F1':<6} | {'Prec':<6} | {'Recall':<6}")
    print("-" * 78)
    for k, m in results.items():
        print(f"{k:<44} | {m['IoU']:.3f}  | {m['F1']:.3f}  | {m['Precision']:.3f}  | {m['Recall']:.3f}")
    print("==========================================================================\n")

    # =====================================================================
    # 5. GENERARE FIGURI PENTRU SLIDE-UL 2 DIN PREZENTAREA CP2
    # =====================================================================
    print("[4/5] Generare 'cp2_harti_comparative.png' pe patch-ul de test...")
    sample_s1, sample_lbl = test_pairs[0]
    X_a_full, X_c_full, y_full, meta = extract_representations(sample_s1, sample_lbl)
    H, W = y_full.shape
    valid_2d = (y_full != -1)

    pred_otsu_2d = (X_a_full[:, :, 1] < otsu_thresh).astype(int) * valid_2d
    pred_alt_2d = rf_alt.predict(X_a_full.reshape(-1, 2)).reshape(H, W) * valid_2d
    pred_chosen_2d = rf_chosen.predict(X_c_full.reshape(-1, 8)).reshape(H, W) * valid_2d

    iou_o = jaccard_score(y_full[valid_2d], pred_otsu_2d[valid_2d], pos_label=1)
    iou_a = jaccard_score(y_full[valid_2d], pred_alt_2d[valid_2d], pos_label=1)
    iou_c = jaccard_score(y_full[valid_2d], pred_chosen_2d[valid_2d], pos_label=1)

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
    axes[0].imshow(meta["vv"], cmap="gray")
    axes[0].set_title(f"Input 1: Sentinel-1 VV\n({os.path.basename(sample_s1)})", fontsize=10.5)

    axes[1].imshow(meta["slope"], cmap="magma", vmin=0, vmax=8)
    axes[1].set_title("Input 2: Pantă DEM (Slope °)\nElimină alarmele la deal", fontsize=10.5)

    axes[2].imshow(np.where(valid_2d, y_full, 0), cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth\nMască Inundație", fontsize=10.5)

    axes[3].imshow(pred_alt_2d, cmap="Blues", vmin=0, vmax=1)
    axes[3].set_title(f"RF Alternativă (Doar SAR)\nIoU = {iou_a:.2f} (zgomot speckle)", fontsize=10.5)

    axes[4].imshow(pred_chosen_2d, cmap="Blues", vmin=0, vmax=1)
    axes[4].set_title(f"RF Reprezentare Aleasă (SAR+DEM)\nIoU = {iou_c:.2f} (+{(iou_c-iou_a)*100:.1f}% IoU)", fontsize=10.5, fontweight="bold")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("cp2_harti_comparative.png", dpi=150, bbox_inches="tight")

    print("[5/5] Generare 'cp2_metrici_protocol.png' (Grafic Metrici + Importanța Trăsăturilor)...")
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))

    labels = ["Prag Otsu\n(SAR VH)", "RF Alternativă\n(Doar SAR: VV, VH)", "RF Reprezentare Aleasă\n(SAR + Speckle + DEM + Pantă)"]
    iou_vals = [results[k]["IoU"] for k in results]
    f1_vals = [results[k]["F1"] for k in results]

    x = np.arange(len(labels))
    width = 0.34
    b1 = ax1.bar(x - width/2, iou_vals, width, label="IoU (Jaccard)", color="#1f77b4")
    b2 = ax1.bar(x + width/2, f1_vals, width, label="F1-Score (Dice)", color="#2ca02c")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylabel("Scor pe Setul de Test Spațial (Hold-out)")
    ax1.set_title("Compararea Reprezentărilor & Baseline-urilor (Split Spațial)", fontsize=11, fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}", ha="center", fontsize=9, fontweight="bold")

    feat_names = ["VV", "VH", "VV-VH", "VV_mean5", "VH_mean5", "VV_std5", "DEM_rel", "Slope°"]
    importances = rf_chosen.feature_importances_
    order = np.argsort(importances)
    ax2.barh(np.array(feat_names)[order], importances[order], color="#ff7f0e")
    ax2.set_xlabel("Importanța trăsăturii în Random Forest (Gini)")
    ax2.set_title("Contribuția Trăsăturilor în Reprezentarea Aleasă", fontsize=11, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("cp2_metrici_protocol.png", dpi=150, bbox_inches="tight")
    print(" -> Gata! Au fost salvate 'cp2_harti_comparative.png' și 'cp2_metrici_protocol.png'!")