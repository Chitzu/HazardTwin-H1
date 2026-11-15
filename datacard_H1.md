# Data Card & State-of-the-Art (SOA) — Cod Temă: H1 (HazardTwin)

* **Sistem modelat:** Bazinul Hidrografic Siret (zona Galați – Pechea, România; activare Copernicus EMSR754)
* **Întrebarea de cercetare:** Cum poate fi cartografiată precis extinderea suprafeței inundate în Bazinul Hidrografic Siret prin segmentarea imaginilor satelitare SAR (Sentinel-1) și a datelor topografice (Copernicus DEM)?
* **Client și decizie:** IGSU și autoritățile locale — planificarea rutelor sigure de evacuare și poziționarea digurilor temporare.
* **Repository GitHub:** `https://github.com/Chitzu/HazardTwin-H1`
* **Fișiere existente în repository pentru CP1:**
  * `pipeline_date.py` — Scriptul care descarcă și co-registrează la $10\text{ m}$ imaginile Sentinel-1 SAR și Copernicus DEM GLO-30 atât pentru patch-urile din `Sen1Floods11`, cât și pentru **Bazinul Siret (Galați – Pechea)**.
  * `datacard_H1.md` — Documentația setului de date și tabelul SOA cu 10 articole.
  * `dovada_pipeline_cp1.png` — Figura generată de `pipeline_date.py` (Rândul 1: patch Sen1Floods11; Rândul 2: Bazinul Siret, RO).

---

## 1. Data Card (Multi-Sursă: Sentinel-1 SAR + Copernicus DEM GLO-30)

| Atribut | Sursa 1: Sen1Floods11 & Sentinel-1 SAR | Sursa 2: Copernicus DEM GLO-30 (Model Digital de Elevație) |
| :--- | :--- | :--- |
| **Sursă** | Cloud to Street (`gs://sen1floods11/v1.1/data/flood_events/HandLabeled/`) + fereastra Sentinel-1 peste Bazinul Siret (`./date_siret/Siret_S1.tif`) | AWS Open Data Registry (`s3://copernicus-dem-30m/`, inclusiv tile-ul `Copernicus_DSM_COG_10_N45_00_E027_00_DEM` pentru Bazinul Siret, Galați) |
| **Perioadă** | 2016 – 2019 (cele 11 evenimente din Sen1Floods11) + Septembrie 2024 (inundațiile din Bazinul Siret – EMSR754) | 2021 – prezent (Copernicus DEM GLO-30) |
| **Rezoluție** | $10\text{ m/pixel}$ (dimensiune patch: $512 \times 512$ pixeli, acoperire $5.12\text{ km} \times 5.12\text{ km}$) | $30\text{ m/pixel}$ nativ ($1\text{ arc-second}$), re-eșantionat bilinear în cod la grila de $10\text{ m/pixel}$ ($512 \times 512$ pixeli) a imaginii SAR |
| **Unități** | Coeficient de retroîmprăștiere radar (backscatter $\sigma^0$) în decibeli ($\text{dB}$) pentru polarizările $\text{VV}$ și $\text{VH}$; etichete $\{-1, 0, 1\}$ | Metri ($\text{m}$) altitudine față de nivelul mării și grade ($^\circ$) pentru panta terenului derivată din DEM |
| **Goluri (Gaps)** | Pixeli `NoData` (`NaN` în S1 și `-1` în măștile de etichete) la marginile orbitei satelitare; zgomot *speckle* și umbre radar pe versanți | Fără goluri pe uscat (golurile de achiziție radar sunt completate prin interpolare în produsul Copernicus GLO-30) |
| **Împărțire Train/Test** | Împărțire oficială Sen1Floods11 (446 patch-uri HandLabeled): **252 Train** / **89 Validation** / **90 Test** (cu bazinul Bolivia ca hold-out spațial) + **1 regiune externă de test pe Bazinul Siret (Galați – Pechea, RO)** | Nu se împarte independent; fiecare fereastră Copernicus DEM este extrasă automat pe *bounding-box*-ul (`EPSG:4326`) al patch-ului Sentinel-1 corespunzător |
| **Licență** | Creative Commons Attribution 4.0 International (CC BY 4.0) | Licență deschisă Copernicus DEM (Open & Free Access) |

---

## 2. Tabel State-of-the-Art (SOA) — 10 Articole (cu coloana Split)

*Notă privind revizuirea conform feedback-ului CP1:*
* **Rândul 4:** Articolul de sinteză *Bentivoglio et al. (2022)* a fost înlocuit cu **Twele et al. (2016)**, care implementează efectiv un lanț de procesare pe **Sentinel-1 SAR + DEM (indicele HAND și pantă)**.
* **Rândul 6:** Pentru **Mateo-Garcia et al. (2021) (WorldFloods)** s-a precizat că folosește **exclusiv Sentinel-2 (Optic)**.
* **Rândul 8:** Pentru **Bai et al. (2021) (BASNet / U-Net++)** s-a precizat că folosește **fuziunea Sentinel-1 SAR + Sentinel-2 Optic**.
* **Coloana Split:** Adăugată pentru a distinge clar studiile evaluate pe **Split Spațial (bazine/evenimente hold-out)** față de cele pe **Split Aleator**.

| Nr. | Articol (Titlu, Autori, An) | Metodă | Date | Split (Protocol Evaluare) | Metrică & Rezultat Raportat |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1* (Bonafilia et al., 2020) | FCN (ResNet-50) vs. Otsu | S1 SAR ($\text{VV}, \text{VH}$), S2 | **Split Spațial (Hold-out Bolivia)** + Split Oficial Patch | $\text{IoU} = 0.50$ (Bolivia S1) / $\text{IoU} = 0.65$ (global); baseline-ul oficial Sen1Floods11. |
| **2** | *Explainable Flood Segmentation on Sentinel-1 SAR Imagery: A Comparative Study of CNN and Transformer Architectures* (2026) | CNN vs. Vision Transformer (ViT) + XAI | Doar S1 SAR ($\text{VV}, \text{VH}$) | **Split Oficial Sen1Floods11** (Train / Val / Test) | $\text{IoU} = 0.74$; oferă explicații Grad-CAM utile pentru deciziile IGSU. |
| **3** | *Fully Convolutional Neural Network for Rapid Flood Mapping in Synthetic Aperture Radar Images* (Nemni et al., 2020) | U-Net (ResNet / Xception) | Doar S1 SAR | **Split Spațial pe Evenimente** (8 inundații UNOSAT) | $\text{F1} = 0.91$, $\text{IoU} = 0.84$ ($+14\%$ peste pragul clasic Otsu). |
| **4** | *Sentinel-1-based flood mapping: a fully automated processing chain* (Twele et al., Int. J. Remote Sens. 2016) | Prag automat + Logică Fuzzy + HAND | **S1 SAR + DEM (HAND & Pantă)** | **Split Spațial pe Bazine** (activări Copernicus EMS) | $\text{Kappa} = 0.85$, $\text{OA} = 94.0\%$; DEM-ul elimină alarmele false din umbrele radar pe versanți. |
| **5** | *Flood extent mapping using Sentinel-1 SAR imagery and Deep Learning* (Katiyar et al., IGARSS 2021) | DeepLabv3+ / U-Net vs. Otsu | S1 SAR ($\text{VV} + \text{VH}$) | **Split Spațial** (test pe regiune separată) | $\text{F1} = 0.86$ pe polarizare duală $\text{VV}+\text{VH}$. |
| **6** | *Towards global flood mapping onboard low cost satellites with machine learning* (Mateo-Garcia et al., 2021) | U-Net / SCNN (*WorldFloods*) | **Doar S2 (Optic Multispectral)** | **Split Spațial** (5 regiuni/evenimente hold-out) | $\text{IoU} = 0.71$ pe bazine noi; blocat însă de plafonul de nori (justifică S1 SAR). |
| **7** | *Exploring Sentinel-1 and Sentinel-2 diversity for flood inundation mapping using deep learning* (Konapala et al., 2021) | U-Net cu fuziune multi-sursă | **S1 SAR + S2 Optic + DEM** | **Split Oficial Sen1Floods11** (cu bazine hold-out) | $\text{F1} = 0.84$; adăugarea DEM-ului și a pantei aduce $+6\%$ în F1-Score. |
| **8** | *Flood Segmentation on Sentinel-1/2 Imagery with BASNet / U-Net++* (Bai et al., 2021) | BASNet / U-Net++ | **Fuziune S1 SAR + S2 Optic** | **Split Oficial Sen1Floods11** (Test pe eveniment hold-out) | $\text{IoU} = 0.73$, $\text{F1} = 0.85$ prin combinarea radarului S1 cu opticul S2. |
| **9** | *Urban Flood Mapping with Residual Patch Similarity Learning in Bi-temporal SAR Imagery* (Peng et al., 2019) | ResNet Patch Similarity | Doar S1 SAR | **Split Spațial** (zone periurbane neincluse în train) | $\text{Kappa} = 0.87$, $\text{F1} = 0.89$; robust la reflexiile din localități. |
| **10** | *Automated Extraction of Surface Water Extent from Sentinel-1 Data* (Huang et al., 2018) | Random Forest + mască pantă | **S1 SAR + DEM Topografic** | **Split Spațial Regional** | $\text{OA} = 93.2\%$; baseline clasic ML pentru fuziunea SAR + DEM. |