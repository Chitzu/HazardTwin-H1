# Data Card & State-of-the-Art (SOA) — Cod Temă: H1 (HazardTwin)

* **Autor:** Botezatu Mihail-Gabriel
* **Sistem modelat:** Bazinul Hidrografic Siret (zona Galați – Pechea, România; activări Copernicus EMSR754 / EMSR482)
* **Întrebarea de cercetare:** Cum poate fi cartografiată precis extinderea suprafeței inundate în Bazinul Hidrografic Siret prin segmentarea imaginilor satelitare SAR (Sentinel-1) și a datelor topografice (DEM)?
* **Client și decizie:** IGSU și autoritățile locale — planificarea rutelor sigure de evacuare și poziționarea digurilor temporare.

---

## 1. Data Card (Multi-Sursă: Sentinel-1 SAR + DEM Topografic)

| Atribut | Sursa 1: Sen1Floods11 & Sentinel-1 SAR | Sursa 2: Model Digital de Elevație (SRTM / Copernicus DEM) |
| :--- | :--- | :--- |
| **Sursă** | Cloud to Street (`gs://sen1floods11`) & Catalogul Copernicus Sentinel-1 GRD (via Google Earth Engine) | NASA / USGS (`USGS/SRTMGL1_003` în GEE) & Copernicus DEM GLO-30 (`s3://copernicus-dem-30m`) |
| **Perioadă** | 2016 – 2019 (evenimentele din Sen1Floods11) + Septembrie 2024 (inundațiile din Bazinul Siret – EMSR754) | 2000 (misiunea SRTM, actualizată void-filled) / 2021 (Copernicus GLO-30) |
| **Rezoluție** | $10\text{ m/pixel}$ (dimensiune patch: $512 \times 512$ pixeli, acoperire $5.12\text{ km} \times 5.12\text{ km}$) | $30\text{ m/pixel}$ nativ, re-eșantionat bilinear la grila de $10\text{ m/pixel}$ ($512 \times 512$ pixeli) a imaginii SAR |
| **Unități** | Coeficient de retroîmprăștiere radar (backscatter $\sigma^0$) măsurat în decibeli ($\text{dB}$) pentru polarizările duale $\text{VV}$ și $\text{VH}$ | Metri ($\text{m}$) altitudine față de nivelul mării (și grade pentru panta terenului derivată din DEM) |
| **Goluri (Gaps)** | Măști `NoData` (`NaN` în S1 și `-1` în măștile de etichete) la marginile orbitei satelitare; artefacte de îmbinare swath; umbre radar pe versanți abrupți | Golurile de achiziție radar (voids) sunt completate prin interpolare în versiunile SRTM Void-Filled / Copernicus DEM |
| **Împărțire Train/Test** | Împărțire oficială Sen1Floods11 (446 patch-uri etichetate manual): **252 Train** / **89 Validation** / **90 Test** + **1 regiune de test externă** (Bazinul Siret, RO) | Nu se împarte independent; fiecare patch DEM este extras și co-registrat spațial pe bounding-box-ul patch-ului Sentinel-1 corespunzător |
| **Licență** | Creative Commons Attribution 4.0 International (CC BY 4.0) | Domeniul Public (USGS SRTM) / Licență deschisă Copernicus (Open Access) |

---

## 2. Pipeline-ul de Încărcare și Preprocesare a Datelor

Pipeline-ul implementat în PyTorch (`pipeline_cp1.py`) realizează următoarele operații:
1. **Descărcare selectivă și filtrare `NoData`:** Accesează bucket-ul `gs://sen1floods11` și selectează patch-urile `S1Hand` și `LabelHand` valide (cu rată `NoData` $< 5\%$).
2. **Co-registrare spațială (SAR + DEM):** Extrage fereastra geografică exactă (`bounds` în proiecția `EPSG:4326`) din modelul digital de elevație de $30\text{ m}$ și o re-eșantionează bilinear la dimensiunea $512 \times 512$ ($10\text{ m/pixel}$).
3. **Normalizare și asamblare tensor multi-canal:**
   * Canalele $\text{VV}$ și $\text{VH}$ sunt trunchiate la intervalul $[-30\text{ dB}, 0\text{ dB}]$ și scalate liniar în $[0, 1]$.
   * Canalul $\text{DEM}$ este normalizat min-max pe fiecare patch în $[0, 1]$.
   * Tensorul de intrare rezultat are forma $(B, 3, 512, 512)$ corespunzătoare canalelor $[\text{VV}, \text{VH}, \text{DEM}]$, iar masca țintă are forma $(B, 512, 512)$ cu clasele $\{-1: \text{NoData}, 0: \text{Uscat}, 1: \text{Apă}\}$.

![Dovada rulării pipeline-ului pe eșantionul Bolivia_129334_S1Hand.tif](dovada_pipeline_cp1_2.jpg)
*Figura 1: Eșantion procesat de pipeline (`Bolivia_129334_S1Hand.tif`). Se observă corelația fizică dintre zonele joase din DEM ($144\text{ m} - 148\text{ m}$) și masca Ground Truth de apă ($65.9\%$ apă vs. $34.1\%$ uscat pe crestele de $152\text{ m} - 157\text{ m}$).*

---

## 3. Tabel State-of-the-Art (SOA) — 10 Articole

| Nr. | Articol (Titlu, Autori, An) | Metodă | Date | Metrică | Rezultat Raportat & Relevanță |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1* (Bonafilia et al., CVPRW 2020) | FCN (ResNet-50) vs. Otsu Threshold | S1 ($\text{VV}, \text{VH}$), S2 | IoU | $\text{IoU} = 0.65$ pe setul de test S1 (depășește pragul clasic Otsu cu $+11\%$); reprezintă baseline-ul oficial al setului de date. |
| **2** | *Explainable Flood Segmentation on Sentinel-1 SAR Imagery: A Comparative Study of CNN and Transformer Architectures* (2026) | Hibrid CNN + Vision Transformer (ViT) + XAI | S1 SAR (Sen1Floods11) | IoU, F1-Score | $\text{IoU} = 0.74$; integrează hărți de atenție și Grad-CAM utile pentru explicarea deciziilor către autorități (IGSU). |
| **3** | *Fully Convolutional Neural Network for Rapid Flood Mapping in Synthetic Aperture Radar Images* (Nemni et al., Remote Sensing 2020) | U-Net (backbone ResNet / Xception) | S1 SAR (activări Copernicus EMS) | F1-Score, IoU | $\text{F1} = 0.91$, $\text{IoU} = 0.84$; reduce timpul de cartografiere cu $80\%$ față de fluxul semi-automat din Copernicus EMS. |
| **4** | *Deep learning methods for flood mapping: a review of existing applications and future research directions* (Bentivoglio et al., HESS 2022 / 2021) | Arhitecturi CNN multi-intrare | S1 SAR + DEM topografic | F1-Score | $\text{F1} = 0.88$; demonstrează că includerea DEM-ului elimină alarmele false cauzate de umbrele radar pe versanți. |
| **5** | *Flood extent mapping using Sentinel-1 SAR imagery and Deep Learning* (Katiyar et al., IGARSS 2021) | DeepLabv3+ vs. U-Net vs. Otsu | S1 SAR ($\text{VV} + \text{VH}$) | Precision, Recall, F1 | $\text{F1} = 0.86$; arată că folosirea ambelor polarizări ($\text{VV} + \text{VH}$) crește precizia la granița apă-vegetație. |
| **6** | *Towards global flood mapping onboard low cost satellites with machine learning* (Mateo-Garcia et al., Nature Scientific Reports 2021) | U-Net / SCNN (*WorldFloods*) | S1, S2 | IoU | $\text{IoU} = 0.71$; validează capacitatea de generalizare a modelelor de segmentare pe bazine hidrografice nemaivăzute la antrenare. |
| **7** | *Exploring Sentinel-1 and Sentinel-2 diversity for flood inundation mapping using deep learning* (Konapala et al., ISPRS 2021) | U-Net cu fuziune multi-sursă | S1, S2, SRTM DEM (Sen1Floods11) | F1-Score, IoU | $\text{F1} = 0.84$; demonstrează că adăugarea canalelor topografice (DEM + pantă) la S1 aduce un câștig de $+6\%$ în F1-Score. |
| **8** | *Flood Segmentation on Sentinel-1 SAR Imagery with Semi-Supervised Learning / BASNet* (Bai et al., 2021) | U-Net++ / BASNet (Boundary-Aware) | S1 SAR (Sen1Floods11) | IoU, F1-Score | $\text{IoU} = 0.73$, $\text{F1} = 0.85$; conexiunile dense păstrează contururile fine ale canalelor secundare inundate. |
| **9** | *Urban Flood Mapping with Residual Patch Similarity Learning in Bi-temporal SAR Imagery* (Peng et al., IEEE TGRS 2019) | ResNet Patch Similarity | S1 SAR | Coeficient Kappa, OA | $\text{Kappa} = 0.87$; arhitectura reziduală este robustă la reflexiile duble (double-bounce) din localitățile inundate. |
| **10** | *Automated Extraction of Surface Water Extent from Sentinel-1 Data* (Huang et al., Remote Sensing 2018) | Random Forest + filtre morfologice | S1 SAR + SRTM DEM | Overall Accuracy (OA) | $\text{OA} = 93.2\%$; oferă un reper clasic de Machine Learning pentru mascarea pantelor $> 5^\circ$ cu ajutorul DEM-ului. |