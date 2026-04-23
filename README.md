# Network Anomaly Detection with Wireshark (NADW)

A supervised multi-class machine learning system that classifies network traffic captured by Wireshark into attack categories. Feed any Wireshark `.csv` export into the inference pipeline and get a full anomaly report — per-flow predictions, confidence scores, and visualisations.

---

## What it detects

| Class | Attack types |
|---|---|
| **Flood** | SYN Flood, UDP Flood, HTTP Flood, ICMP Flood, Slowloris, NTP Amplification |
| **Malware** | Cerber, GlobeImposter, Hidden Tear, Locky (ransomware) · Agent Tesla, FormBook, Redline (spyware) · Chthonic, Delf Banker, Lokibot, Ratinfected, Squirrel Waffle (trojan) |
| **Brute Force** | Password cracking, MySQL brute force, Redis brute force, SMB brute force |
| **Exploit** | CVE-2020, SQL Injection, Evil Twin, Brobot |
| **Probe** | Port scan, MITM reconnaissance |
| **Benign** | Normal network traffic |

---

## How it works

Raw Wireshark packets are not classified directly — a single packet from an HTTP Flood is indistinguishable from normal browsing. Instead, packets are grouped into **flows** (same source IP + destination IP + protocol within a 5-second window) and 18 statistical features are extracted per flow. The model classifies these behavioral patterns, not individual packets.

```
Wireshark CSVs  →  Label by folder  →  Group into flows  →  Extract 18 features
      →  Train Random Forest + XGBoost  →  Classify live captures
```

This is the same approach used by professional IDS systems like Snort and Suricata.

---

## Project structure

```
nadw_project/
│
├── 01_data_loading.ipynb          # Download dataset, label by folder, save parquet
├── 02_eda.ipynb                   # Exploratory data analysis — 7 visualisations
├── 03_feature_engineering.ipynb   # Group packets into flows, extract 18 features
├── 04_model_training.ipynb        # Train RF + XGBoost, SMOTE, evaluate, save model
├── 05_inference_pipeline.ipynb    # Analyse your own Wireshark captures
│
├── utils/
│   └── feature_engineering.py     # Shared feature extraction (used by NB3 + NB5)
│
├── data/                          # Created at runtime
│   ├── raw_combined.parquet       # All labelled packets combined
│   ├── features.parquet           # Flow-level features ready for training
│   └── plots/                     # EDA and evaluation charts
│
├── models/                        # Created at runtime
│   ├── best_model.pkl             # Best classifier (RF or XGBoost)
│   ├── scaler.pkl                 # Fitted StandardScaler
│   ├── label_encoder.pkl          # Class name ↔ integer mapping
│   └── model_metadata.json        # Accuracy scores and config
│
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** [`boy177/NADW-network-attacks-dataset`](https://huggingface.co/datasets/boy177/NADW-network-attacks-dataset) (duplicated from `onurkya7/NADW-network-attacks-dataset`)

Each CSV file is a clean, isolated Wireshark capture of one simulated attack type. Labels are derived from folder names — there is no label column inside the files.

```
benign/           → benign-1.csv … benign-5.csv
brute_force/      → brute_force.csv, mysql_brute_force.csv, redis_brute_force.csv,
                    SMB_brute_force.csv, SMB_brute_force-2.csv
exploit/          → brobot.csv, cve-2020.csv, sql-i.csv, tch-ssl.csv
flood/            → http-flood.csv, http_slowloris.csv, icmp_flood.csv,
                    ntp-amplification.csv, syn-flood.csv, udp_flood.csv
malware/
  ransomware/     → cerber.csv, globeImposter.csv, hidden_tear.csv, locky.csv, lord.csv
  spyware/        → agent_tesla.csv, form_book.csv, redline.csv
  trojan/         → chthonic.csv, delf-banker.csv, lokibot.csv,
                    ratinfected.csv, squirrel_waffle.csv
probe/            → probe.csv
```

**25 CSV files · 6 attack classes · ~151,000 packets**

---

## Features extracted per flow (18 total)

| Feature | Description |
|---|---|
| `packet_count` | Total packets in the flow |
| `avg_length` / `std_length` / `min_length` / `max_length` | Packet size statistics |
| `avg_iat` / `std_iat` | Inter-arrival time mean and standard deviation |
| `duration` | Flow duration in seconds |
| `bytes_per_second` | Total bytes ÷ duration |
| `packets_per_second` | Packet count ÷ duration |
| `syn_count` / `ack_count` / `fin_count` / `rst_count` | TCP flag counts parsed from the Info column |
| `http_request_count` / `http_response_count` | HTTP GET/POST and 200/404 counts |
| `unique_dst_ports` | Number of distinct destination ports in the flow |
| `protocol_encoded` | Dominant protocol encoded as integer |

---

## Setup

### 1. Clone or download

```bash
git clone <your-repo-url>
cd nadw_project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order

```bash
jupyter notebook
```

Open and run each notebook top to bottom in this order:

```
01_data_loading.ipynb
02_eda.ipynb
03_feature_engineering.ipynb
04_model_training.ipynb
```

### 4. (Recommended) Add your own normal traffic

The lab benign data does not contain much QUIC or TLSv1.3, which dominate modern real-world traffic. Without augmenting the benign class, the model may flag legitimate QUIC traffic as Malware.

In `01_data_loading.ipynb`, add paths to your own clean Wireshark captures:

```python
EXTRA_BENIGN_PATHS = [
    Path('normalTraffic.csv'),  # export a 10-min Wireshark capture of normal traffic
]
```

Then re-run notebooks 1 → 3 → 4 to retrain with the augmented benign class.

### 5. Analyse your own Wireshark captures

1. Open Wireshark and record traffic
2. **File → Export Packet Dissections → As CSV...**
3. Ensure these columns are included: `No.`, `Time`, `Source`, `Destination`, `Protocol`, `Length`, `Info`
4. Open `05_inference_pipeline.ipynb` and set:

```python
CAPTURE_CSV = Path('your_capture.csv')
```

5. Run all cells — you will get a per-flow prediction table, summary report, and two charts

---

## Outputs from the inference pipeline

| Output | Description |
|---|---|
| `analysis_<name>.png` | Bar chart + pie chart of detected traffic types |
| `confidence_<name>.png` | Confidence score distribution per predicted class |
| `results_<name>.csv` | Full per-flow table with `raw_prediction`, `prediction`, and `confidence` |

---

## Design decisions

### Why flow-based features?
Packet-level features from clean simulated data overfit badly — UDP Flood packets are all identical 542-byte UDP datagrams. A model trained on raw packets would fail completely on real noisy traffic. Flow-level features capture behavioral patterns (packet rate, IAT, flag ratios) that generalise to real captures.

### Why a shared `feature_engineering.py` module?
If NB3 and NB5 had their own copies of the feature extraction code, they could silently diverge. One shared module guarantees byte-for-byte identical logic at training and inference time, eliminating training/inference skew — one of the most common ML production bugs.

### Why a confidence threshold?
The model assigns a probability score to every prediction. Flows where the model is below 80% confident are demoted to `Benign` — the model is essentially saying "I don't recognise this traffic pattern strongly enough." This significantly reduces false positives on traffic types not well-represented in training data (e.g. QUIC-heavy captures).

### Why both Random Forest and XGBoost?
Different algorithms have different inductive biases. Random Forest is robust and parallelisable. XGBoost achieves higher accuracy on tabular data through gradient boosting. We train both and keep whichever has higher test-set accuracy. Neither is used to evaluate the other.

### Why SMOTE?
The Probe class has far fewer flows than Flood. A model trained on imbalanced data learns to predict the majority class. SMOTE generates synthetic minority-class samples by interpolating between existing samples in feature space, giving all classes equal representation during training.

---

## Requirements

```
Python 3.10+
pandas, numpy, scikit-learn, xgboost
imbalanced-learn, matplotlib, seaborn
huggingface_hub, joblib, pyarrow
jupyter, ipykernel
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## HuggingFace token

The dataset is public — no token needed. If you make your duplicate private, set:

```bash
export HF_TOKEN=hf_your_token_here
```

The notebook reads this automatically via `os.getenv('HF_TOKEN', None)`.

---

## Interpreting results

| Confidence | Meaning |
|---|---|
| ≥ 80% predicted as Flood/Malware/etc. | High confidence anomaly — investigate |
| < 80% any class | Uncertain — demoted to Benign in final output |
| Benign ≥ 80% | Model is confident this is normal traffic |

The `raw_prediction` column shows what the model originally predicted before the threshold was applied. The `prediction` column shows the final decision after demotion. Compare both to understand where the model is uncertain.

---

## Limitations

- Trained on simulated attacks — real-world attacks are noisier and more varied
- New attack types not in the training data (zero-days) will not be detected
- The fixed 5-second window may split slow attacks (like Slowloris) across multiple flows
- IP addresses are not used as features — the model generalises across networks but loses IP-based signal
- Accuracy will degrade over time as network behaviour evolves — periodic retraining is recommended

---

## Acknowledgements

Dataset originally created by [onurkya7](https://huggingface.co/datasets/onurkya7/NADW-network-attacks-dataset). Captures collected using Wireshark on simulated attack environments covering 25 distinct attack scenarios across 6 categories.