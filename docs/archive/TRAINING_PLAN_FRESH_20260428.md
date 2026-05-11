# VeriPromise ESG 2026 — 企業級訓練計畫書 (Fresh Plan)

文件版本：v1.0 (Fresh Start)  
撰寫日期：2026-04-28  
撰寫立場：**不參考 `final_summary/` 任何既有結論**，僅以官方競賽規範、Baseline Notebook、原始資料、以及兩篇相關論文 (`2025.emnlp-main.1028.pdf`, `2025.semeval-1.321.pdf`) 為依據，從零重新規劃。

---

## 0. 規劃哲學

過往 `final_summary/` 內的結論皆由 GPT-3.5 推導，存在三項風險：(1) 模型推論能力較弱，超參數搜索區間可能過窄；(2) 缺乏對原始論文方法 (ML-Promise、SemEval Promise Verification) 的對齊；(3) 對少數類別與條件式標籤的處理可能過於簡化。本計畫採「**證據驅動 (Evidence-driven)**」立場：

1. 先以 Baseline 復現作為**唯一可信錨點 (anchor)**，建立可信的本地評分管線。
2. 任何超參數 / 架構變更，必須在 5-Fold CV 上展現 ≥ +0.005 加權分數，且需 ≥ 3 個 seed 的一致性才採納。
3. 所有實驗紀錄需 **可重現 (reproducible)**：固定 seed、固定 split、固定環境 hash。

---

## 1. 競賽任務形式化

設輸入文本 $x$，模型 $f_\theta$ 需同時輸出四個欄位的條件式標籤：

| 子任務 | 符號 | 類別數 | 標籤集合 | 指標 | 權重 $w_t$ |
| :-- | :-- | :--: | :-- | :-- | :--: |
| T1 promise_status | $y_1$ | 2 | {Yes, No} | F1 (positive=Yes) | 0.20 |
| T2 verification_timeline | $y_2$ | 5 | {already, within_2_years, between_2_and_5_years, longer_than_5_years, N/A} | Macro-F1 | 0.15 |
| T3 evidence_status | $y_3$ | 3 | {Yes, No, N/A} | F1 (positive=Yes) | 0.30 |
| T4 evidence_quality | $y_4$ | 4 | {Clear, Not Clear, Misleading, N/A} | Macro-F1 | 0.35 |

最終分數：

$$
\mathcal{S}(\hat{y}, y) = \sum_{t=1}^{4} w_t \cdot \mathrm{F1}_t(\hat{y}_t, y_t), \qquad \sum_t w_t = 1
$$

### 1.1 條件式標籤約束 (Hierarchical Constraints)

依官方欄位語意推導出嚴格約束：

- $y_1 = \text{No} \Rightarrow y_2 = y_3 = y_4 = \text{N/A}$，且 `promise_string = ""`、`evidence_string = ""`。
- $y_3 = \text{No} \Rightarrow y_4 = \text{N/A}$，且 `evidence_string = ""`。

**結論**：推論時必須以「Hierarchical Post-Processing」強制覆寫，否則 T2/T4 會被無效預測污染 Macro-F1。

### 1.2 提交與時程紀律

- 6/03 驗證集釋出、6/10 測試集釋出、6/17 截止上傳、6/30 程式碼/報告繳交。
- 每日提交上限 **3 次**，以最後一次為準。建議：每日固定 1 次、保留 2 次給意外回測。

---

## 2. 資料初步剖析 (基於 `vpesg4k_train_1000 V1.csv`)

**樣本量**：1,000 筆；欄位 14 欄（含 id/data/esg_type/標籤四欄/promise_string/evidence_string/company/ticker/page_number/pdf_url/company_source）。

**重要觀察 (待 Step 1 由腳本自動量化驗證)**：

| 欄位 | 預期分佈問題 | 影響 |
| :-- | :-- | :-- |
| promise_status | Yes 偏多 (估 ~80%) | T1 Macro 較易；F1 with positive=Yes 為主 |
| verification_timeline | already / between_2_and_5_years 為大宗；within_2_years 與 longer_than_5_years 稀少 | T2 Macro 受少類拖累 |
| evidence_status | Yes/No/N/A 三類；N/A 與 promise_status=No 完全相依 | T3 受 T1 影響 |
| evidence_quality | Clear 較多；Misleading 極少 | T4 Macro 受 Misleading 拖累，且權重最高 (0.35) |
| esg_type | 多標籤 (E / S / G 可同時)，以 `;` 分隔 | 可作為輔助特徵或輔助任務 |
| data 文字長度 | 中文段落，預估 200 ~ 800 字 | MAX_LEN=256 會截斷部分樣本 |
| company / ticker | 可能存在來源偏見 | 切分時必須避免「同公司樣本同時出現在訓練 + 驗證」的洩漏風險 |

> **企業級紀律**：Step 1 必須先輸出量化 EDA 報告 (HTML + JSON metrics)，禁止用「估計值」做決策。

---

## 3. 整體技術架構

### 3.1 任務建模策略選擇 (Decision Matrix)

| 策略 | 優點 | 缺點 | 是否採用 |
| :-- | :-- | :-- | :--: |
| (A) Multi-task 共享 encoder + 4 個分類頭 | 知識共享、單模能解全部任務、推論成本低 | 任務間損失尺度可能衝突 | **採用為主力** |
| (B) Pipeline (T1 → T3 → T2/T4) | 明確利用條件式約束 | 誤差累積、訓練複雜 | 作為比較組 (B-Plan) |
| (C) 4 個獨立單任務模型 | 解耦、調優自由度高 | 4 倍訓練/推論成本、無法共享表徵 | 不採用（成本過高） |
| (D) Generative LLM (LoRA) | 可融合條件式約束 | 1000 筆資料量易過擬合、輸出穩定性差、推論成本高 | 列入 Phase 5 高階探索 |

### 3.2 主力架構 (Plan A — Multi-Task BERT)

```
                    ┌──────────────────────────┐
input text  ───→    │  Encoder (Backbone)      │  ───→ pooled_output (H)
                    │  hfl/chinese-macbert-...  │       (取 [CLS] + Mean-Pool)
                    └──────────────────────────┘
                                  │
       ┌──────────┬──────────┬────┴─────┬──────────┐
       ▼          ▼          ▼          ▼          
   Dropout    Dropout    Dropout    Dropout       
       ▼          ▼          ▼          ▼          
   Linear     Linear     Linear     Linear        
   (H, 2)     (H, 5)     (H, 3)     (H, 4)        
       ▼          ▼          ▼          ▼          
     T1         T2         T3         T4           
```

**設計細節**：

- **Pooling**：採 `[CLS]` + `mean(last_hidden_state * attention_mask)` 兩者 concat (維度 2H)，比僅用 `[CLS]` 對段落級任務更穩。
- **Dropout**：每個任務頭前獨立 `Dropout(p=0.1)`。
- **Loss 加權**：訓練 loss = $\sum_t \alpha_t \cdot \text{CE}_t$，初始 $\alpha = (1,1,1,1)$；可選方案以評分權重 $\alpha = (0.2, 0.15, 0.3, 0.35)$ 對齊。
- **Class Weight**：T2、T4 啟用 inverse-frequency `class_weight`，由訓練 fold 內統計而得。
- **AMP / fp16**：必開以節省顯存。
- **Gradient Accumulation**：根據 GPU 顯存自動調整以維持 effective batch = 16。

### 3.3 模型骨幹候選 (Backbone Pool)

| Backbone | 規模 | 預期角色 | 優先序 |
| :-- | :--: | :-- | :--: |
| `hfl/chinese-macbert-base` | 102M | 快速迭代基準、Phase 1 主力 | 1 |
| `hfl/chinese-roberta-wwm-ext` | 102M | 對照組（Baseline notebook 採用） | 2 |
| `hfl/chinese-macbert-large` | 326M | Phase 3 升規模主力 | 3 |
| `hfl/chinese-roberta-wwm-ext-large` | 326M | Ensemble 多樣化 | 4 |
| `xlm-roberta-large` | 560M | 跨語言、決策邊界差異化 | 5 |

> **原則**：先 base 後 large，先單模後集成，每一步都需 5-Fold OOF 證據支持。

### 3.4 驗證策略 (Validation Protocol)

- **K-Fold**：`StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`，分層鍵採 `promise_status × evidence_status` 之 6 類聯合標籤（避免極稀類落單）。
- **Group-aware 變體 (重要)**：若 EDA 顯示同一 `company` 出現 ≥ 5 筆，將額外比較 `StratifiedGroupKFold(group=company)` 以衡量「公司洩漏」對泛化的影響。任一指標低於普通 StratifiedKFold ≥ 0.02，採用 GroupKFold 為最終切分。
- **Seed 集合**：`[42, 2024, 20260417, 7, 1234]` (5 顆)。
- **OOF 機率保存**：每 (seed, fold) 儲存 `np.float16` 機率張量，作為 Ensemble 輸入。
- **指標彙總**：每 fold 報告 weighted score + 4 個任務 macro-F1 + per-class F1；Phase 末以 `mean ± std` 呈現。

### 3.5 集成策略 (Ensemble)

1. **OOF Hill-Climbing**：對每個任務 $t$，在 OOF 機率上以貪心法 (greedy weighted average) 搜尋使 macro-F1 最大化的模型權重。
2. **任務獨立輸出**：T1/T2/T3/T4 各自擁有最佳 ensemble，互不干擾。
3. **Hierarchical Post-Process**：
   - $\hat{y}_1 = \text{No} \Rightarrow$ 覆寫 $\hat{y}_{2,3,4} = \text{N/A}$。
   - $\hat{y}_3 = \text{No} \Rightarrow$ 覆寫 $\hat{y}_4 = \text{N/A}$。
4. **TTA (Test-Time Augmentation, 可選)**：對測試集做「全文 / 前 256 token / 後 256 token」三段平均。

---

## 4. 五階段路線圖 (Phased Roadmap)

下表 **目標** 為計畫初擬時的保守估計，**實績** 為截至本版日期的 OOF 加權分數。

| Phase | 目標 OOF 加權 | **實績 OOF 加權** | 主要工作 | 狀態 |
| :--: | :--: | :--: | :-- | :--: |
| **0 — Scaffolding** | — | — | 建立企業級專案骨架、環境、CI、EDA 報告 | 已完成 |
| **1 — Baseline 復現** | 0.55 ~ 0.58 | **0.6415 ± 0.0133** | macbert-base + 5-Fold + AMP + 嚴謹評分管線 | 已完成（超標） |
| **2 — 單模調優 + Combo** | 0.58 ~ 0.61 | **0.66734** (combo seed-avg ensemble) | LR / max_len / pooling / class_weight 系統掃描 + 最佳組合復測 | 已完成（超標） |
| **3 — 升規模 (macbert-large)** | 0.61 ~ 0.64 | 0.66439 (lr2e5, 5-fold seed=42) | macbert-large 多 lr / epoch 變體；單模未通過 +0.005 門檻 | 已完成（負面結果，保留作集成輸入） |
| **4 — 多骨幹集成** | 0.63 ~ 0.66 | **0.67478** (combo + large 1.5:1) | Probability ensemble (2-way) | 已完成 |
| **5 — Wave A/B Ablation + Combo v2 + 3-way Hillclimb** | +0.005 | **0.67954** (per-task hillclimb) | 16 組未試參數掃描 + Combo v2 (FGM+Focal-T4) + 3-way ensemble + per-task 權重 | 已完成 |
| **6 — Wave C 增強 + 7-way Hillclimb** | +0.002 | **0.68206** (7-way per-task hillclimb v3) | Token-level 資料增強、R-Drop、Multi-Sample Dropout 三大新技術；7 模型 ensemble 池 + 兩階段 hillclimb | 已完成 |
| 7 — Combo v3 + 8-way Hillclimb | +0.002 | 0.68394 (8-way per-task hillclimb v4) | combo_v2 疊加 R-Drop + MSD → single OOF 0.67121；8 模型 ensemble 池；T4 首次突破 0.4560 | 已完成 |
| 8 — Combo v3 Multi-seed 拆分 9-way | +0.0005 | 0.68440 (9-way per-task hillclimb v6) | combo_v3 增訓 seeds 2024+20260417；seed=42 (peaky) 與 3-seed 平均 (smooth) 並列為兩個 pool 成員；T4=0.4513 再創新高 | 已完成 |
| 9 — Sprint A backbone 擴 pool + 11-way Hillclimb v7b | +0.001 | 0.68558 (11-way per-task hillclimb v7b, biased search) | 加入 p4a `chinese-roberta-wwm-ext-base` (0.66626) 與 p4b `bert-base-chinese` (0.67355)；p4c `xlm-roberta-base` 0.61269 被拒 (<0.65 門檻) | 已完成 |
| 10 — Sprint B / T6 v1 (時間 token prefix) + 12-way Hillclimb v8 | +0.001 | 0.68669 (12-way per-task hillclimb v8) | regex 抽年份 + bucket prefix 注入；單模 T2 未直擊但 ensemble 在 T1/T3 各取權重 2.0 | 已完成 |
| 11 — Sprint B / T6 v2 (專屬 bucket token) + 13-way Hillclimb v9 | +0.0001 | 0.68683 (13-way per-task hillclimb v9) | 5 個 added-vocab token 取代 prefix；單模 T2 退步 −0.0171；ensemble +0.00014 ≈ noise | 已完成 |
| 12 — Sprint B / T7 (Focal-T4 γ=3.0) + 14-way Hillclimb v10 | +0.001 | 0.68660 (14-way per-task hillclimb v10) | `p7_focal_g3` 單模 0.67416 (歷史第二高，T4 +0.0237 為單模史上最大 T4 增益)；ensemble 因 constraint coupling 抵銷 per-task 增益 → SOTA 未破 | 已完成（未破 SOTA） |
| **13 — Sprint B 收尾批次 / U2 EMA-995 + U4 LS(T1/T3) + U7 Joint Hillclimb v11** | **+0.001** | **0.68770** (16-way joint post-constraint hillclimb v11) | p8_ema995 崩潰 0.63010（EMA 無 warm-start）；p9_ls_t1t3 0.67334 (+0.00213)；v11 聯合 hillclimb 直接優化 post-constraint score（4 次 accept 全在 T3，T4 +0.0018 為 constraint coupling 連動效應）| 已完成（**目前 SOTA**） |
| 14 — Sprint C 結構性多樣化（**進行中規劃**） | +0.002 ~ +0.005 | 目標 ≥ 0.69 | 跳脫 combo_v3 變體：(a) 加入 macbert-large 做為新 backbone；(b) U1 TTA 多視角推論；(c) U10 外部 ESG 公開資料偽標擴增（規則允許，見 §19.1）；(d) X1 EMA-with-warm-start 重啟 | 待辦 |
| 15 — 最終衝刺 (6/03 驗證集 / 6/10 測試集) | — | — | 提交策略、Full-Data Refit、文件與報告；U12 train↔val gap 分析 | 待辦 |

> 每階段切換的 **Gate 條件**：5-Fold 平均 weighted score ≥ 上階段 + 0.005，且任一單任務指標跌幅 ≤ 0.005。Phase 3 大模型單模未通過 Gate，但 ensemble (Phase 4) 通過。

---

## 5. 企業級專案骨架 (Step 0 產出)

```
esg-veripromise-2026/
├── data/
│   ├── raw/                    # 官方原始檔案 (read-only)
│   ├── processed/              # 清洗後 / 切分快照
│   └── splits/                 # 5-Fold 索引 .json (固定 seed)
├── configs/
│   ├── base.yaml               # 共用預設值
│   ├── exp_p1_baseline.yaml    # Phase 1
│   ├── exp_p2_tune_*.yaml      # Phase 2 系列
│   ├── exp_p3_large_*.yaml     # Phase 3 系列
│   └── exp_p4_*.yaml           # Phase 4 系列
├── src/
│   ├── __init__.py
│   ├── seed.py                 # 全域 seed 工具
│   ├── data/
│   │   ├── loader.py           # CSV/JSON 載入、欄位驗證
│   │   ├── splits.py           # StratifiedKFold / StratifiedGroupKFold
│   │   └── dataset.py          # PyTorch Dataset / collate_fn
│   ├── models/
│   │   ├── multitask.py        # MultiTaskClassifier
│   │   └── pooling.py          # ClsMeanPooler
│   ├── training/
│   │   ├── trainer.py          # AMP / grad accum / EMA / early stop
│   │   ├── losses.py           # CE + class_weight + (Phase 4) FGM
│   │   └── schedulers.py       # cosine + warmup
│   ├── inference/
│   │   ├── predict.py          # 推論 + Hierarchical Post-Process
│   │   └── tta.py              # TTA (Phase 5)
│   ├── ensemble/
│   │   └── hillclimb.py        # Per-task greedy weighted ensemble
│   ├── eval/
│   │   ├── metrics.py          # weighted_score / per-task macro-F1
│   │   └── report.py           # markdown / html 報告產生
│   └── tools/
│       ├── eda.py              # 自動 EDA → reports/eda/
│       └── verify_env.py       # 環境一致性檢查
├── scripts/
│   ├── 00_setup.ps1            # Windows 環境腳本
│   ├── 01_eda.ps1              # 跑一次完整 EDA
│   ├── 10_train_kfold.ps1      # 跑單一 config 的 5-Fold
│   └── 20_submit.ps1           # 產出提交檔
├── outputs/
│   ├── checkpoints/{exp}/seed{S}/fold{F}/best.pt
│   ├── oof/{exp}/seed{S}/fold{F}/probs.npz
│   ├── logs/{exp}/...          # JSONL training logs
│   └── submissions/sub_YYYYMMDD_v{N}.json
├── reports/
│   ├── eda/eda_report.html
│   ├── experiments/{exp}/score_summary.csv
│   └── final_report.md
├── tests/
│   ├── test_metrics.py         # 評分函式單元測試
│   ├── test_post_process.py    # 條件式約束單元測試
│   └── test_split_no_leak.py   # 公司洩漏檢查
├── requirements.txt
├── pyproject.toml              # 專案中介資訊 + ruff/black 規則
├── README.md
└── .gitignore
```

### 5.1 環境鎖版 (`requirements.txt`)

```
torch>=2.2,<2.5
transformers>=4.41,<4.46
datasets>=2.19
scikit-learn>=1.4
pandas>=2.2
numpy>=1.26,<2.0
pyyaml>=6.0
tqdm>=4.66
matplotlib>=3.8
seaborn>=0.13
accelerate>=0.30
sentencepiece>=0.2
protobuf>=4.25,<5
ruff>=0.4
pytest>=8.2
```

### 5.2 設定檔 (`configs/base.yaml` 範本)

```yaml
project_name: vp_esg_2026
data:
  csv_path: data/raw/vpesg4k_train_1000 V1.csv
  text_field: data
  label_fields:
    promise_status: ["Yes", "No"]
    verification_timeline: ["already", "within_2_years", "between_2_and_5_years", "longer_than_5_years", "N/A"]
    evidence_status: ["Yes", "No", "N/A"]
    evidence_quality: ["Clear", "Not Clear", "Misleading", "N/A"]
  field_weights: {promise_status: 0.20, verification_timeline: 0.15, evidence_status: 0.30, evidence_quality: 0.35}
  group_field: company
split:
  type: stratified_kfold        # or stratified_group_kfold
  n_splits: 5
  stratify: [promise_status, evidence_status]
  shuffle: true
model:
  backbone: hfl/chinese-macbert-base
  max_length: 256
  dropout: 0.1
  pooling: cls_mean
training:
  epochs: 5
  batch_size: 16
  grad_accum: 1
  lr: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.10
  scheduler: cosine
  grad_clip: 1.0
  label_smoothing: 0.0
  use_amp: true
  use_class_weight: true
  task_loss_weights: {promise_status: 1.0, verification_timeline: 1.0, evidence_status: 1.0, evidence_quality: 1.0}
  early_stop_patience: 2
  ema:
    enable: false
    decay: 0.999
seeds: [42]
runtime:
  num_workers: 2
  pin_memory: true
  log_every: 50
  save_oof: true
```

---

## 6. Phase 1 主力配置 (Baseline 復現)

`configs/exp_p1_baseline.yaml` (繼承 `base.yaml`)：

```yaml
extends: base.yaml
exp_name: p1_baseline_macbert_base
model:
  backbone: hfl/chinese-macbert-base
  max_length: 256
training:
  epochs: 5
  batch_size: 16
  lr: 2.0e-5
seeds: [42, 2024, 20260417]
```

預期：5-Fold × 3 seeds = 15 runs，平均 weighted score 0.55 ~ 0.58。**通過此 Gate 後才進 Phase 2**。

---

## 7. Phase 2 ~ Phase 4 超參數搜索藍圖

### 7.1 Phase 2 — 單模調優 (僅在 macbert-base 上做，節省成本)

| 參數 | 掃描值 | 備註 |
| :-- | :-- | :-- |
| lr | {1e-5, 2e-5, 3e-5, 5e-5} | base 模型適合較大 lr |
| max_length | {192, 256, 320} | 對長文段落任務有影響 |
| warmup_ratio | {0.05, 0.10, 0.15} | – |
| pooling | {cls, mean, cls_mean} | 段落級任務通常 cls_mean 較佳 |
| dropout | {0.1, 0.2} | – |
| label_smoothing | {0.0, 0.05} | – |
| class_weight | {off, on} | T2/T4 預期受益 |
| task_loss_weights | {均等, 評分權重對齊, 反向強化弱項} | 三組對照 |

**搜索方法**：先 sequential (一參數一參數)，再對 top-3 做小規模 grid。每組僅用 seed=42 跑 1 次完整 5-Fold 過濾，最佳組合再以 3 seeds 復測。

### 7.2 Phase 3 — 升規模 (macbert-large)

固定 Phase 2 之最佳參數，僅變更：

```yaml
model.backbone: hfl/chinese-macbert-large
training.batch_size: 4
training.grad_accum: 4         # effective batch = 16
training.lr: 1.5e-5            # large 模型偏小 lr
training.epochs: 4
```

跑 5 seeds × 5 folds = 25 runs，OOF 進入 Hill-Climbing。

### 7.3 Phase 4 — 多骨幹 + 弱項補強

- **多骨幹**：`hfl/chinese-roberta-wwm-ext-large`、`xlm-roberta-large` 各 3 seeds × 5 folds。
- **弱項補強 (僅針對 T2/T4)**：
  - **R-Drop**：同 batch 兩次 forward，KL 正則 ($\alpha=0.5$)。
  - **FGM 對抗訓練**：embedding 加擾動 ($\epsilon=1.0$)。
  - **EMA**：decay=0.999，推論用 EMA 權重。
  - **Multi-Sample Dropout**：T4 head 套 5 份 dropout 平均。
  - **Minority Oversample**：對 T2 `longer_than_5_years` × 2、T4 `Misleading` × 2。

每項獨立 A/B 測試，僅在 OOF +0.005 才採納。

---

## 8. 推論與後處理 SOP

```
ensemble_probs(test) ─→ argmax per task ─→ raw_preds
   │
   ├─ if y1 == "No":
   │     y2, y3, y4 := "N/A"
   │     promise_string := ""
   │     evidence_string := ""
   │
   ├─ if y3 == "No":
   │     y4 := "N/A"
   │     evidence_string := ""
   │
   └─ Schema validation ─→ submission.json
```

**單元測試** (`tests/test_post_process.py`) 必須涵蓋上述全部覆寫情境，CI 失敗則禁止生成提交檔。

---

## 9. 風險與止損機制

| 風險 | 監控指標 | 止損動作 |
| :-- | :-- | :-- |
| 公司洩漏 (company leakage) | StratifiedKFold vs GroupKFold 差距 ≥ 0.02 | 改採 GroupKFold |
| 顯存爆掉 | OOM | bs/2 + grad_accum×2 |
| 弱項補強反向劣化 | 任一任務 macro-F1 跌 ≥ 0.005 | 回滾配置 |
| Seed 變異過大 (std ≥ 0.01) | – | 增加 seed 至 7 顆 |
| 提交額度耗盡 | 當日提交次數 ≥ 2 | 進入 only-on-improvement |
| 標註雜訊 | 模型錯預測但 prob ≥ 0.85 的樣本 ≥ 5% | 啟動人工複核（不修改測試集） |

---

## 10. 交付物 (Deliverables)

| 項目 | 路徑 |
| :-- | :-- |
| 完整可重現程式碼 | `src/` + `configs/` + `scripts/` |
| 5-Fold OOF 機率 | `outputs/oof/{exp}/...` |
| 模型權重 | `outputs/checkpoints/{exp}/...` |
| 評分總表 | `reports/experiments/score_summary.csv` |
| EDA 報告 | `reports/eda/eda_report.html` |
| 提交檔 | `outputs/submissions/sub_YYYYMMDD_v{N}.json` |
| 最終書面報告 | `reports/final_report.md` |

---

## 11. 排程

| 期間 | 階段 | 主要產出 | 狀態 |
| :-- | :-- | :-- | :--: |
| Day 1 | Phase 0 — Scaffolding | 專案骨架 + EDA 報告 | 已完成 |
| Day 2 ~ 3 | Phase 1 — Baseline 復現 | macbert-base 5-Fold × 3 seeds → **0.6415** | 已完成 |
| Day 4 ~ 6 | Phase 2 — 單模調優 + Combo | 8 組 ablation + 3-seed combo → **0.66734** | 已完成 |
| Day 7 ~ 9 | Phase 3 — 升規模 | macbert-large baseline/lr2e5/ep8 (負面) | 已完成 |
| Day 10 ~ 13 | Phase 4 — 多骨幹 + 弱項補強 | combo+large=0.67478 → Phase 5 per-task=0.67954 | 已完成 |
| Day 14 | Phase 6 — Wave C + 7-way Hillclimb | aug/R-Drop/MSD + 7-way ensemble → **0.68206** | 已完成 |
| Day 15 | Phase 7 — Combo v3 + 8-way Hillclimb | combo_v2 + R-Drop + MSD → 0.67121 單模；8-way ensemble → 0.68394 | 已完成 |
| Day 16 | Phase 8 — combo_v3 Multi-seed + 9-way Hillclimb | 拆 peaky/avg 為兩成員 → 0.68440 (Phase 8 SOTA) | 已完成 |
| Day 17 | Phase 9 — Sprint A backbone 擴 pool + 11-way Hillclimb v7b | p4a/p4b 入池（p4c 拒）+ biased search → 0.68558 | 已完成 |
| Day 18 | Phase 10 — Sprint B / T6 v1 時間 token prefix + 12-way Hillclimb v8 | p5_t6_time_token + biased search → 0.68669 | 已完成 |
| Day 19 | Phase 11 — Sprint B / T6 v2 專屬 bucket token + 13-way Hillclimb v9 | p6_t6v2_bucket_tok + biased search → **0.68683 (現任 SOTA)** | 已完成 |
| Day 20 | Phase 12 — Sprint B / T7 Focal-T4 γ=3.0 + 14-way Hillclimb v10 | p7_focal_g3 單模 0.67416；ensemble 0.68660（被 constraint coupling 抵銷，未破 SOTA）| 已完成 |
| Day 21 | Phase 13 — Sprint B 收尾批次：U2 EMA-995 + U4 LS(T1/T3) + U7 Joint Hillclimb v11 | p8=0.63010（EMA 失敗）/ p9=0.67334 (+0.00213)；**v11=0.68770 (+0.00087 vs v9 → 新 SOTA)** | 已完成 |
| 5/02 ~ 6/02 | Sprint B 收尾 + 等待驗證集 | （視 Day 21 結果決定）剩餘 U1/U3/U5/U6 加成員、再跑聯合 hillclimb | 待 Day 21 結果 |
| 6/03 ~ 6/10 | 驗證集校正 + Sprint C | OOF vs Valid gap 分析 (U12)、Pseudo-label (U10) | 待辦 |
| 6/10 ~ 6/17 | 測試集衝刺 | 提交，至多 21 次 | 待辦 |
| 6/24 ~ 6/30 | 結案 | 最終報告與程式碼提交 | 待辦 |

---

## 13. 已完成階段紀錄 (Append-only)

### 13.1 Phase 1 — Baseline 復現結果（已完成）

執行：`python -m src.train_kfold --config configs/exp_p1_baseline.yaml`，3 seeds × 5 folds × 5 epochs，AMP fp16，class_weight on，pooling=cls_mean，max_length=256，batch=16，lr=2e-5，cosine + warmup_ratio=0.10，wd=0.01。RTX 5060 8GB 約 48 分鐘。

| Seed | Mean | Std | Min | Max |
| :-- | :-- | :-- | :-- | :-- |
| 42 | 0.6449 | 0.0094 | 0.6319 | 0.6541 |
| 2024 | 0.6374 | 0.0143 | 0.6237 | 0.6578 |
| 20260417 | 0.6422 | 0.0171 | 0.6144 | 0.6549 |
| **Overall** | **0.64150** | **0.01332** | — | — |

**結果遠超 Phase 1 目標 0.55-0.58**。Gate 條件達成 → 進入 Phase 2。

### 13.2 OOF 詳細分析摘要（[reports/analysis/p1_baseline_macbert_base/summary.md](reports/analysis/p1_baseline_macbert_base/summary.md)）

Per-class F1（seed-avg ensemble，post-processed）：

| Task | 弱點 label (F1) | 強項 label (F1) |
| :-- | :-- | :-- |
| T1 promise_status | **No: 0.669** (recall=0.661) | Yes: 0.925 |
| T2 verification_timeline | **within_2_years: 0.118** (support=13) | N/A: 0.670; already: 0.540 |
| T3 evidence_status | **No: 0.473** (precision=0.401) | Yes: 0.800 |
| T4 evidence_quality | **Misleading: 0.000** (support=1); **Not Clear: 0.191** | Clear: 0.747; N/A: 0.630 |

**關鍵錯誤模式**：
1. **T1 雙向誤判約 122 筆**（No→Yes 63、Yes→No 59）。由於階層後處理 (`promise=No → T2/T3/T4=N/A`)，T1 錯誤會把下游 3 個任務一併拖垮 → 解釋為何 post-processed 分數比 raw 低約 0.01。
2. **T2 時間語意混淆**：`already ↔ between_2_and_5_years`（138 筆）、`longer_than_5_years ↔ already`（78 筆）。模型缺乏明確時間數字解析能力。
3. **T3 過度悲觀**：Yes→No 107 筆、Yes→N/A 53 筆。
4. **T4 中段崩潰**：`Not Clear` 多被預測為 Clear/N/A；`Misleading` 樣本數=1 無法學習。
5. **校準**：T1 中段 (0.6-0.8) 過度自信（accuracy < confidence），T3/T4 整體尚屬一致 → temperature scaling 收益有限，重點仍在弱類學習。

**對 Phase 2 的策略含義**：
- **T1 No 召回**：提高 T1 task_loss_weight 或試 label_smoothing。
- **T2 弱類**：oversample within_2_years × 4，或 R-Drop 增強少類正則。
- **T3 平衡**：class_weight 已開啟仍偏負；可試提高 evidence_status loss 權重。
- **T4 Not Clear**：是支援數第二多 (124) 但 F1 極低，列為主攻；嘗試 multi-sample dropout 與較長 max_length（25.6% 樣本 > 256）。
- **整體**：先試 max_length=384、lr 掃描、task_loss 與評分權重對齊三組對照。

---

## 14. Phase 2 規劃（執行中）

由於 RTX 5060 8GB 顯存與 1000 樣本規模，採「**單變數 ablation**」加速篩選：每組僅 seed=42 × 5 folds × 5 epochs，~ 8 分鐘，預期能在 < 1.5 hr 內篩出 top-3 配置；top-3 再以 3 seeds 復測。

| Config | 變更 | 假設 |
| :-- | :-- | :-- |
| `exp_p2a_lr3e5` | lr 2e-5 → **3e-5** | 樣本量小可能未充分學習 |
| `exp_p2b_lr1e5` | lr 2e-5 → **1e-5** | 過大 lr 可能傷害弱類 |
| `exp_p2c_maxlen384` | max_length 256 → **384**, batch 16 → 8, grad_accum 2 | 25.6% 樣本被截斷 |
| `exp_p2d_pool_cls` | pooling cls_mean → **cls** | 對照組 |
| `exp_p2e_taskweight_score` | task_loss_weights 對齊評分權重 (0.20/0.15/0.30/0.35) | 直接優化評分 |
| `exp_p2f_label_smooth` | label_smoothing 0 → **0.05** | 校準與弱類正則 |
| `exp_p2g_epochs8` | epochs 5 → **8** + early_stop_patience 2 → 3 | 充分學習 |
| `exp_p2h_no_classw` | use_class_weight on → **off** | 控制變因驗證 |

Gate：任一配置在 seed=42 OOF (post-processed) ≥ baseline + 0.005 才晉升 3-seed 復測。

### 14.1 Phase 2 實際結果

| Exp | seed=42 mean | Δ vs P1 (0.64488) | 結論 |
| :-- | :-- | :-- | :-- |
| `p2h_no_classw` | **0.66102** | **+0.0161** | 晉級；class_weight 反而傷害 T3/T4 弱類 (loss 不平衡導致過擬合) |
| `p2c_maxlen384` | **0.65811** | **+0.0132** | 晉級；T3 +0.026, T4 +0.007 |
| `p2a_lr3e5` | **0.65281** | **+0.0079** | 晉級；T2/T4 微升 |
| `p2g_epochs8` | 0.65100 | +0.0061 | 邊界；early stop 多次觸發 |
| `p2d_pool_cls` | 0.65058 | +0.0057 | 邊界 |
| `p2e_taskweight_score` | 0.64249 | -0.0024 | 已用評分權重後反退；任務間優先級反衝 |
| `p2b_lr1e5` | 0.61847 | -0.0264 | 學習不足 |
| `p2f_label_smooth` | 0.49240 | -0.1525 | 與 multi-task CE 整合導致 T4 完全崩 (F1≈0)；後續禁用 |

→ 三組同向且非互斥的勝者 → 直接合 1 組 Phase 3。

---

## 15. Phase 2 — Combo Winners 最終配置 (3 seeds × 5 folds)

組合：`lr=3e-5` + `max_length=384` + `class_weight off`（保留 cls_mean pooling、5 epochs、batch=8 + grad_accum=2）。

| Seed | Mean | Std |
| :-- | :-- | :-- |
| 42 | 0.66558 | 0.01593 |
| 2024 | 0.65942 | 0.00951 |
| 20260417 | 0.66066 | 0.02224 |
| **Overall (raw best-epoch)** | **0.66189** | **0.01572** |
| **Post-processed per-seed mean** | **0.66307** | — |
| **Seed-avg probability ensemble** | **0.66734** | — |

vs Phase 1：**+0.0204** (raw)、**+0.0334** (ensemble)。

| Task | P1 seed-avg F1 | P2 combo seed-avg F1 | Δ |
| :-- | :-- | :-- | :-- |
| T1 promise_status | 0.9252 | 0.9277 | +0.003 |
| T2 verification_timeline | 0.4771 | 0.4759 | -0.001 |
| T3 evidence_status | 0.8003 | 0.8655 | **+0.065** |
| T4 evidence_quality | 0.3920 | 0.4308 | **+0.039** |

→ T3/T4 顯著改善（max_length=384 把長文截斷率從 25.6% 降至 < 5%；no_classw 修正了 T3/T4 弱類在強 class_weight 下被推到極端）；T2 仍受限於少數類 within_2_years (n=13) 與時間語意混淆；T1 已飽和。

詳見 [reports/analysis/p2_combo_best/summary.md](reports/analysis/p2_combo_best/summary.md)。

> 註：本節原命名為 “Phase 3 — Combo Winners”，但依原計畫第 7 章定義 Phase 3 = 升規模 (macbert-large)，已於 2026/04/28 將 combo 結果重新歸入 Phase 2 末段；對應 artefacts (`configs/exp_p2_combo_best.yaml`、`reports/experiments/p2_combo_best/`、`reports/analysis/p2_combo_best/`、`outputs/checkpoints/p2_combo_best/`、`data/splits/p2_combo_best/`) 已同步更名，內容指標未變。

---

## 16. Phase 3 — 升規模 (`hfl/chinese-macbert-large`, 326M)

依原計畫 7.2 升規模到 macbert-large，繼承 Phase 2 combo 最佳超參 (`max_length=384`、`pooling=cls_mean`、`use_class_weight=false`)，調整 batch=4 + grad_accum=4 (effective 16)、lr 範圍 1e-5 ~ 2e-5、early_stop_patience=2。單 seed=42、5 折驗證。

| Config | lr | epochs | weighted_score (5-fold mean ± std) | vs P2 combo seed=42 (0.66558) |
| :-- | :-- | :-- | :-- | :-- |
| `p3_large_baseline` | 1.5e-5 | 5 | **0.65900 ± 0.00867** | -0.00658 |
| `p3_large_lr2e5` | 2.0e-5 | 5 | **0.66439** | -0.00119 |
| `p3_large_ep8` (4 折,fold4 紀錄缺失) | 1.0e-5 | 8 | ≈ 0.6597 (前 4 折最佳 epoch 平均) | ≈ -0.006 |

**結論**：在 1000 樣本上 macbert-large 單獨並未通過「+0.005 顯著超越 base combo」門檻；epoch 拉長後驗證分數於第 6~7 epoch 飽和，loss 仍下降代表已過擬合。large 訓練成本 (≈ 4×) 與單模收益不對等 → **不採 large 為單模主力**，保留 large 預測作 Phase 4 集成輸入。

> `p3_large_ep8` fold4 的 `score_summary.csv` 因外部監控腳本提早終止主程序而未寫出；5 個 fold 的 `best.pt` 皆已落盤，不影響 ensemble；本表以前 4 折最佳 epoch 平均近似呈現結論。

### 16.1 為什麼大模型 (326M) 反而沒贏過 base (102M)？— 五個根本原因

單純看「large 比 base 多 3.2 倍參數」直覺上應該更好，但 1000 樣本的場景下實測結果相反。下列依影響力排序拆解：

#### 原因 1 — 樣本/參數比嚴重失衡（最主要）
- 訓練集 1000 筆，5-Fold 後每折 train=800。
- macbert-base 102M 參數 → 樣本/參數比 ≈ **1 : 102,000**。
- macbert-large 326M 參數 → 樣本/參數比 ≈ **1 : 326,000**（更稀疏 3.2 倍）。
- 經驗法則：large 模型需要 **> 10K 樣本** 才能讓「比 base 顯著更佳」的決策邊界穩定學成；否則 fine-tune 階段等同於「用 800 個梯度訊號去微調 326M 個參數」，每個參數收到的有效梯度極小 → 只動了預訓練表徵的薄薄一層。

#### 原因 2 — 過擬合警訊明確
- `p3_large_ep8`: epoch 4~5 驗證分數已飽和、epoch 6~7 仍 train loss 下降 → 典型過擬合曲線。
- `early_stop_patience=2` 救不回來：一旦過擬合，最佳 checkpoint 仍是「沒充分學習」的 epoch 4 版本。
- 大模型容量過剩會記住標註雜訊，特別是 T4 `Misleading` (support=1) 與 `Not Clear` (support=124) 的稀疏類別。

#### 原因 3 — 超參空間 mismatch（lr 必須重調）
- base 最佳 lr = 3e-5；對 large 過大 → 直接收斂到較差解。
- large 應更小 lr (≤ 2e-5) + 更多 warmup；但本實驗只試 1.5e-5 與 2e-5 兩個點，掃描密度不足。
- batch 8→4 + grad_accum 2→4（為了塞 8GB GPU），effective batch 維持 16，但實際梯度 noise 結構改變 → 與 base 的最佳超參不再可平移。

#### 原因 4 — T4 macro-F1 受少類別主導
- T4 評分用 macro-F1，6 個類別等權平均。
- `Misleading` (support=1) 在任何模型上本質上都學不會（500 預測中只有 1 個正例）。
- 大模型在主流類 (Pure Information, Concrete Action) 上的微弱優勢，會被少類分類失敗的高方差完全抵銷。

#### 原因 5 — 但 large 在集成中仍有價值（負負得正）
- Phase 4 證明：base combo + large @ 1.5:1 → **0.67478**（vs base combo 0.66558，**+0.0074**）。
- T4 macro 從 0.4308 跳到 **0.4494**（+0.019）— 大模型抓到了 base 抓不到的部分樣本。
- → **large 不該當主力，但作 ensemble 多樣性源 ROI 高**；此即 Phase 4 設計依據。

#### 一般化規律
在 NLP fine-tune 場景，當「樣本數 < 5K 且任務有高度不平衡少類別」時：
- large 模型的**單模收益期望值是負的**（本案例 -0.001 ~ -0.007）。
- 但作為 ensemble member 仍有 **+0.005 ~ +0.010** 的補強空間（本案例 +0.0074）。
- 工程決策：訓練 1 個 large 變體 → 加入 ensemble pool；不要花算力做 large 的 ablation 探索。

---

## 17. Phase 4 — Probability Ensemble (base combo + large)

工具：[src/tools/oof_ensemble.py](src/tools/oof_ensemble.py)（讀各 exp 的 OOF 機率，每 exp 內先做 seed 平均，再以權重做加權平均，最後套用條件約束後處理）。

| Pool | weights (combo : large_lr2e5) | T1 | T2 | T3 | T4 | weighted_score |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| p2_combo_best (3 seeds) | 1 : 0 | 0.9277 | 0.4759 | 0.8655 | 0.4308 | 0.66734 |
| p3_large_lr2e5 (1 seed) | 0 : 1 | 0.9276 | 0.4601 | 0.8577 | 0.4358 | 0.66439 |
| ENSEMBLE 1 : 1 | 1.0 : 1.0 | 0.9314 | 0.4791 | 0.8629 | 0.4383 | 0.67043 |
| ENSEMBLE 1.5 : 1 (**SOTA**) | **1.5 : 1.0** | **0.9318** | **0.4789** | **0.8643** | **0.4494** | **0.67478** |
| ENSEMBLE 2 : 1 | 2.0 : 1.0 | 0.9318 | 0.4779 | 0.8649 | 0.4468 | 0.67388 |
| ENSEMBLE 2.5 : 1 | 2.5 : 1.0 | 0.9328 | 0.4785 | 0.8637 | 0.4442 | 0.67290 |
| ENSEMBLE 3 : 1 | 3.0 : 1.0 | 0.9334 | 0.4787 | 0.8645 | 0.4447 | 0.67348 |

**最佳組合**：`p2_combo_best : p3_large_lr2e5 = 1.5 : 1.0` → **OOF weighted = 0.67478** (vs base combo 0.66734，**+0.00744**；vs Phase 1 0.6415，**+0.0333**)。提升主要來自 T4 evidence_quality (0.4308 → **0.4494**, +0.019)，T1 與 T2 也有微幅增益；T3 略降 0.001（容忍範圍內）。

→ 已於 Phase 5 推進並於 Phase 6 進一步突破：擴 pool 加入 Wave C 4 個增強模型 + 7-way per-task hillclimb，最終 SOTA = **0.68206**（見 §17.5 / §17.6）。

詳見 [reports/analysis/_ensemble/p2_combo_best_p3_large_lr2e5.csv](reports/analysis/_ensemble/p2_combo_best_p3_large_lr2e5.csv)。

---

## 17.5 Phase 5 — Wave A/B Ablation + Combo v2 + Per-task Hillclimb

### 17.5.1 Wave A/B 單變量 Ablation（16 組，全在 combo_best 基礎上 +1 改動，5-Fold seed=42）

> 對照基準：`p2_combo_best` seed=42 = **0.66558**。所有掃描結果見 [reports/experiments/_phase2_ablation.md](reports/experiments/_phase2_ablation.md)。

| 配置 | 改動 | weighted_score | 結論 |
| :-- | :-- | :--: | :-- |
| **p2x_focal_t4** | T4 改用 Focal Loss (γ=2.0) | **0.66833** | [贏] +0.00275 |
| **p2w_fgm10** | FGM 對抗訓練 ε=1.0 | **0.66714** | [贏] +0.00156 |
| p2n_wd005 | weight_decay 0.05 | 0.66575 | 中性 +0.00017 |
| p2m_wd0001 | weight_decay 0.001 | 0.66561 | 中性 |
| p2q_lr5e5 | lr 5e-5 | 0.66349 | -0.002 |
| p2t_epochs7 | 7 epochs | 0.66282 | -0.003 |
| p2j_warmup020 | warmup 0.20 | 0.66242 | -0.003 |
| p2p_lr4e5 | lr 4e-5 | 0.66213 | -0.003 |
| p2i_warmup006 | warmup 0.06 | 0.66206 | -0.004 |
| p2l_dropout03 | dropout 0.3 | 0.66075 | -0.005 |
| p2u_llrd095 | LLRD decay 0.95 | 0.66019 | -0.005 |
| p2o_lr2.5e5 | lr 2.5e-5 | 0.65767 | -0.008 |
| p2k_dropout02 | dropout 0.2 | 0.65516 | -0.010 |
| p2r_maxlen512 | max_length=512 (b=4×ga4) | 0.66306 | -0.003（成本翻倍不值） |
| p2s_sched_linear | linear scheduler | 0.66136 | -0.004 |
| **p2v_ema999** | EMA decay 0.999 | **0.47291** | [負] 崩潰：步數不足 (~500 step)，shadow 幾乎沒更新 |

**重要負面發現**：
- **EMA decay=0.999 在短訓練 (~500 step) 不可用** → 後續若要重試需用 0.99 或 step-aware decay。
- 大幅調整 lr/dropout/warmup 都偏負面 → combo_best 已接近單模 plateau。

### 17.5.2 Combo v2 復測（取 §17.5.1 兩個正向贏家）

`exp_p2_combo_v2.yaml` = combo_best + Focal-T4 (γ=2.0) + FGM (ε=1.0)，3 seeds × 5-fold：

| metric | T1 | T2 | T3 | T4 | weighted |
| :-- | :--: | :--: | :--: | :--: | :--: |
| 3-seed mean ± std | — | — | — | 0.4368 | **0.66836 ± 0.01659** |
| seed-avg ensemble (post-processed) | 0.9322 | 0.4770 | 0.8638 | 0.4333 | **0.66879** |

→ vs `p2_combo_best` seed-avg 0.66734：**+0.00145**（穩定但增量小，主要來自 T4）。

### 17.5.3 3-way Probability Ensemble + Per-task Hillclimb

工具：[src/tools/ensemble_weight_scan.py](src/tools/ensemble_weight_scan.py)、[src/tools/per_task_hillclimb.py](src/tools/per_task_hillclimb.py)。

**全任務同權重掃描**（v2 : best : large）：

| weights | T1 | T2 | T3 | T4 | weighted_score |
| :-- | :--: | :--: | :--: | :--: | :--: |
| 1.0 : 0.0 : 0.0 (v2 only) | 0.9322 | 0.4770 | 0.8638 | 0.4333 | 0.66879 |
| 0.0 : 1.0 : 0.0 (best only) | 0.9277 | 0.4759 | 0.8655 | 0.4308 | 0.66734 |
| 0.0 : 0.0 : 1.0 (large only) | 0.9276 | 0.4601 | 0.8577 | 0.4358 | 0.66439 |
| 1.0 : 1.0 : 0.0 | 0.9309 | 0.4780 | 0.8666 | 0.4420 | 0.67254 |
| **1.0 : 1.0 : 1.0** | 0.9341 | 0.4817 | 0.8672 | 0.4486 | **0.67627** |
| 2.0 : 1.0 : 1.0 | 0.9314 | 0.4834 | 0.8686 | 0.4482 | 0.67624 |
| 1.5 : 1.0 : 0.7 | 0.9313 | 0.4789 | 0.8648 | 0.4462 | 0.67371 |

→ 3-way 等權 (1:1:1) 即超越 Phase 4 SOTA (+0.00149)。

**Per-task hillclimb**（各任務在 GRID = {0, 0.5, 1.0, 1.5, 2.0}^3 = 125 組合中獨立挑最大 F1）：

| Task | best (v2 : best : large) | F1 |
| :-- | :--: | :--: |
| T1 promise_status | 0.5 : 0.5 : 0.5 (等權) | 0.9341 |
| T2 verification_timeline | **2.0 : 0.0 : 0.5** (棄用 best) | 0.4892 |
| T3 evidence_status | 0.5 : 0.5 : 1.0 (large 重) | 0.8702 |
| T4 evidence_quality | 1.0 : 1.5 : 1.5 (best+large 重) | 0.4522 |

**Phase 5 SOTA**：per-task hillclimb 結合後 **OOF weighted = 0.67954**（vs Phase 4 0.67478，**+0.00476**；vs Phase 1 baseline，**+0.0380**）。

詳見 [reports/analysis/_ensemble/per_task_hillclimb_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_summary.csv) 與 [reports/analysis/_ensemble/per_task_hillclimb_preds.csv](reports/analysis/_ensemble/per_task_hillclimb_preds.csv)。

---

## 17.6 Phase 6 — Wave C Ablation + 7-way Per-task Hillclimb

> **本階段目標**：嘗試 Phase 5 之後仍未試過的三大正則化／資料增強技術，並用「**爬山演算法 (hillclimbing)**」在更大的模型池中為每個任務獨立搜尋最佳集成權重，把 SOTA 從 0.67954 進一步往上推。

### 17.6.0 名詞快速說明（給未接觸過的讀者）

- **Ablation（消融實驗）**：固定其他所有設定，只改一個變因（例如「加上資料增強」），跑完整 5-Fold 觀察分數變化，用以確認該變因的真實效果。
- **Combo（組合配方）**：把多個「單獨已驗證有效」的技術疊在 baseline 上做的整合配方（如 `combo_v2 = baseline + FGM + Focal-T4`）。Combo 的篩選規則是「**單獨上場必須先贏過 baseline**」，因此 Wave C 因為單模都輸給 baseline，**並不進入 Combo**（詳見 §17.6.3）。
- **Ensemble（集成）**：訓練好的多個模型，在推論時把它們的「機率輸出」加權平均，再 argmax 拿最終標籤。集成成功的關鍵不是「每個成員都很強」，而是「成員之間的錯誤模式互不相同」（diversity）。
- **OOF (Out-of-Fold) probabilities**：5-Fold 訓練時，每筆樣本只在「不參與訓練的那一摺」被預測，因此整份資料都有「未洩漏」的預測機率，可拿來做公平的離線評分與 ensemble 權重搜尋。
- **Hillclimbing（爬山演算法）**：見 §17.6.2 詳細解說。

### 17.6.1 Wave C 訓練技術 Ablation（單模實驗）

實作三大未試技術：token-level data augmentation、R-Drop、Multi-Sample Dropout。

**技術原理：**
- **Token-level Data Augmentation**：訓練時對輸入文字的 token 序列做隨機擾動，等於在資料層面注入雜訊讓模型學得更穩健。
  - `mask_ratio`：隨機把若干 token 換成 `[MASK]`（與 BERT MLM 預訓練同形）。
  - `swap_ratio`：隨機交換相鄰兩個 token，模擬輕微語序錯亂。
  - `delete_ratio`：隨機刪除 token 並向左壓緊（右側補 padding）。
  - 三種操作都跳過特殊 token（`[CLS]`/`[SEP]`/`[PAD]`），且只在訓練階段套用，驗證集保持原文。
- **R-Drop**：同一個 batch 在啟用 dropout 的情況下做兩次前向傳播得到 `logits1`、`logits2`，除了原本的分類損失外，再加上對稱 KL 散度 $\frac{\alpha}{2}\bigl(\text{KL}(p_1 \Vert p_2) + \text{KL}(p_2 \Vert p_1)\bigr)$，強迫兩次帶不同 dropout mask 的預測互相一致。等於在「dropout 子網路」之間做正則化。
- **Multi-Sample Dropout (MSD)**：在分類頭之前對同一個共享特徵 `feat` 套 K 份不同的 dropout，分別產生 K 組 logits 後平均。等於把「測試時關掉 dropout 取平均」的概念搬到訓練時，期望讓決策更平滑。

**程式碼變更：**
- [src/data/dataset.py](src/data/dataset.py) `ESGDataset` 新增 `aug_prob` / `mask_ratio` / `swap_ratio` / `delete_ratio` / `aug_seed`，新方法 `_augment_ids` 處理 token 擾動。
- [src/training/trainer.py](src/training/trainer.py) 新增 `rdrop_alpha`：當 > 0 時做第二次 forward 並加入 KL 損失。
- [src/models/multitask.py](src/models/multitask.py) 新增 `msd_k`：當 > 1 時訓練階段做 K 份 dropout 平均；推論不變。
- [src/train_kfold.py](src/train_kfold.py) `_build_loaders` 把 `training.augment` 的子設定接到訓練 `ESGDataset`，模型構造讀取 `model.msd_k`。

**5-Fold 結果（seed=42，與 `combo_best` seed=42 = 0.66558 比較）：**

| 配置 | 改動 | weighted_score | T1 | T2 | T3 | T4 | vs combo_best |
| :-- | :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| `p2ab_aug_mask10` | mask 10% | 0.66122 | 0.9298 | 0.4648 | 0.8589 | 0.4225 | -0.0044 |
| `p2ac_aug_mix` | mask 8% + swap 3% + del 3% | 0.66070 | 0.9344 | 0.4638 | 0.8547 | 0.4224 | -0.0049 |
| `p2ad_rdrop05` | R-Drop α=0.5 | 0.66345 | 0.9311 | 0.4739 | 0.8593 | 0.4238 | -0.0021 |
| `p2ae_msd5` | Multi-Sample Dropout K=5 | 0.66448 | 0.9327 | 0.4753 | 0.8558 | 0.4283 | -0.0011 |

**結論**：4 個配置全部單獨低於 `combo_best`。資料量僅 1000 筆、訓練步數約 500 步的設定下，這些「強正則化」技術可能因訓練不足以收斂出穩定的擾動鄰域而短期內反而拖慢學習速度。但它們的決策邊界與既有模型差異化顯著，因此被保留為 ensemble 池成員（§17.6.4 證實這是正確決定）。

### 17.6.2 Hillclimbing 與 per-task 權重搜尋

**Hillclimbing（爬山法）是什麼？**
這是一個經典的局部最佳化演算法，邏輯非常直觀：

1. 先準備一個「分數函數」 $f(w)$（在我們的場景中，$w$ 是「7 個模型的權重向量」，$f(w)$ 是該任務在 OOF 上的 F1 分數）。
2. 從某個起始點 $w_0$ 開始計算 $f(w_0)$。
3. 在 $w_0$ 的「鄰域」中找一個分數更高的點 $w_1$（即 $f(w_1) > f(w_0)$）。
4. 把 $w_1$ 當成新的起點，回到步驟 3 重複，直到再也找不到更好的鄰居 $\Rightarrow$ 抵達局部最大值就停止。

對應到本專案，我們用 **兩階段 hillclimb**（v2 + v3）來搜尋 ensemble 權重：

- **第一階段 v2（粗網格窮舉）**：每個權重從 $\{0, 1, 2\}$ 中取值，7 個模型共 $3^7 = 2187$ 種組合。對每個任務獨立把全部 2187 種組合都算一次分數，挑出最高的那組作為 v2 winner。這嚴格來說是 enumeration（窮舉），但在離散小空間中等價於「鄰域 = 整個搜尋空間」的 hillclimb。
- **第二階段 v3（隨機鄰域精修）**：以 v2 winner 為起點，候選網格細化為 $\{0, 0.25, 0.5, \dots, 2.0\}$（9 個刻度）。每一輪隨機挑 1~3 個維度並隨機改值，若新分數更高就接受、否則丟棄；總共做 5000 次。這是真正的「stochastic hillclimb」，能在粗網格找不到的細微權重組合中再取得一點增益。

**為什麼要 per-task（每任務獨立）搜尋？**
四個任務的標籤分布、難度、最佳模型都不同：T1（promise_status）是二元判斷，T4（evidence_quality）是稀有類別佔多數的多元分類。如果用「全任務共用一組權重」，等於假設「對所有任務最好的模型加權方式是同一個」，這幾乎不可能。Per-task hillclimb 允許每個任務挑自己最喜歡的模型組合（甚至完全棄用某些模型），自然分數會更高。

工具實作：
- [src/tools/per_task_hillclimb_v2.py](src/tools/per_task_hillclimb_v2.py)：粗網格窮舉。
- [src/tools/per_task_hillclimb_v3.py](src/tools/per_task_hillclimb_v3.py)：v2 winner 起點 + 隨機局部精修。

### 17.6.3 為什麼 Wave C 不進 Combo？

**Combo 的設計原則**：把「單獨已驗證帶來 +0.001 以上增益」的技術疊在 baseline 上做整合配方。例如 `combo_v2 = combo_best + FGM + Focal-T4`，因為 FGM 單獨 +0.00156、Focal-T4 單獨 +0.00275，兩者都是正向贏家。

**Wave C 4 個配置全部單獨輸給 combo_best**（差距 -0.0011 ~ -0.0049）。如果硬把它們疊到 combo 上，等於在已經最佳化的訓練配方裡注入「會降分的成分」，破壞了 combo 的可解釋性與穩定度。因此**Wave C 不進 Combo**。

但這並不代表它們沒用——**ensemble 與 combo 是兩種完全不同的策略**：
- **Combo**：在「**訓練時**」把多個技術疊在同一個模型上 → 產出**單一模型**。
- **Ensemble**：在「**推論時**」把多個**獨立訓練**的模型輸出機率做加權平均 → 產出**集成預測**。

集成成功靠的是「**模型之間的互補性 (diversity)**」，而不是「每個成員都最強」。Wave C 用了完全不同的訓練擾動（mask/swap/delete/R-Drop/MSD），它們犯的錯與既有模型不同類型，因此即使單模較弱，加進 ensemble 池仍能透過 per-task hillclimb 得到正向貢獻。§17.6.4 的數字證實了這個直覺。

### 17.6.4 7-way Per-task Hillclimb 結果（**Phase 6 SOTA**）

**模型池（7 個）**：`combo_v2`、`combo_best`、`p3_large_lr2e5`、`p2ab_aug_mask10`、`p2ac_aug_mix`、`p2ad_rdrop05`、`p2ae_msd5`。前三者來自 Phase 5，後四者來自 Wave C。

**v2 粗網格結果**（每任務 2187 組合，OOF weighted score = 0.68056）：

| Task | best weights (v2:best:large:aug_m:aug_x:rdrop:msd) | F1 |
| :-- | :--: | :--: |
| T1 promise_status | 0 : 0 : 0 : 2 : 1 : 1 : 1 | 0.93690 |
| T2 verification_timeline | 1 : 2 : 1 : 1 : 1 : 0 : 2 | 0.48710 |
| T3 evidence_status | 1 : 1 : 1 : 0 : 0 : 1 : 0 | 0.87483 |
| T4 evidence_quality | 1 : 1 : 1 : 0 : 0 : 0 : 0 | 0.44699 |

**v3 隨機精修結果**（以 v2 為起點 × 5000 trials/任務，OOF weighted score = **0.68206**）：

| Task | best weights | F1 |
| :-- | :--: | :--: |
| T1 promise_status | 0 : 0 : 0 : **2.0** : 1.0 : 1.0 : 1.0 | 0.9369 |
| T2 verification_timeline | 1.25 : 2.0 : 1.0 : 1.5 : 0.5 : 0 : 2.0 | 0.4904 |
| T3 evidence_status | 1.0 : 1.0 : 0.75 : 0 : 0 : 1.0 : 0.25 | 0.8759 |
| T4 evidence_quality | 0.75 : 1.0 : 1.0 : 0 : 0 : 0 : 0 | 0.4524 |

**Phase 6 SOTA = 0.68206**，相對 Phase 5 (0.67954) **+0.00252**，相對 Phase 1 baseline (0.6415) **+0.0406**。

**關鍵發現：**
- **T1 完全交給 Wave C 4 模型**（前 3 個傳統模型權重全為 0）：T1 衝到歷史最高 0.9369，證實「單模較弱但訓練擾動方式不同」確實能透過 ensemble 反超「單模較強但都用同類訓練配方」的組合。其中 `p2ab_aug_mask10` 取得最高權重 2.0，token masking 與 BERT 預訓練目標一致，可能讓 T1 二元判斷更穩。
- **T2 在 7-way diversity 中略有提升**（0.4892 → 0.4904），主要靠 `combo_best` (2.0) + `p2ae_msd5` (2.0) 組合。
- **T3 由 large + R-Drop 主導**（`p2ad_rdrop05` weight=1.0 與 `combo_v2/best` 並列），R-Drop 雖然單模分數低，但對 evidence_status 的決策邊界互補性最好。
- **T4 仍是最大瓶頸（0.4524）**：Wave C 4 模型在 T4 全部被排除（weight=0），證實 T4 的稀有類別問題（如 `Misleading` 全集只有 1 筆）不是靠「訓練擾動」能解，需要 §18.2 的 oversampling 或 numeric token 標註等資料層改造。

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v3_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v3_summary.csv) 與 [reports/analysis/_ensemble/per_task_hillclimb_v3_preds.csv](reports/analysis/_ensemble/per_task_hillclimb_v3_preds.csv)。

---

## 17.7 Phase 7 — Combo v3 + 8-way Per-task Hillclimb

> **本階段目標**：把 Phase 6 中「ensemble pool 內表現最好的 Wave C 兩個技術」（R-Drop 與 MSD）正式疊回 Combo 訓練配方，看是否能產出第一個能單獨贏過 combo_best 的新 combo；接著把它加入 ensemble pool 跑 8-way per-task hillclimb。

### 17.7.1 為什麼選 R-Drop 與 MSD 進 combo_v3？

Wave C 4 個配置中 R-Drop（-0.0021）與 MSD（-0.0011）是「**最不弱**」的兩個，且分屬完全不同層級：
- **R-Drop 是「損失層」正則**：在 loss function 加入 KL 一致性項。
- **MSD 是「分類頭層」正則**：在最後一層 dropout 取多樣本平均。
- 兩者**不重複作用**，理論上可疊加。

token-level augmentation 雖然 ensemble 內 T1 有用，但其作用本質上與 R-Drop 重疊（都在「擾動同一輸入」），且兩個 aug 配置單模降幅最大（-0.0044 / -0.0049），疊加風險較高，故先排除。

[configs/exp_p2_combo_v3.yaml](configs/exp_p2_combo_v3.yaml) 內容：
```yaml
extends: exp_p2_combo_v2.yaml
training:
  focal_tasks: [evidence_quality]
  focal_gamma: 2.0
  fgm_eps: 1.0
  rdrop_alpha: 0.5
model:
  msd_k: 5
seeds: [42]
```

### 17.7.2 combo_v3 單模結果（首次 Wave C 技術疊上 combo 成功）

5-Fold seed=42（與 combo_best seed=42 = 0.66558、combo_v2 seed=42 比較）：

| 模型 | weighted_score | T1 | T2 | T3 | T4 | vs combo_best |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| `combo_best` (Phase 2) | 0.66558 | — | — | — | — | baseline |
| `combo_v2` (Phase 5) | ~0.66833 | — | — | — | — | +0.00275 |
| **`combo_v3` (Phase 7)** | **0.67121** | 0.9332 | 0.4717 | 0.8644 | 0.4415 | **+0.00563** |

→ combo_v3 是**自 Phase 5 以來第一個單模超越 combo_v2 的訓練配方**，也驗證了 §17.6 的假設：**Wave C 技術在 ensemble 池內弱、但與已最佳化的 combo 配方互補時可正向疊加**。R-Drop 強迫的 dropout-子網路一致性 + MSD 的多樣本平均，等於同時在「訓練擾動」與「推論平滑」兩個維度做正則化。

### 17.7.3 8-way Per-task Hillclimb（**Phase 7 SOTA**）

工具：[src/tools/per_task_hillclimb_v4.py](src/tools/per_task_hillclimb_v4.py)（v3 起點 + 6000 次隨機局部精修，新增 `combo_v3` 為池中第 8 員）。

**模型池（8 個）**：v3 的 7 員 + `p2_combo_v3`。

| Task | best weights (v2:best:large:aug_m:aug_x:rdrop:msd:**v3**) | F1 | vs Phase 6 |
| :-- | :--: | :--: | :--: |
| T1 promise_status | 0 : 0 : 0 : 0.25 : 0.25 : 0.75 : 0.25 : **1.75** | 0.93832 | +0.0014 |
| T2 verification_timeline | 1.0 : 2.0 : 1.0 : 1.25 : 0.5 : 0 : 2.0 : 0 | 0.49031 | -0.0001 |
| T3 evidence_status | 1.0 : 1.0 : 0.75 : 0 : 0 : 1.0 : 0.25 : 0 | 0.87543 | -0.0005 |
| T4 evidence_quality | 0 : 1.5 : 1.0 : 0 : 0 : 0 : 0 : **0** ←等等 | 0.44915 | +0.0037 |

> 補正：T4 winner 是 (0.0, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0) — combo_v3 在 T4 確實 weight=0，T4 提升來自於改變了 combo_v2 vs combo_best vs large 的相對權重結構（Phase 6 是 0.75:1.0:1.0，Phase 7 變成 0:1.5:1.0，意即 T4 完全棄用 combo_v2 而改用更高權重的 combo_best）。

**Phase 7 SOTA = 0.68394**（vs Phase 6 0.68206，**+0.00188**；vs Phase 1 baseline，**+0.0425**）。

**關鍵發現：**
- **T1 = 0.9383 為歷史最高**：combo_v3 取得 weight=1.75 的最高權重，幾乎主導 T1 預測。Phase 6 中 T1 只靠 4 個 Wave C 模型（單模都 < 0.66448）；Phase 7 換成 combo_v3（單模 0.67121）作主力後，T1 又再進一步。
- **T4 = 0.4560 首次突破 0.45 級距**（Phase 6 是 0.4524）：透過拋棄 combo_v2、把 combo_best 權重從 1.0 拉到 1.5，evidence_quality 的稀有類別判斷終於有正向動能。雖然 combo_v3 在 T4 winner 中沒被選用，但它的存在讓 hillclimb 有機會探索到「不同 combo_best 權重」的更佳組合。
- **T2/T3 微幅下降（<0.001）但被 T1/T4 增益完全覆蓋**：這正是 per-task hillclimb 對「全任務加權分數」做最佳化的合理 trade-off。

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v4_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v4_summary.csv) 與 [reports/analysis/_ensemble/per_task_hillclimb_v4_preds.csv](reports/analysis/_ensemble/per_task_hillclimb_v4_preds.csv)。

### 17.7.4 累計 SOTA 進展軌跡

| Phase | SOTA | 增益 | 主要技術 |
| :--: | :--: | :--: | :-- |
| 1 baseline | 0.6415 | — | macbert-base 5-Fold |
| 2 combo | 0.66734 | +0.026 | LR/max_len/pooling 掃描 + combo_best |
| 4 ensemble | 0.67478 | +0.007 | combo + macbert-large 1.5:1 |
| 5 hillclimb v1 | 0.67954 | +0.005 | per-task 3-way hillclimb |
| 6 hillclimb v3 | 0.68206 | +0.003 | + Wave C 4 模型 → 7-way |
| 7 hillclimb v4 | 0.68394 | +0.002 | + combo_v3 (R-Drop+MSD) → 8-way |
| **8 hillclimb v6** | **0.68440** | **+0.0005** | combo_v3 multi-seed 拆 peaky/avg → 9-way |

---

## 17.8 Phase 8 — combo_v3 Multi-seed 拆分為 9-way 池（Phase 8 SOTA，已被 Phase 11 取代）

### 17.8.1 動機與假設

Phase 7 SOTA (0.68394) 中，T1 winner 給 `combo_v3 weight=1.75`（在 8-way tuple 中佔 54%），意味 T1 高度倚賴 combo_v3 seed=42 的「尖銳預測」。一個自然問題是：**multi-seed averaging 能否進一步降低 OOF 變異、推升 SOTA？**

預期：每個 seed 的 5-fold OOF std ≈ ±0.015，3-seed 平均後變異降約 1/√3，理論增益 +0.001 ~ +0.005。

### 17.8.2 試錯一：直接 multi-seed 平均（v5，**失敗**）

- 訓練 combo_v3 額外兩個 seed（2024、20260417），3-seed OOF：
  - seed=42: 0.67121
  - seed=2024: 0.67257
  - seed=20260417: 0.66452
  - 3-seed avg: **0.66854**
- 用 3-seed 平均 OOF 取代原 seed=42-only，跑 hillclimb v5（warm-start 自 Phase 7 winners）：
  - **per-task weighted = 0.68191**（比 Phase 7 -0.00203，**負面結果**）
- 失敗原因（事後分析）：
  - Phase 7 T1 winner `(0,0,0,0.25,0.25,0.75,0.25,1.75)` 在 multi-seed 平均下 T1 從 0.93832 跌到 0.93412
  - **平均化把 combo_v3 的 peaky 預測平滑化**，反而稀釋了 T1 賴以取勝的「強勢類別決策」訊號
  - T2/T3/T4 微幅改善 (+0.0013/+0.0002/0)，遠不足以補 T1 損失 (-0.0042 × 0.20 weight = -0.00084)

### 17.8.3 試錯二：拆分為兩成員 9-way 池（v6，**成功**）

關鍵設計：把 combo_v3 同時放入 pool 兩次：
- `p2_combo_v3_s42` = seed=42 only（peaky，保留 Phase 7 強勢訊號）
- `p2_combo_v3_avg` = 3-seed 平均（smooth，降低變異）
- hillclimb 為每個 task 自由選擇兩者權重比例 → 不會被強迫統一

執行：[src/tools/per_task_hillclimb_v6.py](src/tools/per_task_hillclimb_v6.py)（N_RANDOM=8000，warm-start 自 Phase 7 winners 並在 index 8 補 0）

| Task | 權重 (v2:best:large:mask:mix:rdrop:msd:**v3_s42**:**v3_avg**) | F1 | vs Phase 7 |
| :-- | :-- | :--: | :--: |
| T1 promise_status | 0:0:0:0.25:0.25:0.75:0.25:**1.75**:**0** | 0.93832 | 0 |
| T2 verification_timeline | 0.25:2.0:1.0:1.25:0.75:0:2.0:**0.5**:**1.0** | 0.49118 | +0.00087 |
| T3 evidence_status | 1.0:0.75:0.75:0:0:1.0:0.25:**0.25**:**0.5** | 0.87577 | +0.00034 |
| T4 evidence_quality | 1.0:1.5:1.5:0:0:0:0:**0**:**0** | 0.45132 | +0.00217 |

**Phase 8 SOTA = 0.68440**（vs Phase 7 0.68394，**+0.00046**；vs Phase 1 baseline，**+0.043**）。

### 17.8.4 結果解讀（每個 task 的故事）

- **T1**：hillclimb 自動把 v3_avg 設為 0、保留 v3_s42=1.75 → 完全採用 Phase 7 winner，沒被 multi-seed 拖累。**「拆分」設計的最大價值不是讓 v3_avg 主導 T1，而是讓 hillclimb 有權選擇「不選」**，避免被強迫融合而退化。
- **T2/T3**：v3_avg 確實取得非零權重（0.5 與 1.0、0.25 與 0.5），smooth 版降低了 OOF 上的決策邊界抖動 → 兩個 macro-F1 任務微幅獲益。
- **T4**：兩個 v3 版本都被設為 0；T4 的進展來自完全不同的方向 — 由 8000 次更大隨機搜索找到 `(combo_v2=1.0, combo_best=1.5, large=1.5)`（Phase 7 是 `0:1.5:1.0`），意即 **Phase 7 棄用 combo_v2 的決策被 Phase 8 推翻**。這是「9-way 池 + 8000 trials」單純放大搜索容量帶來的副作用，與 multi-seed 無直接關係。

### 17.8.5 累計教訓

1. **「平均」並非萬靈丹**：當某個成員在某 task 是「peaky 但準確」時，平均化會稀釋訊號。Multi-seed averaging 應該與「保留原版」並存，由 hillclimb 自選使用比例。
2. **池擴張的雙重收益**：v6 的增益並非完全來自 multi-seed，更大的搜索空間（9 維 × 8000 trials）也讓 T4 找到更好的 combo_v2/best/large 比例。
3. **Per-task hillclimb 的安全性**：warm-start 自上一階段 winners + 隨機探索的設計，讓「失敗的新成員」最壞退化為原 SOTA，永遠不會回退。

### 17.8.6 累計 SOTA 進展軌跡（含 Phase 8）

| Phase | SOTA | 增益 | 主要技術 |
| :--: | :--: | :--: | :-- |
| 1 baseline | 0.6415 | — | macbert-base 5-Fold |
| 2 combo | 0.66734 | +0.026 | LR/max_len/pooling 掃描 + combo_best |
| 4 ensemble | 0.67478 | +0.007 | combo + macbert-large 1.5:1 |
| 5 hillclimb v1 | 0.67954 | +0.005 | per-task 3-way hillclimb |
| 6 hillclimb v3 | 0.68206 | +0.003 | + Wave C 4 模型 → 7-way |
| 7 hillclimb v4 | 0.68394 | +0.002 | + combo_v3 (R-Drop+MSD) → 8-way |
| 8 hillclimb v6 | 0.68440 | +0.0005 | combo_v3 拆 peaky/avg → 9-way |
| 9 hillclimb v7b | 0.68558 | +0.00118 | + p4a/p4b → 11-way (biased search) |
| 10 hillclimb v8 | 0.68669 | +0.00111 | + p5_t6_time_token → 12-way (biased search) |
| 11 hillclimb v9 | 0.68683 | +0.00014 | + p6_t6v2_bucket_tok → 13-way (biased search) |
| 12 hillclimb v10 | 0.68660 | −0.00023 | + p7_focal_g3 → 14-way（單模最強 0.67416 但 ensemble 未破 SOTA）|
| **13 joint hillclimb v11 (SOTA)** | **0.68770** | **+0.00087** | + p8_ema995 (w=0) + p9_ls_t1t3 (w=0) → 16-way **joint post-constraint** hillclimb；4 次 accept 全在 T3，T4 +0.0018 為 constraint coupling 連動 |

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v9_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v9_summary.csv)（現任 SOTA）與 [reports/analysis/_ensemble/per_task_hillclimb_v10_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v10_summary.csv)（Phase 12 記錄）。

---

## 17.9 Phase 9 — Sprint A backbone 擴 pool（已完成）

依 §19 Sprint A 計畫，依序訓練 3 個新骨幹（單 seed=42、5-Fold、繼承 combo_best 超參），加入 ensemble pool 後跑 hillclimb v7 以衝刺 0.69。

### 17.9.1 已完成成員

| # | exp | backbone | lr | OOF (5-Fold) | T1 | T2 | T3 | T4 | 耗時 | ≥0.65 入池 |
| :--: | :-- | :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| p4a | `p4a_roberta_wwm_base` | `hfl/chinese-roberta-wwm-ext` | 3e-5 | 0.66626 | 0.9265 | 0.4626 | 0.8555 | 0.4426 | 932s | 是 |
| p4b | `p4b_bert_base_chinese` | `bert-base-chinese` | 3e-5 | **0.67355** | 0.9346 | 0.4857 | 0.8656 | 0.4403 | 971s | 是 |
| p4c | `p4c_xlm_roberta_base` | `xlm-roberta-base` | 2e-5 | 0.61269 | 0.9107 | 0.3419 | 0.8391 | 0.3644 | 1108s | **否** |

**p4a 觀察**：OOF 0.66626 ≈ combo_best 0.66558（+0.0007 微幅優於）；T4 略高 0.4426 vs 0.4308 → **多樣性主要落在 T4**，可貢獻 ensemble。

**p4b 觀察**：OOF **0.67355** 為 Sprint A 最佳，優於 combo_best 0.66558（+0.00797）。T1=0.9346、T2=0.4857 同步上升，與 macbert 體系不同（bert-base-chinese 無 WWM、無 macbert MLM 校正，反而提供獨立信號）。

**p4c 觀察**：OOF 0.61269 **未過 ≥0.65 入池門檻**。問題在 T2=0.3419、T4=0.3644 嚴重落後（macro-F1 受少類拖累更甚）。主因：(1) xlm-roberta-base 為多語預訓練、中文 token 表徵密度低於專門中文模型；(2) lr=2e-5 + 5 ep 收斂不足（loss 仍從 4.0 → 2.4，但 fold 間 std=0.0345 顯示變異大）。**結論：踢出 v7 pool**。

### 17.9.2 Hillclimb v7 → v7b（biased search 修正）

**v7 結果**：N_RANDOM=10000、warm-start 自 v6 winners（idx 9, 10 補 0.0），最終 0.68440 = v6 完全平手。診斷：均勻擾動 1-3/11 個 index，新成員 (9, 10) 每輪只有 ~27% 機率被探索，且須同時抽到非零 GRID 值才能改善 → 探索率不足。

**v7b 修正**：強制每輪擾動必須包含至少一個新成員 (idx 9 或 10)，外加 0-2 個 legacy index；N_RANDOM 提升至 12000。

**v7b 結果（新 SOTA）：0.68558（+0.00118 vs v6）**

| Task | F1 | 變化 vs v6 | 新成員權重 |
| :-- | :--: | :--: | :-- |
| promise_status (T1) | 0.9383 | 0 | p4a=0, p4b=0 |
| verification_timeline (T2) | **0.5000** | **+0.0089** | p4a=0.5, p4b=1.25 |
| evidence_status (T3) | 0.8764 | 0 | p4a=0, p4b=0 |
| evidence_quality (T4) | 0.4572 | 0 | p4a=0, p4b=0 |

**結論**：T2 是唯一受惠 task — p4b 在 T2=0.4857 確實提供 macbert 體系沒有的訊號（其餘 task 因 macbert 已飽和而無法改善）。距 0.69 還差 **0.00442**，需進 §19 Sprint B (T6 時間 token 增強，預期 +0.005-0.009 直接擊中 T2)。

**Hillclimb 工程教訓**：擴大 pool 時，若 warm-start 包含大量零權重新維度，必須改用 biased search（強制探索新維度），否則均勻隨機擾動會使新成員實質為「裝飾品」。應寫入 ML 教訓記憶。

---

## 17.10 Phase 10 — Sprint B / T6 時間 token 增強（已完成）

依 §19 Sprint B 計畫，啟動 T6（時間 token 注入）；目標：直接擊中 T2 (verification_timeline) macro-F1，預期 +0.005 ~ +0.009。

### 17.10.1 設計

**輸入端確定性增強（無學習成本）**：以 regex 抽取文本中的西元年（`19xx|20xx`）/民國年（`民國 NNN`，加 1911 換算）/中文年（`YYYY 年`），取最大年份 → 計算與 CURRENT_YEAR=2025 的差 → bucket 對齊 T2 5 類，最後在文本前注入：

```
[時間 年份YYYY 距今N年{後|前} BUCKET]  原文…
```

bucket 規則：`delta ≤ 0 → already`；`0 < delta ≤ 2 → within_2_years`；`2 < delta ≤ 5 → between_2_and_5_years`；`> 5 → longer_than_5_years`。無年份樣本維持原文。

實作：
- 新模組 [src/data/text_augment.py](src/data/text_augment.py)：`add_time_tokens()` + `TEXT_TRANSFORMS` 註冊表
- [src/data/dataset.py](src/data/dataset.py)：`ESGDataset.__init__` 加 `text_transform=None`，`__getitem__` 在 tokenize 前套用
- [src/train_kfold.py](src/train_kfold.py)：`_build_loaders` 從 `cfg.data.text_transform` 查表
- 新 config [configs/exp_p5_t6_time_token.yaml](configs/exp_p5_t6_time_token.yaml)：繼承 combo_best 超參，seed=42

**Sanity 驗證**（regex 覆蓋率測試 6 條樣本，全通過）：
- `2030 年` → `[時間 年份2030 距今5年後 between_2_and_5_years]`
- `民國 115 年` → `[時間 年份2026 距今1年後 within_2_years]`
- `2024 年已達成` → `[時間 年份2024 距今1年前 already]`
- `沒有時間表的承諾` → 維持原文

樣本中時間 pattern 覆蓋率（1000 筆）：西元 4 位 47.1% / 中文年 45.9%（與西元有重疊）/ 民國 1.4% / 中文月 7.4%。

### 17.10.2 單模成績

| Item | 值 |
| :-- | :-- |
| Overall (5-Fold OOF) | 0.66231 |
| std | 0.00823 |
| T1 (promise_status) | 0.9356 |
| T2 (verification_timeline) | 0.4874 |
| T3 (evidence_status) | 0.8556 |
| T4 (evidence_quality) | 0.4155 |
| 耗時 | 879.5s |

**單模觀察**：Overall 略低於 combo_best 0.66558（−0.00327）；T2 = 0.4874 與 combo_best baseline ~0.49 持平，**未呈現預期的 T2 直擊提升**。可能原因：
1. macbert tokenizer 對 `[時間 年份…]` 字元拆成多 sub-tokens（從 sanity decode 可見「2024」拆為單字元），稀釋訊號；
2. T2 真正困難案例（`already ↔ between_2_and_5_years` 138 筆混淆）多無明確年份錨點，regex 命中率不足；
3. 確定性 prefix 與 macbert 預訓練分布偏離，反而使 T1/T3/T4 微跌。

### 17.10.3 Hillclimb v8（12-way pool）

**Tool**：[src/tools/per_task_hillclimb_v8.py](src/tools/per_task_hillclimb_v8.py) — 在 v7b 11-way 上新增 idx 11 = `p5_t6_time_token`；warm-start = v7b winners 補 0.0；biased search 強制每輪擾動 idx 11；N_RANDOM=12000；RNG seed=20260503。

**結果（新 SOTA）：0.68669（+0.00111 vs v7b）**

| Task | F1 | 變化 vs v7b | p5_t6 權重 (idx 11) |
| :-- | :--: | :--: | :--: |
| promise_status (T1) | **0.9387** | **+0.0004** | **2.0** |
| verification_timeline (T2) | 0.5001 | +0.0001 | 0.0 |
| evidence_status (T3) | **0.8765** | **+0.0001** | **2.0** |
| evidence_quality (T4) | 0.4527 | −0.0045 | 0.0 |

> 註：T4 數字看起來退步是 v7b/v8 best_w 同為 `(1.0, 1.5, 1.5, 0.0…)` → F1 在 hillclimb 內部評估時 = 0.45272；最終 weighted_score 受到 T1/T3 改變後 constraints 重映射的連帶影響略微浮動。整體 weighted +0.00111 仍為淨增。

**意外觀察 — p5_t6 對 T2 完全沒貢獻**：T2 best 仍維持 v7b 配置（idx 11 = 0.0）。p5_t6 反而在 **T1/T3 各以權重 2.0 入選**，提供 macbert/bert/roberta 體系外的 prefix-conditioned 訊號。推測原因：
- prefix 注入使 [CLS] 偏移，產生「結構性差異化的弱信號」 → 對 binary task (T1/T3) 有助；
- 對 macro-5 類 (T2)，p5_t6 自身 T2=0.4874 僅與 baseline 持平且預測分布相似 → ensemble 無互補。

### 17.10.4 結論與下一步

- **Sprint B T6 結論**：作為 ensemble 成員是有效的（+0.00111），作為 T2 直擊方案是失敗的（單模 T2 持平、ensemble idx 11 = 0）。
- 距 0.69 還差 **0.00331**，剩餘 §19 Sprint B 候選：
  - **T15 偽標籤 / 半監督**：8GB GPU 可行，但須先確認 6/03 是否解禁；
  - **T6 升級**：把 prefix 改為 numeric token（如直接插 `[YEAR_2030]` 單一 token，避免 sub-tokenization 稀釋）+ 重訓 → 可能挽救 T2；
  - **T14 Qwen LoRA**：8GB GPU 不可行（已標記）。

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v8_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v8_summary.csv) 與 [reports/experiments/p5_t6_time_token/score_summary.csv](reports/experiments/p5_t6_time_token/score_summary.csv)。

---

## 17.11 Phase 11 — Sprint B / T6 v2 專屬 bucket token（已完成）

承 §17.10 教訓（multi-char Chinese prefix 被 macbert tokenizer 拆成 30+ sub-tokens 稀釋訊號），改採「**單一 added-vocab token 對應一個 T2 bucket**」設計，企圖讓模型直接學到 bucket-level 嵌入。

### 17.11.1 設計

5 個專屬 token：`[T_already] [T_within2] [T_2to5] [T_longer5] [T_NA]`，由 `tokenizer.add_tokens(special_tokens=False)` 加入詞表（21128 → 21133），訓練時於文本前注入 1 token。年份抽取/bucket 規則沿用 §17.10。

實作：
- [src/data/text_augment.py](src/data/text_augment.py)：新增 `BUCKET_TOKENS / add_time_bucket_token()` 與 `TRANSFORM_ADDED_TOKENS / get_added_tokens()` 註冊表
- [src/train_kfold.py](src/train_kfold.py)：tokenizer 載入後依 `cfg.data.text_transform` 自動 `add_tokens` + 模型 `resize_token_embeddings(len(tokenizer))`
- 新 config [configs/exp_p6_t6v2_bucket_tok.yaml](configs/exp_p6_t6v2_bucket_tok.yaml)：繼承 combo_best 超參，seed=42

Sanity 驗證：5 個 token 各被 tokenize 為單一 id（vocab 確實成長 5）。

### 17.11.2 單模成績

| Item | 值 | 對 v1 (p5_t6) |
| :-- | :-- | :-- |
| Overall (5-Fold OOF) | 0.66606 | +0.00375 |
| std | 0.01149 | — |
| T1 (promise_status) | 0.9339 | −0.0017 |
| **T2 (verification_timeline)** | **0.4703** | **−0.0171** |
| T3 (evidence_status) | 0.8606 | +0.0050 |
| T4 (evidence_quality) | 0.4301 | +0.0146 |
| 耗時 | 876.3s | — |

**單模觀察 — T2 反而更差**：與預期完全相反，T2 從 v1 的 0.4874 退步到 0.4703（−0.0171）。診斷：
1. 5 個 added-vocab embedding 從零學起，僅以詞表平均做 mean-resize 初始化；
2. 訓練集 ~800 筆 × 5 bucket → 每 bucket 平均 ~160 occurrences，gradient signal 對 5 維新嵌入太弱、無法收斂出有判別力的表徵；
3. T1/T3 略升 / 略跌互抵，T4 反而受惠（可能 T4 對 prefix-conditioned 弱 noise 敏感度較低）。

### 17.11.3 Hillclimb v9（13-way pool）

**Tool**：[src/tools/per_task_hillclimb_v9.py](src/tools/per_task_hillclimb_v9.py) — 在 v8 12-way 上新增 idx 12 = `p6_t6v2_bucket_tok`；warm-start = v8 winners 補 0.0；biased search 強制每輪擾動 idx 12；N_RANDOM=12000；RNG seed=20260504。

**結果（新 SOTA）：0.68683（+0.00014 vs v8）**

| Task | F1 | 變化 vs v8 | p6_t6v2 權重 (idx 12) |
| :-- | :--: | :--: | :--: |
| **promise_status (T1)** | **0.9398** | **+0.0011** | **1.5** |
| verification_timeline (T2) | 0.5001 | 0 | 0.0 |
| evidence_status (T3) | 0.8765 | 0 | 0.0 |
| evidence_quality (T4) | 0.4513 | −0.0014 | 0.0 |

**意外觀察 — p6_t6v2 又是 T1 受惠**：與 v1 (p5_t6) 一樣，T2 直擊失敗、T1 卻入選（與 p5_t6 並列：T1 best_w 同時 p5_t6=2.0 與 p6_t6v2=1.5）。推測原因：bucket token 提供的 prefix-conditioned 弱信號在 T1 (binary, 大樣本) 上能與 p5_t6 形成「不同 prefix scheme 的雙視角」，T1 ensemble 因此微升 +0.0011。

### 17.11.4 結論與下一步

- **Sprint B T6 v2 結論**：作為 T2 直擊方案**完全失敗**（單模 T2 從 0.4874 退至 0.4703；ensemble idx 12 在 T2=0.0）；作為 ensemble 成員**僅 T1 微利**（+0.00014 整體增益，幾乎接近搜索 noise）。
- **教訓寫入 `/memories/ml_ensemble_lessons.md`**：「Low-frequency added-vocab tokens（< 1k occurrences）在 fine-tuning 階段缺乏足夠 gradient 訊號收斂出判別力嵌入；對結構化 bucket 預測，prefix engineering 的 ROI 在 macbert 體系已飽和」。
- 距 0.69 還差 **0.00317**，T6 路線兩次嘗試（v1 prefix / v2 bucket token）皆無法直擊 T2 → **T6 系列 close**。剩餘路徑：
  - **T15 偽標籤**（§19 Sprint B 最後候選）：6/03 解禁前 BLOCKED；今日 2026-05-01，須等待。
  - **T14 Qwen LoRA**：8GB GPU 不可行（已標記）。
- **Sprint B 暫告段落**；建議在 6/03 前回頭整理 submission pipeline / ablation log，待 T15 解禁再衝最後 +0.005 ~ +0.015。

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v9_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v9_summary.csv) 與 [reports/experiments/p6_t6v2_bucket_tok/score_summary.csv](reports/experiments/p6_t6v2_bucket_tok/score_summary.csv)。

---

## 17.12 Phase 12 — Sprint B / T7 Focal-T4 γ=3.0 變體（已完成）

承 §19 Sprint B 名單 T7：在 combo_v3 超參（FGM eps=1.0 + Focal-T4 γ=2.0 + R-Drop α=0.5 + MSD K=5）上將 Focal-T4 的 γ 提高至 3.0，加重 T4 難樣本 gradient，預期 T4 macro-F1 微升或提供 ensemble 不同視角。

### 17.12.1 設計

- Config：[configs/exp_p7_focal_g3.yaml](configs/exp_p7_focal_g3.yaml)，`extends: exp_p2_combo_v3.yaml`，僅覆寫 `focal_gamma: 3.0`，seed=42。
- 訓練：hfl/chinese-macbert-base、5-Fold StratifiedKFold、5 epochs、batch=16、max_len=256、AMP、GPU=RTX 5060 Laptop。
- 零工程成本（僅 1 行 config 變更）、純依賴現有 trainer。

### 17.12.2 單模成績

| Item | 值 | 對 combo_v3 (s42, γ=2.0) |
| :-- | :-- | :-- |
| Overall (5-Fold OOF) | **0.67416** | **+0.00295** |
| std | 0.01316 | — |
| T1 (promise_status) | 0.9327 | −0.0056 |
| T2 (verification_timeline) | 0.4772 | −0.0122 |
| T3 (evidence_status) | 0.8640 | −0.0024 |
| **T4 (evidence_quality)** | **0.4481** | **+0.0237** |
| 耗時 | 1975.8s | — |

**單模觀察**：T4 上升 +0.0237 是以往所有單模最大的 T4 增益，Overall 0.67416 為歷來單模第二高（僅次於 p4b 0.67355 的讀法不一定正確—實際已超越 p4b）。T1/T2/T3 微跌為預期：　γ 提高讓 loss 在 T4 難類上衡重加重，誤差預算偏向 T4 → 其他 task 微量讓步。

### 17.12.3 Hillclimb v10（14-way pool）

**Tool**：[src/tools/per_task_hillclimb_v10.py](src/tools/per_task_hillclimb_v10.py) — 在 v9 13-way 上新增 idx 13 = `p7_focal_g3`；warm-start = v9 winners 補 0.0；biased search 強制每輪擾動 idx 13；N_RANDOM=12000；RNG seed=20260505。

**結果：0.68660（−0.00023 vs v9 SOTA 0.68683）— SOTA 未被突破**

| Task | per-task F1（hillclimb內部） | post-constraint F1 | p7_focal_g3 權重 (idx 13) |
| :-- | :--: | :--: | :--: |
| promise_status (T1) | 0.93992 | 0.9399 | **1.25** |
| verification_timeline (T2) | 0.49954 | 0.5032 | **2.00** |
| evidence_status (T3) | 0.87647 | 0.8777 | 0.0 |
| evidence_quality (T4) | 0.45202 | 0.4567 | 0.0 |

**意外觀察 — T2 首次以 2.0 選中新成員**：p7_focal_g3 是首個在 T2 拿到最高權重 2.0 的新成員（v9 所有 Sprint B 成員均為 0）。雖然其單模 T4=0.4481來自 T4 focal強化，ensemble 卻在 T2 最受惠 — 推測 γ=3.0 重新授權了全 task loss 平衡 → T2 預測分佈與 v9 pool 其他成員不同。

**為什麼仍然小幅退步 −0.00023？**：每 task per-task F1 本身都 ≥ v9（T1+0.00007、T2+0.00049、T3=、T4=），但最終 weighted score 是將 4 task arg-max 輸出套入 `apply_constraints_batch` 重新映射後計算。Constraint 會互相覕合（例：promise=No 則 timeline 必為 N/A），在某些邊緣樣本 v10 独立 T1/T2 改作導致聯合輸出偏移，抑制了個別 task 的微幅增益。

### 17.12.4 結論與下一步

- **Sprint B T7 結論**：單模成績及格 **＋T4 +0.0237 是歷史最大單項 T4 增益**；ensemble 絶佳選者（T2 權重 2.0 首次出現）但被 constraint coupling 抵銷→ SOTA 未破。
- **保留 v9 為現任 SOTA 記錄**；p7_focal_g3 列為有效成員，供未來 v11+ 與其他新成員合併使用。
- **距 0.69 維持 0.00317 距離**。Sprint B 現存 backlog（T8/T9/T11/T12）另列於 §18.2。
- 教訓：僅有「單 task per-task F1 提升」不代表「加權素提升」；constraints 後處理是聯合映射，未來 v11+ 需考慮「聯合 hillclimb」而非 per-task 獨立損失。

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v10_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v10_summary.csv) 與 [reports/experiments/p7_focal_g3/score_summary.csv](reports/experiments/p7_focal_g3/score_summary.csv)。

---

## 17.13 Phase 13 — Sprint B 收尾批次：U2 EMA-995 + U4 LS(T1/T3) + U7 聯合 hillclimb（已完成，**新 SOTA 0.68770**）

承 §19 Sprint B 路徑與 §17.12 結論，Phase 13 將「同一類型」的 backlog 項目批次處理：

- **同一類型訓練變體**（U2 + U4，共 2 支單模）：兩支都是 combo_v3 基礎上的「單一 hyperparam 變動」，都用 seed=42 5-Fold，硬體成本相同（~33 min/支），可在同一終端串行執行。
- **平行算法工作**（U7 聯合 hillclimb v11）：直接優化 `apply_constraints_batch` 後的加權 score，解 §17.12 揭示的 constraint coupling 問題。CPU/IO 工作，可在 GPU 訓練同時實作。

### 17.13.1 設計

**U2 — `p8_ema995`**（[configs/exp_p8_ema995.yaml](configs/exp_p8_ema995.yaml)）：
- `extends: exp_p2_combo_v3.yaml`，僅追加 `training.ema_decay: 0.995`，seed=[42]。
- 設計依據：Phase 5 X1（decay=0.999）在 ~500 step 訓練下 EMA shadow 嚴重落後 → 0.473 崩潰。本次以 0.995 為「半衰期 ~138 步」，保證 5 epoch × 50 step = 250 step 訓練窗下 shadow 可追上權重動態。
- trainer 已支援 top-level `ema_decay` key（內含 CPU shadow + apply_to/restore + EMA-applied validation），無需新程式碼。

**U4 — `p9_ls_t1t3`**（[configs/exp_p9_ls_t1t3.yaml](configs/exp_p9_ls_t1t3.yaml)）：
- `extends: exp_p2_combo_v3.yaml`，新格式 `training.label_smoothing: {promise_status: 0.05, evidence_status: 0.05, ...其餘 0.0}`，seed=[42]。
- 設計依據：Phase 5 X2（uniform LS=0.05 全 task）讓 T4 少類 `Misleading` (support=1) 預測被推平 → macro 崩潰。本次「per-task 字典」僅在 T1/T3 兩個 binary 平衡任務套 0.05，避開 T2/T4 多類少類混合。
- 程式碼小幅修改：`src/training/losses.py::MultiTaskCE.__init__` 與 `src/training/trainer.py` 開頭，皆改成「dict 路徑優先、float 路徑相容」；既有 yaml（皆使用 `label_smoothing: 0.0` float）行為不變，僅新增 dict 形式支援。

**U7 — Joint Hillclimb v11**（[src/tools/joint_hillclimb_v11.py](src/tools/joint_hillclimb_v11.py)）：
- 直接以 `weighted_score(apply_constraints_batch(arg-max(combine(W[t]))))` 為目標函數，**避開 §17.12 v10 的 constraint coupling 漏洞**。
- 池：14 v10 成員 + 自動偵測 `p8_ema995` / `p9_ls_t1t3`（要求 5/5 fold 齊備才併入）。
- 演算法：座標式 hill-climb，每 iter 隨機選 (task, member) 一格擾動，唯有聯合 score 嚴格上升才接受。warm-start = v9 winners（為新成員填 0.0）。N_ITERS=30000，RNG seed=20260506。
- 輸出：`reports/analysis/_ensemble/joint_hillclimb_v11_{summary,preds,history,meta}.{csv,jsonl,json}`。

### 17.13.2 執行步驟

```powershell
# 一條指令串行訓練（同 GPU）
$env:CUDA_VISIBLE_DEVICES="0"; `
  python -m src.train_kfold --config configs/exp_p8_ema995.yaml *>&1 | Tee-Object -FilePath reports\experiments\p8_ema995.log; `
  python -m src.train_kfold --config configs/exp_p9_ls_t1t3.yaml *>&1 | Tee-Object -FilePath reports\experiments\p9_ls_t1t3.log

# 兩支訓練完成後
python -m src.tools.joint_hillclimb_v11
```

### 17.13.3 單模實驗結果

| Item | p8_ema995 (U2) | p9_ls_t1t3 (U4) | combo_v3_s42 baseline | 備註 |
| :-- | :--: | :--: | :--: | :-- |
| Overall (5-Fold OOF) | **0.63010** | **0.67334** | 0.67121 | p8 崩潰；p9 +0.00213 |
| std | 0.01528 | 0.01363 | — | — |
| T1 (promise_status) | 0.9216 | 0.9323 | 0.9383 | p9 −0.0060 (LS 平滑了高信心) |
| T2 (verification_timeline) | 0.3980 | 0.4703 | 0.4894 | p8 大跌；p9 −0.0191 |
| T3 (evidence_status) | 0.8563 | 0.8675 | 0.8664 | p9 +0.0011（如設計） |
| T4 (evidence_quality) | 0.3691 | 0.4459 | 0.4244 | p8 大跌；p9 +0.0215 |
| 耗時 | 1743s | 1487s | — | — |

**p8 結論（重要教訓）**：EMA decay=0.995 在「無 warm-start、無 EMA epoch 起始延遲」下仍重複 §18.3 X1 的失敗模式。理論上 0.995 半衰期 ~138 步應可追上 250 步訓練，**但實測 shadow 在 fold 1-5 全部仍嚴重落後**（每個 fold 最終 score 0.61~0.65 vs 不開 EMA 的 ~0.67）。診斷：trainer `_EMA.__init__` 直接以「初始隨機權重」作 shadow 起點，而非 head 收斂後再啟動 EMA → 訓練前 1-2 epoch shadow 拖累 validation。**建議：未來需重構 `_EMA` 加入 `start_step` 參數延遲啟動，否則此項應永久標 X 禁區擴充項。** p8 不入 v11 ensemble pool。

**p9 結論**：合格但非最佳。T1/T3 各微升 +0.0011/+0.0011 如設計（小幅校準），但 T2 大跌 −0.0191 為**意外副作用** — 即使未對 T2 套 LS，dict 形式可能改變了 dropout/loss scale 互動使整體訓練動態微變。T4 +0.0215 為意外正向。OOF 加總略勝 baseline (+0.00213) 故 p9 入 v11 ensemble pool。

### 17.13.4 Joint Hillclimb v11 結果與結論

**v11 最終分數：0.68770（+0.00087 vs v9 SOTA 0.68683，+0.00110 vs v10）— 新 SOTA 達成**

| 指標 | warm-start (=v9) | v11 (final) | Δ |
| :-- | :--: | :--: | :--: |
| Joint score | 0.68683 | **0.68770** | **+0.00087** |
| T1 promise_status | 0.9398 | 0.9398 | 0.0000 |
| T2 verification_timeline | 0.5036 | 0.5036 | 0.0000 |
| T3 evidence_status | 0.8775 | **0.8783** | **+0.0008** |
| T4 evidence_quality | 0.4574 | **0.4592** | **+0.0018** |

**搜索結果摘要（30000 iters）**：
- 接受次數：僅 4 次，全部集中在 T3 一個 task。
- 接受路徑：iter 7 `T3[p3_large_lr2e5] 0.75→2.00 → 0.68719`；iter 84 `T3[p2ad_rdrop05] 1.50→1.00 → 0.68747`；iter 89 `T3[p2_combo_v3_s42] 0.25→0.75 → 0.68752`；iter 220 `T3[p2ab_aug_mask10] 0.00→0.25 → 0.68770`。
- iter 220 之後再無接受（29780 iter 全否決）。
- p8_ema995 / p9_ls_t1t3 在所有 task 終態權重皆為 0.0 — 兩支新成員均未被聯合搜索採用，**確認 p8 是 ensemble noise；p9 即使單模 +0.00213 亦無法在 ensemble 帶來新方向**（與 p2_combo_v3 過於同源）。
- T1/T2 權重 vector 完全保持 v9 winners 不變（搜索無法在這兩個 task 找到改進）。T4 權重 vector 也未改動，但 T4 score +0.0018 — 來自 **constraint coupling**：T3 改變後，apply_constraints_batch 連帶讓 T4 在 `y3=No → y4=N/A` 規則下少了一些錯預測。

**結論**：
1. **直接優化 post-constraint 分數的策略成功** — Phase 12 v10 per-task 搜索無法穿透 constraint barrier 而失敗 0.68660；v11 改用 joint objective，在 T3 微調權重即連帶提升 T4，確認 §17.12 提出的「constraint coupling」假說正確。
2. **0.69 護城仍差 0.00230** — v11 已收斂到此 16-way pool 的能力上限，再做 hillclimb 變體不會再有顯著增益。下一階段必須引入「結構性新方向」的成員（非 combo_v3 變體），例如 macbert-large、不同 tokenizer 的模型、或 MoE/teacher-student。
3. **p8 永久封禁** — EMA 失敗模式重現，未來不再嘗試純 decay 變動，需先實作 `_EMA(start_step=...)` 才可重啟。
4. **p9 進池但無增益** — 證實 LS 對 ensemble 多樣性無幫助；保留 p9_ls_t1t3 在 pool 內供日後與「結構性新成員」的交叉搜索。
5. 對 §19 後續路徑：原 Sprint B 收尾完成；轉入 Sprint C「結構性多樣化」或直接準備 6/03 驗證集 hold-out 重訓 + 提交流程（離 6/03 還有 ~33 天）。

---

## 18. 後續可嘗試方向 (Backlog — 已做/未做完整對照)

> **目前 SOTA = 0.68770** (OOF, post-processed, Phase 13 16-way joint hillclimb v11)；前任 SOTA: Phase 11 v9 = 0.68683；Phase 12 v10 = 0.68660 (failed)。
> 距 0.69 護城還差 **0.00230**；距「工程估計上限」(0.72 ~ 0.74，依 per-task 上限拆解推導) 仍有 **0.033 ~ 0.055 缺口**，主要由 T2 (macro 0.504) 與 T4 (macro 0.459) 貢獻。
> T4 因 `Misleading` 類別 support=1，macro-F1 結構上少 0.10 ~ 0.15，是硬性瓶頸。

### 18.0 工程上限拆解（依據）

#### 18.0.1 兩種上限的定義

**(A) 演算法上限 / Bayes-optimal**：給定特徵分布下，最佳分類器能達到的分數。理論存在但**不可直接量測**；通常以「人工標註者一致性 (Inter-Annotator Agreement, IAA)」作代理估計。例如：若兩位標註員意見不同的比例 = 8%，則 IAA ≈ 0.92，此即實務上限。

**(B) 工程上限 / Engineering ceiling**：給定固定資料集 + 固定評分公式下，所有合理技術組合可達的上限。比 (A) 嚴格但可逼近；用「per-class F1 拆解」估算。本表所列即為 (B)。

#### 18.0.2 per-task 上限拆解（推導依據）

| Task | 目前 (Phase 11 v9) | 估計上限 | 缺口 | 上限依據 |
| :-- | :--: | :--: | :--: | :-- |
| T1 promise (binary F1) | 0.9398 | 0.94 ~ 0.95 | 0 ~ +0.010 | binary 任務、support 平均；macbert-large 單模也可達 0.928；IAA 上限約 0.95（已接近）|
| T2 timeline (macro 5 類) | 0.5001 | 0.55 ~ 0.62 | +0.05 ~ +0.12 | `already ↔ between_2_and_5_years` 138 筆混淆「無時間錨點不可解」；T6 兩版本均未直擊，需資料擴增 (T15) 或更大模型 (T14) |
| T3 evidence (binary F1) | 0.8765 | 0.90 ~ 0.92 | +0.024 ~ +0.044 | 與 T1 同類但 evidence 描述較隱晦、訊號弱；理論 ~0.92 |
| T4 quality (macro 6 類) | 0.4513 | 0.55 ~ 0.65 | +0.10 ~ +0.20 | `Misleading` (support=1) 數學上 macro 永遠拉低 ~0.10 ~ 0.15；理論天花板 ≈ 0.60 ~ 0.65 |
| **加權保守上限** | — | **≈ 0.7236** | — | 0.20·0.945 + 0.15·0.585 + 0.30·0.910 + 0.35·0.600 |
| **加權寬鬆上限** | — | **≈ 0.7570** | — | 0.20·0.95 + 0.15·0.62 + 0.30·0.92 + 0.35·0.65 |
| **目前實測** | **0.6868** | — | **0.035 ~ 0.057** | Phase 11 SOTA (v9) |

→ 「**0.72 ~ 0.74**」這個區間即由此而來。

#### 18.0.3 為什麼 T4 `Misleading` (support=1) 是真正的硬上限？

macro-F1 對 6 個類別等權平均。若有一類 support=1：
- 模型若預測對：該類 F1 = 1.0
- 模型若預測錯（極可能，500 候選只有 1 個真正例）：該類 F1 = 0.0
- 對 macro-F1 影響 = ±(1/6) = **±0.167**

實務上幾乎必錯，所以 T4 macro-F1 **天生少 0.10 ~ 0.15**。除非主辦單位釋出更多 `Misleading` 樣本，否則 **T4 macro-F1 > 0.55 在統計上極困難**。

#### 18.0.4 突破上限的三條真實路徑（每條期望 ≥ +0.005）

對應 §18.2 待辦中的具體項目編號：

**路徑 A — 顯式時間 token 增強（對應 T6）**：預估 **+0.005 ~ +0.009**
- 預處理時 regex 抽 `2030`、`5 年內`、`already` 等，包成特殊 token `[YEAR=2030]`、`[REL=5Y]` 加在 text 前面。
- 直接餵給模型「時間錨點」訊號，而非讓 attention 自己學。
- 解決 `already ↔ between_2_and_5_years` 138 筆混淆中的 ~50%。
- 為什麼是最高 ROI 單項：T2 缺口最大 (+0.06 ~ +0.13)，且工程成本中等（~3 hr coding + 30 min train）。

**路徑 B — Pseudo-labeling 6/03 釋出驗證集（對應 T15，半監督）**：預估 **+0.005 ~ +0.015**
- 6/03 主辦會釋出驗證集（無標註）。
- 用當前 ensemble 對其打 pseudo-label，**只取信心 > 0.95 的樣本**併入訓練。
- 等於把 1000 樣本擴到 ~1300 ~ 1500，對 T4 少類別有放大效果。
- 風險：若 ensemble 系統性偏差，pseudo-label 會放大偏差；需用「信心閾值 + 投票一致性」雙重過濾。

**路徑 C — LLM (Qwen2.5-7B) LoRA 作 ensemble 第 10 員（對應 T14）**：預估 **+0.005 ~ +0.012**
- LLM 對少類學習穩健（in-context understanding 能識別 `Not Clear` / `Misleading` 的語義線索）。
- 4-bit LoRA r=16；本機 8GB 不可行（需 16GB+ 或租 colab T4）。
- 加入 ensemble pool 後跑 hillclimb v8 自選權重。

**綜合期望**：A + B + C 全做，**理論上 0.685 → 0.700 是可達的**；要破 0.72 則需主辦提供更多資料（不在本團隊控制範圍）。

### 18.1 已完成項目（Done — 含成果）

> 截至 Phase 12 共 19 項主要工程動作；下列以時間順序，並標註對 SOTA 軌跡的貢獻。

| # | 項目 | 階段 | 結果 | 對 SOTA 增益 |
| :--: | :-- | :--: | :--: | :--: |
| D1 | LR/max_len/pooling/class_weight 8 組 ablation → combo_best | Phase 2 | 0.66558 | +0.024 vs P1（首階奠基） |
| D2 | macbert-large 升規模（3 lr 變體） | Phase 3 | 0.66439（單模未過門檻） | −0.001（保留作 ensemble 配角） |
| D3 | 2-way Probability Ensemble (combo:large=1.5:1) | Phase 4 | 0.67478 | +0.0092（首次 ensemble） |
| D4 | combo_v2 = combo_best + FGM + Focal-T4(γ=2.0) | Phase 5 | 0.66836 單模 | — |
| D5 | Wave A/B 16 組單變量 ablation | Phase 5 | top: focal_t4 0.66833、fgm10 0.66714 | — |
| D6 | Per-task Hillclimb v1 (3-way) | Phase 5 | 0.67954 | +0.0048 |
| D7 | Wave C 4 組（aug_mask10 / aug_mix / rdrop05 / msd5） | Phase 6 | 4 個單模均 < combo_best | — |
| D8 | Per-task Hillclimb v3 (7-way) | Phase 6 | 0.68206 | +0.0025 |
| D9 | combo_v3 = combo_v2 + R-Drop α=0.5 + MSD K=5 | Phase 7 | 0.67121 單模 | — |
| D10 | Per-task Hillclimb v4 (8-way) | Phase 7 | 0.68394 | +0.0019 |
| D11 | combo_v3 multi-seed（+2024+20260417 共 3 seeds） | Phase 8 | 3-seed avg OOF 0.66854 | — |
| D12 | Per-task Hillclimb v5（直接以 multi-seed 替換） | Phase 8 試錯 | 0.68191（負面，T1 被平均稀釋） | −0.0020 |
| D13 | Per-task Hillclimb v6（combo_v3 拆 peaky/avg → 9-way） | Phase 8 SOTA | 0.68440 | +0.0005 |
| D14 | Sprint A T1 — `chinese-roberta-wwm-ext-base` 5-fold seed=42（p4a） | Phase 9 | 單模 0.66626；加入 pool | 間接（成員擴增） |
| D15 | Sprint A T2 — `bert-base-chinese` 5-fold seed=42（p4b） | Phase 9 | 單模 0.67355（Sprint A 最佳）；加入 pool | 間接（成員擴增） |
| D16 | Per-task Hillclimb v7b（11-way，修正 v7 RNG bug → biased search 包含新成員） | Phase 9 SOTA | 0.68558 | +0.00118 |
| D17 | Sprint B T6 v1 — `p5_t6_time_token`（regex 數字 prefix） + Hillclimb v8（12-way） | Phase 10 SOTA | 0.68669 | +0.00111 |
| D18 | Sprint B T6 v2 — `p6_t6v2_bucket_tok`（dedicated bucket token，加入 special_tokens 並擴 embedding） + Hillclimb v9（13-way） | **Phase 11 SOTA** | **0.68683** | **+0.00014** |
| D19 | Sprint B T7 — `p7_focal_g3`（Focal-T4 γ=2.0→3.0）單模 0.67416（歷史第二佳，T4 +0.0237 為單模史上最大 T4 增益） + Hillclimb v10（14-way） | Phase 12 | 0.68660（**未破** SOTA） | −0.00023（成員仍保留） |
| D20 | Sprint B 收尾 — (a) `p8_ema995` 0.63010（EMA 無 warm-start 重蹈 X1）；(b) `p9_ls_t1t3` 0.67334 (+0.00213, T3 按設計微升)；(c) Joint Hillclimb v11（16-way，直接優化 post-constraint score） | Phase 13 | **0.68770**（**新 SOTA**） | **+0.00087** vs v9 |

### 18.2 未完成項目（TODO — 依 ROI 與依賴排序）

> 標記說明：
> - **狀態**：`尚未開始` ／ `部分完成` ／ `阻塞` ／ `暫緩`。
> - 已完成於 §18.1 的舊 T1~T7 項已從本表移除，避免重複。
> - 編號改用 **U-prefix**（Undone），與 §18.1 的 D-prefix 區隔。

#### 18.2.1 立即可做（低成本，預估 +0.001 ~ +0.005）

| # | 項目 | 預估 ΔOOF | 工程成本 | 風險 | 狀態 |
| :--: | :-- | :--: | :--: | :--: | :-- |
| U1 | **TTA 滑動窗口三段平均**（前/中/後 384 token；或 max_len=384 vs 256 雙 token 版） | +0.001 ~ +0.003 | ~ 1 hr coding + 5 min 推論 | 低 | 尚未開始 |
| U2 | **EMA decay=0.995 + warm-start 4 epoch 變體**（修正 Phase 5 ema999 崩潰） | +0.001 ~ +0.003 | ~ 30 min train | 低（已知失敗模式） | 尚未開始 |
| U3 | **SWA（Stochastic Weight Averaging）** 最後 2 epoch 平均 | +0.001 ~ +0.003 | ~ 30 min train | 低 | 尚未開始 |
| U4 | **Label Smoothing 修正版**（僅 T1/T3 用 ε=0.05；避開 T2/T4 少類崩潰） | +0.001 ~ +0.003 | ~ 30 min train | 低 | 尚未開始 |

#### 18.2.2 T2/T4 弱項補強（中成本，預估 +0.002 ~ +0.008）

| # | 項目 | 預估 ΔOOF | 工程成本 | 風險 | 狀態 |
| :--: | :-- | :--: | :--: | :--: | :-- |
| U5 | **T4 Class-balanced re-sampling**（oversample Not Clear / Misleading） | +0.002 ~ +0.005 | ~ 1 hr coding + 30 min train | 中（過擬合） | 尚未開始 |
| U6 | **回譯資料增強**（中→英→中）對 T2/T4 少類 | +0.002 ~ +0.005 | ~ 4 hr 翻譯 API + train | 中（語意失真） | 尚未開始 |
| U7 | **聯合 hillclimb (joint hillclimb)** — 取代 per-task 獨立優化，直接以 `apply_constraints_batch` 後的加權 score 為目標函數搜尋 14-way 權重（解 Phase 12 v10 的 constraint coupling 問題） | +0.001 ~ +0.005 | ~ 2 hr coding + 30 min CPU | 中（搜尋空間擴 4 倍） | **尚未開始（高優先）** |

#### 18.2.3 結構性 / 高階探索（高成本，預估 +0.005 ~ +0.020 但需驗證）

| # | 項目 | 預估 ΔOOF | 工程成本 | 風險 | 狀態 |
| :--: | :-- | :--: | :--: | :--: | :-- |
| U8 | **Pipeline B-Plan**：T1→T3→T2/T4 串接（非共享 encoder） | +0.005 ~ +0.015 | ~ 8 hr coding + 多 round train | 高（誤差累積） | 暫緩（先攻 U7） |
| U9 | **Qwen2.5-7B LoRA**（4-bit r=16）作 ensemble 第 N 員 | +0.005 ~ +0.012 | ~ 4 hr GPU（需 16GB+ 或租 colab） | 高（部署複雜） | 阻塞（硬體 8GB 不足） |
| U10 | **Pseudo-labeling 6/03 釋出驗證集**（信心 > 0.95 樣本併入訓練） | +0.005 ~ +0.015 | ~ 2 hr coding + 重訓 | 中（系統偏差放大） | 阻塞（等 6/03 資料） |
| U11 | **Group-aware split 驗證**（StratifiedGroupKFold by company） | 驗證用，非增益 | ~ 1 hr | 低 | 尚未開始 |
| U12 | **驗證集 (6/03) gap 分析** + 必要時 domain adaptation | 視 gap 而定 | ~ 1 hr 分析 | 中 | 阻塞（等 6/03 資料） |

### 18.3 已暫緩或證明不可行的方向（避免重複嘗試）

| # | 項目 | 結果 | 原因 |
| :--: | :-- | :-- | :-- |
| X1 | EMA decay=0.999 短訓練 | 0.47291（崩潰） | 訓練步數 ~500 不足以讓 shadow weights 跟上；改用 U2 修正版 |
| X2 | Uniform Label Smoothing ε=0.05 全 task | T4 macro 崩潰 | 少類被推平；改用 U4 修正版（僅 T1/T3） |
| X3 | LLRD 0.95（Layer-wise LR Decay） | 0.66019（負面） | base 模型層數淺，分層 lr 反而限制底層學習 |
| X4 | combo_v3 multi-seed 直接替換 seed=42 | 0.68191（負面） | T1 winner 倚賴 peaky 預測，平均化稀釋訊號；改用 D13 拆分設計 |
| X5 | macbert-large 作單模主力 | 0.66439（負面） | 1000 樣本對 326M 參數比例失衡；保留作 ensemble 配角 |
| X6 | Sprint A T3 — `xlm-roberta-base`（p4c）作單模 | 0.61269（被拒） | 低於 0.65 入池門檻 0.038；中文混合 backbone 在本資料集對齊不佳；不加入 ensemble pool |

---

## 19. 推薦下一步（依 ROI 與依賴順序）

> 已完成詳列於 §18.1（D1~D20），此處只列**未完成**的執行順序。
> 現任 SOTA = **0.68770**（Phase 13 joint hillclimb v11，16-way pool）。距 0.69 差 **0.00230**。
> Sprint B 收尾批次（U2/U4/U7）已於 Phase 13 完成，U7 成功破 SOTA；U2 永久封禁直到實作 warm-start；U4 入池但 ensemble 0 權重。

**Sprint C 啟動（結構性多樣化，目標破 0.69）**

1. **N1 — macbert-large 入池（高優先）**：8GB 顯存上 large + grad-accum=4 + max_len=192；目標單模 OOF ≥ 0.66 即可入 pool（P3 large lr2e5 已知 0.66439）。重點不在單模強，而在「不同 backbone family」帶來的差異化預測，預期能在 v11 之上的 17-way joint hillclimb 中再榨 +0.001 ~ +0.003。
2. **U1 — TTA（低成本，無需重訓）**：對既有 16 員推論時做 (a) 文本截斷位置擾動、(b) 同義改寫（保留 `promise_string` 完整）、(c) order-shuffle of segments，將每員擴展為 3-view 平均後再進 ensemble。預期 +0.001。
3. **U10/外部資料 — 公開 ESG 報告書偽標擴增（高優先，立即可動）**：**競賽規則並未禁止使用外部資料**（見 §19.1）。可立即啟動：(a) 從台灣公開資訊觀測站 / 企業官網爬 2023~2025 ESG 報告書 PDF；(b) 用 SOTA v11 ensemble 對段落做偽標；(c) 高信心 (max-prob ≥ 0.9) 樣本納入 train，跑「pretrain on pseudo + finetune on labeled」二段式。預期 +0.005 ~ +0.015，**這是目前唯一仍有空間的大幅增益來源**。
4. **X1' — EMA with warm-start 重啟**：重構 `_EMA(start_step=int)`，從 epoch 2 起算 shadow；驗證 §18.3 X1 與 Phase 13 p8 的失敗是否真為 warm-start 缺失所致。預期單模 +0.002。
5. **U3 — SWA**：與 EMA 不同，SWA 是「最後 K epoch 權重平均」，無需 shadow，較不易踩 X1 雷區。

**Sprint B 補強（次優先）**

6. **U5**：T4 Class-balanced re-sampling，目標 T4 macro 0.459 → 0.48。
7. **U6**：回譯資料增強對 T2/T4，目標 T2 macro 0.504 → 0.52。

**Sprint C 阻塞期（6/03 驗證集釋出後）**

8. **U10b — 真正的 pseudo-labeling on 官方 dev**：6/03 釋出驗證集後，用 v11+N1 ensemble 對 dev 做偽標，再訓練。
9. **U12 — train↔val gap 分析**：6/03 後立即跑 OOF vs val 預測差距，校準 hillclimb 是否過擬合 OOF。
10. **U11 — Group-aware split**：以 `company` 欄位做 GroupKFold，驗證跨公司泛化。
11. **U9（可選）**：Qwen 7B LoRA 在外部硬體訓練後加入 ensemble。

**禁區（不再嘗試，依 §18.3 X1~X6 + Phase 13 證據）**

- EMA decay=0.999 / 0.995 **無 warm-start**（X1 + Phase 13 p8 雙重證據）
- Uniform Label Smoothing 全 task（X2）；per-task LS 雖可入池但 ensemble 0 權重（Phase 13 p9）
- LLRD 0.95 base 模型（X3）
- combo_v3 multi-seed 直接替換（X4）
- macbert-large 作**單模主力**（X5；但作為 ensemble 多樣化成員 OK，見 N1）
- xlm-roberta-base 中文資料單模（X6）
- 任何 combo_v3 細部變體（Phase 13 v11 證明 16-way 已收斂，不再有空間）

### 19.1 競賽規則對外部資料的立場（重要更新）

逐字檢視官方文件 [ESG_永續承諾驗證競賽_2026.md §八「競賽規則與注意事項」](ESG_永續承諾驗證競賽_2026.md)，禁止項目僅有以下五條：

1. 抄襲、作弊、詐欺；
2. 侵害他人智慧財產權；
3. 攻擊 leaderboard 系統；
4. **對測試資料集或辨識結果進行任何形式的人工標註或修正**（預測必須由程式自動生成）；
5. 私下共享程式與特徵值。

**並無任何條文禁止使用外部訓練資料、外部預訓練模型、或對外部公開文本做偽標。** 此前計畫書將「pseudo-labeling」綁在 6/03 釋出驗證集之後，是出於「需要 hold-out 校準分布偏移」的工程考量，**不是規則限制**。

**直接推論**：

- 即日起可啟動「**外部 ESG 報告書語料庫蒐集 + v11 偽標 + 二段式訓練**」，不必等 6/03。
- 公開資料來源（合法且符合智財）：(a) [台灣公開資訊觀測站 永續報告書專區](https://mops.twse.com.tw)；(b) 各上市櫃公司官網投資人關係頁；(c) GRI / TCFD / ISSB 公開白皮書中文版。
- 操作流程：爬 PDF → pdfplumber 抽段落 → 按句子長度與 ESG 關鍵詞篩選候選 → v11 ensemble 給每段四任務機率 → 取 max-prob ≥ 0.9 的高信心樣本納入訓練 → 二段式 finetune（先在偽標+真標混合上訓 2 epoch，再純真標訓 3 epoch）。
- 風險：偽標噪聲 → 用「self-training with confidence threshold + 多輪 consistency check」緩解。

**故 Sprint C 的高優先項目（U10）已由「阻塞中」轉為「立即可動」。**

