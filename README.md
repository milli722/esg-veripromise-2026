# VeriPromise ESG 2026 — Phase 36 SOTA Pipeline

> 繁中 ESG 永續承諾驗證競賽（**ESG VeriPromise 2026 / AI CUP 2026**）參賽程式碼。
> 採 6 stems × 3 seeds × 5-fold × 3-view TTA × per-task hillclimb 集成法，
> 在官方 1,000 筆訓練集上以 5-Fold Stratified CV 取得 **OOF weighted score = 0.71018**。

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2%2B-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 目錄

1. [競賽簡介](#1-競賽簡介)
2. [SOTA 結果速覽](#2-sota-結果速覽)
3. [專案結構](#3-專案結構)
4. [快速開始](#4-快速開始)
5. [完整重現流程](#5-完整重現流程)
6. [方法概要](#6-方法概要)
7. [文件總覽](#7-文件總覽)
8. [硬體與環境](#8-硬體與環境)
9. [常見問題](#9-常見問題)
10. [授權與引用](#10-授權與引用)

---

## 1. 競賽簡介

ESG VeriPromise 2026 競賽要求對企業永續報告書中的中文段落同時預測四個任務：

| 代號 | 欄位 | 標籤域 | 評分 | 權重 |
| :-: | :-- | :-- | :-- | :-: |
| T1 | `promise_status` | Yes / No | F1（positive=Yes） | 0.20 |
| T2 | `verification_timeline` | already / within_2_years / between_2_and_5_years / longer_than_5_years / N/A | macro F1 | 0.15 |
| T3 | `evidence_status` | Yes / No / N/A | F1（positive=Yes） | 0.30 |
| T4 | `evidence_quality` | Clear / Not Clear / Misleading / N/A | macro F1 | 0.35 |

最終分數：

```
S = 0.20 · F1(T1, Yes) + 0.15 · macroF1(T2)
  + 0.30 · F1(T3, Yes) + 0.35 · macroF1(T4)
```

訓練樣本 **1,000 筆**；測試集競賽方未公開。詳見 [`ESG_永續承諾驗證競賽_2026.md`](ESG_永續承諾驗證競賽_2026.md)。

---

## 2. SOTA 結果速覽

Phase 36（2026-05-10）— 5-Fold OOF：

| 指標 | 分數 |
| :-- | :-: |
| **加權總分** | **0.71018** |
| T1 promise_status (F1, Yes) | 0.94210 |
| T2 verification_timeline (macro F1) | 0.62778 |
| T3 evidence_status (F1, Yes) | 0.87774 |
| T4 evidence_quality (macro F1) | 0.46934 |

### Phase 32 → 36 進化軌跡

| 階段 | 主要差異 | OOF |
| :-: | :-- | :-: |
| Phase 32 | p2_combo_best baseline + per-task hillclimb（單 stem） | 0.6694 |
| Phase 33 | + class-weighted CE / Focal γ=3 on T4 | 0.6741 |
| Phase 34 | + U10 v1/v2 偽標籤兩階段微調 | 0.6878 |
| Phase 35 | + 5-way 6 stems × 3-view TTA | 0.7034 |
| **Phase 36** | + U6-pro 反翻譯擴增 & 6-way × 3-view × per-task hillclimb | **0.71018** |

完整實驗紀錄見 [`MASTER_PLAN_AND_PROGRESS_20260502.md`](MASTER_PLAN_AND_PROGRESS_20260502.md)。

---

## 3. 專案結構

```
esg-veripromise-2026/
├── SOTA_Reproduction_Phase36.ipynb    # ★ 主要交付：SOTA 重現手冊（17 章）
├── MASTER_PLAN_AND_PROGRESS_20260502.md  # 完整研究日誌（Phase 1~36）
├── ESG_永續承諾驗證競賽_2026.md         # 競賽規則
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── vpesg4k_train_1000 V1.csv          # 官方訓練集（1,000 筆）
├── vpesg4k_train_1000 V1.json
├── [External]_VeriPromiseESG_..._Baseline_Code_ZH.ipynb  # 官方 baseline
│
├── configs/                # 6 個 SOTA stem 的 YAML 配方
│   ├── base.yaml
│   ├── exp_p2_combo_best.yaml
│   ├── exp_p2_combo_best_u10_pseudo.yaml
│   ├── exp_p2_combo_best_u10_pseudo_v2.yaml
│   ├── exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3.yaml
│   ├── exp_p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3.yaml
│   └── exp_p2_combo_best_classw_focal_u6pro.yaml
│
├── src/                    # 框架原始碼
│   ├── seed.py             # 全域種子
│   ├── config.py           # YAML 遞迴 extends 載入
│   ├── train_kfold.py      # 5-fold 訓練入口
│   ├── train_pseudo_kfold.py  # 兩階段（pseudo + real → real-only）
│   ├── data/               # loader / dataset / splits
│   ├── models/             # MultiTaskClassifier + Pooler
│   ├── training/           # trainer / losses / schedulers
│   ├── inference/          # post_process（階層約束）
│   ├── eval/               # metrics（weighted_score）
│   └── tools/              # OOF 集成 / hillclimb / 3-view TTA
│
├── scripts/
│   ├── build_sota_notebook.py   # 重新產生 SOTA notebook
│   ├── u10_collect_sr.py        # M1: 永續報告書 PDF 爬取
│   ├── u10_pdf_extract*.py      # M3: PDF 段落擷取（v1/v2/v3）
│   ├── u10_pseudo_label*.py     # M4: 偽標籤閘門（v1/v2/v3）
│   ├── u10_v{2,3}_sustaihub_crawl.py  # 擴充來源
│   ├── u10_sources.py           # 公司清單
│   └── u6_backtranslate_pro.py  # NLLB-200 雙樞紐回譯 + ESG 術語表
│
├── tests/                  # pytest 單元測試（評分函式 + 後處理）
└── docs/archive/           # 早期計畫存檔
```

> **本機獨有（不上傳）**：
> - `outputs/checkpoints/` ≈ 188 GB（6 stems × 3 seeds × 5 folds 的 best.pt 與 oof_probs.npz）
> - `data/raw/` ≈ 1.3 GB（U10 永續報告書原始 PDF）
> - `data/processed/`、`data/splits/`（可重生）
> - `outputs/cache/`（per-view OOF 快取）
> - `reports/{logs,runs,experiments,analysis,u6}/`（per-run 實驗中介物）
>
> 完整理由與重生方式見 §5 與 [`.gitignore`](.gitignore)。

---

## 4. 快速開始

### 4.1 安裝

```powershell
# Windows / PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2 跑單元測試

```powershell
pytest -q
```

### 4.3 開啟 SOTA 重現手冊

```powershell
jupyter notebook SOTA_Reproduction_Phase36.ipynb
```

筆記本第 §1 章可選擇執行模式：

| 模式 | 用途 | 預估耗時（RTX 5060 Laptop 8GB） |
| :-- | :-- | :-- |
| `explain` | 僅閱讀，不執行模型 | 0 分鐘 |
| `load_oof` | **預設**。從既有 checkpoints 重建 OOF + hillclimb 重現 0.71018 | 8 ~ 12 分鐘 |
| `demo` | 對單一 stem 跑 1 fold smoke-train | 3 ~ 5 分鐘 |
| `full` | 從零訓練 6 stems × 3 seeds × 5 folds | 30 ~ 40 GPU 小時 |

> **注意**：`load_oof` 與 `demo`/`full` 都需要在本機自行訓練／下載 `outputs/checkpoints/`。
> 公開 GitHub 倉儲僅含程式碼與設定檔，不含模型權重。

---

## 5. 完整重現流程

從零端到端重訓 Phase 36 SOTA（不依賴任何既有 checkpoint）。

### 5.1 重生資料層

```powershell
# 5.1.1 確認官方資料就位（已隨 repo 上傳）
ls "vpesg4k_train_1000 V1.csv", "vpesg4k_train_1000 V1.json"

# 5.1.2 （可選）重跑 U10 永續報告書管線
#   M1: 從公開資訊觀測站爬 30 家公司 PDF（約 1.3 GB / 60 分鐘）
python scripts\u10_collect_sr.py
#   M3: PDF 段落擷取 + ESG 關鍵字過濾 + SimHash 去重
python scripts\u10_pdf_extract_v3.py
#   M4: 對未標註段落跑教師模型 + 四任務聯合閘門（先需先有 p2_combo_best 15 ckpts）
python scripts\u10_pseudo_label_v3.py

# 5.1.3 （可選）重跑 U6-pro 反翻譯擴增（NLLB-200，約 3 GB 模型 / 90 分鐘 GPU）
python scripts\u6_backtranslate_pro.py
```

> 上述步驟涉及外部網站結構與 PDF 解析版本依賴，**byte-exact 重現可能漂移 ±0.003**。
> 若僅需重現官方資料 baseline 而不在意 U10/U6 增強，請跳過 5.1.2 / 5.1.3。

### 5.2 訓練 6 個 SOTA stem

```powershell
# 預估 30~40 GPU 小時 / RTX 4060 8GB；單 fold 約 3~5 分鐘
python -m src.train_kfold        --config configs\exp_p2_combo_best.yaml
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo.yaml
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo_v2.yaml
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3.yaml
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3.yaml
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_classw_focal_u6pro.yaml
```

每個 stem 會在 `outputs/checkpoints/{stem}/seed{S}/fold{F}/` 寫入 `best.pt` 與 `oof_probs.npz`。

### 5.3 集成與重現分數

開啟 [`SOTA_Reproduction_Phase36.ipynb`](SOTA_Reproduction_Phase36.ipynb) 並設定 `MODE = "load_oof"`，依序執行所有 cell。最末 cell 將輸出：

```
[Phase 36 重現] weighted_score = 0.71018
  promise_status         F1 = 0.94210
  verification_timeline  F1 = 0.62778
  evidence_status        F1 = 0.87774
  evidence_quality       F1 = 0.46934
```

實際數字會因 cuDNN 非決定性、torch 浮點累加在 ±0.0005 內漂移。

---

## 6. 方法概要

詳細推導與消融實驗見 [`MASTER_PLAN_AND_PROGRESS_20260502.md`](MASTER_PLAN_AND_PROGRESS_20260502.md) §10、§47–§54。

### 6.1 模型

- 主幹：`hfl/chinese-macbert-base`（中文 MacBERT，BERT 架構＋MLM-as-correction 預訓練）
- 池化：`cls_mean`（拼接 [CLS] 與 attention-mask 加權平均向量；2 × hidden_size = 1536）
- 分類頭：4 個獨立的 `nn.Linear(1536, C_t)` + 任務獨立 dropout

> 已驗證 large 模型（macbert-large、xlm-r-base、nezha-base、ernie-base、electra-base）在 1k 樣本下均 < base，故維持 base 主幹。

### 6.2 訓練策略

| 組件 | 設定 |
| :-- | :-- |
| 切分 | 5-Fold StratifiedKFold（複合鍵 `promise_status \| evidence_status`），3 seeds |
| 優化器 | AdamW，lr=3e-5，wd=0.01 |
| 學習率排程 | Cosine + 10% warmup |
| 數值精度 | AMP fp16 + grad_clip 1.0 |
| 批量 | bs=8 + grad_accum=2，max_len=384 |
| 損失 | 多任務 CE；T1/T2/T3 用類別加權；T4 用 Focal Loss (γ=3) |
| 提前停止 | patience=2，依 fold OOF 加權分數 |

### 6.3 半監督與資料擴增

- **U10 弱監督管線**（M1~M5）：從公開資訊觀測站爬 30 家公司永續報告書 → PDF 段落擷取＋SimHash 去重 → 教師模型四任務聯合閘門（T1≥0.80 / T2≥0.60 / T3≥0.70 / T4≥0.60）→ 兩階段微調（Stage A：pseudo+real / Stage B：real-only）。
- **U6-pro 反翻譯**：NLLB-200-distilled-600M × 雙樞紐（en、ja）× 雙溫度，搭配 109 條 ESG 術語表保護與 ChrF 過濾，對 minority 類別（T2 within_2y / longer_5y、T4 Misleading / Not Clear）產生 434 筆增強樣本，僅在 fold-train 端注入，**永不洩漏 val/OOF**。

### 6.4 集成（Phase 36 核心）

對每個任務同時搜尋兩組 simplex 權重：

- **Stem 權重** $w^{(t)} \in \Delta^5$（6 個 stem 加總為 1）
- **View 權重** $\alpha^{(t)} \in \Delta^2$（stored + middle + tail 三視角加總為 1）

使用 grid step = 0.05 的座標下降，A↔B 交替 2 輪即收斂。最終 hillclimb 收斂至：

| 任務 | Stem w*（依 stem #1~#6 順序） | View α*（stored, middle, tail） |
| :-: | :-- | :-- |
| T1 | (0.10, 0, 0, 0, 0.50, 0.40) | (0, 0, 1) |
| T2 | (0.10, 0.10, 0, 0.40, 0.30, 0.10) | (0.4, 0, 0.6) |
| T3 | (0.30, 0.20, 0.30, 0, 0, 0.20) | (0.5, 0.5, 0) |
| T4 | (0.10, 0.10, 0, 0.40, 0, 0.40) | (0.5, 0.5, 0) |

### 6.5 階層後處理

```python
if promise_status == "No":
    verification_timeline = evidence_status = evidence_quality = "N/A"
elif evidence_status == "No":
    evidence_quality = "N/A"
```

該約束**必須在 hillclimb 評分迴圈內即套用**，否則 hillclimb 會選到不合法但 OOF F1 較高的權重。

---

## 7. 文件總覽

| 文件 | 內容 |
| :-- | :-- |
| [`SOTA_Reproduction_Phase36.ipynb`](SOTA_Reproduction_Phase36.ipynb) | 17 章可執行重現手冊；首選入口 |
| [`MASTER_PLAN_AND_PROGRESS_20260502.md`](MASTER_PLAN_AND_PROGRESS_20260502.md) | 從 Phase 1 到 Phase 36 的完整研究日誌（5 部 / 55 節 / 243 錨點） |
| [`ESG_永續承諾驗證競賽_2026.md`](ESG_永續承諾驗證競賽_2026.md) | 競賽官方規則 |
| [`[External]_VeriPromiseESG_..._Baseline_Code_ZH.ipynb`](%5BExternal%5D_VeriPromiseESG_2026_ESG_Promise_Verification_Competition_Baseline_Code_ZH.ipynb) | 主辦單位提供的 baseline notebook |
| [`docs/archive/`](docs/archive/) | 早期計畫文件存檔 |

---

## 8. 硬體與環境

### 開發環境

| 項目 | 版本 |
| :-- | :-- |
| OS | Windows 11 / WSL2 Ubuntu 22.04 |
| Python | 3.10 ~ 3.13 |
| PyTorch | 2.2 ~ 2.4 + CUDA 12.x |
| transformers | 4.41 ~ 4.45 |
| GPU | RTX 4060/5060 Laptop 8GB（最低）／ A6000 48GB（推薦） |

### 訓練成本

| 項目 | 估計 |
| :-- | :-- |
| 單 fold 訓練（base，max_len=384，bs=8） | 3~5 分鐘 |
| 單 stem（3 seeds × 5 folds） | 1.5~2.5 GPU 小時 |
| 6 stems 全部訓練 | 約 9~15 GPU 小時（不含 U10 偽標籤產生的兩階段乘數） |
| Phase 36 完整管線（含 U10 教師推論 / U6-pro NLLB 回譯） | 30~40 GPU 小時 |

---

## 9. 常見問題

**Q1. clone 之後為何沒有 `outputs/checkpoints/`？**
A. 全部 6 個 stem 的 best.pt 約 188 GB，超過 GitHub 與 Git LFS 的合理上限。請依 §5 重訓，或洽作者取得權重備份。

**Q2. `MODE="load_oof"` 但筆記本提示 checkpoint missing？**
A. 你需要先完成 §5.2 訓練。或將別處備份的 `outputs/checkpoints/{stem}/seed{S}/fold{F}/best.pt` 與 `oof_probs.npz` 放回原位。

**Q3. 為什麼用 base 而非 large 主幹？**
A. 1k 樣本對 large 容量過度。已驗證 macbert-large 在 8GB GPU 上 bs 必須降至 2，OOF 反而 −0.018。詳見 MASTER_PLAN §A.7。

**Q4. T4 為何單獨用 Focal γ=3？**
A. T4 的 `Misleading` 與 `Not Clear` 類別計數合計 < 80，是分數天花板。Focal Loss 透過 $(1-p_t)^\gamma$ 抑制易類別主導，γ=3 為 Phase 33 grid search 結果。

**Q5. U10 偽標籤如何避免污染驗證集？**
A. 偽標籤只進 fold-train，永遠不寫入 fold-val；OOF 索引嚴格依官方 1,000 筆全域索引拼接。

**Q6. 我的分數比 0.71018 低 / 高 0.005，正常嗎？**
A. 正常。cuDNN 非決定性、torch 浮點累加順序、tokenizer 版本（4.41 ↔ 4.45）皆會引入 ±0.0005 漂移；U10 / U6 重跑可能再帶 ±0.003。

---

## 10. 授權與引用

### 程式碼

本倉儲程式碼（`src/`、`scripts/`、`configs/`、notebook）採 MIT 授權。

### 資料

- `vpesg4k_train_1000 V1.csv` / `.json`：競賽主辦單位釋出，僅供 ESG VeriPromise 2026 競賽研究用。轉用請參照官方授權條款。
- U10 永續報告書（本機 `data/raw/`）：來源為公開資訊觀測站；著作權歸各企業所有，**不隨本倉儲發佈**。

### 引用

若本程式碼對您的研究有幫助，請引用 MASTER_PLAN：

```bibtex
@misc{veripromise2026_phase36,
  title  = {VeriPromise ESG 2026 — Phase 36 SOTA Pipeline (0.71018 OOF)},
  author = {ericchen2023},
  year   = {2026},
  howpublished = {\url{https://github.com/ericchen2023/esg-veripromise-2026}}
}
```

---

> 任何問題、bug、再現失敗，歡迎開 issue 或 pull request。
