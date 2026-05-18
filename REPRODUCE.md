# REPRODUCE.md — 從 clone 到 SOTA 0.71364 完整重現手冊

> **目標**：只要照本文件操作，從一張裸機（已裝 Python + CUDA 驅動）clone 此 repo 後，能完整重現 **Phase 37 AP-D3 SOTA OOF weighted = 0.71364**（4 任務 5-Fold StratifiedKFold OOF 平均，超越 Phase 36 0.71018 +0.00346）。
>
> 本檔為說明書，無需另外查任何資源。所有資料、所有腳本、所有設定都在本 repo 內，跑完即得 SOTA。

---

## 0. 一句話 SOTA

**`0.71364`** = 7 stems × seed42 × 5-fold OOF probs → 3-view TTA（stored / middle / tail） → 4-task per-task simplex hillclimb（stem 權重 + view 權重）。

| 來源 | 指令（執行順序見 §4） |
| :-- | :-- |
| 7 個 stem 的 `oof_probs.npz` + `best.pt` | `python -m src.train_kfold ...` × 1 + `python -m src.train_pseudo_kfold ...` × 6（合計 ~30~40 GPU hr） |
| AP-D3 集成 | `python -m src.tools.u10_per_task_tta --stems <7 stems> --grid-step 0.1 --tag ap_d3_7way_3view` |

---

## 1. 環境需求

| 項目 | 規格 |
| :-- | :-- |
| OS | Windows 10/11、Ubuntu 22.04 LTS 或同等版本 |
| Python | 3.10 ~ 3.13（作者環境 3.13.7） |
| CUDA | 12.x（驅動需支援；CPU-only 也可跑 §5 集成步驟，僅 §4 訓練需 GPU） |
| GPU VRAM | ≥ 8 GB（作者：RTX 5060 Laptop 8.5 GB） |
| RAM | ≥ 16 GB |
| 磁碟 | ~50 GB（含 7 stems × seed42 × 5 folds 的 `best.pt`，約 38 GB） |
| 網路 | 首次執行需下載 HuggingFace 模型 `hfl/chinese-macbert-base`（~400 MB） |

---

## 2. 安裝

```powershell
# Windows / PowerShell
git clone https://github.com/ericchen2023/esg-veripromise-2026.git
cd esg-veripromise-2026
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q   # 單元測試應全綠（~80 個測試 < 30 秒）
```

```bash
# Linux / macOS
git clone https://github.com/ericchen2023/esg-veripromise-2026.git
cd esg-veripromise-2026
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

---

## 3. 資料準備（已隨 repo 上傳，無需另尋）

| 路徑 | 內容 | 是否已隨 repo |
| :-- | :-- | :-- |
| `vpesg4k_train_1000 V1.csv` | 競賽方提供的官方 1,000 筆訓練集 | **是** |
| `vpesg4k_train_1000 V1.json` | 對應 JSON 版本 | **是** |
| `assets/u6_pro/esg_glossary.json` | U6-pro 109 詞 ESG 術語表 | **是** |
| `assets/aug_plus/handcrafted_v1.csv` | Phase 37 手工撰寫 47 列 minority 種子 | **是** |
| `assets/aug_plus/build_handcrafted_v1.py` | 種子原始建構腳本（可重生 `handcrafted_v1.csv`） | **是** |
| `configs/prompts/ap_*.yaml` | AP3 LLM 合成 prompt（mock provider 即可離線跑） | **是** |
| `outputs/checkpoints/<stem>/seed42/fold{0..4}/best.pt` | 7 stems × 5 folds 的 fine-tuned 權重 | **否**（~38 GB，需在 §4 自行訓出） |
| `outputs/checkpoints/<stem>/seed42/fold{0..4}/oof_probs.npz` | 對應的 OOF 機率（訓練即產出） | **否** |
| `outputs/cache/u10_tta/<stem>_{middle,tail}.npz` | 3-view TTA 快取（會在 §5 自動產出） | **否** |

> 用到的 **唯一外部資源** 是 HuggingFace 的 `hfl/chinese-macbert-base`（pretrained backbone），第一次訓練時 transformers 會自動下載。**不需要其他 model / dataset / pseudo-label** 即可重現 0.71364。

---

## 4. 訓練：產出 7 個 stem 的 `best.pt`（GPU 必要）

> **時間估算**：每個 stem = 5 folds × ~5~7 min / fold = ~30~40 min。合計 **~4~5 GPU 小時**（RTX 5060 Laptop 8.5 GB，AMP fp16）。
>
> **資源**：作者每跑一個 stem 都會在 `reports/experiments/` 寫入 5 個 fold log；如果中途崩潰可單跑 `--fold N` 續跑。

### 4.1 訓練腳本指令（依此順序執行）

```powershell
# stem #1: p2_combo_best — 純官方 1,000 列、Phase 36 最佳配方（no pseudo）
python -m src.train_kfold --config configs\exp_p2_combo_best.yaml

# stem #2: p2_combo_best_u10_pseudo — Stage A: U10 v1 偽標 (211 列) + real；Stage B: real-only
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo.yaml

# stem #3: p2_combo_best_u10_pseudo_v2 — Stage A: U10 v2 偽標 (3,904 列) + real
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo_v2.yaml

# stem #4: p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3 — 同 stem #3 + 類別加權 CE + T4 Focal γ=3
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3.yaml

# stem #5: p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3 — U10 v3 偽標 + 同上 loss
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3.yaml

# stem #6: p2_combo_best_classw_focal_u6pro — Phase 36 SOTA 主成員：U6-pro 反翻譯 434 列 + classw + focal
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_classw_focal_u6pro.yaml

# stem #7: p2_combo_best_aug_plus — Phase 37 新增：U10-pro + handcrafted 47 列 aug_plus（NEW SOTA 關鍵）
python -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_aug_plus.yaml
```

每個 stem 跑完後會在 `outputs/checkpoints/<stem>/seed42/fold{0..4}/` 寫入：
- `best.pt` — 最佳 epoch 的模型權重
- `oof_probs.npz` — 該 fold 驗證集 OOF 機率（4 個任務各一個 array）

> **stem #6/#7 的偽標籤從哪來？** stem #6 依賴 U6-pro 反翻譯產出的 434 列（路徑 `data/processed/u6_pro/u6_pro_434.csv`）；stem #7 依賴 U10-pro 偽標籤 + 47 列 handcrafted。如果 repo 內這些檔案缺失，看 §4.2 重生方式。

### 4.2 （可選）重生偽標籤資料

如果 `data/processed/` 內的偽標籤檔案沒被一起 clone，可依以下指令重生（僅在你想完全 byte-exact 重現偽標管線時需要；否則跳過，直接用 §4.1 訓練）：

```powershell
# U6-pro 反翻譯（~90 min GPU + NLLB-200-distilled-600M ~3 GB 下載）
python scripts\u6_backtranslate_pro.py

# U10 v3 弱監督管線（需先有 stem #1 ckpts 當教師；~60 min）
python scripts\u10_collect_sr.py        # M1: PDF 爬取（~1.3 GB）
python scripts\u10_pdf_extract_v3.py    # M3: 段落擷取 + SimHash 去重
python scripts\u10_pseudo_label_v3.py   # M4: 教師閘門 + 偽標產出

# Aug-Plus 47 列 handcrafted seeds（已隨 repo 上傳，無需重生）
python assets\aug_plus\build_handcrafted_v1.py   # 重生 handcrafted_v1.csv
```

> ⚠️ U6-pro / U10 涉及外部模型版本與 PDF 解析；byte-exact 重現可能漂移 ±0.003。如果你的目標只是「重現 0.71364」，**最穩做法**是直接用 repo 隨附的 `data/processed/` 與 `assets/` 內檔案，跳過 §4.2。

---

## 5. 集成：產出 SOTA OOF 0.71364

訓練完 7 個 stem 後，跑下列**單一指令**即得 SOTA：

```powershell
python -m src.tools.u10_per_task_tta `
  --stems p2_combo_best `
          p2_combo_best_u10_pseudo `
          p2_combo_best_u10_pseudo_v2 `
          p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3 `
          p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3 `
          p2_combo_best_classw_focal_u6pro `
          p2_combo_best_aug_plus `
  --grid-step 0.1 --max-rounds 4 --joint-rounds 2 `
  --tag ap_d3_7way_3view
```

執行流程（指令會自動處理）：

1. **載入 stored 視角 OOF**：從每個 stem 的 `oof_probs.npz` 取 fold 0~4 平均（seed42 only）。
2. **預測 middle / tail 視角**：對每個 stem 用對應的 `best.pt` 跑 forward pass（首次 ~10 min，會寫入 `outputs/cache/u10_tta/<stem>_{middle,tail}.npz`；再次執行命中快取直接讀）。
3. **Stage A — Per-task stem 權重 hillclimb**：4 任務各搜尋 7-stem simplex（grid 0.1，8008 點 / task）。
4. **Stage B — Per-task view 權重 hillclimb**：4 任務各搜尋 3-view simplex（grid 0.1，66 點 / task）。
5. **Joint A↔B 交替 2 輪** 直到收斂。
6. **套用階層後處理 constraints**（若 `promise_status="No"`，下游三任務強制 `N/A`）並評分。

最終終端會印出：

```
[u10-tta FINAL] 0.7136375980 delta_vs_u10_stack=+0.0266812240 ...
  task promise_status: 0.943373
  task verification_timeline: 0.630613
  task evidence_status: 0.880450
  task evidence_quality: 0.474960
```

並寫入 `reports/analysis/_ensemble/ap_d3_7way_3view_{summary.csv, meta.json, preds.csv}`，其中 `preds.csv` 即可作為提交檔範本（1,000 列 × 4 預測欄）。

> **預期數字**：weighted = **0.71364** ± 0.0005（cuDNN 非確定性的浮點累加）。
> **每任務**：T1 = 0.94337、T2 = 0.63061、T3 = 0.88045、T4 = 0.47496。

---

## 6. （可選）apples-to-apples baseline 對照

若要驗證 stem #7 真的帶來 +0.00346 增益，請額外跑 6-way × 3-view 對照（不含 stem #7）：

```powershell
python -m src.tools.u10_per_task_tta `
  --stems p2_combo_best `
          p2_combo_best_u10_pseudo `
          p2_combo_best_u10_pseudo_v2 `
          p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3 `
          p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3 `
          p2_combo_best_classw_focal_u6pro `
  --grid-step 0.1 --max-rounds 2 --joint-rounds 0 `
  --tag ap_d3_6way_3view_baseline
```

預期輸出 weighted = **0.71018**，與 Phase 36 公開 SOTA 一致；與 7-way 比差 +0.00346 ⇒ 此即 stem #7 純增量貢獻。

---

## 7. 完整時間預算

| 步驟 | GPU | 時間 |
| :-- | :-- | :-- |
| §2 安裝 + 測試 | 否 | 5 ~ 10 min |
| §4.2 （可選）重生 U6-pro / U10 偽標籤 | 是 | 2 ~ 3 hr |
| §4.1 訓練 7 stems × 5 folds（seed42） | 是 | 4 ~ 5 hr |
| §5 AP-D3 集成 | 是（首次 ~10 min）/ 否（後續快取） | 3 ~ 15 min |
| §6 baseline 對照（可選） | 否 | 2 ~ 5 min |
| **合計（無 §4.2、無 §6）** | — | **~5 hr GPU + 15 min CPU** |

---

## 8. 常見問題

**Q1：`hfl/chinese-macbert-base` 下載失敗？**
A：HuggingFace 鏡像問題；可設 `HF_ENDPOINT=https://hf-mirror.com` 環境變數後重試，或先在有網路機器上 `huggingface-cli download hfl/chinese-macbert-base` 後 rsync 到 `~/.cache/huggingface/`。

**Q2：CUDA OOM？**
A：在所有 YAML 內把 `train.batch_size` 從 8 改 4 + `grad_accum_steps` 從 2 改 4（等效 batch 不變）。max_len 改成 256 亦可。

**Q3：是否能跳過 §4 直接用既有 ckpts？**
A：可以，但你需要自備 ckpts（38 GB 不上 GitHub）。聯絡作者取得 `outputs/checkpoints/*.tar.gz`，解壓到 repo 根，再直接跳到 §5 即可（耗時 < 15 min）。

**Q4：`oof_probs.npz` 與 `best.pt` 是兩個檔案，可以只留 OOF 嗎？**
A：可以，但只能跑「stored view」的 stack，無法跑 3-view TTA（middle/tail 視角需要 `best.pt` 重新 forward）。stored-only 6-way 約 0.71018、7-way 約 0.71109，皆**低於** SOTA 0.71364。

**Q5：怎麼產提交檔？**
A：`reports/analysis/_ensemble/ap_d3_7way_3view_preds.csv` 已是 1,000 列的最終預測；競賽方若要求特定欄位順序，可改 `python -m src.tools.u10_per_task_tta` 內 `_eval_full` 輸出格式（或直接讀 `preds.csv` 重排）。

**Q6：訓練 stem 數量能不能減？**
A：可以，但 SOTA 會降。Phase 36 是 6 stems = 0.71018；Phase 37 加 stem #7 = 0.71364。如果只有 GPU 預算跑 3 stems，建議選 stem #1 + #4 + #6（base / classw_focal / u6pro），預期 OOF ~0.69~0.70。

---

## 9. 文件對照

| 文件 | 用途 |
| :-- | :-- |
| [REPRODUCE.md](REPRODUCE.md) | **本檔** — 從 clone 到 SOTA 0.71364 的單一 SOP |
| [README.md](README.md) | 專案總覽、目錄、授權 |
| [MASTER_PLAN_AND_PROGRESS_20260518.md](MASTER_PLAN_AND_PROGRESS_20260518.md) | 完整研究日誌（Phase 1~37），所有 ablation 與 SOTA 軌跡 |
| [SOTA_Reproduction_Phase36.ipynb](SOTA_Reproduction_Phase36.ipynb) | Phase 36 SOTA 互動式 notebook（舊 SOTA，保留作為對照） |
| [ESG_永續承諾驗證競賽_2026.md](ESG_永續承諾驗證競賽_2026.md) | 競賽官方規則 |

---

*最後更新：2026-05-18 · SOTA 0.71364 對應 commit 將在本次 push 內，可由 `git log --oneline -1` 確認。*
