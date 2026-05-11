# ESG 永續承諾驗證競賽 2026 — 進度審視報告

**撰寫日期**：2026-05-01（Day 21 結束）
**目的**：站在第三方角度，回答三個問題：
1. 每一次「下一步」是否合理？根據什麼證據決定？
2. 目前所有已完成步驟、輸出是否完整？
3. 整理已做 / 未做、推薦下一步（並確認是否已做過）。

> 本文是「面向沒看過完整訓練計畫書的人」的精煉版總覽。完整實作細節仍在 [TRAINING_PLAN_FRESH_20260428.md](TRAINING_PLAN_FRESH_20260428.md)。

---

## 0. 一句話總結

13 個 Phase、21 天訓練、22 個單模、4 代 ensemble pool（3→16-way）、11 代 hillclimb（v1→v11）。**目前最佳分數 OOF 5-Fold weighted = 0.68770**（Phase 13 joint hillclimb v11，2026-05-01 達成），距 0.69 護城線剩 **0.00230**。下一階段必須跳脫 combo_v3 變體，引入結構性新方向（不同 backbone family / 外部 ESG 偽標）才有破護城的可能。

---

## 1. 競賽基礎資訊（讓新讀者一次看懂）

### 1.1 任務

對每段 ESG 報告書文本，同時預測四個欄位：

| 子任務 | 名稱 | 類別數 | 評分指標 | 權重 |
| :--: | :-- | :--: | :-- | :--: |
| **T1** | promise_status（是否承諾）| 2 | F1 (positive=Yes) | **0.20** |
| **T2** | verification_timeline（驗證時程）| 5 | Macro-F1 | **0.15** |
| **T3** | evidence_status（是否有證據）| 3 | F1 (positive=Yes) | **0.30** |
| **T4** | evidence_quality（證據品質）| 4 | Macro-F1 | **0.35** |

最終分數 $\mathcal{S} = 0.20 \cdot \mathrm{T1} + 0.15 \cdot \mathrm{T2} + 0.30 \cdot \mathrm{T3} + 0.35 \cdot \mathrm{T4}$。

### 1.2 條件式約束（極重要）

- **T1=No → T2 / T3 / T4 全部必須=N/A**（且 promise_string、evidence_string 必須空字串）。
- **T3=No → T4 必須=N/A**（且 evidence_string 空字串）。

推論時必須以 `apply_constraints_batch` 強制覆寫，否則 T2/T4 macro-F1 會被無效預測污染。**這個約束是 Phase 12/13 兩次重大突破（v10 失敗、v11 成功）的關鍵理論根源。**

### 1.3 資料

- 訓練集 1,000 筆（已釋出）
- 驗證集：2026-06-03 釋出
- 測試集：2026-06-10 釋出，6/17 截止上傳
- 每日提交上限 3 次

### 1.4 硬體與環境

- 單卡 RTX 5060 Laptop **8GB** CUDA
- Windows + PowerShell 5.1 + Python 3.13.7 + torch 2.11.0+cu128 + transformers 5.6.2

### 1.5 評分管線

5-Fold StratifiedKFold（依 promise_status + evidence_status 分層）→ AdamW + cosine LR + 10% warmup → AMP fp16 → 5 epochs → early_stop_patience=2 → 每 fold 存 OOF float16 機率 → ensemble 階段做 per-task 或 joint hillclimb。

---

## 2. 13 個 Phase 一覽（每一步「為什麼這麼做」）

> 讀法：每行的「決策依據」說明上一個 Phase 結束後，**為什麼**選擇這個下一步，而不是其他選項。

| Phase | 名稱 | OOF | Δ | 決策依據（為什麼是下一步）|
| :--: | :-- | :--: | :--: | :-- |
| 1 | Baseline 復現（macbert-base + 5-Fold）| 0.6415 | — | 唯一可信錨點。先確認本地評分管線 = 官方評分。 |
| 2 | 單模調優（lr / max_len / pooling / class_weight）| 0.66734 | +0.026 | Phase 1 確認管線後，掃單模超參找最甜蜜點。發現 lr=3e-5 / max_len=384 / **關閉 class_weight** 為三大贏家。 |
| 3 | 升規模 macbert-large | 0.66439 | −0.003 | Phase 2 顯示 base 模型已收斂；想試大模型能否再進。**結果負面**（單模反退），但保留作 ensemble 輸入。**X5 禁區教訓**：大模型在小資料 + 8GB 顯存環境不適合作主力。 |
| 4 | 多骨幹 ensemble（base + large 1.5:1）| 0.67478 | +0.010 | Phase 3 雖負面，但「不同 backbone 的錯誤模式不一樣」這一假設用 ensemble 驗證。果然 +0.010。 |
| 5 | Wave A/B Ablation + Combo v2 + 3-way Hillclimb | 0.67954 | +0.005 | Phase 4 確認 ensemble 路線可行 → 系統補完 Phase 2 沒掃完的 16 組變體（FGM、Focal、LLRD 等），勝出者疊成 combo_v2 = base + Focal-T4 + FGM。並引入 per-task hillclimb（每個任務獨立選權重）。 |
| 6 | Wave C 增強 + 7-way Hillclimb | 0.68206 | +0.003 | Phase 5 顯示「弱模型也能在 ensemble 中加分」→ 故意做 4 個「單模都會輸」的變體（token aug、R-Drop、MSD、TTA-train），全部塞進 pool 看 hillclimb 怎麼選。**重要規則發現**：「diversity beats individual strength in per-task hillclimb」。 |
| 7 | Combo v3 + 8-way Hillclimb | 0.68394 | +0.002 | Phase 6 的 R-Drop 與 MSD 是「最不弱」的兩個 → 試試疊到 combo_v2 上做 combo_v3。成功，單模 0.67121（base 史上最高）。 |
| 8 | Combo v3 多 seed 拆分 9-way | 0.68440 | +0.0005 | combo_v3 多訓 2 個 seed 後想用 3-seed 平均；但「直接替換」會輸（X4 教訓）。改成 **拆兩員**進池：seed=42（peaky）+ 3-seed 平均（smooth），讓 hillclimb 自己選。**規則：variance reduction 永遠用「並列」不用「替換」**。 |
| 9 | Sprint A 換骨幹 + 11-way v7b | 0.68558 | +0.001 | 8-way pool 全是 macbert-base 變體，想引入 backbone 多樣性。試 3 個：roberta-wwm（入池）、bert-base-chinese（入池）、xlm-roberta-base（拒，0.61 < 0.65 門檻 → X6 禁區）。新成員加入後，**warm-start 不能用均勻隨機搜索** → 改 biased search 強迫每次採樣含新成員。 |
| 10 | Sprint B / T6 v1 時間 token prefix + 12-way v8 | 0.68669 | +0.001 | 觀察到 T2 macro 仍卡 0.49，是最大短板；嘗試「regex 抽年份 → bucket 前綴注入」做 task-specific feature engineering。**單模 T2 沒進步**，但 ensemble 中 T1/T3 反取得權重 2.0（意外 diversity 效應）。 |
| 11 | Sprint B / T6 v2 專屬 bucket token + 13-way v9 | **0.68683** | +0.0001 | T6 v1 失敗推測是「prefix 被 tokenizer 拆碎」；改用 5 個專屬 added-vocab token。**結果單模 T2 反退步 0.0171**（5 個低頻新嵌入學不起來 → Phase 10/11 合併教訓：T6 路線正式關閉）。 |
| 12 | Sprint B / T7 Focal-T4 γ=3.0 + 14-way v10 | 0.68660 | **−0.00023** | T4 macro 仍弱（0.45），試提高 Focal γ。**單模成功**：p7_focal_g3 OOF=0.67416（史上第二）、T4=0.4481（單模史上最大 +0.0237 增益）。**但 14-way v10 ensemble 失敗** → 揭露關鍵問題：per-task hillclimb 各自優化會被 `apply_constraints_batch` 抵銷（**constraint coupling**）。 |
| 13 | Sprint B 收尾 / U2 EMA + U4 LS + U7 Joint Hillclimb v11 | **0.68770** | **+0.00087** | 三事齊發：(a) p8 EMA-995 試解 X1 失敗 → **再失敗 0.63010**（永久封禁 EMA without warm-start）；(b) p9 per-task LS(T1/T3) 修正 X2 → 單模 0.67334 OK 但 ensemble 0 權重；(c) **U7 Joint Hillclimb v11**：直接以 `weighted_score(apply_constraints_batch(·))` 為目標 → 4 次 accept 全在 T3，T4 +0.0018 為**約束連動效應**自動取得 → **破 SOTA**。 |
| **14** | **Sprint C 結構性多樣化（待辦）** | 目標 ≥ 0.69 | +0.002 ~ +0.005 | 16-way pool 已收斂（30000 iter 僅 4 accept）→ combo_v3 變體已榨乾，必須引入結構性新方向。詳見 §5。 |

---

## 3. 已完成步驟與輸出完整性檢查

### 3.1 訓練產物（22 個單模，每個 5-Fold OOF 機率張量）

| 編號 | 實驗 | OOF | 狀態 | 入 v11 pool |
| :-- | :-- | :--: | :--: | :--: |
| p1 | baseline | 0.6415 | OK | — |
| p2 | combo_best | 0.66734 | OK | v (idx 1) |
| p2 | combo_v2 | ~0.668 | OK | v (idx 0) |
| p2 | combo_v3 (seed=42) | 0.67121 | OK | v (idx 7) |
| p2 | combo_v3 (3-seed avg) | 0.66854 | OK | v (idx 8) |
| p2ab | aug_mask10 | ~0.665 | OK | v (idx 3) |
| p2ac | aug_mix | ~0.666 | OK | v (idx 4) |
| p2ad | rdrop05 | ~0.669 | OK | v (idx 5) |
| p2ae | msd5 | ~0.670 | OK | v (idx 6) |
| p3 | large_lr2e5 | 0.66439 | OK | v (idx 2) |
| p4a | roberta_wwm_base | 0.66626 | OK | v (idx 9) |
| p4b | bert_base_chinese | 0.67355 | OK | v (idx 10) |
| p4c | xlm_roberta_base | 0.61269 | 拒（X6）| — |
| p5 | t6_time_token | 0.66231 | OK | v (idx 11) |
| p6 | t6v2_bucket_tok | 0.66606 | OK | v (idx 12) |
| p7 | focal_g3 | **0.67416** (史上第二高) | OK | v (idx 13) |
| p8 | ema995 | 0.63010 | **FAIL**（保留 ckpt 但 v11 0 權重）| v (idx 14, w=0) |
| p9 | ls_t1t3 | 0.67334 | OK | v (idx 15, w=0) |

完整性：**v 22 員實驗的 fold0~fold4 best.pt + OOF 機率張量皆齊**。`reports/experiments/<exp>/score_summary.csv` 與 `<exp>/oof_*.npy` 都在。

### 3.2 Hillclimb 產物（11 代）

| 版本 | pool | 目標函數 | 結果 | Status |
| :--: | :--: | :-- | :--: | :--: |
| v1 | 3-way | per-task F1 | 0.67954 | 已完成 |
| v3 | 7-way | per-task F1 | 0.68206 | 已完成 |
| v4 | 8-way | per-task F1 | 0.68394 | 已完成 |
| v6 | 9-way | per-task F1 | 0.68440 | 已完成 |
| v7b | 11-way | per-task F1 (biased) | 0.68558 | 已完成 |
| v8 | 12-way | per-task F1 | 0.68669 | 已完成 |
| v9 | 13-way | per-task F1 | 0.68683 | 已完成（前任 SOTA）|
| v10 | 14-way | per-task F1 | 0.68660 | 已完成（**失敗** — constraint coupling）|
| **v11** | **16-way** | **post-constraint joint score** | **0.68770** | **目前 SOTA** |

完整性：**v 每代 hillclimb 的 winner weights / per-task F1 / 預測表 / log 都在 `reports/analysis/_ensemble/`。**

### 3.3 文件產物

- [TRAINING_PLAN_FRESH_20260428.md](TRAINING_PLAN_FRESH_20260428.md) — 主計畫書（13 Phase 完整、§17.1 ~ §17.13、§18.1 D1~D20、§18.3 X1~X6 禁區、§19 推薦下一步、§19.1 規則確認）。
- [README.md](README.md) — Phase 表 + SOTA 段落已更新到 Phase 13。
- /memories/ml_ensemble_lessons.md — 教訓累積（最新含 Phase 13 三條）。
- /memories/session/esg2026_phase8_state.md — session 狀態（含 Phase 13 完整數據）。

完整性：**v 4 份文件均同步至 SOTA = 0.68770。**

---

## 4. 已做 / 未做 詳細對照

### 4.1 已做（請列為「不再嘗試」或「已產品化」）

#### A. 已產品化（仍在 v11 pool 中）
- 所有上述 16 員 ensemble pool 成員。

#### B. 已嘗試 + 失敗 + 永久禁區（X 系列 + Phase 13 新增）
| 代號 | 失敗動作 | 結果 | 教訓 |
| :--: | :-- | :--: | :-- |
| X1 | EMA decay=0.999 短訓練 | 0.47291 | EMA shadow 從 step 0 開始追，短訓無法收斂 |
| **X1'** | **EMA decay=0.995 短訓練（Phase 13 p8）**| **0.63010** | **僅縮短半衰期不夠，必須加 warm-start** |
| X2 | Uniform LS=0.05 全 task | 0.49 | T4 少類 `Misleading` (support=1) 被推平 |
| **X2'** | **per-task LS T1/T3=0.05（Phase 13 p9）**| 單模 0.67334 OK，**但 ensemble 0 權重** | LS 變體與 combo_v3 太相似，無 diversity |
| X3 | LLRD 0.95 base 模型 | −0.005 | base 12 層太淺，LLRD 反讓底層學不動 |
| X4 | combo_v3 multi-seed 直接替換 | −0.002 | 變異減少破壞 peaky 任務的判決訊號 |
| X5 | macbert-large 作單模主力 | −0.003 | 8GB 顯存 + 1k 樣本不適合大模 |
| X6 | xlm-roberta-base 中文 | 0.61 | 多語模型對純中文劣勢 |
| **新** | **任何 combo_v3 細部變體** | — | Phase 13 v11 證明 16-way 已收斂，再榨無意義 |

#### C. 已嘗試 + 成功 + 已內化進程式碼
- FGM 對抗訓練（eps=1.0）
- Focal Loss on T4（γ=2.0）
- R-Drop（α=0.5）
- Multi-Sample Dropout（K=5）
- Two-stage hillclimb（粗網格 + 隨機局部精修）
- Biased random search（warm-start 含新成員時必用）
- Joint hillclimb（post-constraint objective）

### 4.2 未做（Backlog）

| 代號 | 動作 | 預期增益 | 狀態 | 備註 |
| :--: | :-- | :--: | :--: | :-- |
| **N1** | macbert-large 入 ensemble pool | +0.001 ~ +0.003 | **未做、立即可動** | 核心：用 grad_accum=4 + max_len=192 在 8GB 跑 large；不要當主力，只要 ≥ 0.66 入池即可 |
| **U1** | TTA（推論時擾動）| +0.001 | **未做、立即可動** | 截斷位置 / 同義改寫 / segment shuffle 三視角平均 |
| **U10** | 外部 ESG 報告偽標擴增 | **+0.005 ~ +0.015** | **未做、立即可動**（規則允許，見 §6 Q1） | 全項目剩餘空間最大的單一動作 |
| X1' | EMA with warm-start 重啟 | +0.002 單模 | 未做 | 需重構 `_EMA(start_step=...)` |
| U3 | SWA（最後 K epoch 權重平均）| +0.001 ~ +0.002 | 未做 | 比 EMA 安全（無 shadow） |
| U5 | T4 class-balanced re-sampling | T4 +0.02 | 未做 | 寫 sampler |
| U6 | 回譯資料增強對 T2/T4 | T2 +0.02 | 未做 | 需翻譯 API 或 NLLB |
| U10b | 6/03 釋出官方 dev 後做 pseudo-labeling | +0.003 | **6/03 後可做** | 真正的「校準分布偏移」用 |
| U12 | OOF vs val gap 分析 | 校準工具 | **6/03 後可做** | 用於診斷 hillclimb 是否過擬 OOF |
| U11 | GroupKFold by company | 驗證工具 | **6/03 後可做** | 跨公司泛化 sanity check |
| U9 | Qwen 7B LoRA（外部硬體）| +0.005? | 可選 | 8GB 跑不動，需雲端 |

---

## 5. 推薦下一步（並確認是否做過）

### 5.1 立即可動的三件事（依 ROI 排序）

| 優先 | 動作 | 預期 Δ | 為什麼是下一步 | 是否做過 |
| :--: | :-- | :--: | :-- | :--: |
| **1** | **U10 外部 ESG 偽標擴增** | **+0.005 ~ +0.015** | (a) §6 Q1 確認規則允許；(b) 現有 1k 樣本是最大瓶頸（T2/T4 少類 support=1）；(c) v11 ensemble 信心 ≥ 0.9 的偽標噪聲低 | [NG] 未做 |
| **2** | **N1 macbert-large 入池** | +0.001 ~ +0.003 | 16-way pool 已證明收斂於 combo_v3 同源變體；唯一突破方法是「不同 backbone family」 | [NG] 未做（Phase 3 作單模失敗 ≠ 作 ensemble 成員失敗）|
| **3** | **U1 TTA 推論擴增** | +0.001 | 不需重訓，最便宜的免費午餐 | [NG] 未做 |

### 5.2 為何不再做：曾經考慮但證據不足
- [NG] U2 EMA-995 — Phase 13 已試，0.63010 失敗。（除非先做 X1' warm-start 重構）
- [NG] U4 LS(T1/T3) — Phase 13 已試，單模 OK 但 ensemble 0 權重。
- [NG] T6 路線（time token / bucket vocab）— Phase 10/11 兩次失敗。
- [NG] T7 Focal-T4 γ=3.0 直接擴大 — Phase 12 v10 證明單模強 ≠ ensemble 強。
- [NG] Pseudo-label on 官方 dev — 6/03 才釋出。
- [NG] 任何 hillclimb v12+ on 16-way pool — 30000 iter 僅 4 accept，已飽和。

### 5.3 6/03 才能做的事
- U10b（官方 dev 偽標）、U12（gap 分析）、U11（GroupKFold sanity check）、提交策略。

---

## 6. 用戶兩個額外問題的精確回答

### Q1：為什麼需要等驗證集才能提高上限？規則有說可以自己找資料嗎？

**答**：**之前的「等驗證集」是工程選擇，不是規則限制。規則完全允許使用外部資料。**

逐字檢視 [ESG_永續承諾驗證競賽_2026.md §八](ESG_永續承諾驗證競賽_2026.md)，禁止項目只有：

1. 抄襲、作弊、詐欺
2. 侵害他人智慧財產權
3. 攻擊 leaderboard 系統
4. **對測試集做人工標註或修正**（預測必須由程式自動產生）
5. 私下共享程式與特徵值

**沒有任何條文禁止：**
- 使用外部訓練資料（包括公開 ESG 報告書、白皮書、新聞）
- 使用外部預訓練模型（HF 上各種中文 PLM 都可）
- 對外部公開文本做偽標再加入訓練

**之前說「等 6/03」的真實原因**：原計畫想用「官方驗證集」做 pseudo-label，因為它的分布最接近測試集。但這不是必需 — **可以即刻啟動「外部公開 ESG 報告書偽標」**，雖然分布偏移風險較高，但用 confidence threshold ≥ 0.9 + multi-round consistency check 可顯著緩解。

**合法且可立即取得的外部資料源**：
- 台灣公開資訊觀測站「永續報告書專區」（mops.twse.com.tw）
- 各上市櫃公司官網「投資人關係」/ ESG / 永續發展頁
- GRI、TCFD、ISSB 的中文公開白皮書

**操作流程建議**：
```
1. 爬 PDF (~50~200 份)
2. pdfplumber 抽段落（保留 100~500 字段落）
3. 過濾：含 ESG 關鍵詞 + 句長 30~500
4. v11 ensemble 推論四任務機率
5. 篩 max-prob ≥ 0.9 的「高信心偽標」（預估 30%~50% 留下）
6. 二段式訓練：
   階段 A：偽標 + 真標混合，2 epochs
   階段 B：純真標 finetune，3 epochs
7. 重新跑 v12 hillclimb
```

預期增益 +0.005 ~ +0.015（**目前唯一仍有大幅空間的單一動作**）。

> 詳見 [TRAINING_PLAN_FRESH_20260428.md §19.1](TRAINING_PLAN_FRESH_20260428.md)。

### Q2：現在的最佳參數模型配置（給同學的 Excel）

已產出獨立 Excel 檔：`final_summary/BEST_MODEL_CONFIG_FOR_TEAMMATE.xlsx`

包含 4 張工作表：
1. **總覽** — SOTA 分數、pool 組成、檔案位置
2. **單模超參** — combo_v3（最佳單模骨幹）的全部超參
3. **Pool 16 員清單** — 每員的 backbone / tweak / 單模 OOF
4. **Joint Hillclimb v11 權重表** — 每個任務、每個成員的最終權重

---

## 7. 結語：本階段所有「下一步」的合理性審視

| Phase 切換 | 上一步發現 | 下一步動作 | 合理性審視 |
| :--: | :-- | :-- | :-- |
| 1→2 | 復現成功 | 掃單模超參 | [OK] 必要先擇單模甜蜜點 |
| 2→3 | base 收斂 | 試大模 | [OK] 標準做法，雖負面但保留 |
| 3→4 | 大模單模負面 | 兩骨幹 ensemble | [OK] 「不同 backbone 錯誤分布」假說驗證 |
| 4→5 | ensemble 可行 | 系統補完 ablation + per-task hillclimb | [OK] 從「直覺加減」進化到「資料驅動選擇」|
| 5→6 | 弱模也可加分 | 故意做弱模測 diversity | [OK] 大膽假設驗證 |
| 6→7 | 兩個最不弱可疊 | 疊成 combo_v3 | [OK] 證據驅動 |
| 7→8 | seed 變異需引入 | 拆兩員不替換 | [OK] X4 教訓避坑 |
| 8→9 | base 變體飽和 | 換 backbone | [OK] 必要但 xlm-r 應預判到（事後檢討：可選 PLM 前先看 token coverage）|
| 9→10 | T2 仍弱 | 時間 token | [OK] 任務針對性嘗試 |
| 10→11 | T6 v1 prefix 失敗 | 換專屬 vocab | [注意] 證據不足就上 — 5 個低頻新嵌入學不起來是可預判的（事後新教訓）|
| 11→12 | T4 仍弱 | Focal γ↑ | [OK] 順理成章 |
| 12→13 | v10 失敗揭露 coupling | 三批次（含 Joint Hillclimb）| [OK] **本次最佳決策** — 一次解 (i) X1 重試確認、(ii) X2 修正版確認、(iii) coupling 突破，三鳥一石 |
| **13→14** | **v11 收斂、外部資料規則允許** | **U10 外部偽標 + N1 large 入池 + U1 TTA** | [OK] **建議路徑** — 詳見 §5.1 |

**結論**：除 Phase 11 (T6 v2) 是「執著於失敗 idea」的次優決策外，其他 12 次「下一步」都有清楚的證據支撐，並嚴格遵守 §18.3 禁區與 X 教訓。Phase 13 的「批次三事」更是教科書級的 ROI 安排。
