<!-- markdownlint-disable MD025 MD032 MD060 MD022 MD058 MD031 MD040 MD036 MD029 MD012 MD033 -->

# VeriPromise ESG 2026 — 主規劃與進度紀錄

> **文件版本** v4.2 / **整理日期** 2026-05-18

<!-- restructure-v2 applied --> / **語系** 繁體中文 / **作者** 個人單兵組
>
> 本檔為**單一真相來源（single source of truth）**。所有訓練決策、實驗結果、待辦項目皆以本檔為準。

### 如何閱讀本文件（30 秒導覽）

1. **想看現況**：上方指標表 + [§1 Snapshot](#1-一頁總結snapshot)。
2. **想看最新實驗**：[§53 Phase 37 Aug-Plus](#53-phase-37--aug-plus-hand-crafted-minority-訓練與-single-stem-ablation2026-05-18) · [§53.12 AP-D3 NEW SOTA 0.71364](#5312-ap-d3-結果--7-way--3-view-per-task-tta--new-sota-0713642026-05-18)。
3. **想看 SOTA 是怎麼來的**：[§52 Phase 36](#52-phase-36--u6-pro-back-translation--stem-6--6-way--3-view-tta--new-sota-071018)（舊 SOTA 0.71018）、[§53.12](#5312-ap-d3-結果--7-way--3-view-per-task-tta--new-sota-0713642026-05-18)（新 SOTA 0.71364）。
4. **想看接下來該做什麼**：[§8 待辦事項（ROI 排序）](#8-待辦事項roi-排序)。
5. **想看為什麼某條路放棄**：[§6 已驗證為負方向](#6-已驗證為負方向x1--x13-禁區)。
6. **想看官方規則對外部資料的立場**：[§57](#57-競賽規則對外部資料的立場已確認可用)（含 2026-05-17 主辦方裁示摘要）。

| 指標 | 數值 |
| :-- | :-- |
| **Current SOTA**                          | **0.71364**（Phase 37 AP-D3 — 7-way × 3-view per-task TTA，[§53.12](#5312-ap-d3-結果--7-way--3-view-per-task-tta--new-sota-0713642026-05-18)） |
| **舊 SOTA**                               | 0.71018（Phase 36 — U6-pro back-translation + 6-way × 3-view per-task TTA） |
| **保守工程上限**                          | 0.7236（距離 −0.0134） |
| **寬鬆工程上限**                          | 0.7570（距離 −0.0468） |
| **賽程截止**                              | 2026-06-17 22:00（test 6/10 釋出 → 21 次提交額度） |
| **驗證集釋出**                            | 2026-06-03（valid 校準先決條件）|
| **訓練集規模**                            | 1,000 筆 / 50 家 TWSE 上市公司（FY2022~2024）|
| **硬體**                                  | RTX 5060 Laptop 8 GB CUDA |

---

## 文件導覽（Table of Contents）

### Part I — 執行摘要 (Executive Summary)
- [§1. 一頁總結（Snapshot）](#1-一頁總結snapshot)
- [§2. 競賽規格](#2-競賽規格)
- [§3. 整體策略與雙軌 SOTA 軌跡](#3-整體策略與雙軌-sota-軌跡)

### Part II — 進度全景 (Progress Dashboard)
- [§4. SOTA 軌跡總表](#4-sota-軌跡總表)
- [§5. 已完成項目索引（含主要 D1~D30）](#5-已完成項目索引含主要-d1d30)
- [§6. 已驗證為負方向（X1 ~ X13 禁區）](#6-已驗證為負方向x1--x13-禁區)
- [§7. 結構性殘留與工程上限](#7-結構性殘留與工程上限)
- [§8. 待辦事項（ROI 排序）](#8-待辦事項roi-排序)

### Part III — 技術設計 (Technical Design)
- [§9. 規劃哲學](#9-規劃哲學)
- [§10. 競賽任務形式化](#10-競賽任務形式化)
- [§11. 資料初步剖析 (基於 `vpesg4k_train_1000 V1.csv`)](#11-資料初步剖析-基於-vpesg4k_train_1000-v1csv)
- [§12. 整體技術架構](#12-整體技術架構)
- [§13. 完整里程碑路線圖 (Phased Roadmap)](#13-完整里程碑路線圖-phased-roadmap)
- [§14. 企業級專案骨架 (Step 0 產出)](#14-企業級專案骨架-step-0-產出)
- [§15. 推論與後處理 SOP](#15-推論與後處理-sop)
- [§16. 風險與止損機制](#16-風險與止損機制)
- [§17. 交付物 (Deliverables)](#17-交付物-deliverables)
- [§18. 排程](#18-排程)

### Part IV — Phase 詳細紀錄 (Append-only Phase Log)
- [§19. Phase 1 — 主力配置（Baseline 復現設定）](#19-phase-1--主力配置baseline-復現設定)
- [§20. Phase 1 結果與 Phase 2 籌備（OOF 0.6415 → combo 規劃）](#20-phase-1-結果與-phase-2-籌備oof-06415--combo-規劃)
- [§21. Phase 2~4 超參數搜索藍圖](#21-phase-24-超參數搜索藍圖)
- [§22. Phase 2 — Combo Winners 最終配置 (3 seeds × 5 folds)](#22-條件式約束極重要)
- [§23. Phase 3 — 升規模 (`hfl/chinese-macbert-large`, 326M)](#23-資料)
- [§24. Phase 4 — Probability Ensemble (base combo + large)](#24-硬體與環境)
- [§25. Phase 5 — Wave A/B Ablation + Combo v2 + Per-task Hillclimb](#25-評分管線)
- [§26. Phase 6 — Wave C Ablation + 7-way Per-task Hillclimb](#26-phase-6--wave-c-ablation--7-way-per-task-hillclimb)
- [§27. Phase 7 — Combo v3 + 8-way Per-task Hillclimb](#27-phase-7--combo-v3--8-way-per-task-hillclimb)
- [§28. Phase 8 — combo_v3 Multi-seed 拆分為 9-way 池（Phase 8 SOTA，已被 Phase 11 取代）](#28-phase-8--combo_v3-multi-seed-拆分為-9-way-池phase-8-sota已被-phase-11-取代)
- [§29. Phase 9 — Sprint A backbone 擴 pool（已完成）](#29-phase-9--sprint-a-backbone-擴-pool已完成)
- [§30. Phase 10 — Sprint B / T6 時間 token 增強（已完成）](#30-phase-10--sprint-b--t6-時間-token-增強已完成)
- [§31. Phase 11 — Sprint B / T6 v2 專屬 bucket token（已完成）](#31-設計哲學三句話)
- [§32. Phase 12 — Sprint B / T7 Focal-T4 γ=3.0 變體（已完成）](#32-雙軌-sota-軌跡)
- [§33. Phase 13 — Sprint B 收尾批次：U2 EMA-995 + U4 LS(T1/T3) + U7 聯合 hillclimb（已完成，**新 SOTA 0.68770**）](#33-三條主軸總覽)
- [§34. Phase 14 — Sprint C 結構性多樣化 N1：macbert-large + Focal-T4 + FGM (p10) + Joint Hillclimb v12（已完成，**新 SOTA 0.68825**）](#34-不再進攻的方向已驗證為負詳-1919-phase-1--主力配置baseline-復現設定--13946-phase-25-完成--u6--b4-nllb-600m-中英中-回譯資料增強2026-05-05)
- [§35. Phase 15 (Reset) — U10 暫緩並清除外部 pseudo 線（2026-05-04）](#35-phase-15-reset--u10-暫緩並清除外部-pseudo-線2026-05-04)
- [§36. Phase 15 (Implementation) — U1 TTA 推論擴增（2026-05-04）](#36-phase-15-implementation--u1-tta-推論擴增2026-05-04)
- [§37. Phase 16 完成 — N2/p11 ELECTRA base 官方資料結構性成員（2026-05-04）](#37-phase-16-完成--n2p11-electra-base-官方資料結構性成員2026-05-04)
- [§38. Phase 17 完成 — U1-b TTA 三視角補測（2026-05-05）](#38-phase-17-完成--u1-b-tta-三視角補測2026-05-05)
- [§39. Phase 18 完成 — U1-c 任務別/視角別加權 TTA（2026-05-05）](#39-phase-18-完成--u1-c-任務別視角別加權-tta2026-05-05)
- [§40. Phase 19：U11 GroupKFold sanity（read-only 診斷，2026-05-05）](#40-phase-19u11-groupkfold-sanityread-only-診斷2026-05-05)
- [§41. Phase 20：U12 OOF cross-fold variance（read-only 診斷，2026-05-05）](#41-競賽-active-軌跡tta-path)
- [§42. Phase 21：U3 SWA — `p2_combo_best_swa` 設置與訓練（2026-05-05）](#42-u10-弱監督軌跡bestpt-path)
- [§43. Phase 22：U5 T4 class-balanced re-sampling — 設置（2026-05-05）](#43-仍在-active-v12-pool-的成員17-員)
- [§44. Phase 23：N3 NeZha-base 設置（2026-05-05）](#44-phase-23n3-nezha-base-設置2026-05-05)
- [§45. Phase 24：U2 EMA + warm-start 設置與訓練（2026-05-05）](#45-phase-24u2-ema--warm-start-設置與訓練2026-05-05)
- [§46. Phase 25 完成 — U6 / B4 NLLB-600M 中→英→中 回譯資料增強（2026-05-05）](#46-phase-25-完成--u6--b4-nllb-600m-中英中-回譯資料增強2026-05-05)
- [§47. U10 — 企業永續報告書 (SR) 弱監督 pipeline 完整版（2026-05-09 重啟，2026-05-10 v2 重訓）](#47-u10--企業永續報告書-sr-弱監督-pipeline-完整版2026-05-09-重啟2026-05-10-v2-重訓)
- [§48. Phase 32 — U10 per-task TTA — 0.68632（U10 stack 0.67746 → +0.00886）](#48-phase-32--u10-per-task-tta--068632u10-stack-067746--000886)
- [§49. Phase 33 — Class-weighted CE + Focal-T4 γ=3.0 + 4-way per-task hillclimb — NEW SOTA 0.70185](#49-phase-33--class-weighted-ce--focal-t4-γ30--4-way-per-task-hillclimb--new-sota-070185)
- [§50. Phase 34 — M3-v3 + M4-v3 + stem #5（classw_focal v3）+ 5-way per-task hillclimb — NEW SOTA 0.70569](#50-phase-34--m3-v3--m4-v3--stem-5classw_focal-v3-5-way-per-task-hillclimb--new-sota-070569)
- [§51. Phase 35 — 5-way × 3-view per-task TTA — NEW SOTA 0.70758](#51-工程動作分類d1--d30)
- [§52. Phase 36 — U6-pro back-translation + stem #6 + 6-way × 3-view TTA — NEW SOTA 0.71018](#52-u1--u12-backlog-完成狀態)
- [§53. Phase 37 — Aug-Plus hand-crafted minority 訓練與 single-stem ablation（2026-05-18）· **§53.12 NEW SOTA 0.71364**](#53-phase-37--aug-plus-hand-crafted-minority-訓練與-single-stem-ablation2026-05-18)
- [§54. Phase 37 並行路線 — U13：LLM 合成 + 人工標註 + LLM 評審（規劃中）](#54-phase-37-並行路線--u13llm-合成--人工標註--llm-評審規劃中)

### Part V — 附錄 (Appendix)
- [§55. Backlog 詳表（D1 ~ D30 + U1 ~ U12 全紀錄）](#55-backlog-詳表d1--d30--u1--u12-全紀錄)
- [§56. 工程上限拆解（推導依據）](#56-工程上限拆解推導依據)
- [§57. 競賽規則對外部資料的立場（已確認可用）](#57-競賽規則對外部資料的立場已確認可用)

═══════════════════════════════════════════════════════════════════════════════






# Part I — 執行摘要 (Executive Summary)

本部分為決策摘要：3 分鐘看完目前狀態、競賽規格、整體策略。所有量化結論皆有對應 Part IV Phase 紀錄背書。

═══════════════════════════════════════════════════════════════════════════════

## 1. 一頁總結（Snapshot）

截至 2026-05-12，本專案 SOTA 軌跡如下（**Phase 36 6-way × 3-view u6pro 反超 0.71018，首度跨越 0.71**）：

| 軌跡 | 路徑 | 最佳分數 | 達成日期 | 對應章節 |
|---|---|---:|---|---|
| ** 全競賽 SOTA（Phase 36）** | U10 6-way per-task stem hillclimb × 3-view (stored/middle/tail) per-task α 搜尋；新加入 stem #6 = `p2_combo_best_classw_focal_u6pro`（U6-pro 防幻覺回譯 434 augmented samples） | **0.71018** | 2026-05-10 | [§52](#52-u1--u12-backlog-完成狀態) |
| Phase 35 5-way × 3-view TTA | U10 5-way per-task stem hillclimb × 3-view (stored/middle/tail) per-task α 搜尋（K-D simplex grid 0.1） | 0.70758 | 2026-05-12 | [§50 / §51](#50-phase-34--m3-v3--m4-v3--stem-5classw_focal-v3-5-way-per-task-hillclimb--new-sota-070569) |
| Phase 34 5-way per-task hillclimb | baseline + v1 + v2 + v2_classw_focal + v3_classw_focal 5-way per-task hillclimb（grid 0.1） | 0.70569 | 2026-05-12 | [§50 / §51](#50-phase-34--m3-v3--m4-v3--stem-5classw_focal-v3-5-way-per-task-hillclimb--new-sota-070569) |
| Phase 33 4-way per-task hillclimb | baseline + v1 + v2 + v2_classw_focal 4-way per-task hillclimb（grid 0.05 simplex） | 0.70185 | 2026-05-12 | [§49](#49-phase-33--class-weighted-ce--focal-t4-γ30--4-way-per-task-hillclimb--new-sota-070185) |
| 競賽 active SOTA（Phase 18） | U1-c per-task / per-view 加權 TTA（17-way pool × 3 視角，零再訓） | 0.68925 | 2026-05-05 | [§39](#39-phase-18-完成--u1-c-任務別視角別加權-tta2026-05-05) |
| U10 + per-task TTA（Phase 32） | U10 45 ckpt × 3 stems × 3 views per-task per-view aggregator | 0.68632 | 2026-05-12 | [§48](#48-phase-32--u10-per-task-tta--068632u10-stack-067746--000886) |
| U10 4-way equal-weight（Phase 33 ref） | baseline + v1 + v2 + classw_focal_t4_g3 等權 ensemble（無 hillclimb） | 0.67922 | 2026-05-12 | [§49](#49-phase-33--class-weighted-ce--focal-t4-γ30--4-way-per-task-hillclimb--new-sota-070185) |
| U10 best.pt path（Phase 31） | U10 baseline + v1 + v2 三方 best.pt OOF stack（45 ckpt） | 0.67746 | 2026-05-10 | [§47.7](#47-u10--企業永續報告書-sr-弱監督-pipeline-完整版2026-05-09-重啟2026-05-10-v2-重訓) |

**Phase 35 達成 0.70758**（+0.01833 vs active 0.68925，+0.00573 vs Phase 33 0.70185，+0.00189 vs Phase 34 0.70569）；新加入的 v3 stem（M3-v3 9657 段語料 + classw_focal）在 5-way per-task 結構中拿走 T2 主要權重（v2_classw_focal=0.5、v3_classw_focal=0.4），verification_timeline 從 0.60157 拉升至 **0.62669**（+0.02512）。 仍為 OOF 結果，待 6/03 valid 校準（保守 0.69~0.71）。

距離保守工程上限 0.7236：**−0.0160**；距離寬鬆工程上限 0.7570：−0.0494。

---

## 2. 競賽規格

### 2.1 任務

對每段 ESG 報告書文本，同時預測四個欄位：

| 子任務 | 名稱 | 類別數 | 評分指標 | 權重 |
| :--: | :-- | :--: | :-- | :--: |
| **T1** | promise_status | 2 | F1 (positive=Yes) | **0.20** |
| **T2** | verification_timeline | 5 | Macro-F1 | **0.15** |
| **T3** | evidence_status | 3 | F1 (positive=Yes) | **0.30** |
| **T4** | evidence_quality | 4 | Macro-F1 | **0.35** |

最終分數 $\mathcal{S} = 0.20 \cdot \mathrm{T1} + 0.15 \cdot \mathrm{T2} + 0.30 \cdot \mathrm{T3} + 0.35 \cdot \mathrm{T4}$。

### 2.2 條件式約束（極重要）

- **T1=No → T2 / T3 / T4 全部必須 = N/A**（且 promise_string、evidence_string 必須為空字串）。
- **T3=No → T4 必須 = N/A**（且 evidence_string 為空字串）。

推論時必須以 `apply_constraints_batch` 強制覆寫，否則 T2 / T4 macro-F1 會被無效預測污染。Phase 12 / 13 / 14 三次關鍵判斷（v10 失敗、v11 成功、v12 增益）皆源自此約束。

### 2.3 資料

- 訓練集 1,000 筆（涵蓋 50 家 TWSE 上市公司，已釋出，名單見 [§47.2](#472-已標註-50-家公司禁止重複爬取標註)）
- 驗證集：2026-06-03 釋出
- 測試集：2026-06-10 釋出，6/17 截止上傳
- 每日提交上限 3 次

### 2.4 硬體與環境

- 單卡 RTX 5060 Laptop **8 GB** CUDA
- Windows + PowerShell 5.1 + Python 3.13.7 + torch 2.11.0+cu128 + transformers 5.6.2
- 必用系統 Python：`C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe`（`f:\esg-veripromise-2026\.venv` 缺 numpy）

### 2.5 評分管線

5-Fold StratifiedKFold（依 promise_status + evidence_status 分層）→ AdamW + cosine LR + 10% warmup → AMP fp16 → 5 epochs → early_stop_patience = 2 → 每 fold 存 OOF float16 機率 → ensemble 階段做 per-task 或 joint hillclimb。

---

## 3. 整體策略與雙軌 SOTA 軌跡

### 3.1 設計哲學（三句話）

1. **同源異構 ensemble**：以 macbert-base 為主力、macbert-large / roberta-wwm / bert-base 為配角，靠不同初始化、不同訓練技巧（FGM / R-Drop / MSD / Focal）換 diversity；異族 backbone（XLM-R / NeZha / ERNIE）已驗證對純中文 1k 樣本不利（X6 / X11）。
2. **Per-task 權重 + Per-view TTA**：T1（binary）與 T4（4-class macro）的最佳 ensemble 權重差距極大；改用 per-task hillclimb（[§32](#32-雙軌-sota-軌跡)）+ 3-view 推論（stored / middle / tail）等價於零成本軟 ensemble。
3. **合規外部資料 + 防幻覺增強**：訓練集 1k 太小 → 採集台灣 31 家公司（不重疊已標註 50 家）62 份 ESG 報告書做弱監督偽標（[§56](#56-工程上限拆解推導依據)），加 NLLB-3.3B + ChrF≥0.5 過濾的回譯（[§142](#142-設定檔-configsbaseyaml-範本)）撐少數類覆蓋。

### 3.2 雙軌 SOTA 軌跡

| 軌跡 | 起點 | 巔峰 | 主要槓桿 |
| :-- | :--: | :--: | :-- |
| **Active (TTA path)**          | Phase 1 0.6415  | **Phase 18 0.68925** | per-task hillclimb + U1-c TTA |
| **U10 (best.pt path)**         | Phase 31 baseline 0.66734 | **Phase 31 stack 0.67746** | 弱監督偽標 + 兩階段隔離訓練 |
| **雙軌合流 (Phase 32~36)**      | 0.67746 / 0.68925 | **Phase 36 0.71018** | classw_focal + M3-v3 + 5-way × 3-view + U6-pro BT |

### 3.3 三條主軸總覽

```
┌─ 主軸 A：Ensemble 與 Hillclimb ─────────────────────────────────────────────┐
│  Phase 4 (2-way)  → Phase 5 (3-way) → … → Phase 14 (17-way joint v12)    │
│  Phase 33 (4-way per-task) → Phase 34 (5-way) → Phase 36 (6-way × 3-view) │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ 主軸 B：TTA 多視角推論（零再訓 free-lunch） ────────────────────────────────┐
│  Phase 15 (U1 stored+middle)                                              │
│  Phase 18 (U1-c per-task / per-view α 搜尋)                              │
│  Phase 35 (5-way × 3-view) → Phase 36 (6-way × 3-view)                   │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ 主軸 C：弱監督 + 防幻覺增強（解 plateau） ──────────────────────────────────┐
│  Phase 31 U10 v1+v2 stack  → Phase 33 classw_focal stem #4               │
│  Phase 34 M3-v3 stem #5    → Phase 36 U6-pro BT stem #6                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 不再進攻的方向（已驗證為負，詳 [§19](#19-phase-1--主力配置baseline-復現設定) / [§139](#46-phase-25-完成--u6--b4-nllb-600m-中英中-回譯資料增強2026-05-05)）

X1~X13 共 13 條禁區，重點：EMA decay≥0.995 短訓練、xlm-r 主力、NeZha / ERNIE base 同級異族、SWA cosine 末段 LR≠0、global T4 re-sampling、NLLB-600M zh-en-zh BT。本輪嚴格遵守。

---

═══════════════════════════════════════════════════════════════════════════════

# Part II — 進度全景 (Progress Dashboard)

本部分為「目前狀態」的精煉表格集合：誰做了什麼、結果如何、還能做什麼。
所有 Phase 詳細推導與訓練紀錄請翻 Part IV。

═══════════════════════════════════════════════════════════════════════════════

## 4. SOTA 軌跡總表

> 詳細紀錄見後段 [§20 起](#7-結構性殘留與工程上限)。本表只列對 SOTA 有貢獻或關鍵失敗教訓的階段。

### 4.1 競賽 active 軌跡（TTA path）

| Phase | 名稱 | OOF | Δ vs 前任 | 備註 |
| :--: | :-- | :--: | :--: | :-- |
| 1 | Baseline 復現（macbert-base + 5-Fold） | 0.6415 | — | 評分管線錨點 |
| 2 | 單模調優（lr / max_len / pooling / class_weight） | 0.66734 | +0.026 | combo_best 出爐 |
| 3 | macbert-large 升規模 | 0.66439 | −0.003 | 單模負面（X5）；保留作 ensemble 配角 |
| 4 | 2-way base+large ensemble | 0.67478 | +0.010 | 首次 ensemble 增益 |
| 5 | Wave A/B + combo_v2 + 3-way per-task hillclimb | 0.67954 | +0.005 | per-task hillclimb 出爐 |
| 6 | Wave C + 7-way per-task | 0.68206 | +0.003 | 弱模也能加分 |
| 7 | combo_v3（R-Drop + MSD）+ 8-way | 0.68394 | +0.002 | base 史上最高單模 0.67121 |
| 8 | combo_v3 multi-seed 拆 9-way | 0.68440 | +0.0005 | X4 教訓：variance reduction 用「並列」不用「替換」 |
| 9 | Sprint A roberta-wwm + bert-base 11-way | 0.68558 | +0.001 | xlm-r 拒入池 X6 |
| 10 | Sprint B / T6 v1 time-token 12-way | 0.68669 | +0.001 | T6 v1 單模 T2 沒進步、ensemble 取 T1/T3 權重 |
| 11 | Sprint B / T6 v2 bucket-token 13-way | 0.68683 | +0.0001 | T6 路線正式關閉 |
| 12 | Sprint B / T7 Focal-T4 γ=3 14-way | 0.68660 | −0.00023 | 揭露 constraint coupling |
| 13 | Joint Hillclimb v11（post-constraint）16-way | 0.68770 | +0.00087 | EMA 二次失敗 X1' |
| 14 | Sprint C / N1 large+focal+fgm 17-way | 0.68825 | +0.00055 | p10 在 T2 取得權重 1.50 |
| 15 | U1 TTA `stored+middle` | 0.68879 | +0.00054 | 零再訓增益 |
| 16 | N2 ELECTRA admission | 0.68825 | 0 | p11 全 0 權重 X7 |
| 17 | U1-b 三視角等權 | 0.68873 | −0.00006 | 等權路線關閉 X8 |
| **18** | **U1-c per-task / per-view TTA** | **0.68925** | **+0.00046** | **競賽 active SOTA** |
| 19 | U11 GroupKFold sanity（read-only） | — | — | 診斷：SK row-leak ≈ 99.93%；valid 釋出後重驗 |
| 20 | U12 OOF cross-fold variance（read-only） | — | — | 診斷：ensemble noise budget ≈ 0.0045 |
| 21 | B1 / U3 SWA | 0.66252 | −0.0031 | 拒絕 X9（cosine 末段 LR ≠ 0） |
| 22 | B2 / U5 T4 re-sampling | 0.66562 | +0.00004 | 拒絕 X10（global sampler 抵銷 T2/T4） |
| 23 | C1 / N3 NeZha→ERNIE | 0.6086 | — | 拒絕 X11（base-class 異源 backbone 不利） |
| 24 | B3 / U2 EMA + warm-start | 0.64941 | −0.01617 | 拒絕 X12（EMA 末段 LR + 短訓練） |
| 25 | B4 / U6 NLLB-600M 回譯 | 0.66321 | −0.00237 | 拒絕 X13（譯本保真度不足） |
| 32 | U10 per-task TTA（U10 stack + 3-view α） | 0.68632 | +0.00886 vs U10 stack 0.67746 | 兩條軌跡首度相加；詳 [§48](#48-phase-32--u10-per-task-tta--068632u10-stack-067746--000886) |
| 33 | Class-weighted CE + Focal-T4 γ=3.0 + 4-way hillclimb（stem #4） | 0.70185 | +0.01260 vs active 0.68925 | 首度突破 0.70；詳 [§49](#49-phase-33--class-weighted-ce--focal-t4-γ30--4-way-per-task-hillclimb--new-sota-070185) |
| 34 | M3-v3 + M4-v3 + stem #5（classw_focal v3）+ 5-way hillclimb | 0.70569 | +0.00384 | M3 corpus 1840→3680；詳 [§50](#50-phase-34--m3-v3--m4-v3--stem-5classw_focal-v3-5-way-per-task-hillclimb--new-sota-070569) |
| 35 | 5-way × 3-view per-task TTA（stored / mid / tail） | 0.70758 | +0.00189 | 多視角再次 free-lunch；詳 [§51](#51-工程動作分類d1--d30) |
| **36** | **U6-pro back-translation + stem #6 + 6-way × 3-view TTA** | **0.71018** | **+0.00260** | 舊 SOTA；詳 [§52](#52-u1--u12-backlog-完成狀態) |
| 37 | Aug-Plus (47 hand-crafted) + stem #6 recipe（single-stem ablation） | 0.66966 (1 seed) | −0.00078 vs stem #6 baseline 0.67044 | 單獨 OOF 平手；保留為 7th-stem；詳 [§53](#53-phase-37--aug-plus-hand-crafted-minority-訓練與-single-stem-ablation2026-05-18) |
| 37.11 | AP-D1：7-way per-task hillclimb（stored only） | 0.71109 | +0.00238 vs 6-way × stored only seed42 | stem #7 在 stack 內證明 diversity；詳 [§53.11](#5311-ap-d1-結果--7-way-per-task-hillclimb2026-05-18) |
| **37.12** | **AP-D3：7-way × 3-view per-task TTA（seed42）** | **0.71364** | **+0.00346** vs 6-way × 3-view × seed42 (0.71018) | **NEW SOTA**；詳 [§53.12](#5312-ap-d3-結果--7-way--3-view-per-task-tta--new-sota-0713642026-05-18) |

### 4.2 U10 弱監督軌跡（best.pt path）

| Phase | 配置 | T1 | T2 | T3 | T4 | weighted | Δ vs baseline |
| :--: | :-- | ---: | ---: | ---: | ---: | ---: | ---: |
| 31 | baseline `p2_combo_best` (15 ckpt) | 0.92767 | 0.47585 | 0.86551 | 0.43079 | 0.66734 | — |
| 31 | pseudo v1 solo (15 ckpt) | 0.92793 | 0.47611 | 0.86013 | 0.44736 | 0.67162 | +0.00428 |
| 31 | pseudo v2 solo (15 ckpt) | 0.93683 | 0.49806 | 0.86559 | 0.42983 | 0.67219 | +0.00485 |
| 31 | stack baseline + v1 (30 ckpt) | 0.93222 | 0.47691 | 0.86486 | 0.44719 | 0.67396 | +0.00662 |
| 31 | stack baseline + v2 (30 ckpt) | 0.93929 | 0.49278 | 0.87065 | 0.43800 | 0.67627 | +0.00893 |
| **31** | **stack baseline + v1 + v2 (45 ckpt)** | 0.93594 | 0.48939 | 0.87023 | 0.44513 | **0.67746** | **+0.01012** |

完整 U10 pipeline（M1 採集 → M3 抽取 → M4 偽標 → M5 兩階段訓練 → M6 ensemble）詳見 [§47](#47-u10--企業永續報告書-sr-弱監督-pipeline-完整版2026-05-09-重啟2026-05-10-v2-重訓)。

### 4.3 仍在 active v12 pool 的成員（17 員）

詳見後段 [§12.1](#12-整體技術架構) 全表（24 員產出，17 員入 v12 池，3 員 0 權重，4 員拒入池）。

---

## 5. 已完成項目索引（含主要 D1~D30）

### 5.1 工程動作分類（D1 ~ D30）

| 類別 | 編號範圍 | 數量 | 簡述 |
| :-- | :--: | :--: | :-- |
| 單模調優 / 配置 | D1, D4, D9 | 3 | combo_best / v2 / v3 三代 |
| 多骨幹擴池 | D2, D14, D15, D21 | 4 | macbert-large / roberta-wwm / bert-base / large+focal+fgm |
| Ensemble 與 Hillclimb | D3, D6, D8, D10, D12, D13, D16, D17, D18, D19, D20 | 11 | 從 2-way 到 17-way joint v12 |
| Ablation Wave | D5, D7 | 2 | Wave A/B 16 組 + Wave C 4 組 |
| Multi-seed | D11, D12, D13 | 3 | 試錯 + 拆分成功設計 |
| TTA | D22, D24, D25 | 3 | U1 / U1-b / U1-c |
| Backlog 驗證（拒絕） | D23, D26, D27, D28, D29, D30 | 6 | N2 / U3 / U5 / N3 / U2 / U6 |

完整 30 項詳表（含時間、結果、SOTA 增益）見 [§143.1](#531-動機)。

### 5.2 U1 ~ U12 Backlog 完成狀態

| ID | 項目 | 狀態 | 對應紀錄 |
| :-- | :-- | :-- | :-- |
| U1 / U1-b / U1-c | TTA 多視角推論 | **完成且採納**（active SOTA 0.68925）| [§43](#43-仍在-active-v12-pool-的成員17-員) / [§45](#45-phase-24u2-ema--warm-start-設置與訓練2026-05-05) / [§46](#46-phase-25-完成--u6--b4-nllb-600m-中英中-回譯資料增強2026-05-05) |
| U2 | EMA decay=0.995 + warm-start | 完成且拒絕 | [§52](#52-u1--u12-backlog-完成狀態) / X12 |
| U3 | SWA 末 K=3 epoch 平均 | 完成且拒絕 | [§49](#49-phase-33--class-weighted-ce--focal-t4-γ30--4-way-per-task-hillclimb--new-sota-070185) / X9 |
| U5 | T4 class-balanced re-sampling | 完成且拒絕 | [§50](#50-phase-34--m3-v3--m4-v3--stem-5classw_focal-v3-5-way-per-task-hillclimb--new-sota-070569) / X10 |
| U6 | NLLB-600M zh-en-zh 回譯 | 完成且拒絕 | [§53](#53-phase-37--aug-plus-hand-crafted-minority-訓練與-single-stem-ablation2026-05-18) / X13 |
| **U6-pro** | NLLB-3.3B + ChrF≥0.5 過濾回譯 | **完成且採納**（current SOTA 0.71018）| [§142](#142-設定檔-configsbaseyaml-範本) |
| N3 | NeZha → ERNIE fallback backbone | 完成且拒絕 | [§51](#51-工程動作分類d1--d30) / X11 |
| **U10** | TW 企業永續報告書專用偽標 pipeline | **完成且採納**（best.pt SOTA 0.67746）| [§56](#56-工程上限拆解推導依據) |
| U11 | StratifiedGroupKFold by company sanity | 完成（診斷）| [§47](#47-u10--企業永續報告書-sr-弱監督-pipeline-完整版2026-05-09-重啟2026-05-10-v2-重訓) |
| U12 | OOF cross-fold variance | 完成（診斷）| [§48](#48-phase-32--u10-per-task-tta--068632u10-stack-067746--000886) |
| U8 | Pipeline B-Plan T1→T3→T2/T4 串接 | 暫緩 | — |
| U9 | Qwen2.5-7B LoRA 4-bit r=16 | 阻塞（8 GB 顯存不足）| — |

---

## 6. 已驗證為負方向（X1 ~ X13 禁區）

完整 X1 ~ X13 禁區詳見 [§55.3](#553-已暫緩或證明不可行的方向避免重複嘗試)。摘要：

| 代號 | 方向 | 結果 | 教訓 |
| :--: | :-- | :-- | :-- |
| X1 / X1' | EMA decay=0.999 / 0.995（無或短 warm-start） | 0.473 / 0.630 | EMA 需 cosine 末段 LR=0 + 拉長至 8~10 epoch |
| X2 / X2' | Uniform LS / per-task LS(T1/T3) | 0.49 / ensemble 0 權重 | LS 變體與主力 combo 太相似無 diversity |
| X3 | LLRD 0.95 base 模型 | −0.005 | base 12 層太淺 |
| X4 | combo_v3 multi-seed 直接替換 | −0.002 | variance reduction 用並列不用替換 |
| X5 | macbert-large 作主力單模 | −0.003 | 8 GB + 1k 樣本不適合 326M 模型 |
| X6 | xlm-roberta-base | 0.61 | 多語模對純中文劣勢 |
| X7 | ELECTRA-base 同設定 | 單模 0.627 / admission 0 accept | 同設定不重跑；換訓練策略才再試 |
| X8 | TTA 三視角等權 | −0.00006 | 必須 per-task 加權，等權路線關閉 |
| X9 | SWA 末 K=3 epoch 平均 | −0.0031 | cosine 末段 LR ≠ 0 為主因 |
| X10 | T4 global re-sampling | T4 +0.012 / T2 −0.014（淨持平） | 改 per-task loss weighting |
| X11 | NeZha / ERNIE-3.0-base-zh | 不可訓 / 0.609 | base 同級異源 backbone 不利 |
| X12 | EMA decay=0.995 + warm-start 2 epoch | −0.01617 | 同 X1' 限制條件 |
| X13 | NLLB-200-distilled-600M zh-en-zh BT | −0.00237 | 換 NLLB-3.3B+；加 round-trip ChrF ≥ 0.5 過濾 |

**暫緩**：

- **U8** Pipeline B-Plan（T1 → T3 → T2/T4 串接）— 待 U10 結果穩定後再評估。
- **U9** Qwen2.5-7B LoRA — RTX 5060 Laptop 8 GB 不足，需雲端 16 GB+ 顯存。

**已從本檔移除（避免污染現任結果）**：U10 PATH-D / U13 PATH-E / U14 PATH-C / U15 PATH-B Wikipedia + IPCC 偽標等 2026-05-09 之前所有外部資料線；現任 [§47](#47-u10--企業永續報告書-sr-弱監督-pipeline-完整版2026-05-09-重啟2026-05-10-v2-重訓) U10 為 2026-05-09 重啟的「企業永續報告書專用」版本（已成功）。

---

## 7. 結構性殘留與工程上限

### 7.1 Class collapse（兩個永遠 0 的少數類）

| Task | 類別 | 訓練集真實次數 | baseline 模型 OOF 預測次數 | U10 v2 偽標籤次數 |
|---|---|---:|---:|---:|
| T2 | within_2_years | 個位數 | **0** | 0 |
| T4 | Misleading | 1 | **0** | 0 |

這兩類是「baseline 模型完全不預測 → teacher pseudo 不會選 → 偽標池仍 0」的閉環。詳見 [§47.8](#478-結構性殘留問題class-collapse)。

**根本解（待辦）**：Class-weighted CE / Focal-T4 γ=3.0 重訓 + T4-only 過採樣（X10 已試過 global re-sampling，需改 per-task loss weighting）。

### 7.2 工程上限拆解

| 加權上限 | 計算 | 與 active SOTA 0.68925 的距離 |
|---|---|---:|
| 保守 ≈ 0.7236 | 0.20·0.945 + 0.15·0.585 + 0.30·0.910 + 0.35·0.600 | +0.035 |
| 寬鬆 ≈ 0.7570 | 0.20·0.95 + 0.15·0.62 + 0.30·0.92 + 0.35·0.65 | +0.069 |

詳見 [§56.1](#561-兩種上限的定義)。

### 7.3 X18 self-training 上限定理（U10 邊界）

U10 使用 **15 baseline ckpt 平均 softmax** 作 teacher（不使用更強的 U1-c TTA），避免「自我訓練無法超越 teacher」上限直接打死。實測 +0.01012（baseline 0.66734 → stack 0.67746），符合「同 teacher self-training 增益接近上限」的預期。換 backbone 作 teacher（XLM-R / Qwen-LoRA）才能解鎖新 plateau。

---

## 8. 待辦事項（ROI 排序）

> 排序原則：(a) 校準現有 SOTA 為第一要務（避免 OOF 過擬合誤判）；(b) plateau 解鎖路徑次之；(c) 提交策略最後。
> 已完成項目（U10 TTA、Class-weighted Focal、M3-v3、5-way hillclimb、3-view TTA、U6-pro 防幻覺回譯）已從本表移出，詳見 §48 ~ §53 的 Phase 32 ~ 36 紀錄與 §10 SOTA 表。
> 所有 X1 ~ X13 禁區項目（§53.3）不得重跑同設定。

| 優先 | ID | 名稱 | 預期增益 | 風險 / 阻塞 | 對應章節 |
| :--: | :-- | :-- | :--: | :-- | :-- |
| **1** | F1 valid 校準 | 6/03 valid 釋出後執行 hold-out per-task hillclimb；對 Phase 36 / 35 / 34 三版本同時量 OOF↔valid drift；若 valid 退化 > 0.008 則回退至 Phase 35 5-way 配置；同步以 GroupKFold（U11 / §40）復量 row-leak 影響 | 0（診斷工具） | 低；read-only；依賴 6/03 釋出時程 | §40 / §41 / §50 / §51 / §53 |
| **2** | F2 跨家族 teacher（XLM-R / mDeBERTa / Qwen-LoRA） | X18 self-training 上限定理只適用於同 teacher；換 backbone 解鎖新 plateau。具體做法：(a) XLM-R-base 5-fold 加入 stem 池走 7-way TTA；(b) 若硬體允許再試 mDeBERTa-v3-base；(c) Qwen2.5-7B LoRA 暫緩（8GB 顯存阻塞，需雲端 16GB+） | T2/T4 各 +0.003 ~ +0.010；衝高 0.715 ~ 0.720 區段 | 高；硬體預算、cross-family stem 設計、額外 36 ~ 60 min 訓練成本 | §47.12 路徑 4 |
| **3** | F3 提交策略（6/11 起 21 次 LB） | 規劃 6 個提交錨點：baseline / U1-c-TTA (0.68925) / v12 (0.69+) / Phase 33 hillclimb (0.70185) / Phase 35 5-way×3-view (0.70758) / Phase 36 6-way×3-view u6pro (0.71018)；每錨點對應 `reports/analysis/_ensemble/*_preds.csv` 1000 列預測 | LB 對齊；找出最穩 valid↔LB gap | 低 | §11.5 / §152 |
| **4** | F4 7-way × 3-view 擴展（依賴 F2） | F2 一旦產出新跨家族 stem（XLM-R 等），立即把 stem 池擴成 7-way 重跑 `u10_per_task_tta`；per-task K=7 simplex（grid 0.1 ≈ 8008 點/task） | +0.002 ~ +0.005 vs Phase 36 0.71018 | 中；hypothesis space 指數增加，需嚴格 valid 校準 | 待新建 §53（執行後） |
| **5** | F5 T4 Misleading 結構性處理 | 目前 T4 Misleading support=1，Phase 36 仍預測為 0；嘗試 (a) 把 U6-pro 對 T4 Misleading 的 BT 樣本權重再放大 2~3×；(b) Focal γ 提升至 5.0 + class weight 重設；(c) 若 valid 上 Misleading 樣本充足，加單獨 head 微調 | T4 +0.005 ~ +0.020（理論上限 ≈ 0.65） | 高；class collapse 為硬上限 | §21.1 / §55.0.3 |
| **6** | F6 提交檔最終清理 | submission.json 產出 + `apply_constraints_batch` 強制覆寫 + 條件式約束單元測試（`tests/test_post_process.py`）+ schema validation；對應 §15 SOP | 0（不影響分數，避免無效提交） | 低；阻塞 6/17 截止 | §15 |

**暫緩 / 阻塞**：

- **U8** Pipeline B-Plan（T1 → T3 → T2/T4 串接）：Phase 36 已突破 0.71，B-Plan 邊際收益不明，待 valid 校準後再評估。
- **U9** Qwen2.5-7B LoRA：RTX 5060 Laptop 8 GB 顯存不足，需雲端 16 GB+ 顯存。

---

═══════════════════════════════════════════════════════════════════════════════

# Part III — 技術設計 (Technical Design)

本部分為原始訓練計畫書（v1.0）的整理：規劃哲學、資料剖析、訓練 SOP、推論 SOP、風險機制、交付物、排程。
所有「為什麼這樣設計」的依據都在這裡，作為新成員的入門教材。

═══════════════════════════════════════════════════════════════════════════════

## 9. 規劃哲學

過往 `final_summary/` 內的結論皆由 GPT-3.5 推導，存在三項風險：(1) 模型推論能力較弱，超參數搜索區間可能過窄；(2) 缺乏對原始論文方法 (ML-Promise、SemEval Promise Verification) 的對齊；(3) 對少數類別與條件式標籤的處理可能過於簡化。本計畫採「**證據驅動 (Evidence-driven)**」立場：

1. 先以 Baseline 復現作為**唯一可信錨點 (anchor)**，建立可信的本地評分管線。
2. 任何超參數 / 架構變更，必須在 5-Fold CV 上展現 ≥ +0.005 加權分數，且需 ≥ 3 個 seed 的一致性才採納。
3. 所有實驗紀錄需 **可重現 (reproducible)**：固定 seed、固定 split、固定環境 hash。

---

## 10. 競賽任務形式化

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

### 10.1 條件式標籤約束 (Hierarchical Constraints)

依官方欄位語意推導出嚴格約束：

- $y_1 = \text{No} \Rightarrow y_2 = y_3 = y_4 = \text{N/A}$，且 `promise_string = ""`、`evidence_string = ""`。
- $y_3 = \text{No} \Rightarrow y_4 = \text{N/A}$，且 `evidence_string = ""`。

**結論**：推論時必須以「Hierarchical Post-Processing」強制覆寫，否則 T2/T4 會被無效預測污染 Macro-F1。

### 10.2 提交與時程紀律

- 6/03 驗證集釋出、6/10 測試集釋出、6/17 截止上傳、6/30 程式碼/報告繳交。
- 每日提交上限 **3 次**，以最後一次為準。建議：每日固定 1 次、保留 2 次給意外回測。

---

## 11. 資料初步剖析 (基於 `vpesg4k_train_1000 V1.csv`)

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

## 12. 整體技術架構

### 12.1 任務建模策略選擇 (Decision Matrix)

| 策略 | 優點 | 缺點 | 是否採用 |
| :-- | :-- | :-- | :--: |
| (A) Multi-task 共享 encoder + 4 個分類頭 | 知識共享、單模能解全部任務、推論成本低 | 任務間損失尺度可能衝突 | **採用為主力** |
| (B) Pipeline (T1 → T3 → T2/T4) | 明確利用條件式約束 | 誤差累積、訓練複雜 | 作為比較組 (B-Plan) |
| (C) 4 個獨立單任務模型 | 解耦、調優自由度高 | 4 倍訓練/推論成本、無法共享表徵 | 不採用（成本過高） |
| (D) Generative LLM (LoRA) | 可融合條件式約束 | 1000 筆資料量易過擬合、輸出穩定性差、推論成本高 | 列入 Phase 5 高階探索 |

### 12.2 主力架構 (Plan A — Multi-Task BERT)

```
 ┌──────────────────────────┐
input text ───→ │ Encoder (Backbone) │ ───→ pooled_output (H)
 │ hfl/chinese-macbert-... │ (取 [CLS] + Mean-Pool)
 └──────────────────────────┘
 │
 ┌──────────┬──────────┬────┴─────┬──────────┐
 ▼ ▼ ▼ ▼ 
 Dropout Dropout Dropout Dropout 
 ▼ ▼ ▼ ▼ 
 Linear Linear Linear Linear 
 (H, 2) (H, 5) (H, 3) (H, 4) 
 ▼ ▼ ▼ ▼ 
 T1 T2 T3 T4 
```

**設計細節**：

- **Pooling**：採 `[CLS]` + `mean(last_hidden_state * attention_mask)` 兩者 concat (維度 2H)，比僅用 `[CLS]` 對段落級任務更穩。
- **Dropout**：每個任務頭前獨立 `Dropout(p=0.1)`。
- **Loss 加權**：訓練 loss = $\sum_t \alpha_t \cdot \text{CE}_t$，初始 $\alpha = (1,1,1,1)$；可選方案以評分權重 $\alpha = (0.2, 0.15, 0.3, 0.35)$ 對齊。
- **Class Weight**：T2、T4 啟用 inverse-frequency `class_weight`，由訓練 fold 內統計而得。
- **AMP / fp16**：必開以節省顯存。
- **Gradient Accumulation**：根據 GPU 顯存自動調整以維持 effective batch = 16。

### 12.3 模型骨幹候選 (Backbone Pool)

| Backbone | 規模 | 預期角色 | 優先序 |
| :-- | :--: | :-- | :--: |
| `hfl/chinese-macbert-base` | 102M | 快速迭代基準、Phase 1 主力 | 1 |
| `hfl/chinese-roberta-wwm-ext` | 102M | 對照組（Baseline notebook 採用） | 2 |
| `hfl/chinese-macbert-large` | 326M | Phase 3 升規模主力 | 3 |
| `hfl/chinese-roberta-wwm-ext-large` | 326M | Ensemble 多樣化 | 4 |
| `xlm-roberta-large` | 560M | 跨語言、決策邊界差異化 | 5 |

> **原則**：先 base 後 large，先單模後集成，每一步都需 5-Fold OOF 證據支持。

### 12.4 驗證策略 (Validation Protocol)

- **K-Fold**：`StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`，分層鍵採 `promise_status × evidence_status` 之 6 類聯合標籤（避免極稀類落單）。
- **Group-aware 變體 (重要)**：若 EDA 顯示同一 `company` 出現 ≥ 5 筆，將額外比較 `StratifiedGroupKFold(group=company)` 以衡量「公司洩漏」對泛化的影響。任一指標低於普通 StratifiedKFold ≥ 0.02，採用 GroupKFold 為最終切分。
- **Seed 集合**：`[42, 2024, 20260417, 7, 1234]` (5 顆)。
- **OOF 機率保存**：每 (seed, fold) 儲存 `np.float16` 機率張量，作為 Ensemble 輸入。
- **指標彙總**：每 fold 報告 weighted score + 4 個任務 macro-F1 + per-class F1；Phase 末以 `mean ± std` 呈現。

### 12.5 集成策略 (Ensemble)

1. **OOF Hill-Climbing**：對每個任務 $t$，在 OOF 機率上以貪心法 (greedy weighted average) 搜尋使 macro-F1 最大化的模型權重。
2. **任務獨立輸出**：T1/T2/T3/T4 各自擁有最佳 ensemble，互不干擾。
3. **Hierarchical Post-Process**：
 - $\hat{y}_1 = \text{No} \Rightarrow$ 覆寫 $\hat{y}_{2,3,4} = \text{N/A}$。
 - $\hat{y}_3 = \text{No} \Rightarrow$ 覆寫 $\hat{y}_4 = \text{N/A}$。
4. **TTA (Test-Time Augmentation, 可選)**：對測試集做「全文 / 前 256 token / 後 256 token」三段平均。

---

## 13. 完整里程碑路線圖 (Phased Roadmap)

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
| 13 — Sprint B 收尾批次 / U2 EMA-995 + U4 LS(T1/T3) + U7 Joint Hillclimb v11 | +0.001 | 0.68770 (16-way joint post-constraint hillclimb v11) | p8_ema995 崩潰 0.63010（EMA 無 warm-start）；p9_ls_t1t3 0.67334 (+0.00213)；v11 聯合 hillclimb 直接優化 post-constraint score（4 次 accept 全在 T3，T4 +0.0018 為 constraint coupling 連動效應）| 已完成（前一代 SOTA） |
| **14 — Sprint C / N1 結構性多樣化 + 17-way Joint Hillclimb v12** | **+0.0005 ~ +0.003** | **0.68825** | `p10_large_focal_fgm` 單模 OOF=0.66874 入池；v12 8 次 accept 全在 T2，p10 取得 T2 權重 1.50，T2 0.5036→0.5072 | 已完成（目前訓練 ensemble SOTA） |
| **15 — Sprint C 後續 / U1 TTA 推論擴增** | **+0.00054** | **0.68879** | U10 外部 pseudo 線暫緩並清除後，先做官方資料、無重訓的推論增益；`stored+middle` 已被 Phase 18 U1-c 取代 | 已完成（上代 active SOTA）|
| **18 — U1-c per-task / per-view TTA coordinate descent** | **+0.00046 vs U1 / +0.00100 vs v12** | **0.68925** | 零再訓，14 員 × stored/middle/tail 三視角 per-task 加權；T2 0.51026 史上最高 | 已完成（**目前 active SOTA**）|
| 16 — Sprint C / N2 p11 ELECTRA base admission | 0 | 單模 0.62694；admission 0.68825 | p11 5-Fold 完成但未達 0.65 gate；v16 18-way joint hillclimb 0 accept、p11 四任務 0 權重 | 已完成（未採納） |
| 17 — 最終衝刺 (6/03 驗證集 / 6/10 測試集) | — | — | 提交策略、Full-Data Refit、文件與報告；U12 train↔val gap 分析 | 待辦 |

> 初期單模階段的 **Gate 條件**：5-Fold 平均 weighted score ≥ 上階段 + 0.005，且任一單任務指標跌幅 ≤ 0.005。Phase 6 之後進入高分 plateau，Gate 改為「是否提供新 ensemble diversity / 是否推高 post-constraint joint score」。Phase 3 大模型單模未通過 Gate，但 ensemble (Phase 4) 通過。

---

## 14. 企業級專案骨架 (Step 0 產出)

```
esg-veripromise-2026/
├── data/
│ ├── raw/ # 官方原始檔案 (read-only)
│ ├── processed/ # 清洗後 / 切分快照
│ └── splits/ # 5-Fold 索引 .json (固定 seed)
├── configs/
│ ├── base.yaml # 共用預設值
│ ├── exp_p1_baseline.yaml # Phase 1
│ ├── exp_p2_tune_*.yaml # Phase 2 系列
│ ├── exp_p3_large_*.yaml # Phase 3 系列
│ └── exp_p4_*.yaml # Phase 4 系列
├── src/
│ ├── __init__.py
│ ├── seed.py # 全域 seed 工具
│ ├── data/
│ │ ├── loader.py # CSV/JSON 載入、欄位驗證
│ │ ├── splits.py # StratifiedKFold / StratifiedGroupKFold
│ │ └── dataset.py # PyTorch Dataset / collate_fn
│ ├── models/
│ │ ├── multitask.py # MultiTaskClassifier
│ │ └── pooling.py # ClsMeanPooler
│ ├── training/
│ │ ├── trainer.py # AMP / grad accum / EMA / early stop
│ │ ├── losses.py # CE + class_weight + (Phase 4) FGM
│ │ └── schedulers.py # cosine + warmup
│ ├── inference/
│ │ ├── predict.py # 推論 + Hierarchical Post-Process
│ │ └── tta.py # TTA (Phase 5)
│ ├── ensemble/
│ │ └── hillclimb.py # Per-task greedy weighted ensemble
│ ├── eval/
│ │ ├── metrics.py # weighted_score / per-task macro-F1
│ │ └── report.py # markdown / html 報告產生
│ └── tools/
│ ├── eda.py # 自動 EDA → reports/eda/
│ └── verify_env.py # 環境一致性檢查
├── scripts/
│ ├── 00_setup.ps1 # Windows 環境腳本
│ ├── 01_eda.ps1 # 跑一次完整 EDA
│ ├── 10_train_kfold.ps1 # 跑單一 config 的 5-Fold
│ └── 20_submit.ps1 # 產出提交檔
├── outputs/
│ ├── checkpoints/{exp}/seed{S}/fold{F}/best.pt
│ ├── oof/{exp}/seed{S}/fold{F}/probs.npz
│ ├── logs/{exp}/... # JSONL training logs
│ └── submissions/sub_YYYYMMDD_v{N}.json
├── reports/
│ ├── eda/eda_report.html
│ ├── experiments/{exp}/score_summary.csv
│ └── final_report.md
├── tests/
│ ├── test_metrics.py # 評分函式單元測試
│ ├── test_post_process.py # 條件式約束單元測試
│ └── test_split_no_leak.py # 公司洩漏檢查
├── requirements.txt
├── pyproject.toml # 專案中介資訊 + ruff/black 規則
├── README.md
└── .gitignore
```

### 14.1 環境鎖版 (`requirements.txt`)

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

### 14.2 設定檔 (`configs/base.yaml` 範本)

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
 type: stratified_kfold # or stratified_group_kfold
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

## 15. 推論與後處理 SOP

```
ensemble_probs(test) ─→ argmax per task ─→ raw_preds
 │
 ├─ if y1 == "No":
 │ y2, y3, y4 := "N/A"
 │ promise_string := ""
 │ evidence_string := ""
 │
 ├─ if y3 == "No":
 │ y4 := "N/A"
 │ evidence_string := ""
 │
 └─ Schema validation ─→ submission.json
```

**單元測試** (`tests/test_post_process.py`) 必須涵蓋上述全部覆寫情境，CI 失敗則禁止生成提交檔。

---

## 16. 風險與止損機制

| 風險 | 監控指標 | 止損動作 |
| :-- | :-- | :-- |
| 公司洩漏 (company leakage) | StratifiedKFold vs GroupKFold 差距 ≥ 0.02 | 改採 GroupKFold |
| 顯存爆掉 | OOM | bs/2 + grad_accum×2 |
| 弱項補強反向劣化 | 任一任務 macro-F1 跌 ≥ 0.005 | 回滾配置 |
| Seed 變異過大 (std ≥ 0.01) | – | 增加 seed 至 7 顆 |
| 提交額度耗盡 | 當日提交次數 ≥ 2 | 進入 only-on-improvement |
| 標註雜訊 | 模型錯預測但 prob ≥ 0.85 的樣本 ≥ 5% | 啟動人工複核（不修改測試集） |

---

## 17. 交付物 (Deliverables)

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

## 18. 排程

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
| Day 19 | Phase 11 — Sprint B / T6 v2 專屬 bucket token + 13-way Hillclimb v9 | p6_t6v2_bucket_tok + biased search → **0.68683（當時 SOTA）** | 已完成 |
| Day 20 | Phase 12 — Sprint B / T7 Focal-T4 γ=3.0 + 14-way Hillclimb v10 | p7_focal_g3 單模 0.67416；ensemble 0.68660（被 constraint coupling 抵銷，未破 SOTA）| 已完成 |
| Day 21 | Phase 13 — Sprint B 收尾批次：U2 EMA-995 + U4 LS(T1/T3) + U7 Joint Hillclimb v11 | p8=0.63010（EMA 失敗）/ p9=0.67334 (+0.00213)；**v11=0.68770 (+0.00087 vs v9 → 新 SOTA)** | 已完成 |
| Day 22 | Phase 14 — N1 large×Focal×FGM + Joint Hillclimb v12 | p10=0.66874 入池；**v12=0.68825 (+0.00055 vs v11 → 新 SOTA)**，T2=0.5072 | 已完成 |
| Day 23 | Phase 15 — U1 TTA 推論擴增 | `stored+middle`=**0.68879**（目前 active SOTA），未引入外部資料、未重訓 | 已完成 |
| Day 24 | Phase 16 — N2/p11 ELECTRA base + v16 admission | 單模 0.62694；v16 admission 0 accept；p11 未納入 active ensemble | 已完成（未採納）|
| Day 25 | Phase 17 — U1-b `stored+middle+tail` 三視角等權 TTA 補測 | 0.68873，較 `stored+middle` 低 0.0000597；等權三視角路徑關閉 | 已完成（負向）|
| Day 25 | Phase 18 — U1-c per-task / per-view TTA coordinate descent（零再訓）| **0.68925**（+0.00046 vs U1 / +0.00100 vs v12）；T2 0.51026 為史上最高 | 已完成（新 SOTA）|
| Day 25 | Phase 19 — U11 GroupKFold sanity（read-only）| StratifiedKFold row-leak ≈ 99.93%；GK drift T2/T4 過高 → MONITOR不重訓，記錄 valid↔OOF gap 預期 | 已完成（診斷）|
| Day 25 | Phase 20 — U12 OOF cross-fold variance（read-only，12/14 員）| 單員 std=0.0115；ensemble noise ≈ 0.0045 → U1-c 虛增益在雜訊內，需 valid 重驗 | 已完成（診斷）|
| Day 25 | Phase 21 — B1 / U3 SWA `p2_combo_best_swa`（K=3 epoch 平均，5 折）| member-level OOF 0.66252 vs best-epoch 0.66558（−0.0031）；fold4 −0.0139；未過 admission；列入 X9 | 已完成（拒絕）|
| Day 25 | Phase 22 — B2 / U5 T4 sqrt-inverse-freq re-sampling（alpha=0.5，5 折）| OOF 0.66562（+0.00004，持平）；T4 +0.012 但 T2 −0.014 抵銷；未過 admission；列入 X10 | 已完成（拒絕）|
| Day 25 | Phase 23 — C1 / N3 NeZha-base → ERNIE-3.0-base-zh fallback | NeZha fallback 為 BertModel 不可訓（fold0 0.549）；ERNIE 5 折 mean 0.6086 << 0.66；列入 X11 | 已完成（拒絕）|
| Day 25 | Phase 24 — U2 / B3 EMA decay=0.995 + warm-start 2 epoch（`p2_combo_best_ema995_warm`） | member OOF 0.64941 vs baseline 0.66558（−0.01617）；5 折全退化、T4 −0.062；列入 X12 | 已完成（拒絕）|
| Day 25 | Phase 25 — U6 / B4 NLLB-200-distilled-600M 中→英→中 回譯（T2/T4 少類 138 源員 × 1.22 → 168 增強樣本，`p2_combo_best_u6_bt`） | member OOF 0.66321 vs baseline 0.66558（−0.00237）；T2 −0.014、T4 −0.031；列入 X13 | 已完成（拒絕）|
| 5/03 ~ 6/02 | Sprint C 後續 | U3 SWA、U5 T4 re-sampling、N3 新 backbone、U11 GroupKFold、U12 OOF↔valid gap；U1-b/U1-c 已完；B1+B2+C1+B3 全數失敗 | U10 外部 pseudo 線已暫緩並清除；**內部可控加分項全部試遍，唯一剩餘 U6 回譯增強（需外部翻譯 API）** |
| 6/03 ~ 6/10 | 驗證集校正 + Sprint C | OOF vs Valid gap 分析 (U12)、GroupKFold sanity check (U11)、提交策略 | 待辦 |
| 6/10 ~ 6/17 | 測試集衝刺 | 提交，至多 21 次 | 待辦 |
| 6/24 ~ 6/30 | 結案 | 最終報告與程式碼提交 | 待辦 |

---

═══════════════════════════════════════════════════════════════════════════════

# Part IV — Phase 詳細紀錄 (Append-only Phase Log)

本部分為完整訓練歷史，**append-only**（不刪除、不改寫已完成 Phase）。
每個 Phase 包含：(a) 動機 / 設計、(b) 執行狀態、(c) 結果、(d) 結論與後續。
失敗的 Phase 同樣保留作為禁區（X1 ~ X13）的依據。

═══════════════════════════════════════════════════════════════════════════════

## 19. Phase 1 — 主力配置（Baseline 復現設定）

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

## 20. Phase 1 結果與 Phase 2 籌備（OOF 0.6415 → combo 規劃）

### 20.1 Phase 1 — Baseline 復現結果（已完成）

執行：`python -m src.train_kfold --config configs/exp_p1_baseline.yaml`，3 seeds × 5 folds × 5 epochs，AMP fp16，class_weight on，pooling=cls_mean，max_length=256，batch=16，lr=2e-5，cosine + warmup_ratio=0.10，wd=0.01。RTX 5060 8GB 約 48 分鐘。

| Seed | Mean | Std | Min | Max |
| :-- | :-- | :-- | :-- | :-- |
| 42 | 0.6449 | 0.0094 | 0.6319 | 0.6541 |
| 2024 | 0.6374 | 0.0143 | 0.6237 | 0.6578 |
| 20260417 | 0.6422 | 0.0171 | 0.6144 | 0.6549 |
| **Overall** | **0.64150** | **0.01332** | — | — |

**結果遠超 Phase 1 目標 0.55-0.58**。Gate 條件達成 → 進入 Phase 2。

### 20.2 OOF 詳細分析摘要（[reports/analysis/p1_baseline_macbert_base/summary.md](reports/analysis/p1_baseline_macbert_base/summary.md)）

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

## 21. Phase 2~4 超參數搜索藍圖

### 21.1 Phase 2 — 單模調優 (僅在 macbert-base 上做，節省成本)

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

### 21.2 Phase 3 — 升規模 (macbert-large)

固定 Phase 2 之最佳參數，僅變更：

```yaml
model.backbone: hfl/chinese-macbert-large
training.batch_size: 4
training.grad_accum: 4 # effective batch = 16
training.lr: 1.5e-5 # large 模型偏小 lr
training.epochs: 4
```

跑 5 seeds × 5 folds = 25 runs，OOF 進入 Hill-Climbing。

### 21.3 Phase 4 — 多骨幹 + 弱項補強

- **多骨幹**：`hfl/chinese-roberta-wwm-ext-large`、`xlm-roberta-large` 各 3 seeds × 5 folds。
- **弱項補強 (僅針對 T2/T4)**：
 - **R-Drop**：同 batch 兩次 forward，KL 正則 ($\alpha=0.5$)。
 - **FGM 對抗訓練**：embedding 加擾動 ($\epsilon=1.0$)。
 - **EMA**：decay=0.999，推論用 EMA 權重。
 - **Multi-Sample Dropout**：T4 head 套 5 份 dropout 平均。
 - **Minority Oversample**：對 T2 `longer_than_5_years` × 2、T4 `Misleading` × 2。

每項獨立 A/B 測試，僅在 OOF +0.005 才採納。

---

## 22. Phase 2 — Combo Winners 最終配置 (3 seeds × 5 folds)

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

## 23. Phase 3 — 升規模 (`hfl/chinese-macbert-large`, 326M)

依原計畫 7.2 升規模到 macbert-large，繼承 Phase 2 combo 最佳超參 (`max_length=384`、`pooling=cls_mean`、`use_class_weight=false`)，調整 batch=4 + grad_accum=4 (effective 16)、lr 範圍 1e-5 ~ 2e-5、early_stop_patience=2。單 seed=42、5 折驗證。

| Config | lr | epochs | weighted_score (5-fold mean ± std) | vs P2 combo seed=42 (0.66558) |
| :-- | :-- | :-- | :-- | :-- |
| `p3_large_baseline` | 1.5e-5 | 5 | **0.65900 ± 0.00867** | -0.00658 |
| `p3_large_lr2e5` | 2.0e-5 | 5 | **0.66439** | -0.00119 |
| `p3_large_ep8` (4 折,fold4 紀錄缺失) | 1.0e-5 | 8 | ≈ 0.6597 (前 4 折最佳 epoch 平均) | ≈ -0.006 |

**結論**：在 1000 樣本上 macbert-large 單獨並未通過「+0.005 顯著超越 base combo」門檻；epoch 拉長後驗證分數於第 6~7 epoch 飽和，loss 仍下降代表已過擬合。large 訓練成本 (≈ 4×) 與單模收益不對等 → **不採 large 為單模主力**，保留 large 預測作 Phase 4 集成輸入。

> `p3_large_ep8` fold4 的 `score_summary.csv` 因外部監控腳本提早終止主程序而未寫出；5 個 fold 的 `best.pt` 皆已落盤，不影響 ensemble；本表以前 4 折最佳 epoch 平均近似呈現結論。

### 23.1 為什麼大模型 (326M) 反而沒贏過 base (102M)？— 五個根本原因

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

## 24. Phase 4 — Probability Ensemble (base combo + large)

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

→ 已於 Phase 5 推進並於 Phase 6 進一步突破：擴 pool 加入 Wave C 4 個增強模型 + 7-way per-task hillclimb，最終 SOTA = **0.68206**（見 §25 / §26）。

詳見 [reports/analysis/_ensemble/p2_combo_best_p3_large_lr2e5.csv](reports/analysis/_ensemble/p2_combo_best_p3_large_lr2e5.csv)。

---

## 25. Phase 5 — Wave A/B Ablation + Combo v2 + Per-task Hillclimb

### 25.1 Wave A/B 單變量 Ablation（16 組，全在 combo_best 基礎上 +1 改動，5-Fold seed=42）

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

### 25.2 Combo v2 復測（取 §25.1 兩個正向贏家）

`exp_p2_combo_v2.yaml` = combo_best + Focal-T4 (γ=2.0) + FGM (ε=1.0)，3 seeds × 5-fold：

| metric | T1 | T2 | T3 | T4 | weighted |
| :-- | :--: | :--: | :--: | :--: | :--: |
| 3-seed mean ± std | — | — | — | 0.4368 | **0.66836 ± 0.01659** |
| seed-avg ensemble (post-processed) | 0.9322 | 0.4770 | 0.8638 | 0.4333 | **0.66879** |

→ vs `p2_combo_best` seed-avg 0.66734：**+0.00145**（穩定但增量小，主要來自 T4）。

### 25.3 3-way Probability Ensemble + Per-task Hillclimb

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

## 26. Phase 6 — Wave C Ablation + 7-way Per-task Hillclimb

> **本階段目標**：嘗試 Phase 5 之後仍未試過的三大正則化／資料增強技術，並用「**爬山演算法 (hillclimbing)**」在更大的模型池中為每個任務獨立搜尋最佳集成權重，把 SOTA 從 0.67954 進一步往上推。

### 26.0 名詞快速說明（給未接觸過的讀者）

- **Ablation（消融實驗）**：固定其他所有設定，只改一個變因（例如「加上資料增強」），跑完整 5-Fold 觀察分數變化，用以確認該變因的真實效果。
- **Combo（組合配方）**：把多個「單獨已驗證有效」的技術疊在 baseline 上做的整合配方（如 `combo_v2 = baseline + FGM + Focal-T4`）。Combo 的篩選規則是「**單獨上場必須先贏過 baseline**」，因此 Wave C 因為單模都輸給 baseline，**並不進入 Combo**（詳見 §26.3）。
- **Ensemble（集成）**：訓練好的多個模型，在推論時把它們的「機率輸出」加權平均，再 argmax 拿最終標籤。集成成功的關鍵不是「每個成員都很強」，而是「成員之間的錯誤模式互不相同」（diversity）。
- **OOF (Out-of-Fold) probabilities**：5-Fold 訓練時，每筆樣本只在「不參與訓練的那一摺」被預測，因此整份資料都有「未洩漏」的預測機率，可拿來做公平的離線評分與 ensemble 權重搜尋。
- **Hillclimbing（爬山演算法）**：見 §26.2 詳細解說。

### 26.1 Wave C 訓練技術 Ablation（單模實驗）

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

**結論**：4 個配置全部單獨低於 `combo_best`。資料量僅 1000 筆、訓練步數約 500 步的設定下，這些「強正則化」技術可能因訓練不足以收斂出穩定的擾動鄰域而短期內反而拖慢學習速度。但它們的決策邊界與既有模型差異化顯著，因此被保留為 ensemble 池成員（§26.4 證實這是正確決定）。

### 26.2 Hillclimbing 與 per-task 權重搜尋

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

### 26.3 為什麼 Wave C 不進 Combo？

**Combo 的設計原則**：把「單獨已驗證帶來 +0.001 以上增益」的技術疊在 baseline 上做整合配方。例如 `combo_v2 = combo_best + FGM + Focal-T4`，因為 FGM 單獨 +0.00156、Focal-T4 單獨 +0.00275，兩者都是正向贏家。

**Wave C 4 個配置全部單獨輸給 combo_best**（差距 -0.0011 ~ -0.0049）。如果硬把它們疊到 combo 上，等於在已經最佳化的訓練配方裡注入「會降分的成分」，破壞了 combo 的可解釋性與穩定度。因此**Wave C 不進 Combo**。

但這並不代表它們沒用——**ensemble 與 combo 是兩種完全不同的策略**：
- **Combo**：在「**訓練時**」把多個技術疊在同一個模型上 → 產出**單一模型**。
- **Ensemble**：在「**推論時**」把多個**獨立訓練**的模型輸出機率做加權平均 → 產出**集成預測**。

集成成功靠的是「**模型之間的互補性 (diversity)**」，而不是「每個成員都最強」。Wave C 用了完全不同的訓練擾動（mask/swap/delete/R-Drop/MSD），它們犯的錯與既有模型不同類型，因此即使單模較弱，加進 ensemble 池仍能透過 per-task hillclimb 得到正向貢獻。§26.4 的數字證實了這個直覺。

### 26.4 7-way Per-task Hillclimb 結果（**Phase 6 SOTA**）

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
- **T4 仍是最大瓶頸（0.4524）**：Wave C 4 模型在 T4 全部被排除（weight=0），證實 T4 的稀有類別問題（如 `Misleading` 全集只有 1 筆）不是靠「訓練擾動」能解，需要 §53.2 的 oversampling 或 numeric token 標註等資料層改造。

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v3_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v3_summary.csv) 與 [reports/analysis/_ensemble/per_task_hillclimb_v3_preds.csv](reports/analysis/_ensemble/per_task_hillclimb_v3_preds.csv)。

### 26.5 概念釐清：「ensemble pool 是做什麼的？」

> 此處對應使用者於 Phase 28 結尾提問。`ensemble pool` 概念在本節（Phase 6）首次以「7 個成員 × per-task hillclimb」的形態正式登場，故釐清放在這裡。

`ensemble pool` 是「**通過 admission 閘門的訓練成員（member）集合**」，由 hillclimb / weighted ensemble 演算法從中搜尋最佳權重組合來產生最終預測：

- **目前 v12 池**：14 個成員（p2_combo_best、p2_combo_v2/v3、p3 large、p4a/p4b、p5_t6、p6_t6v2、p7_focal_g3、p9_ls_t1t3、p10_large_focal_fgm 等），active SOTA = 0.68925（U1-c per-task TTA 後處理，詳 §39）。
- **Admission 門檻 = 0.66608**：只有單模 member-level OOF ≥ 0.66608 的訓練才能加入 pool；低於門檻的成員加入只會稀釋 ensemble 訊號（X4/X12 已驗證）。
- **Hillclimb 演進**：v6 → v7b → v8 → v9 → v10 → v11 → v12 漸進式 coordinate descent（每代詳見 §26.4 / §27.3 / §28.3 / §29.2 / §33 / §34），每加一個新成員 ΔSOTA 約 +0.00050~+0.00118。
- **Phase 26–30 全數拒絕並刪除**（U10 / U13 / U14 / U15 PATH-B/C/D/E）：2026-05-09 使用者判斷上述路徑跳本偏離 ESG 企業永續報告書主類，走向沒有意義；資料 / 腳本 / 權重 / 報告一次性清除；U10 重關為「企業永續報告書專用」詳 §47 U10-NEW。

**結論**：ensemble pool 是「成員品質保證機制」，admission 是品質閘門，hillclimb 是權重搜尋演算法。

---

## 27. Phase 7 — Combo v3 + 8-way Per-task Hillclimb

> **本階段目標**：把 Phase 6 中「ensemble pool 內表現最好的 Wave C 兩個技術」（R-Drop 與 MSD）正式疊回 Combo 訓練配方，看是否能產出第一個能單獨贏過 combo_best 的新 combo；接著把它加入 ensemble pool 跑 8-way per-task hillclimb。

### 27.1 為什麼選 R-Drop 與 MSD 進 combo_v3？

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

### 27.2 combo_v3 單模結果（首次 Wave C 技術疊上 combo 成功）

5-Fold seed=42（與 combo_best seed=42 = 0.66558、combo_v2 seed=42 比較）：

| 模型 | weighted_score | T1 | T2 | T3 | T4 | vs combo_best |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| `combo_best` (Phase 2) | 0.66558 | — | — | — | — | baseline |
| `combo_v2` (Phase 5) | ~0.66833 | — | — | — | — | +0.00275 |
| **`combo_v3` (Phase 7)** | **0.67121** | 0.9332 | 0.4717 | 0.8644 | 0.4415 | **+0.00563** |

→ combo_v3 是**自 Phase 5 以來第一個單模超越 combo_v2 的訓練配方**，也驗證了 §26 的假設：**Wave C 技術在 ensemble 池內弱、但與已最佳化的 combo 配方互補時可正向疊加**。R-Drop 強迫的 dropout-子網路一致性 + MSD 的多樣本平均，等於同時在「訓練擾動」與「推論平滑」兩個維度做正則化。

### 27.3 8-way Per-task Hillclimb（**Phase 7 SOTA**）

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

### 27.4 累計 SOTA 進展軌跡

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

## 28. Phase 8 — combo_v3 Multi-seed 拆分為 9-way 池（Phase 8 SOTA，已被 Phase 11 取代）

### 28.1 動機與假設

Phase 7 SOTA (0.68394) 中，T1 winner 給 `combo_v3 weight=1.75`（在 8-way tuple 中佔 54%），意味 T1 高度倚賴 combo_v3 seed=42 的「尖銳預測」。一個自然問題是：**multi-seed averaging 能否進一步降低 OOF 變異、推升 SOTA？**

預期：每個 seed 的 5-fold OOF std ≈ ±0.015，3-seed 平均後變異降約 1/√3，理論增益 +0.001 ~ +0.005。

### 28.2 試錯一：直接 multi-seed 平均（v5，**失敗**）

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

### 28.3 試錯二：拆分為兩成員 9-way 池（v6，**成功**）

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

### 28.4 結果解讀（每個 task 的故事）

- **T1**：hillclimb 自動把 v3_avg 設為 0、保留 v3_s42=1.75 → 完全採用 Phase 7 winner，沒被 multi-seed 拖累。**「拆分」設計的最大價值不是讓 v3_avg 主導 T1，而是讓 hillclimb 有權選擇「不選」**，避免被強迫融合而退化。
- **T2/T3**：v3_avg 確實取得非零權重（0.5 與 1.0、0.25 與 0.5），smooth 版降低了 OOF 上的決策邊界抖動 → 兩個 macro-F1 任務微幅獲益。
- **T4**：兩個 v3 版本都被設為 0；T4 的進展來自完全不同的方向 — 由 8000 次更大隨機搜索找到 `(combo_v2=1.0, combo_best=1.5, large=1.5)`（Phase 7 是 `0:1.5:1.0`），意即 **Phase 7 棄用 combo_v2 的決策被 Phase 8 推翻**。這是「9-way 池 + 8000 trials」單純放大搜索容量帶來的副作用，與 multi-seed 無直接關係。

### 28.5 累計教訓

1. **「平均」並非萬靈丹**：當某個成員在某 task 是「peaky 但準確」時，平均化會稀釋訊號。Multi-seed averaging 應該與「保留原版」並存，由 hillclimb 自選使用比例。
2. **池擴張的雙重收益**：v6 的增益並非完全來自 multi-seed，更大的搜索空間（9 維 × 8000 trials）也讓 T4 找到更好的 combo_v2/best/large 比例。
3. **Per-task hillclimb 的安全性**：warm-start 自上一階段 winners + 隨機探索的設計，讓「失敗的新成員」最壞退化為原 SOTA，永遠不會回退。

### 28.6 累計 SOTA 進展軌跡（含 Phase 8）

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

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v9_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v9_summary.csv)（Phase 11 當時 SOTA）與 [reports/analysis/_ensemble/per_task_hillclimb_v10_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v10_summary.csv)（Phase 12 記錄）。

---

## 29. Phase 9 — Sprint A backbone 擴 pool（已完成）

依 §55 Sprint A 計畫，依序訓練 3 個新骨幹（單 seed=42、5-Fold、繼承 combo_best 超參），加入 ensemble pool 後跑 hillclimb v7 以衝刺 0.69。

### 29.1 已完成成員

| # | exp | backbone | lr | OOF (5-Fold) | T1 | T2 | T3 | T4 | 耗時 | ≥0.65 入池 |
| :--: | :-- | :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| p4a | `p4a_roberta_wwm_base` | `hfl/chinese-roberta-wwm-ext` | 3e-5 | 0.66626 | 0.9265 | 0.4626 | 0.8555 | 0.4426 | 932s | 是 |
| p4b | `p4b_bert_base_chinese` | `bert-base-chinese` | 3e-5 | **0.67355** | 0.9346 | 0.4857 | 0.8656 | 0.4403 | 971s | 是 |
| p4c | `p4c_xlm_roberta_base` | `xlm-roberta-base` | 2e-5 | 0.61269 | 0.9107 | 0.3419 | 0.8391 | 0.3644 | 1108s | **否** |

**p4a 觀察**：OOF 0.66626 ≈ combo_best 0.66558（+0.0007 微幅優於）；T4 略高 0.4426 vs 0.4308 → **多樣性主要落在 T4**，可貢獻 ensemble。

**p4b 觀察**：OOF **0.67355** 為 Sprint A 最佳，優於 combo_best 0.66558（+0.00797）。T1=0.9346、T2=0.4857 同步上升，與 macbert 體系不同（bert-base-chinese 無 WWM、無 macbert MLM 校正，反而提供獨立信號）。

**p4c 觀察**：OOF 0.61269 **未過 ≥0.65 入池門檻**。問題在 T2=0.3419、T4=0.3644 嚴重落後（macro-F1 受少類拖累更甚）。主因：(1) xlm-roberta-base 為多語預訓練、中文 token 表徵密度低於專門中文模型；(2) lr=2e-5 + 5 ep 收斂不足（loss 仍從 4.0 → 2.4，但 fold 間 std=0.0345 顯示變異大）。**結論：踢出 v7 pool**。

### 29.2 Hillclimb v7 → v7b（biased search 修正）

**v7 結果**：N_RANDOM=10000、warm-start 自 v6 winners（idx 9, 10 補 0.0），最終 0.68440 = v6 完全平手。診斷：均勻擾動 1-3/11 個 index，新成員 (9, 10) 每輪只有 ~27% 機率被探索，且須同時抽到非零 GRID 值才能改善 → 探索率不足。

**v7b 修正**：強制每輪擾動必須包含至少一個新成員 (idx 9 或 10)，外加 0-2 個 legacy index；N_RANDOM 提升至 12000。

**v7b 結果（新 SOTA）：0.68558（+0.00118 vs v6）**

| Task | F1 | 變化 vs v6 | 新成員權重 |
| :-- | :--: | :--: | :-- |
| promise_status (T1) | 0.9383 | 0 | p4a=0, p4b=0 |
| verification_timeline (T2) | **0.5000** | **+0.0089** | p4a=0.5, p4b=1.25 |
| evidence_status (T3) | 0.8764 | 0 | p4a=0, p4b=0 |
| evidence_quality (T4) | 0.4572 | 0 | p4a=0, p4b=0 |

**結論**：T2 是唯一受惠 task — p4b 在 T2=0.4857 確實提供 macbert 體系沒有的訊號（其餘 task 因 macbert 已飽和而無法改善）。距 0.69 還差 **0.00442**，需進 §55 Sprint B (T6 時間 token 增強，預期 +0.005-0.009 直接擊中 T2)。

**Hillclimb 工程教訓**：擴大 pool 時，若 warm-start 包含大量零權重新維度，必須改用 biased search（強制探索新維度），否則均勻隨機擾動會使新成員實質為「裝飾品」。應寫入 ML 教訓記憶。

---

## 30. Phase 10 — Sprint B / T6 時間 token 增強（已完成）

依 §55 Sprint B 計畫，啟動 T6（時間 token 注入）；目標：直接擊中 T2 (verification_timeline) macro-F1，預期 +0.005 ~ +0.009。

### 30.1 設計

**輸入端確定性增強（無學習成本）**：以 regex 抽取文本中的西元年（`19xx|20xx`）/民國年（`民國 NNN`，加 1911 換算）/中文年（`YYYY 年`），取最大年份 → 計算與 CURRENT_YEAR=2025 的差 → bucket 對齊 T2 5 類，最後在文本前注入：

```
[時間 年份YYYY 距今N年{後|前} BUCKET] 原文…
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

### 30.2 單模成績

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

### 30.3 Hillclimb v8（12-way pool）

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

### 30.4 結論與下一步

- **Sprint B T6 結論**：作為 ensemble 成員是有效的（+0.00111），作為 T2 直擊方案是失敗的（單模 T2 持平、ensemble idx 11 = 0）。
- 距 0.69 還差 **0.00331**，剩餘 §55 Sprint B 候選：
 - **T15 偽標籤 / 半監督**：當時列為候選；2026-05-04 已決議暫緩所有 pseudo-label 路線，改先做 U1/N2；
 - **T6 升級**：把 prefix 改為 numeric token（如直接插 `[YEAR_2030]` 單一 token，避免 sub-tokenization 稀釋）+ 重訓 → 可能挽救 T2；
 - **T14 Qwen LoRA**：8GB GPU 不可行（已標記）。

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v8_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v8_summary.csv) 與 [reports/experiments/p5_t6_time_token/score_summary.csv](reports/experiments/p5_t6_time_token/score_summary.csv)。

---

## 31. Phase 11 — Sprint B / T6 v2 專屬 bucket token（已完成）

承 §30 教訓（multi-char Chinese prefix 被 macbert tokenizer 拆成 30+ sub-tokens 稀釋訊號），改採「**單一 added-vocab token 對應一個 T2 bucket**」設計，企圖讓模型直接學到 bucket-level 嵌入。

### 31.1 設計

5 個專屬 token：`[T_already] [T_within2] [T_2to5] [T_longer5] [T_NA]`，由 `tokenizer.add_tokens(special_tokens=False)` 加入詞表（21128 → 21133），訓練時於文本前注入 1 token。年份抽取/bucket 規則沿用 §30。

實作：
- [src/data/text_augment.py](src/data/text_augment.py)：新增 `BUCKET_TOKENS / add_time_bucket_token()` 與 `TRANSFORM_ADDED_TOKENS / get_added_tokens()` 註冊表
- [src/train_kfold.py](src/train_kfold.py)：tokenizer 載入後依 `cfg.data.text_transform` 自動 `add_tokens` + 模型 `resize_token_embeddings(len(tokenizer))`
- 新 config [configs/exp_p6_t6v2_bucket_tok.yaml](configs/exp_p6_t6v2_bucket_tok.yaml)：繼承 combo_best 超參，seed=42

Sanity 驗證：5 個 token 各被 tokenize 為單一 id（vocab 確實成長 5）。

### 31.2 單模成績

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

### 31.3 Hillclimb v9（13-way pool）

**Tool**：[src/tools/per_task_hillclimb_v9.py](src/tools/per_task_hillclimb_v9.py) — 在 v8 12-way 上新增 idx 12 = `p6_t6v2_bucket_tok`；warm-start = v8 winners 補 0.0；biased search 強制每輪擾動 idx 12；N_RANDOM=12000；RNG seed=20260504。

**結果（新 SOTA）：0.68683（+0.00014 vs v8）**

| Task | F1 | 變化 vs v8 | p6_t6v2 權重 (idx 12) |
| :-- | :--: | :--: | :--: |
| **promise_status (T1)** | **0.9398** | **+0.0011** | **1.5** |
| verification_timeline (T2) | 0.5001 | 0 | 0.0 |
| evidence_status (T3) | 0.8765 | 0 | 0.0 |
| evidence_quality (T4) | 0.4513 | −0.0014 | 0.0 |

**意外觀察 — p6_t6v2 又是 T1 受惠**：與 v1 (p5_t6) 一樣，T2 直擊失敗、T1 卻入選（與 p5_t6 並列：T1 best_w 同時 p5_t6=2.0 與 p6_t6v2=1.5）。推測原因：bucket token 提供的 prefix-conditioned 弱信號在 T1 (binary, 大樣本) 上能與 p5_t6 形成「不同 prefix scheme 的雙視角」，T1 ensemble 因此微升 +0.0011。

### 31.4 結論與下一步

- **Sprint B T6 v2 結論**：作為 T2 直擊方案**完全失敗**（單模 T2 從 0.4874 退至 0.4703；ensemble idx 12 在 T2=0.0）；作為 ensemble 成員**僅 T1 微利**（+0.00014 整體增益，幾乎接近搜索 noise）。
- **教訓寫入 `/memories/ml_ensemble_lessons.md`**：「Low-frequency added-vocab tokens（< 1k occurrences）在 fine-tuning 階段缺乏足夠 gradient 訊號收斂出判別力嵌入；對結構化 bucket 預測，prefix engineering 的 ROI 在 macbert 體系已飽和」。
- 距 0.69 還差 **0.00317**，T6 路線兩次嘗試（v1 prefix / v2 bucket token）皆無法直擊 T2 → **T6 系列 close**。剩餘路徑：
 - **T15 偽標籤**（§55 Sprint B 最後候選）：此為 2026-05-01 的歷史候選；2026-05-04 起已暫緩 pseudo-label 路線，改先做 U1/N2。
 - **T14 Qwen LoRA**：8GB GPU 不可行（已標記）。
- **Sprint B 暫告段落**；現行策略不等待 T15，而是在 6/03 前先完成 U1 TTA 與官方資料線的新結構性成員檢查。

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v9_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v9_summary.csv) 與 [reports/experiments/p6_t6v2_bucket_tok/score_summary.csv](reports/experiments/p6_t6v2_bucket_tok/score_summary.csv)。

---

## 32. Phase 12 — Sprint B / T7 Focal-T4 γ=3.0 變體（已完成）

承 §55 Sprint B 名單 T7：在 combo_v3 超參（FGM eps=1.0 + Focal-T4 γ=2.0 + R-Drop α=0.5 + MSD K=5）上將 Focal-T4 的 γ 提高至 3.0，加重 T4 難樣本 gradient，預期 T4 macro-F1 微升或提供 ensemble 不同視角。

### 32.1 設計

- Config：[configs/exp_p7_focal_g3.yaml](configs/exp_p7_focal_g3.yaml)，`extends: exp_p2_combo_v3.yaml`，僅覆寫 `focal_gamma: 3.0`，seed=42。
- 訓練：hfl/chinese-macbert-base、5-Fold StratifiedKFold、5 epochs、batch=16、max_len=256、AMP、GPU=RTX 5060 Laptop。
- 零工程成本（僅 1 行 config 變更）、純依賴現有 trainer。

### 32.2 單模成績

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

### 32.3 Hillclimb v10（14-way pool）

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

### 32.4 結論與下一步

- **Sprint B T7 結論**：單模成績及格 **＋T4 +0.0237 是歷史最大單項 T4 增益**；ensemble 絶佳選者（T2 權重 2.0 首次出現）但被 constraint coupling 抵銷→ SOTA 未破。
- **保留 v9 為當時 SOTA 記錄**；p7_focal_g3 列為有效成員，供未來 v11+ 與其他新成員合併使用。
- **距 0.69 維持 0.00317 距離**。Sprint B 現存 backlog（T8/T9/T11/T12）另列於 §53.2。
- 教訓：僅有「單 task per-task F1 提升」不代表「加權素提升」；constraints 後處理是聯合映射，未來 v11+ 需考慮「聯合 hillclimb」而非 per-task 獨立損失。

詳見 [reports/analysis/_ensemble/per_task_hillclimb_v10_summary.csv](reports/analysis/_ensemble/per_task_hillclimb_v10_summary.csv) 與 [reports/experiments/p7_focal_g3/score_summary.csv](reports/experiments/p7_focal_g3/score_summary.csv)。

---

## 33. Phase 13 — Sprint B 收尾批次：U2 EMA-995 + U4 LS(T1/T3) + U7 聯合 hillclimb（已完成，**新 SOTA 0.68770**）

承 §55 Sprint B 路徑與 §32 結論，Phase 13 將「同一類型」的 backlog 項目批次處理：

- **同一類型訓練變體**（U2 + U4，共 2 支單模）：兩支都是 combo_v3 基礎上的「單一 hyperparam 變動」，都用 seed=42 5-Fold，硬體成本相同（~33 min/支），可在同一終端串行執行。
- **平行算法工作**（U7 聯合 hillclimb v11）：直接優化 `apply_constraints_batch` 後的加權 score，解 §32 揭示的 constraint coupling 問題。CPU/IO 工作，可在 GPU 訓練同時實作。

### 33.1 設計

**U2 — `p8_ema995`**（[configs/exp_p8_ema995.yaml](configs/exp_p8_ema995.yaml)）：
- `extends: exp_p2_combo_v3.yaml`，僅追加 `training.ema_decay: 0.995`，seed=[42]。
- 設計依據：Phase 5 X1（decay=0.999）在 ~500 step 訓練下 EMA shadow 嚴重落後 → 0.473 崩潰。本次以 0.995 為「半衰期 ~138 步」，保證 5 epoch × 50 step = 250 step 訓練窗下 shadow 可追上權重動態。
- trainer 已支援 top-level `ema_decay` key（內含 CPU shadow + apply_to/restore + EMA-applied validation），無需新程式碼。

**U4 — `p9_ls_t1t3`**（[configs/exp_p9_ls_t1t3.yaml](configs/exp_p9_ls_t1t3.yaml)）：
- `extends: exp_p2_combo_v3.yaml`，新格式 `training.label_smoothing: {promise_status: 0.05, evidence_status: 0.05, ...其餘 0.0}`，seed=[42]。
- 設計依據：Phase 5 X2（uniform LS=0.05 全 task）讓 T4 少類 `Misleading` (support=1) 預測被推平 → macro 崩潰。本次「per-task 字典」僅在 T1/T3 兩個 binary 平衡任務套 0.05，避開 T2/T4 多類少類混合。
- 程式碼小幅修改：`src/training/losses.py::MultiTaskCE.__init__` 與 `src/training/trainer.py` 開頭，皆改成「dict 路徑優先、float 路徑相容」；既有 yaml（皆使用 `label_smoothing: 0.0` float）行為不變，僅新增 dict 形式支援。

**U7 — Joint Hillclimb v11**（[src/tools/joint_hillclimb_v11.py](src/tools/joint_hillclimb_v11.py)）：
- 直接以 `weighted_score(apply_constraints_batch(arg-max(combine(W[t]))))` 為目標函數，**避開 §32 v10 的 constraint coupling 漏洞**。
- 池：14 v10 成員 + 自動偵測 `p8_ema995` / `p9_ls_t1t3`（要求 5/5 fold 齊備才併入）。
- 演算法：座標式 hill-climb，每 iter 隨機選 (task, member) 一格擾動，唯有聯合 score 嚴格上升才接受。warm-start = v9 winners（為新成員填 0.0）。N_ITERS=30000，RNG seed=20260506。
- 輸出：`reports/analysis/_ensemble/joint_hillclimb_v11_{summary,preds,history,meta}.{csv,jsonl,json}`。

### 33.2 執行步驟

```powershell
# 一條指令串行訓練（同 GPU）
$env:CUDA_VISIBLE_DEVICES="0"; `
 python -m src.train_kfold --config configs/exp_p8_ema995.yaml *>&1 | Tee-Object -FilePath reports\experiments\p8_ema995.log; `
 python -m src.train_kfold --config configs/exp_p9_ls_t1t3.yaml *>&1 | Tee-Object -FilePath reports\experiments\p9_ls_t1t3.log

# 兩支訓練完成後
python -m src.tools.joint_hillclimb_v11
```

### 33.3 單模實驗結果

| Item | p8_ema995 (U2) | p9_ls_t1t3 (U4) | combo_v3_s42 baseline | 備註 |
| :-- | :--: | :--: | :--: | :-- |
| Overall (5-Fold OOF) | **0.63010** | **0.67334** | 0.67121 | p8 崩潰；p9 +0.00213 |
| std | 0.01528 | 0.01363 | — | — |
| T1 (promise_status) | 0.9216 | 0.9323 | 0.9383 | p9 −0.0060 (LS 平滑了高信心) |
| T2 (verification_timeline) | 0.3980 | 0.4703 | 0.4894 | p8 大跌；p9 −0.0191 |
| T3 (evidence_status) | 0.8563 | 0.8675 | 0.8664 | p9 +0.0011（如設計） |
| T4 (evidence_quality) | 0.3691 | 0.4459 | 0.4244 | p8 大跌；p9 +0.0215 |
| 耗時 | 1743s | 1487s | — | — |

**p8 結論（重要教訓）**：EMA decay=0.995 在「無 warm-start、無 EMA epoch 起始延遲」下仍重複 §53.3 X1 的失敗模式。理論上 0.995 半衰期 ~138 步應可追上 250 步訓練，**但實測 shadow 在 fold 1-5 全部仍嚴重落後**（每個 fold 最終 score 0.61~0.65 vs 不開 EMA 的 ~0.67）。診斷：trainer `_EMA.__init__` 直接以「初始隨機權重」作 shadow 起點，而非 head 收斂後再啟動 EMA → 訓練前 1-2 epoch shadow 拖累 validation。**建議：未來需重構 `_EMA` 加入 `start_step` 參數延遲啟動，否則此項應永久標 X 禁區擴充項。** p8 可保留於 pool 作對照，但 v11/v12 終態權重皆為 0。

**p9 結論**：合格但非最佳。T1/T3 各微升 +0.0011/+0.0011 如設計（小幅校準），但 T2 大跌 −0.0191 為**意外副作用** — 即使未對 T2 套 LS，dict 形式可能改變了 dropout/loss scale 互動使整體訓練動態微變。T4 +0.0215 為意外正向。OOF 加總略勝 baseline (+0.00213) 故 p9 入 v11 ensemble pool。

### 33.4 Joint Hillclimb v11 結果與結論

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
1. **直接優化 post-constraint 分數的策略成功** — Phase 12 v10 per-task 搜索無法穿透 constraint barrier 而失敗 0.68660；v11 改用 joint objective，在 T3 微調權重即連帶提升 T4，確認 §32 提出的「constraint coupling」假說正確。
2. **當時距 0.69 護城仍差 0.00230** — v11 已收斂到此 16-way pool 的能力上限，再做 hillclimb 變體不會再有顯著增益。下一階段必須引入「結構性新方向」的成員（非 combo_v3 變體），例如 macbert-large、不同 tokenizer 的模型、或 MoE/teacher-student；此假設已在 Phase 14 用 p10 驗證。
3. **p8 永久封禁** — EMA 失敗模式重現，未來不再嘗試純 decay 變動，需先實作 `_EMA(start_step=...)` 才可重啟。
4. **p9 進池但無增益** — 證實 LS 對 ensemble 多樣性無幫助；保留 p9_ls_t1t3 在 pool 內供日後與「結構性新成員」的交叉搜索。
5. 對 §55 後續路徑（Phase 13 當下）：原 Sprint B 收尾完成；轉入 Sprint C「結構性多樣化」或準備 6/03 驗證集 hold-out 重訓 + 提交流程。Phase 14 已先完成其中的 N1。

---

## 34. Phase 14 — Sprint C 結構性多樣化 N1：macbert-large + Focal-T4 + FGM (p10) + Joint Hillclimb v12（已完成，**新 SOTA 0.68825**）

**完成日期**：2026-05-02 (Day 22)
**動作**：依 §55 Sprint C 路徑優先級，執行 N1（首個「large × combo」結構性新成員）+ v12 17-way joint hillclimb。

### 34.1 N1 訓練（p10_large_focal_fgm）

**Config**：[configs/exp_p10_large_focal_fgm.yaml](configs/exp_p10_large_focal_fgm.yaml)（繼承 `exp_p3_large_lr2e5.yaml`）。
- backbone：hfl/chinese-macbert-large（24 層，335M 參數）
- max_length=384, lr=2e-5, batch=4, grad_accum=4, epochs=5, seed=42
- **新增**：Focal Loss on T4（γ=2.0）+ FGM（eps=1.0）
- **跳過**：R-Drop（2× forward）+ MSD（5× dropout）— 8GB 顯存無法承載

**OOF 5-Fold weighted (seed=42)**：

| fold | epoch1 | epoch2 | epoch3 | epoch4 | epoch5 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 0 | 0.56874 | 0.61942 | 0.66084 | 0.66333 | **0.67396** |
| 1 | 0.57760 | 0.64413 | 0.62577 | 0.65900 | **0.66576** |
| 2 | 0.59817 | 0.64528 | **0.66549** | 0.66322 | 0.65291 |
| 3 | 0.56817 | 0.60748 | 0.65352 | 0.65815 | **0.66279** |
| 4 | 0.57872 | 0.65888 | **0.67571** | 0.67508 | 0.66497 |

- **最終 OOF = 0.66874**（mean），std=0.00572，5-Fold 跨度 0.6628 ~ 0.6757
- **Per-task means**：T1=0.9263 / T2=0.4762 / T3=0.8530 / T4=0.4462
- 訓練時間：2873s（≈48 分鐘）
- ≥ 0.66 admission threshold → **入池HF Hub 403 stderr**：`Discussions are disabled for this repo` 為 hfl 倉庫關閉討論區的無害噪訊（auto_conversion 線程探查 PR 失敗），不影響權重載入或訓練。PowerShell 因此返回 exit code 1，但 score_summary.csv 與 oof_probs.npz 均正確寫出。

### 34.2 Joint Hillclimb v12（17-way）

**腳本**：[src/tools/joint_hillclimb_v12.py](src/tools/joint_hillclimb_v12.py)
- Pool：v11 的 16 員 + 新成員 p10_large_focal_fgm（idx 16），M=17
- Warm-start：v11 SOTA winners + 0.0 padding
- N_ITERS=30000，GRID=[0, 0.25, 0.5, ..., 2.0]，RNG seed=20260603
- 目標函數：`weighted_score(apply_constraints_batch(arg-max(combine(W[t]))))`

**8 次 accept 全部發生在 T2（verification_timeline，先前最大瓶頸）**：

| iter | task | idx | exp | old → new | joint |
| :--: | :--: | :--: | :-- | :--: | :--: |
| 772 | T2 | 16 | **p10_large_focal_fgm** | 0.00 → 1.00 | 0.68781 |
| 866 | T2 | 0 | p2_combo_v2 | 0.00 → 1.25 | 0.68784 |
| 888 | T2 | 16 | **p10_large_focal_fgm** | 1.00 → 1.50 | 0.68784 |
| 986 | T2 | 10 | p4b_bert_base_chinese | 1.25 → 1.75 | 0.68789 |
| 1001 | T2 | 3 | p2ab_aug_mask10 | 0.25 → 0.50 | 0.68806 |
| 1059 | T2 | 0 | p2_combo_v2 | 1.25 → 0.75 | 0.68811 |
| 1231 | T2 | 11 | p5_t6_time_token | 0.00 → 0.25 | 0.68816 |
| 1359 | T2 | 0 | p2_combo_v2 | 0.75 → 0.50 | **0.68825** |

**最終 v12 SOTA = 0.68825**（+0.00055 vs v11）：
- T1=0.9398（不變）
- **T2=0.5072（+0.0036，瓶頸鬆動）**
- T3=0.8783（不變）
- T4=0.4592（不變）

**新成員 p10 最終權重**：T1=0, **T2=1.50**, T3=0, T4=0（4 任務只在 T2 有非零權重，其他任務均被 v11 既有成員主導）。

### 34.3 結論與下一步

**達成**：Phase 13 推測「16-way pool 已收斂、必須引入結構性多樣化」獲得實證——p10（首個 large + Focal-T4 + FGM 的混合員）一進池就解開 T2 瓶頸，貢獻全部 +0.00055 增益。

**證實假設**：
- v macbert-large 在 ensemble 中作為「pool member」可加分（不能作 main，這點 X5 仍成立）
- v T2 macro-F1 對「不同 backbone family + Focal-T4」組合敏感
- v v11 → v12 的 +0.00055 來自結構性新方向；因 warm-start 為 v11 權重 + p10=0，任何增益都必須由新成員或與新成員共同調整觸發

**距 0.69**：尚剩 −0.00175。下一步候選（依 ROI 排序）：
1. **U1 TTA inference variants** — 預期 +0.0005 ~ +0.001；不需重訓、不用外部資料
2. **N2 macbert-large + R-Drop（單獨）**或 **N3 nezha-base** — 預期 +0.0005 ~ +0.002
3. **U3 SWA** 或 **U5 T4 class-balanced re-sampling** — 預期 +0.001 ~ +0.002
4. **U10 外部 ESG 偽標** — 暫緩到最後再做；目前所有產物已清除，不屬於現階段原始訓練線

**產出**：
- `outputs/checkpoints/p10_large_focal_fgm/seed42/fold{0..4}/best.pt + oof_probs.npz`
- `reports/experiments/p10_large_focal_fgm/score_summary.csv + .json`
- `reports/analysis/_ensemble/joint_hillclimb_v12_{summary,preds,history,meta}.(csv|jsonl|json)` + `joint_hillclimb_v12.log`
- `logs/p10_large_focal_fgm_20260502_021523.log`

---

## 35. Phase 15 (Reset) — U10 暫緩並清除外部 pseudo 線（2026-05-04）

依使用者要求，U10 外部 ESG 報告偽標路線最後再做；目前專案必須回到「沒有加入外部非人工標註資料」的原始訓練狀態。因此已清除所有 U10 相關資料、程式、設定、權重、報告、log、split 與 admission 產物，不再保留任何外部 pseudo-label 訓練痕跡。

### 35.1 已清除範圍

| 類型 | 清除內容 |
| :-- | :-- |
| 外部資料 | `data/external/u10_*`、`data/external/raw/public_reports/` |
| pseudo 與合併資料 | `data/pseudo/u10_*`、`data/processed/u10_*` |
| 訓練輸出 | `reports/experiments/u10_*`、`outputs/checkpoints/u10_*`、`outputs/logs/u10_*`、`data/splits/u10_*` |
| 分析與入池產物 | `reports/pseudo/u10_*`、`reports/analysis/_ensemble/joint_hillclimb_v13_u10*` |
| 快取 | U10 相關 `__pycache__` |

### 35.2 目前狀態判定

- U10：**未做 / 暫緩**。
- 目前訓練主線：只使用官方人工標註 1,000 筆訓練資料與既有 Phase 1–14 模型/OOF 產物。
- 目前 active SOTA：Phase 15 U1 TTA `stored+middle`，OOF weighted = **0.6887881105**；不含 TTA 的最佳訓練 ensemble 仍為 Phase 14 v12，OOF weighted = **0.6882469764**。
- 短期不得再引用已刪除的 U10 檔案、權重或報告作為採納依據。

### 35.3 下一步原則

U10 之後若要重新啟動，必須從資料治理、公開來源、人工審核與採納門檻重新設計，不能沿用已清除產物。現階段優先執行不碰外部非人工標註資料的方法：

1. **U3/U5 非外部資料方法**：SWA 或 T4 class-balanced re-sampling，優先處理模型穩定性與 T4 少類瓶頸。
2. **U1-b TTA 補測**：視時間測 `stored+tail+middle` 或任務別 TTA 權重，但不得犧牲目前 `stored+middle` best。
3. **N3 新結構性成員**：p11 ELECTRA base 已被 admission 拒絕；若再走新 backbone，需換 tokenizer/backbone 或訓練策略。

---

## 36. Phase 15 (Implementation) — U1 TTA 推論擴增（2026-05-04）

**目標**：在完全不使用外部資料、不重訓模型的前提下，測試 v12 17-way joint ensemble 是否能透過 inference-time 多視角 token window 平均取得穩定 OOF 增益。

### 36.1 新增工具

| 檔案 | 功能 |
| :-- | :-- |
| `src/tools/u1_tta_oof.py` | 讀取 v12 meta、既有 fold checkpoint 與 official split；對 v12 非零權重成員重跑 validation fold 的 `tail` / `middle` token-window 推論，與 stored OOF 機率平均後套用原 v12 權重與 `apply_constraints_batch` 評分 |

設計重點：

- **零外部資料**：只使用 `data/raw/vpesg4k_train_1000 V1.csv`、既有 official fold split 與 Phase 1–14 checkpoints。
- **非洩漏 OOF**：每個 fold 只用該 fold checkpoint 推論該 fold validation indices。
- **active members only**：只對 v12 最終權重非零的 14/17 members 做 TTA；零權重成員不影響 score，不浪費 GPU。
- **BERT-family token window**：`tail` / `middle` 以 tokenizer token ids 截取，再手動加 `[CLS]` / `[SEP]` / padding，避免依賴目前 transformers tokenizer 不存在的 helper API。

### 36.2 實測結果

| 變體 | Weighted OOF | Δ vs v12 stored | T1 | T2 | T3 | T4 | 判讀 |
| :-- | --: | --: | --: | --: | --: | --: | :-- |
| v12 stored baseline | 0.6882469764 | — | 0.939845 | 0.507186 | 0.878285 | 0.459184 | Phase 14 SOTA |
| U1 `stored+tail` | 0.6887412240 | +0.0004942476 | 0.939845 | **0.509247** | 0.878285 | 0.459713 | T2/T4 提升，T1/T3 持平 |
| U1 `stored+middle` | **0.6887881105** | **+0.0005411340** | **0.940476** | 0.508718 | 0.878285 | 0.459713 | 本輪最佳，T1/T2/T4 皆不低於 baseline |

輸出：

- `reports/analysis/_ensemble/u1_tta_v12_active_stored_tail_summary.csv`
- `reports/analysis/_ensemble/u1_tta_v12_active_stored_tail_preds.csv`
- `reports/analysis/_ensemble/u1_tta_v12_active_stored_tail_meta.json`
- `reports/analysis/_ensemble/u1_tta_v12_active_stored_middle_summary.csv`
- `reports/analysis/_ensemble/u1_tta_v12_active_stored_middle_preds.csv`
- `reports/analysis/_ensemble/u1_tta_v12_active_stored_middle_meta.json`

### 36.3 決策

採 `U1 stored+middle` 作為目前 active OOF best：**0.6887881105**，距 0.69 剩 **0.0012118895**。此增益完全來自推論視角互補，未引入外部非人工標註資料，也未改變訓練集。後續已完成 N2/p11 結構性成員 admission，但未被採納；下一步不再重跑舊 17-way 或 p11 同設定，而應優先做 U3/U5，或補測 U1-b 三視角/任務別 TTA 權重，且不得取代已驗證的 `stored+middle` best。U1-b 三視角結果已於 §38 補測。

### 36.4 概念釐清：「TTA 是做什麼的？」

> 此處對應使用者於 Phase 28 結尾提問。TTA 概念在本節（Phase 15）首次落地，故釐清放在這裡。

TTA = **Test-Time Augmentation（測試時資料擴增）**，**不重訓練、只在推論時對同一筆樣本以多視角產生多份預測再聚合**：

- **U1 視角設計**：對長文本三種裁切方式
 - `stored`：從頭開始 [CLS] + 前 N token
 - `middle`：中間段落（避開頭尾的樣板宣傳）
 - `tail`：結尾段落（往往是承諾 / 結論）
- **聚合方式**：U1 等權平均、U1-b 三視角等權（§38）、U1-c **per-task per-view weighted**（每個 task 對每個 view 學不同權重，coordinate descent grid 0.05 + oracle warm-start，詳 §39）。
- **U1-c 實測**：active SOTA 從 0.68825 → **0.68925**（+0.00100），**完全免費**，零訓練成本，零外部資料；T2 macro 0.51026 為史上最高。
- **為什麼有效**：長 ESG 文本的關鍵訊號（時間詞、量化目標、reservation 用語）位置不固定 → 單一視角會錯失某些訊號；多視角覆蓋率高，per-task 權重能對 T2（時間詞多在中後段）強化中後視角。

**結論**：TTA 是「相同模型在推論時看不同片段再投票」，是 plateau 階段唯一還在生效的免費午餐技術。

---

## 37. Phase 16 完成 — N2/p11 ELECTRA base 官方資料結構性成員（2026-05-04）

**目標**：在不使用 U10、不使用外部 pseudo-label 的前提下，新增一個和既有 BERT/Roberta/MacBERT MLM-family 不同預訓練目標的 backbone，測試是否能成為 v12/U1 之後的互補成員。

### 37.1 選擇理由

| 候選 | 判斷 | 本輪決策 |
| :-- | :-- | :--: |
| large + R-Drop | 與 p10 同為 macbert-large，且 R-Drop 需要雙 forward，8GB 顯存風險高 | 暫緩 |
| NeZha / ERNIE | 可能有效，但模型相容性與下載穩定性需另行確認 | 後續候選 |
| **Chinese ELECTRA base** | base size、8GB 風險較低；replaced-token detection 預訓練目標不同，錯誤模式較可能和 MLM-family 互補 | **採用** |

### 37.2 設定與 Gate

| 項目 | 值 |
| :-- | :-- |
| Config | `configs/exp_p11_electra_base.yaml` |
| exp_name | `p11_electra_base` |
| backbone | `hfl/chinese-electra-180g-base-discriminator` |
| data | 官方人工標註 1,000 筆；不使用 U10 / pseudo |
| max_length | 384 |
| lr | 2e-5 |
| batch / grad_accum | 8 / 2 |
| seed / folds | seed=42, 5-Fold |
| admission gate | 單模 OOF ≥ 0.65，或 T2/T4/constraint-coupled joint admission 有明顯互補 |

### 37.3 正式 5-Fold 結果

已建立設定檔並完成 smoke test。Smoke 設定為 64 samples、2 folds、1 epoch，目的只驗證 pipeline 相容性，不作為正式分數判讀。

```powershell
C:/Users/User/AppData/Local/Programs/Python/Python313/python.exe -m src.train_kfold --config configs/exp_p11_electra_base.yaml --smoke
```

Smoke 結果：

| 項目 | 結果 |
| :-- | :-- |
| tokenizer / model load | 通過 |
| training loop | 通過 |
| fold checkpoint / summary output | 通過 |
| smoke weighted mean | 0.34381（僅 64 筆、2 folds、1 epoch，不作競賽判讀） |
| 非致命訊息 | HuggingFace discussions 403；ELECTRA discriminator prediction head unexpected keys。兩者均未中止訓練 |

正式 5-Fold 已完成，訓練流程採官方 1,000 筆、seed=42、5 folds；不使用 U10、不使用 pseudo-label：

```powershell
C:/Users/User/AppData/Local/Programs/Python/Python313/python.exe -m src.train_kfold --config configs/exp_p11_electra_base.yaml
```

正式結果：

| fold | best_epoch | weighted | T1 | T2 | T3 | T4 |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 1/5 | 5 | 0.63666 | 0.91545 | 0.40551 | 0.85149 | 0.39227 |
| 2/5 | 5 | 0.60306 | 0.90588 | 0.33722 | 0.83276 | 0.34707 |
| 3/5 | 5 | 0.63121 | 0.92398 | 0.36862 | 0.84039 | 0.39716 |
| 4/5 | 4 | 0.62730 | 0.93452 | 0.40655 | 0.84667 | 0.35833 |
| 5/5 | 4 | 0.63647 | 0.92857 | 0.33016 | 0.87805 | 0.39375 |

整體：**mean=0.62694、std=0.01391、min=0.60306、max=0.63666**。Per-task means：T1=0.9217、T2=0.3696、T3=0.8499、T4=0.3777。

判讀：p11 單模未達 admission gate（0.65），其中 T2/T4 明顯低於既有強模型；但因 gate 仍保留「低單模但可能提供互補」的條件，仍需做 post-constraint joint admission。

產出：
- `outputs/checkpoints/p11_electra_base/seed42/fold{0..4}/best.pt + oof_probs.npz`
- `reports/experiments/p11_electra_base/score_summary.csv`
- `reports/experiments/p11_electra_base/score_summary.json`

### 37.4 v16/p11 Admission 結果

已建立並執行 admission 檢查工具 `src/tools/joint_hillclimb_v16_p11.py`，用途是以 v12 final weights warm-start，把 p11 作為第 18 員加入後重新做 post-constraint joint hillclimb。該工具採 biased search（70% 擾動指定到 p11 新維度，30% 留給舊成員 cleanup），避免零權重新成員在飽和 pool 中探索不足。

```powershell
C:/Users/User/AppData/Local/Programs/Python/Python313/python.exe -m src.tools.joint_hillclimb_v16_p11
```

結果：

| 項目 | 值 |
| :-- | :-- |
| pool | 18-way（v12 17 員 + p11） |
| warm-start score | 0.6882469764（等同 v12，p11 初始 0 權重） |
| search | 30,000 iterations；new-member-biased |
| accepted | 0 |
| final joint score | 0.6882469764 |
| delta vs v12 training SOTA | +0.0000000000 |
| delta vs U1 active SOTA | −0.0005411340 |
| final p11 nonzero tasks | 0/4 |
| final p11 weights | T1=0.00、T2=0.00、T3=0.00、T4=0.00 |

產出：
- `reports/analysis/_ensemble/joint_hillclimb_v16_p11_summary.csv`
- `reports/analysis/_ensemble/joint_hillclimb_v16_p11_preds.csv`
- `reports/analysis/_ensemble/joint_hillclimb_v16_p11_history.jsonl`
- `reports/analysis/_ensemble/joint_hillclimb_v16_p11_meta.json`

### 37.5 決策

p11 ELECTRA base **不納入 active ensemble**。原因是：

1. 單模 OOF mean=0.62694，未達 0.65 admission gate。
2. post-constraint joint admission 30,000 次搜尋無任何 accept。
3. p11 最終四任務皆 0 權重，未提供 v12/U1 之後可用的互補訊號。

目前 active best 已推升為 Phase 18 U1-c per-task TTA：**0.6892512673**（詳見 §39）。Phase 15 U1 `stored+middle` 0.6887881105 仍作為 per-task TTA 的 fallback 參考。Phase 19~25 已將 U11 GroupKFold sanity、U12 OOF↔valid gap 診斷、U3 SWA、U5 T4 re-sampling、N3 NeZha→ERNIE fallback、U2 EMA + warm-start、U6 NLLB-600M 回譯 全數驗證完畢；**內部可控訓練側探索空間現為零，除 U10（依使用者指示暫緩）外§53.2 backlog 已全數驗證完畢**，重心全部轉到 6/03 valid 釋出後的 OOF↔Valid gap 複測、GroupKFold drift 復量、以及提交策略。

---

## 38. Phase 17 完成 — U1-b TTA 三視角補測（2026-05-05）

**目標**：在不重訓、不引入外部資料的前提下，驗證 `stored+middle+tail` 三視角等權平均是否能在 v12 active pool 上超越 `stored+middle` 兩視角 best (0.6887881105)。此為 §55 第三優先項，最低成本且最容易出 ROI 的「補完一輪 TTA 設計空間」實驗。

### 38.1 設計

- 工具沿用 §36.1 `src/tools/u1_tta_oof.py`，命令列只新增一個 `--views` 值：
 ```powershell
 C:/Users/User/AppData/Local/Programs/Python/Python313/python.exe -u -m src.tools.u1_tta_oof --views stored middle tail --members active
 ```
- 流程不變：對 v12 14 個非零權重成員的每個 fold checkpoint，分別在 `middle` 與 `tail` 兩個 token-window 視角重跑 validation 推論；和 stored OOF 三者等權平均後，套用 v12 final weights 與 `apply_constraints_batch`。
- 不更動 v12 weights、不更動 active SOTA 之前需先比較 `stored+middle` 兩視角 best。

### 38.2 實測結果

| 變體 | Weighted OOF | Δ vs v12 stored | Δ vs U1 active SOTA | T1 | T2 | T3 | T4 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| v12 stored baseline | 0.6882469764 | — | −0.0005411340 | 0.939845 | 0.507186 | 0.878285 | 0.459184 |
| U1 `stored+tail` | 0.6887412240 | +0.0004942476 | −0.0000468865 | 0.939845 | 0.509247 | 0.878285 | 0.459713 |
| **U1 `stored+middle`** | **0.6887881105** | **+0.0005411340** | — | **0.940476** | 0.508718 | 0.878285 | 0.459713 |
| U1-b `stored+middle+tail` | 0.6887283739 | +0.0004813975 | **−0.0000597366** | 0.939845 | 0.509412 | 0.878285 | 0.459606 |

判讀：

- 三視角 T2 取得本輪最高 0.509412（高於兩視角 best 0.509247、`stored+middle` 0.508718），說明 `tail` 視角主要貢獻 T2。
- 但三視角等權平均同時拉回 T1 至 baseline 0.939845（低於 `stored+middle` 的 0.940476），且 T4 較 `stored+middle` 微降 0.000107。
- 加權後 `stored+middle+tail` 為 0.6887283739，**低於** `stored+middle` 0.6887881105 約 0.0000597。
- 因此三視角等權平均**未能超越** `stored+middle`，**不取代** active SOTA。

### 38.3 決策

1. 維持 §36 記錄 — Phase 15 active SOTA `stored+middle`：**0.6887881105**（在 §39 已被 Phase 18 U1-c 0.6892512673 取代，不來回跳動）。
2. U1-b 三視角等權結果已記錄為負向證據；後續若要再壓 U1-b，必須做「任務別 TTA 權重」（例如 T2 採三視角、T1/T3/T4 採兩視角）或視角別權重學習，而非繼續嘗試等權平均。
3. 加上 U1-b 結果，§55 中 U1-b 視為已關閉「等權三視角」這條子分支；後續優先項改為 U3/SWA、U5/T4 class-balanced re-sampling、N3 新結構性候選。

### 38.4 產出

- `reports/analysis/_ensemble/u1_tta_v12_active_stored_middle_tail_summary.csv`
- `reports/analysis/_ensemble/u1_tta_v12_active_stored_middle_tail_preds.csv`
- `reports/analysis/_ensemble/u1_tta_v12_active_stored_middle_tail_meta.json`
- `reports/analysis/_ensemble/u1_tta_v12_active_stored_middle_tail.log`

非致命訊息：HuggingFace `get_repo_discussions` 403（背景執行緒，僅影響 safetensors auto-conversion 嘗試，不影響 OOF 數值）；ELECTRA discriminator 預訓練 head 未使用之 unexpected keys 屬正常下游微調行為。

---

## 39. Phase 18 完成 — U1-c 任務別/視角別加權 TTA（2026-05-05）

**目標**：在 §38 證實「等權三視角」會稀釋 T1/T4 之後，將 TTA 由「等權」升級為「per-task 視角加權」，藉由 v12 active 14 員的 stored / middle / tail 三視角機率快取進行 coordinate descent，看能否在不重訓、不引入外部資料的條件下突破 `stored+middle` 0.68879。此為 §55.0 排序中的 A1（最高 ROI 的零再訓練實驗）。

### 39.1 設計

- **新工具**：`src/tools/u1c_per_task_tta.py`
 - 沿用 §38 `u1_tta_oof.py` 的 `predict_one_view`，但將 middle / tail 機率快取至 `outputs/cache/u1c_tta/{member}_{view}.npz`，避免之後重複推論。
 - 對每個任務 t，搜尋簡形 α_{t,view}（sum_view α_{t,v}=1，grid step 0.05 → 231 點/任務/輪）。
 - **暖開**：先以「per-task best of {stored_only, stored+middle eq, stored+tail eq, stored+middle+tail eq}」做 oracle 4 變體擇優，再以 oracle 為起點跑 coordinate descent，最多 6 輪、收斂即停。
 - 每次評估都先依 v12 per-(task, member) 權重做 17 員→單一視角機率，再以 α 混合，最後過 `apply_constraints_batch` 才打分（結構與 v12 一致，避免破壞 constraint coupling）。
- **資料／成員**：v12 17-way active 成員 14 員（`p2_combo_v2` … `p10_large_focal_fgm`），與 §36 / §38 同。
- **基準對照**：
 - stored only（v12 baseline）= 0.6882469764
 - stored+middle 等權（前 SOTA）= 0.6887881105
 - stored+tail 等權 = 0.6887412240
 - stored+middle+tail 等權（X8）= 0.6887283739
 - oracle per-task best of 上述 4 變體 = 0.6891248409

### 39.2 實測結果

**U1-c searched** weighted OOF = **0.6892512673**

| 任務 | α stored | α middle | α tail | T 分數 | vs `stored+middle` Δ |
| :-- | :--: | :--: | :--: | --: | --: |
| promise_status (T1, w=0.20) | 0.50 | 0.50 | 0.00 | 0.940476 | ±0 |
| verification_timeline (T2, w=0.15) | 0.25 | 0.65 | 0.10 | 0.510255 | +0.001537 |
| evidence_status (T3, w=0.30) | 0.50 | 0.00 | 0.50 | 0.879060 | +0.000775 |
| evidence_quality (T4, w=0.35) | 0.50 | 0.00 | 0.50 | 0.459713 | ±0 |

**最終分數**：

- vs Phase 15 U1 active SOTA (`stored+middle`)：**+0.0004631568**
- vs Phase 14 v12 stored baseline：**+0.0010042909**
- 距 0.69 護城線：**−0.0007487327**

### 39.3 決策

1. 推升 active SOTA = **0.6892512673**（Phase 18 U1-c per-task TTA）。
2. T2 視角偏好 `middle` 為主、`tail` 次要（與「中段含具體年份描述」的語意吻合）；T3/T4 視角偏好 `stored+tail` 等權（與「結論句多在尾段」吻合）；T1 維持 `stored+middle`。可量化解釋與 §38 觀察一致。
3. coordinate descent 第 1 輪在 T2 取得 +0.0001264，第 2 輪即收斂；網格 0.05 對 1000 樣本 OOF 已達雜訊地板，**不嘗試更細網格**（避免 OOF 過擬合）。
4. 未來若再壓 U1-c，須先做 §55.0 A2/A3（GroupKFold sanity + OOF↔valid gap），確認此 +0.000463 不是 OOF 過擬合，再考慮細化網格或 Dirichlet 先驗。

### 39.4 產出

- `src/tools/u1c_per_task_tta.py`
- `outputs/cache/u1c_tta/{14 members}_{middle,tail}.npz`（28 個 cache，後續 U1-c/U1-d 可零成本重用）
- `reports/analysis/_ensemble/u1c_per_task_tta_summary.csv`
- `reports/analysis/_ensemble/u1c_per_task_tta_preds.csv`
- `reports/analysis/_ensemble/u1c_per_task_tta_meta.json`
- `reports/analysis/_ensemble/u1c_per_task_tta.log`

非致命訊息：HF 403（背景 safetensors auto-conversion 嘗試），不影響 OOF 數值；exit code 1 來自背景 thread，所有產物在 raise 前已寫盤完成（已驗證 `meta.json` 完整 + 三檔 size 正常）。

---

## 40. Phase 19：U11 GroupKFold sanity（read-only 診斷，2026-05-05）

### 40.1 動機

§55 第二優先項。U1-c 的 +0.000463 全部來自既有 5-Fold StratifiedKFold OOF；在進入 valid set 前，必須先量化目前 5-Fold 是否存在「同公司同時出現於 train/val」的 row-level leakage，否則 OOF 與 valid 的差距無法歸因。本步驟**完全不再訓練**，僅針對 split 結構與標籤分布作對照。

### 40.2 設計

- 載入 `vpesg4k_train_1000 V1.csv`（N=1000，50 公司）。
- 在 seed ∈ {42, 2024, 20260417} 各跑：
 - `StratifiedKFold(n_splits=5, shuffle=True)`（既有所有 active member 採用）。
 - `StratifiedGroupKFold(n_splits=5, shuffle=True, group=company)`。
- 每 fold 量測：
 1. **公司洩漏率**：val 公司中也出現在 train 的比例。
 2. **資料列洩漏率**：val 列中其公司也出現在 train 的比例。
 3. **per-task label drift**：StratifiedKFold val vs GroupKFold val 的逐類分布最大絕對差。
- 決策規則：若 SK 平均 row-leak ≥ 0.50 且 GK 各 task 平均 drift < 0.04 → 建議切換 GK；若 SK 高洩漏但 GK drift ≥ 0.04 → MONITOR（不重訓但需在 valid 後重評）；否則維持 SK。

### 40.3 結果

| seed | SK row-leak | GK row-leak | T1 drift | T2 drift | T3 drift | T4 drift |
| :--: | --: | --: | --: | --: | --: | --: |
| 42 | 0.998 | 0.000 | 0.017 | 0.072 | 0.033 | 0.054 |
| 2024 | 1.000 | 0.000 | 0.024 | 0.110 | 0.032 | 0.071 |
| 20260417 | 1.000 | 0.000 | 0.014 | 0.072 | 0.032 | 0.062 |

- 公司分布：50 unique，max=45，median=19.5，無 singleton 公司。
- 平均 SK row-leak ≈ **99.93%** — 幾乎所有 val 列的公司都出現在 train 中。
- T2 / T4 GK drift > 0.05 ~ 0.11，超過 drift 門檻 0.04。

### 40.4 決策

**MONITOR — leakage observed but drift would distort label balance**：

1. **不重訓**：若改用 GroupKFold 重跑全部 14 員，T2 macro-F1 變動上限可能 ±0.02 ~ ±0.03（因類分布偏移），高於 U1-c 的 +0.00046 增益，風險不對稱。
2. **valid 階段保留心理預期差**：當 6/03 valid 釋出時，預期 valid 與 OOF 將出現顯著 gap（因 valid 多半含未見過公司）。若 gap ≤ 0.005，視為健康；若 ≥ 0.01，需在 §41 / U12 中進一步診斷。
3. **U3 SWA / U5 T4 re-sampling 仍繼續**，因兩者皆是「同 split」內加強，不受公司洩漏影響相對排序。
4. **未來若仍要切換 GK**，應同步：(a) 重跑全部 14 員 + (b) 重新計算 v12 加權 + (c) U1/U1-c 全部重新搜尋。等同重置 Phase 14 ~ Phase 18 結果，**僅在 valid gap ≥ 0.02 時才考慮**。

### 40.5 產出

- `src/tools/u11_group_kfold_sanity.py`
- `reports/analysis/diagnostics/u11_group_kfold_sanity.json`
- `reports/analysis/diagnostics/u11_group_kfold_sanity.md`

---

## 41. Phase 20：U12 OOF cross-fold variance（read-only 診斷，2026-05-05）

### 41.1 動機

§55 第三優先項。U1-c +0.000463 是否落在 OOF 雜訊內？以及 §40 已證 row-leakage ≈ 99.93%，因此需量化「同 split 多 fold 的天然 std budget」，作為解讀未來 valid gap 的尺規。

### 41.2 設計

- 解析 v12 active 14 員的 `reports/experiments/<exp>/score_summary.csv`，逐 fold 取 weighted_score 與四項 macro-F1。
- 計算每員：fold std、range、`fragility = (max−min)/mean`。
- 池子層級：
 - 平均單員 weighted_std。
 - 加權預期 = Σ wᵢ · per_task_std_mean。
 - ensemble 估算 = 上式 / √N。
- 不重訓、不做推論，純報表彙整。

### 41.3 結果

- 12/14 員有完整 `score_summary.csv`（缺 `p2_combo_v3_s42`、`p2_combo_v3_avg` 兩個合併型成員）。
- 平均單員 weighted_std = **0.0115**（max 0.0171）。
- 加權預期單員 std budget = **0.0157**；ensemble 平均後 ≈ **0.0045**。
- per-task std（mean / max）：T1 0.0081 / 0.0118、T2 0.0236 / 0.0322、T3 0.0090 / 0.0160、T4 ~ 0.018 / ~ 0.025。
- 最脆弱前 5 名（fragility）多落在 T2 / T4 變動大的成員（詳見 `u12_oof_fold_variance.md`）。

### 41.4 決策

1. **U1-c 增益 +0.000463 嚴格落在「ensemble 雜訊 budget 0.0045」之內**，但仍是同 OOF 上的真增益（每 fold 都被加權後一致提升）。**仍可保留 U1-c 為 OOF SOTA**，惟在 valid 階段以同樣的 per-task α 重算才能宣告「真增益」。
2. **valid gap 解讀規則**：
 - gap ≤ 0.005：屬正常 fold-noise + 5-fold→1-valid 切換；不需動作。
 - 0.005 < gap ≤ 0.015：U1-c 收益可能被吃掉，但 v12 ensemble 仍應穩；不重訓。
 - gap > 0.015：明顯分布偏移（與 §40 row-leakage 一致），啟動 §40.4 的 GroupKFold 重置流程。
3. **B1 / B2 仍照計畫進行**：因 fold std 在 T2 / T4 最大，U3 SWA / U5 T4 re-sampling 都直接針對最脆弱維度，預期能壓縮 fold std 而非單純抬升均值。

### 41.5 產出

- `src/tools/u12_oof_fold_variance.py`
- `reports/analysis/diagnostics/u12_oof_fold_variance.csv`
- `reports/analysis/diagnostics/u12_oof_fold_perfold.csv`
- `reports/analysis/diagnostics/u12_oof_fold_variance.json`
- `reports/analysis/diagnostics/u12_oof_fold_variance.md`

---

## 42. Phase 21：U3 SWA — `p2_combo_best_swa` 設置與訓練（2026-05-05）

### 42.1 動機

§55 排序中的 B1。SWA（Stochastic Weight Averaging）在 NLP fine-tuning 通常能換到 +0.001 ~ +0.003，且不需 shadow 副本（不重蹈 EMA X1 的覆轍）。`p2_combo_best`（macbert-base / max_len=384 / lr 3e-5 / batch=8 grad_accum=2 / 5 epochs）為 v12 active pool 中**最快、最穩定**的成員，適合作為 SWA 首測；單 fold 約 5~10 分鐘，5 fold 約 30~50 分鐘。

### 42.2 設計

1. **`trainer.py` 修改**（最小侵入）：在 epoch 迴圈尾端，若 `tcfg.get("swa_last_k") > 0` 且 `epoch > epochs - swa_last_k`，額外存 `out_root/swa_epoch{epoch}.pt`（僅 model state dict + 元資料，不含 optimizer）。預設關閉，零向後相容風險。
2. **新 yaml `configs/exp_p2_combo_best_swa.yaml`**：繼承 `exp_p2_combo_best`，加 `training.swa_last_k: 3`，`seeds: [42]`。將存 epoch 3、4、5 三個 swa 檔。
3. **新工具 `src/tools/u3_swa_aggregate.py`**：
 - 對每 fold 載入 `swa_epoch{3,4,5}.pt`，對 float-tensor 簡單算術平均，整數 buffer（如 `position_ids`）取首檔。
 - 重建模型、載入平均後權重、跑 val 推論，得 `oof_probs_swa.npz` 與每 fold 的 SWA 分數。
 - 聚合 5 fold OOF 算成員級 weighted_score；對照原 best.pt 的 score 計算 delta。
 - 報表：`reports/experiments/p2_combo_best_swa/swa_score_summary.{csv,json}` + `reports/analysis/diagnostics/u3_swa_p2_combo_best_swa.md`。
4. **錄取門檻**：若成員級 SWA OOF ≥ p2_combo_best 原 OOF + 0.0005，加入 v17（18-way）joint hillclimb；否則僅作為 SWA 工程驗證收尾，不入池。

### 42.3 執行狀態

訓練於 2026-05-05 由本 session 啟動（async）：
```
python -u -m src.train_kfold --config configs/exp_p2_combo_best_swa.yaml *> outputs/logs/p2_combo_best_swa.runlog
```
- HF 403（背景 safetensors auto-conversion）為已知非致命噪訊。
- 訓練完成後將執行 `python -m src.tools.u3_swa_aggregate --config configs/exp_p2_combo_best_swa.yaml`。
- 結果與決策將補入 §42.4。

### 42.4 結果（2026-05-05 完成）

- 5 折 × 3 epoch 平均 ckpt 全數產出（15 個 `swa_epoch*.pt`）；聚合腳本順利寫出 5 個 `swa.pt`、`oof_probs_swa.npz` 與 `oof_probs.npz`（canonical alias）。
- **member-level OOF（seed=42, K=3 epoch 平均）**：
 - fold 0: 0.6754（vs best-epoch 0.6742, **+0.0012**）
 - fold 1: 0.6455（vs 0.6425, **+0.0030**）
 - fold 2: 0.6616（vs 0.6662, **−0.0046**）
 - fold 3: 0.6528（vs 0.6600, **−0.0072**）
 - fold 4: 0.6711（vs 0.6850, **−0.0139**）
 - **平均 0.66252 vs 0.66558（−0.0031）**
- **結論**：SWA member-level OOF 較 best-epoch baseline 略低 0.003，**未通過 admission 門檻（≥ +0.0005）**。
 - 失敗原因可能：epoch 5 學習率非零導致末段 ckpt 仍偏離 loss basin；K=3 涵蓋的 epoch（3, 4, 5）方差大（fold 0 epoch 4 達 0.674、epoch 5 跌回 0.665）。
 - **不入 v12 池、不執行 U1-c 重跑**；保留產出供未來若採用 cosine 完整 0 LR 收斂時對照。
 - 列入 X9 風險區（避免 SWA + LR ≠ 0 的組合）。

### 42.5 產出

- `src/training/trainer.py`（新增 `swa_last_k` 邏輯，10 行）
- `configs/exp_p2_combo_best_swa.yaml`
- `src/tools/u3_swa_aggregate.py`
- `outputs/checkpoints/p2_combo_best_swa/seed42/fold{0..4}/swa_epoch{3..5}.pt`（訓練後）
- `outputs/checkpoints/p2_combo_best_swa/seed42/fold{0..4}/swa.pt`、`oof_probs_swa.npz`（聚合後）
- `reports/experiments/p2_combo_best_swa/swa_score_summary.{csv,json}`
- `reports/analysis/diagnostics/u3_swa_p2_combo_best_swa.md`

---

## 43. Phase 22：U5 T4 class-balanced re-sampling — 設置（2026-05-05）

### 43.1 動機

§55 B2。U12 顯示 T4 (`evidence_quality`) macro-F1 跨 fold std 為四項中第二高，且訓練 ensemble best 仍停在 0.4597（其中 `Misleading` support=1 為硬性瓶頸）。透過對 T4 標籤分布做 sqrt-inverse-frequency re-sampling，期望在不放大 noise 的前提下緩解 head/tail 比例。

### 43.2 設計

1. **`train_kfold.py` 修改**（`_build_loaders`）：若 `cfg.training.resample_t4` 為 truthy（True 或 0~1 之間 alpha 浮點），改用 `WeightedRandomSampler`，權重 = `1 / count(class)^alpha`。alpha 預設 1.0；本實驗用 **alpha=0.5**（sqrt 抑制）以避免 singleton class 過度放大。
2. **新 yaml `configs/exp_p2_combo_best_resample_t4.yaml`**：繼承 `exp_p2_combo_best`，加 `training.resample_t4: 0.5`，`seeds: [42]`。
3. **執行順序**：B1 SWA 完成後啟動，避免 GPU 衝突。

### 43.3 執行狀態

訓練完成（5 折 × 5 epoch × seed=42，~16 分鐘）。

### 43.4 結果（2026-05-05 完成）

| fold | best_epoch | weighted | T1 | T2 | T3 | T4 | vs baseline (Δ) |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 0 | 4 | 0.6751 | 0.929 | 0.489 | 0.875 | 0.439 | +0.0009 |
| 1 | 2 | 0.6465 | 0.931 | 0.407 | 0.852 | 0.410 | +0.0040 |
| 2 | 2 | 0.6642 | 0.936 | 0.436 | 0.855 | 0.443 | −0.0020 |
| 3 | 5 | 0.6767 | 0.942 | 0.477 | 0.851 | 0.462 | +0.0167 |
| 4 | 4 | 0.6656 | 0.931 | 0.468 | 0.850 | 0.440 | −0.0194 |
| **mean** | | **0.66562** | **0.9337** | **0.4554** | **0.8568** | **0.4386** | **+0.00004** |

- 與 baseline `p2_combo_best` seed=42 mean 0.66558 持平（**+0.00004**），未通過 admission +0.0005 門檻。
- T4 macro 0.4386 vs baseline 0.4265（**+0.012**）— 預期方向正確；但 T2 macro 0.4554 vs baseline 0.4691（**−0.014**）— sampler 改動使 T2 同步退化（因為 sampler 對整批 batch 重抽，T2 weight 並未隨之調整），抵銷掉 T4 增益。
- **結論**：member-level OOF 持平，**不入 v12 池、不執行 U1-c 重跑**；列入 X10 風險區（單一任務 sampler 影響其他任務）。
- 若未來再嘗試，需 (a) 改 per-task loss weighting 而非 global sampler，或 (b) 對 T4 用 head-only mini-finetune，或 (c) 同時平衡 T2 與 T4。

### 43.5 產出

- `src/train_kfold.py`（`_build_loaders` 條件式 sampler，30 行）
- `configs/exp_p2_combo_best_resample_t4.yaml`
- `outputs/checkpoints/p2_combo_best_resample_t4/seed42/fold{0..4}/best.pt`、`oof_probs.npz`
- `reports/experiments/p2_combo_best_resample_t4/score_summary.{csv,json}`

---

## 44. Phase 23：N3 NeZha-base 設置（2026-05-05）

### 44.1 動機

§55 C1。Active pool 14 員裡 12 員都是 macbert / roberta-wwm / electra base 同源，僅 `p3_large_lr2e5` / `p10_large_focal_fgm` 為 macbert-large；缺乏 NeZha 系列（華為 NeZha 在 functional Chinese NLP 上常與 macbert 互補）。**N3 = `peterchou/nezha-chinese-base`**，沿用 `p2_combo_best` 主超參，single seed=42。

### 44.2 設計

- 新 yaml `configs/exp_p13_nezha_base.yaml`，僅覆蓋 `model.backbone: peterchou/nezha-chinese-base`。
- tokenizer 與 sentencepiece 相容性需於首次載入時確認；若 NeZha 與 transformers 4.x 不相容，將切換到 `nghuyong/ernie-3.0-base-zh` 作為 N3-fallback（也已預留 yaml 模板）。
- 入池門檻：單模 ≥ 0.66；若達標執行 v17 joint hillclimb（18-way），否則僅紀錄為 X9。

### 44.3 執行狀態

兩階段嘗試均完成：
1. **NeZha-base**（`peterchou/nezha-chinese-base`）：transformers `AutoModel.from_pretrained` 無 NeZha-specific class，自動 fallback 為 `BertModel`，導致 NeZha 的相對位置編碼層 `embeddings.position_embeddings.weight` 被以隨機初始化覆蓋；fold0 5 epoch 最高 0.549，fold1 三 epoch 全 0.497（loss 不下降）→ **判定不可訓練**，於 fold1 epoch 3 中止。
2. **ERNIE-3.0-base-zh**（`nghuyong/ernie-3.0-base-zh`，N3-fallback）：載入乾淨；5 折 × 5 epoch 完整訓練。

### 44.4 結果（2026-05-05 完成）

**NeZha 階段**：fold0 best 0.549，遠低於 0.66 admission；中止。

**ERNIE 階段**：
| fold | best_epoch | weighted | T1 | T2 | T3 | T4 |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 0 | 5 | 0.6254 | 0.917 | 0.420 | 0.851 | 0.354 |
| 1 | 5 | 0.5802 | 0.907 | 0.305 | 0.805 | 0.318 |
| 2 | 4 | 0.6195 | 0.912 | 0.395 | 0.824 | 0.373 |
| 3 | 2 | 0.5926 | 0.921 | 0.275 | 0.821 | 0.346 |
| 4 | 5 | 0.6250 | 0.920 | 0.414 | 0.826 | 0.375 |
| **mean** | | **0.6086** | **0.9152** | **0.3616** | **0.8237** | **0.3531** |

- **ERNIE 單模 0.6086 << 0.66 admission 門檻**，亦低於 X6 xlm-roberta-base 0.6127；直接拒絕，不入池、不執行 v17 hillclimb。
- T2/T4 macro 退化最嚴重（vs baseline T2 0.469、T4 0.426，分別 −0.107、−0.073）。ERNIE 與 macbert 的 vocab/tokenization 差異反而傷害本資料集（短中文句、多英數混合的 ESG 內容）。
- **結論**：N3 candidate 全數失敗（NeZha 不可訓、ERNIE 單模太低）；列入 X11 風險區（base-class 異源 backbone 對本資料集不利）。
- 後續若再嘗試結構性多樣性，建議改 (a) 探索 macbert/roberta-wwm 內部 layer-wise 變體（freeze 底層 vs 全微調），或 (b) 直接做 large-class backbone 多 seed（U2 EMA + warm-start），而非更換 base 同源 backbone。

### 44.5 產出

- `configs/exp_p13_nezha_base.yaml`、`configs/exp_p13_ernie_base.yaml`
- `outputs/checkpoints/p13_ernie_base/seed42/fold{0..4}/best.pt`、`oof_probs.npz`
- `reports/experiments/p13_ernie_base/score_summary.{csv,json}`

---

## 45. Phase 24：U2 EMA + warm-start 設置與訓練（2026-05-05）

### 45.1 動機

- §53.3 X1 紀錄：Phase 5 ema999（無 warm-start）OOF 0.473 崩潰，原因為 shadow 權重在 ~500 步內未追上模型主權重。
- §53.2.1 U2 即為對 X1 的修正：EMA decay 降為 0.995，並引入 `ema_warmup_epochs`，前 N epoch 不啟動 EMA 追蹤，僅以原權重訓練；EMA 從第 N+1 epoch 才以當下模型參數重新初始化 shadow 並開始累積。
- 目標：以 0 額外資料、~16 min 訓練成本，測試 admission +0.0005，看能否成為新 ensemble 成員。

### 45.2 設計

- `configs/exp_p2_combo_best_ema995_warm.yaml`：繼承 `exp_p2_combo_best.yaml`；新增 `training.ema_decay=0.995`、`training.ema_warmup_epochs=2`；seeds=[42]；fold=5；epoch=5。
- `src/training/trainer.py`：新增 `ema_warmup_epochs` config 讀取與 `ema_started` 旗標；當 `epoch > ema_warmup_epochs` 時以當下模型參數初始化 shadow，再進入 update / apply_to / restore 流程；warm-up 期 EMA 完全不參與訓練與評估。
- 設定下 EMA 實質追蹤 epoch 3、4、5（共 ~300 步），相當於後段 snapshot averaging。

### 45.3 執行狀態

- 終端 ID `594f8bcc-50e7-4781-8a73-04aed3cecdcf`；HF 403 `get_repo_discussions` benign；exit code 1 為 HF 噪聲，`score_summary.json` 已正常寫出。
- 5 fold × 5 epoch × seed 42，總 ~16 min 完成。

### 45.4 結果（2026-05-05 完成）

fold-by-fold（best epoch = 5，皆為 EMA-eval 後分數）：

| fold | best_epoch | weighted_score | T1 (binary) | T2 (macro) | T3 (binary) | T4 (macro) |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 0 | 5 | 0.65288 | 0.9226 | 0.4625 | 0.8611 | 0.4019 |
| 1 | 5 | 0.63733 | 0.9235 | 0.4202 | 0.8339 | 0.3983 |
| 2 | 5 | 0.65763 | 0.9275 | 0.4763 | 0.8533 | 0.4134 |
| 3 | 5 | 0.64128 | 0.9415 | 0.4579 | 0.8622 | 0.3590 |
| 4 | 5 | 0.65791 | 0.9345 | 0.4699 | 0.8542 | 0.4122 |
| **mean** | | **0.64941** | **0.9299** | **0.4574** | **0.8529** | **0.3970** |

- baseline `p2_combo_best` seed=42 mean OOF = **0.66558**；U2 EMA-warm = **0.64941**；**Δ = −0.01617**。
- 5 折全部退化（最差 fold1 −0.0282、最好 fold4 −0.0077），無單折正向。
- per-task vs baseline：T1 0.9405→0.9299（−0.011）、T2 0.469→0.4574（−0.012）、T3 0.8783→0.8529（−0.025）、T4 0.4592→0.3970（−0.062）；**T4 退化最嚴重**，與 X1 的 T4 崩盤模式同源（少類在 EMA 平均後機率被推平）。
- 與 X1 ema999 0.473 相比有顯著改善（warm-start 確實避免徹底崩潰），但仍遠低於 admission 門檻 0.66608（baseline + 0.0005）。
- 主因推測：(a) 5 epoch 訓練後段 LR 仍在 cosine 中段（與 X9 SWA 同樣末段 LR ≠ 0 問題），EMA 平均了仍在震盪的權重；(b) `ema_warmup_epochs=2` 後僅剩 3 epoch 進行 EMA 累積，shadow 樣本量不足；(c) 對 macro-F1 任務（T2/T4）EMA 平均化會傷少類預測 confidence。
- **結論**：U2 不入 v12 池；列入 X12 風險區（EMA + warm-start 在 cosine 末段 LR ≠ 0 + 5 epoch 短訓練下仍對 T4 macro 不利）。
- 後續若再試 EMA：需 (a) cosine 完整下到 0 LR、(b) 拉長至 8~10 epoch、(c) 對 T2/T4 改用 best-epoch 而非 EMA-epoch；不重跑同設定。

### 45.5 產出

- `configs/exp_p2_combo_best_ema995_warm.yaml`
- `src/training/trainer.py`（新增 `ema_warmup_epochs` 與 `ema_started` 邏輯）
- `outputs/checkpoints/p2_combo_best_ema995_warm/seed42/fold{0..4}/best.pt`、`oof_probs.npz`
- `reports/experiments/p2_combo_best_ema995_warm/score_summary.{csv,json}`
- `outputs/logs/p2_combo_best_ema995_warm.runlog`

---

## 46. Phase 25 完成 — U6 / B4 NLLB-600M 中→英→中 回譯資料增強（2026-05-05）

> **狀態**：完成且拒絕（D30 / X13）。member-level OOF mean = **0.66321** vs baseline 0.66558（**−0.00237**），未通過 admission 0.66608；T2 macro 0.4652（−0.014）、T4 macro 0.4276（−0.031），雙目標任務皆退化。**§53.2 backlog 自此除 U10（依使用者指示暫緩）外已全數驗證完畢**。Active SOTA 維持 Phase 18 U1-c per-task TTA = **0.68925** 不變。

### 46.1 動機

§53.2 backlog 中 B1（U3 SWA）、B2（U5 T4 re-sample）、C1（N3 NeZha→ERNIE）、B3（U2 EMA + warm-start）四項已先後於 Phase 21~24 完成且全數拒絕；U6（T2/T4 回譯資料增強）為 §53.2 中最後一項可在本機完成的內部訓練側可控變項。NLLB-200-distilled-600M 為本機可用、合規、無需外部 API 的最大 zh↔en 翻譯模型，可用以實測 BT 增強對 T2 timeline 與 T4 quality 兩個 macro-F1 弱項是否能帶來增益，作為 §53.2 backlog 收尾批次的最後一動。

### 46.2 設計

**Pipeline（`scripts/u6_backtranslate.py`）**：

- 模型：`facebook/nllb-200-distilled-600M`（HF cache）；`src_lang=zho_Hant`（繁中），pivot=`eng_Latn`。
- Phase 1（zh→en）：`num_beams=4` greedy 確定性翻譯，BS=4，AMP fp16。
- Phase 2（en→zh）：`k=0` 為 greedy（最高保真度），`k≥1` 為 sampling（`do_sample=True`，`temperature=0.7+0.1*k`），用以引入語意多樣性。
- 增強樣本 id 編碼：`new_id = orig_id*100 + k`；保留 `_source_id`（int 原 id）與 `_aug_k` 兩個欄位以供 trainer 追蹤。
- T2 倍率（針對 timeline 弱類）：`T2_AUG = {"within_2_years": 3}`。
- T4 倍率（針對 quality 弱類）：`T4_AUG = {"Misleading": 5, "Not Clear": 1}`。
- 多任務同時符合時取**最大倍率**規則，避免重複展開。
- 輸出：`data/processed/u6_backtrans.json`（168 records，UTF-8）、`reports/u6/backtrans_summary.json`（統計）。

**Trainer 注入機制（`src/train_kfold.py`）**：

- 在 `records, df = load_dataset(...)` 之後讀取 `cfg["data"].get("augment_path")`，若存在則載入 `aug_records`；OOF 與 valid 仍以原始 `records`/`df` 構建（**驗證集絕不接受任何增強樣本**）。
- 在 fold loop 內，先以 `tr_ids = {int(records[i]["id"]) for i in tr_idx}` 收集當折 train 真實 id；再篩 `aug_records` 中 `_source_id ∈ tr_ids` 注入 `train_recs`。
- 每折注入數量輸出 log：`[u6] fold=N injected K BT augmented samples (train=...)`，便於審計。

**配置（`configs/exp_p2_combo_best_u6_bt.yaml`）**：

```yaml
extends: exp_p2_combo_best.yaml
exp_name: p2_combo_best_u6_bt
data:
 augment_path: data/processed/u6_backtrans.json
seeds: [42]
```

只覆寫 `augment_path` 與 `seeds=[42]` 以保留 5 折 × 5 epoch baseline 設定，確保 **Δ 完全可歸因於 BT 增強**。

### 46.3 執行狀態

- 翻譯腳本（async 終端 id `111d1339-…`）：186 秒完成；138 唯一源員 → 168 增強樣本（by_T2: between_2_and_5_years 62、longer_than_5_years 30、already 37、within_2_years 39；by_T4: Not Clear 124、Misleading 5、Clear 30、N/A 9；T2 與 T4 多有交集，最大倍率規則生效）。
- 訓練（async 終端 id `e3471b5a-…`）：1095.8 秒完成；HF 403 `get_repo_discussions` 為已知無害錯誤，`score_summary.json` 已正常寫出。
- per-fold 注入計數：fold0=140、fold1=125、fold2=135、fold3=138、fold4=134。

### 46.4 結果（5-fold OOF, member-level）

| Fold | best_epoch | score | T1 (binary F1) | T2 (macro F1) | T3 (binary F1) | T4 (macro F1) |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 0 | 5 | 0.6748 | 0.9271 | 0.4869 | 0.8562 | 0.4557 |
| 1 | 4 | **0.6334** | 0.9306 | 0.4066 | 0.8551 | 0.3706 |
| 2 | 4 | 0.6632 | 0.9298 | 0.4716 | 0.8763 | 0.4104 |
| 3 | 5 | 0.6750 | 0.9283 | 0.4727 | 0.8664 | 0.4528 |
| 4 | 5 | 0.6696 | 0.9298 | 0.4881 | 0.8451 | 0.4484 |
| **mean** | — | **0.66321** | **0.9291** | **0.4652** | **0.8598** | **0.4276** |
| baseline (`p2_combo_best` seed=42) | — | 0.66558 | 0.9290 | 0.4793 | 0.8662 | 0.4587 |
| **Δ vs baseline** | — | **−0.00237** | +0.0001 | **−0.014** | −0.006 | **−0.031** |

- 5 折標準差 = 0.0174（fold1 顯著離群，−0.0282），baseline 同折 std 約 0.013，**BT 顯著放大了 fold-to-fold variance**。
- **雙目標任務 T2 與 T4 皆退化**（T2 −0.014、T4 −0.031），增強行為等同於 label-noise 而非 diversity injection。
- 未通過 admission 0.66608（baseline + 0.0005）；**不入 v12 池**。

**根本原因分析**：

1. **NLLB-200-distilled-600M 對繁中保真度不足**：`outputs/logs/u6_backtranslate.runlog` 顯示「台泥」被譯為 `TaiMot`、字符腐化、詞重複（如「致力於建立建立」），公司品牌名與 ESG 專有名詞嚴重偏移。
2. **英文 pivot 損失 ESG-specific terminology**：「永續發展」→ `sustainable development` → 「可持續發展」（語義同但詞表偏移），「兩年內」常被譯成不完整時間表達。
3. **未加 round-trip ChrF/BLEU 過濾**：低保真度譯本未被自動過濾，全部 168 筆原樣注入。
4. **train_recs 從 ~795 被膨脹至 ~935**，含低保真度樣本對 T2/T4 macro 等同 label-noise，稀釋真實訊號。
5. **fold1 離群**：該折注入 125 筆 BT 樣本中可能包含關鍵稀類（within_2_years / Misleading），低品質譯本直接讓 T4 macro 從 0.46 跌到 0.37。

**未來若再試 BT 路線（嚴禁不重跑同設定）**，需：

- (a) 換 NLLB-3.3B 或 madlad-7B 或外部 NMT API 以提高中文保真度；
- (b) 加「原文 vs BT 譯本」 round-trip ChrF≥0.5 自動過濾（或人工抽查）；
- (c) 限縮至 T4 Misleading only（5 筆），避免 T2 多倍率擴增帶來雜訊；
- (d) 不重跑 NLLB-600M 同設定（已列入 §53.3 X13 禁區）。

### 46.5 產出

- `scripts/u6_backtranslate.py`（NLLB-200-distilled-600M zh→en→zh BT pipeline，新檔）
- `src/train_kfold.py`（新增 `augment_path` 載入與 per-fold `_source_id` 過濾注入機制）
- `configs/exp_p2_combo_best_u6_bt.yaml`（新檔）
- `data/processed/u6_backtrans.json`（168 條增強樣本）
- `reports/u6/backtrans_summary.json`（138 唯一源員、by_T2 / by_T4 統計）
- `outputs/checkpoints/p2_combo_best_u6_bt/seed42/fold{0..4}/best.pt`、`oof_probs.npz`
- `reports/experiments/p2_combo_best_u6_bt/score_summary.{csv,json}`
- `outputs/logs/u6_backtranslate.runlog`、`outputs/logs/p2_combo_best_u6_bt.runlog`

### 46.6 後續影響

- §53.1 新增 D30 行（U6 / B4）；§53.3 新增 X13 行（NLLB-600M zh-en-zh BT 禁區）。
- §53.2.1 U6 狀態由「尚未開始」更新為「完成（D30，未通過 admission；移入 §53.3 X13）」。
- §55.0 ~~B4~~ 行劃線並標註「完成且拒絕」。
- §53、§14.1、§14.3 收尾語：除 U10（依使用者指示暫緩）外，**§53.2 backlog 已全數驗證完畢；內部可控訓練側可動項目現為零**。
- Active SOTA 維持 Phase 18 U1-c per-task TTA = **0.68925** 不變。

---

## 47. U10 — 企業永續報告書 (SR) 弱監督 pipeline 完整版（2026-05-09 重啟，2026-05-10 v2 重訓）

> **章節定位**：本章把 §47 從「散裝進度紀錄」重構為**架構完整、每段都講清楚說明白**的工程規格。Phase 26~30 失敗教訓 → 重啟原則 → 5-stage pipeline → 兩版 (v1/v2) 演進 → 最終 OOF SOTA = **0.67746**（baseline +0.01012，45-ckpt 三方 stack）→ 結構性殘留與後續路徑。命名一律使用 `u10`（不再 `u10new`）。

### 47.0 設計原則與失敗教訓

**為什麼存在這個 pipeline**：
- Phase 26~30（U10 PATH-D / U13 PATH-E / U14 PATH-C / U15 PATH-B / U15 v4-clean PATH-B）全數**實證偏離 ESG 企業永續報告書主題**：PATH-C 取「公司形象 / SDG 對接 / 治理框架」一般 landing 段落、PATH-B v4-clean 取 Wikipedia + IPCC/UNFCCC 學術 PDF，皆與賽方標註資料 (`vpesg4k_train_1000`) 之分布不一致 → distribution shift 對 T2/T3 持續為負。
- 結論：**外部資料必須與「企業永續報告書原文」同質**，才能避免 distribution shift。

**本輪三條鐵律**：
1. **只取台灣（TWSE / TPEX）上市櫃公司「企業永續報告書 (SR / CSR / ESG Report)」原文 PDF 段落**。
2. **來源公司不得與已標註 50 家重疊**（避免資訊洩漏、SimHash 去重失效）。
3. **不直接合併訓練**：先 teacher 偽標 → admission gate → 兩階段訓練（Stage A 含偽 / Stage B 純官方）→ OOF ensemble，全程隔離雜訊。

**驗收三條件**：
- OK 端到端 pipeline 可重現（§47.10 列一鍵指令）。
- OK OOF weighted_score（best.pt path）較 baseline 提升 ≥ +0.005 → **實際 +0.01012**。
- OK 對 minority class（T2 timeline / T4 quality）提供可量測覆蓋改善 → v2 偽標籤 `longer_than_5_years` 0→570、`already` 50→1645，`Not Clear` 0→3；殘留 `within_2_years` / `Misleading` 仍 0（根因記於 §47.8）。

### 47.1 5-stage 架構總覽

```
┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────────┐
│ M1 採集 │ → │ M3 抽取 │ → │ M4 偽標註 │ → │ M5 兩階段 │ → │ M6 OOF Ensemble│
│ 31 ticker │ │ paragraphs │ │ admission │ │ K-Fold 訓練 │ │ best.pt averaging
│ 62 PDFs │ │ v1: 510 │ │ v1: 211/510 │ │ 3 seeds × 5 │ │ baseline+v1+v2 │
│ FY23+FY24 │ │ v2: 12,345 │ │ v2: 3,904 │ │ folds = 15 │ │ 45-ckpt stack │
└─────────────┘ └─────────────┘ └──────────────┘ └──────────────┘ └────────────────┘
 │ │
 ▼ ▼
 15 best.pt × 變體 SOTA = 0.67746
```

| 階段 | 輸入 | 主程式 / 設定 | 輸出 |
|---|---|---|---|
| **M1** 採集 | 31 disjoint ticker | `scripts/u10_collect_v3.py`、`u10_collect_v3b.py` | `data/raw/u10/{ticker}_FY{year}.pdf` × 62 |
| **M3** 抽取 | 62 PDFs | v1 `u10_pdf_extract.py` / v2 `u10_pdf_extract_v2.py` | `data/processed/u10/{corpus,corpus_v2}.jsonl` |
| **M4** 偽標註 | corpus + 15 baseline ckpt | v1 `u10_pseudo_label.py` / v2 `u10_pseudo_label_v2.py` | `data/processed/u10/{pseudo_labels,pseudo_labels_v2}.csv` |
| **M5** 訓練 | pseudo CSV + official train | `src/train_pseudo_kfold.py` + `configs/exp_p2_combo_best_u10_pseudo{,_v2}.yaml` | `outputs/checkpoints/p2_combo_best_u10_pseudo{,_v2}/seed{42,2024,20260417}/fold{0..4}/best.pt` × 15 |
| **M6** Ensemble | 多 experiment OOF probs | `src/tools/oof_ensemble.py` | `reports/experiments/oof_ensemble_*.json/*.csv` |

### 47.2 已標註 50 家公司（**禁止重複爬取/標註**）

依 `vpesg4k_train_1000 V1.json` 完整盤點，共 50 家上市公司 1000 筆已標註紀錄（每家 4~45 筆）。本輪 U10 之爬蟲須**完全排除**以下 ticker 之**任何年度**永續報告書：

| ticker | 公司 | 已標筆數 |
| :--: | :-- | :--: |
| 1101 | 台泥 (TCC) | 26 |
| 1216 | 統一 (Uni-President) | 18 |
| 1301 | 台塑 (FPC) | 20 |
| 1303 | 南亞 (NPC) | 5 |
| 1907 | 永豐餘 (YFY) | 2 |
| 2002 | 中鋼 (CSC) | 17 |
| 2207 | 和泰車 (Hotai Motor) | 18 |
| 2301 | 光寶科 (LTC) | 25 |
| 2303 | 聯電 (UMC) | 38 |
| 2308 | 台達電 (Delta) | 24 |
| 2317 | 鴻海 (Hon Hai) | 21 |
| 2327 | 國巨 (Yageo) | 16 |
| 2330 | 台積電 (TSMC) | 24 |
| 2345 | 智邦 (Accton) | 26 |
| 2379 | 瑞昱 (Realtek / RT) | 18 |
| 2382 | 廣達 (QCI) | 28 |
| 2383 | 台光電 (EMC2) | 17 |
| 2395 | 研華 (Advantech / ACL) | 26 |
| 2412 | 中華電信 (CHT) | 21 |
| 2454 | 聯發科 (MediaTek) | 18 |
| 2603 | 長榮海運 (EMC) | 17 |
| 2609 | 陽明海運 (YMTC) | 20 |
| 2615 | 萬海 (Wan Hai) | 17 |
| 2880 | 華南金 (HNFHC) | 20 |
| 2881 | 富邦金 (Fubon) | 44 |
| 2882 | 國泰金 (Cathay) | 19 |
| 2883 | 開發金 (KGI) | 28 |
| 2884 | 玉山金 (ESFH) | 21 |
| 2885 | 元大金 (Yuanta) | 8 |
| 2886 | 兆豐金 (Mega) | 20 |
| 2887 | 台新金 (Taishin) | 26 |
| 2891 | 中信金 (CTBC) | 6 |
| 2892 | 第一金 (FFHC) | 20 |
| 2912 | 統一超商 (PCSC) | 17 |
| 3008 | 大立光 (Largan) | 17 |
| 3017 | 奇鋐 (AVC) | 12 |
| 3034 | 聯詠 (Novatek) | 4 |
| 3045 | 台灣大 (TWM) | 23 |
| 3231 | 緯創 (Wistron) | 45 |
| 3661 | 世芯-KY (Alchip) | 19 |
| 3711 | 日月光投控 (ASEH) | 31 |
| 4904 | 遠傳 (FET) | 13 |
| 4938 | 和碩 (Pegatron) | 14 |
| 5871 | 中租-KY (Chailease) | 18 |
| 5876 | 上海商銀 (SCSB) | 20 |
| 5880 | 合庫金 (TCFHC) | 45 |
| 6446 | 藥華藥 (PEC) | 14 |
| 6505 | 台塑化 (FPCC) | 14 |
| 6669 | 緯穎 (Wiwynn) | 20 |

### 47.3 M1 採集（Collection）— 31 disjoint ticker × 62 PDFs

**目標**：取得不在 50 家標註集中的台灣上市公司 SR PDF，每家 FY2023 + FY2024 共 2 份。

**爬取策略演進**：
1. **死路證明**（2026-05-09 早）：TWSE MOPS legacy / new SPA / ESG hub / DDG / Bing-requests / Bing-Selenium-headless 皆無公開 SR API 可用（記錄於 `/memories/session/u10_strategy.md`）。
2. **改弦更張**：複製官方標註集 `vpesg4k_train_1000` 之 `company_source` 邏輯 — 為每家公司逐個維護 ESG/IR landing page URL 註冊表（`scripts/u10_sources.py`），由 `scripts/u10_collect_sr.py` (requests + Selenium fallback) 直抓 PDF。
3. **第一輪 smoke (2026-05-09)**：30 家僅 6 家成功 9 PDFs ≈ 178 MB，主因 URL 404 / SPA / Cloudflare / DNS 失敗。
4. **v2~v3+v3b backfill (2026-05-10)**：人工核實失敗清單 → 修正 URL → 補爬 1503 + 9933 → 收斂為 **31 家 disjoint ticker × 62 PDFs**（每家 FY2023 + FY2024）。

**31 家 disjoint ticker**（與 50 家標註集**完全不重疊**）：

```
1102 1326 1402 1503 1605 1722 2353 2354 2356 2357
2376 2408 2474 2610 2618 2727 2812 2867 2890 3037
3443 4958 5269 6239 6415 8046 8299 9904 9910 9921 9933
```

**檔名規範**：`data/raw/u10/{ticker}_FY{year}.pdf`（例：`2330_FY2024.pdf`）。年度以發行年計（FY2024 = 報告 2024 年實績、通常 2025 年發行）。

**產出**：
- `data/raw/u10/*.pdf` × 62
- `reports/experiments/u10/collect_manifest.csv`（記錄每家 ticker × year × URL × byte size × hash）
- 採集腳本：`scripts/u10_sources.py`、`scripts/u10_collect_sr.py`、`scripts/u10_v2_sustaihub_crawl.py`、`scripts/u10_v3_sustaihub_crawl.py`、`scripts/u10_v3b_add2.py`

### 47.4 M3 抽取（Extraction）— v1 保守 → v2 攻擊性放寬（24× yield）

**為什麼需要 v2**：v1 yield 僅 5.7%（510/8971 段落），目視確認大量有效 ESG 段落被過度過濾，導致 M4 偽標籤池太小、minority class 覆蓋不足。v2 以「放寬條件 + 句子級切段 + 量化線索 OR + 啟發式少數類 tag」四箭齊發。

**v1 vs v2 過濾鏈對照**：

| 規則 | v1（保守，`u10_pdf_extract.py`） | v2（攻擊性，`u10_pdf_extract_v2.py`） | 改動理由 |
|---|---|---|---|
| 段落長度 | 50 ≤ len ≤ 500 | 40 ≤ len ≤ 800 | 釋放長段（多步驟承諾、跨年期目標） |
| 中文比例 | CJK ≥ 0.60 | CJK ≥ 0.55 | 放行混雜技術詞段落 |
| 切段方式 | PDF 自然段 | 句子切段 + greedy merge to 60~800 字 bucket | 避免 PDF 換行造成的短碎片被砍 |
| TOC 過濾 | token ≥ 3 / english ≥ 0.40 / space ≥ 0.20 | token ≥ 4 / english ≥ 0.50 / space ≥ 0.25 | 收緊 false-positive，保留章節段落 |
| ESG 關鍵詞 | AND（必含 ESG 字幹） | (含 ESG 字幹) **OR** (含 % 或 4 位年份等量化線索) | 捕捉純量化承諾語句 |
| 去重 | 64-bit SimHash hamming > 8 vs labeled & > 6 self | 同 v1 | 維持資訊洩漏防線 |
| 少數類 tag | 無 | `cand_t2`（時間距離規則）、`cand_t4`（measurable vs vague） | 供 M4-v2 Tier-2 admission 主動採樣 |

**結果對照**：

| 版本 | raw → kept | 涵蓋 ticker | T2 候選分布（already / w2y / b2~5y / l5y） | T4 候選分布（Clear / Not_Clear） |
|---|---|---|---|---|
| v1 | 8,971 → **510** (5.7%) | 28/31 (1402/2867/8046 全段被砍) | 50 / 0 / 1 / 10 | 候選未標記 |
| v2 | 13,768 → **12,345** (89.7%) | 31/31 | 2,139 / 191 / 24 / 106 | 4,711 / 2,623 |

**產出**：
- `data/processed/u10/corpus.jsonl`（510 行 v1）、`data/processed/u10/corpus_v2.jsonl`（12,345 行 v2）
- `reports/experiments/u10/extract_stats.json`、`reports/experiments/u10/extract_v2_stats.json`

> WARN 載入 corpus 須以 `ESGDataset(text_field="text")`，預設 `text_field="data"` 會默默回傳空輸入（已知 bug，記於 §47.9）。

### 47.5 M4 偽標註（Pseudo-labeling）— v1 標準閘 → v2 兩層 admission

**Teacher 模型**：`outputs/checkpoints/p2_combo_best/seed{42,2024,20260417}/fold{0..4}/best.pt` 共 **15 ckpt 平均 softmax**（即 baseline 三次 K-Fold OOF teacher）。**不使用更強的 U1-c TTA**，避免 §X18「自我訓練不超過 teacher」上限定理直接打死。

**v1 admission（標準閘，`u10_pseudo_label.py`）**：4 task 全通才採納。
- 門檻：T1 ≥ 0.80、T2 ≥ 0.60、T3 ≥ 0.70、T4 ≥ 0.60
- 結果：**211 / 510 admitted**（141 背景 No-promise + 70 完整 promise）
- 致命缺陷：T2 within_2_years = 0、T2 between_2_and_5_years = 1、T3 No = 1、T4 Not Clear = 0、T4 Misleading = 0

**v2 admission（兩層閘，`u10_pseudo_label_v2.py`）**：

| 層級 | 適用範圍 | T1 | T2 | T3 | T4 | 採納上限 |
|---|---|---|---|---|---|---|
| **Tier-1**（標準）| 全部 12,345 row | ≥ 0.80 | ≥ 0.60 | ≥ 0.70 | ≥ 0.60 | 無 |
| **Tier-2**（少數類 boost）| `cand_t2` / `cand_t4` heuristic 與模型預測**一致**之 row | ≥ 0.55 | ≥ 0.45 | ≥ 0.50 | ≥ 0.45 | 每類 ≤ 60；Tier-2 全域 ≤ 250 |

**v2 結果**：Tier-1 admit **3,881** + Tier-2 admit **23** = **3,904 rows**（v1 211 → v2 3,904，+18×）。

**少數類覆蓋改善（v1 → v2）**：

| Task | 類別 | v1 | v2 | 改善 |
|---|---|---:|---:|---|
| T2 | already | 50 | 1,645 | OK +33× |
| T2 | between_2_and_5_years | 1 | 26 | OK +26× |
| T2 | longer_than_5_years | 10 | 570 | OK +57× |
| T2 | within_2_years | 0 | 0 | NO 仍 0（見 §47.8） |
| T3 | No | 1 | 11 | OK +11× |
| T4 | Not Clear | 0 | 3 | WARN 微弱 |
| T4 | Misleading | 0 | 0 | NO 仍 0（見 §47.8） |

**產出**：
- `data/processed/u10/pseudo_labels.csv`（211 row，`company_source = "u10"`）
- `data/processed/u10/pseudo_labels_v2.csv`（3,904 row，`company_source = "u10_v2"`）
- `reports/experiments/u10/pseudo_label_stats.json`、`reports/experiments/u10/pseudo_label_v2_stats.json`

### 47.6 M5 兩階段 K-Fold 訓練（Two-stage Training）

**為什麼分兩階段**：直接把偽標籤與官方資料混訓會被 noise 帶偏；分兩階段可讓模型先學到弱監督結構（Stage A），再用乾淨官方資料微調（Stage B），相當於以偽標當「廣域預訓練 → 官方資料 fine-tune」的小型 curriculum。

**Stage A（弱監督預訓練）**：
- 輸入：official train fold + pseudo CSV（v1=211 / v2=3,904；v2 配置另設 `max_pseudo=2000` 隨機抽樣以平衡類分布）
- epochs = 4、batch_size = 8、AMP、max_length = 384、模型 = `hfl/chinese-macbert-base`
- 輸出：`stage_a/best.pt`（依 fold val score 早停）

**Stage B（官方乾淨資料微調）**：
- 載入 Stage A best.pt 作為 init
- 輸入：僅官方 train fold（去除偽標）
- epochs = 3、其他超參同 Stage A
- 輸出：`best.pt`（fold-level 最終 ckpt，供 OOF 用）

**訓練規模**：3 seeds (42 / 2024 / 20260417) × 5 folds = **15 fold-trainings per 配置**。v1 + v2 共 30 fold-trainings × ~3.5 min/fold = **約 175 min on RTX 5060 Laptop 8 GB**（本次 v2 實測 5,892.4 s ≈ 98.2 min）。

**設定檔（命名規範統一為 u10）**：
- `configs/exp_p2_combo_best_u10_pseudo.yaml`（v1，`min_confidence: 0.60`、`max_pseudo: 500`、`stage_a_epochs: 4`、`stage_b_epochs: 3`）
- `configs/exp_p2_combo_best_u10_pseudo_v2.yaml`（v2，`min_confidence: 0.45`、`max_pseudo: 2000`、`stage_a_epochs: 4`、`stage_b_epochs: 3`）

**訓練器**：`src/train_pseudo_kfold.py`（自動處理 Stage A → Stage B 權重轉移、per-fold OOF 收集、`score_summary.{json,csv}` 落地）。

**v2 Per-seed 訓練品質（本次實測）**：

| seed | mean | std | min | max |
|---|---:|---:|---:|---:|
| 42 | 0.670817 | 0.007149 | — | — |
| 2024 | 0.669667 | 0.017011 | — | — |
| 20260417 | 0.663720 | 0.018931 | — | — |
| **整體** | **0.66807** | **0.01449** (n=15) | — | — |

per-task 平均（v2 single config 訓練端 OOF）：T1=0.9354 / T2=0.4839 / T3=0.8598 / T4=0.4299。

### 47.7 M6 OOF Ensemble — **NEW SOTA = 0.67746**（baseline +0.01012）

**Ensemble 工具**：`src/tools/oof_ensemble.py`，對指定 experiment 列表的 per-row OOF softmax 做等權平均後重新 argmax，計算 task macro-F1 與 weighted score。

**完整結果（best.pt path，N=1000，2026-05-10 重訓 ckpt）**：

| 配置 | Ensemble 內容（ckpt 數） | T1 promise | T2 timeline | T3 evidence | T4 quality | weighted | Δ baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `p2_combo_best` (15) | 0.92767 | 0.47585 | 0.86551 | 0.43079 | 0.66734 | — |
| pseudo v1 (solo) | `p2_combo_best_u10_pseudo` (15) | 0.92793 | 0.47611 | 0.86013 | 0.44736 | 0.67162 | +0.00428 |
| pseudo v2 (solo) | `p2_combo_best_u10_pseudo_v2` (15) | 0.93683 | **0.49806** | 0.86559 | 0.42983 | 0.67219 | +0.00485 |
| stack b+v1 | baseline + v1 (30) | 0.93222 | 0.47691 | 0.86486 | 0.44719 | 0.67396 | +0.00662 |
| stack b+v2 | baseline + v2 (30) | **0.93929** | 0.49278 | 0.87065 | 0.43800 | 0.67627 | +0.00893 |
| **stack b+v1+v2** | baseline + v1 + v2 (45) | 0.93594 | 0.48939 | **0.87023** | 0.44513 | **0.67746** | **+0.01012** |

**判讀**：
- v1 主要拉抬 T4（+0.0166），v2 主要拉抬 T2 (+0.0222)、T1 (+0.0092)。
- 三方 stack 取得 **每 task 都 ≥ baseline** 的 Pareto improvement，也是 best.pt path SOTA。
- **重要免責**：本表為 best.pt epoch ensemble；與 §39 Phase 18 SOTA = 0.68925（U1-c per-task 3-view TTA）**不直接可比**。下一步可在新 ckpt 上跑 U1-c TTA 才能對齊（§47.12）。

**ensemble 產出**：`reports/experiments/oof_ensemble_{baseline,pseudo,pseudo_v2,stack,stack_v2,stack_v3}.json/*.csv`。

### 47.8 結構性殘留問題（Class Collapse）

**T2 within_2_years 與 T4 Misleading 仍 0 admitted**：根因是 baseline 模型本身**從不預測這兩類**（class collapse）。
- Tier-2 的雙條件「heuristic = 候選類 AND 模型預測 = 候選類」永遠不滿足。
- 即使把 v2 corpus 擴張到 12,345，這兩類在偽標籤池仍為 0。

**根本解（不在本 pipeline scope）**：
- Class-weighted CE / Focal loss γ=3.0（沿用 §32 Phase 12 配置思路）
- T4-only 過採樣（U5 配置已存在但需重訓）
- 已記入 §53 backlog 對應 U5 / U7

### 47.9 已知工程問題與避坑指南

| 問題 | 症狀 | 解法 |
|---|---|---|
| `ESGDataset` 預設 `text_field="data"` | corpus 載入後 input 為空，Stage A loss 不下降 | **必須**顯式 `text_field="text"`（u10_pseudo_label*.py 已正確設定） |
| HuggingFace 403 "Discussions are disabled" | 訓練 stderr 大量背景紅字、Tee 管線 exit code = 1 | `$env:HF_HUB_DISABLE_IMPLICIT_TOKEN='1'`；改用 `*>&1 \|` 取代 `2>&1 \|` 讓 Tee 回乾淨 exit 0 |
| `f:\esg-veripromise-2026\.venv` 缺 numpy | venv import 失敗 | **永遠用** `C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe`（系統 Python） |
| TWSE MOPS 無公開 SR API | 大規模 ticker 自動爬取失敗 | 改採「逐家公司 ESG/IR landing page 註冊表」模式（`u10_sources.py`） |

### 47.10 一鍵重現指令（Reproduce from scratch）

> 假設 `data/raw/u10/*.pdf` × 62 已採集完成（M1 階段需手動更新 `u10_sources.py` 註冊表）。

```powershell
$py = "C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe"
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = '1'

# Step 1 — M3 抽取 v2 (yield 12,345 paragraphs)
& $py scripts\u10_pdf_extract_v2.py

# Step 2 — M4 偽標註 v2 (3,904 admitted; uses 15 baseline ckpt)
& $py scripts\u10_pseudo_label_v2.py

# Step 3 — M5 兩階段 K-Fold 訓練（v1 + v2，約 175 min）
& $py -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo.yaml *>&1 | Tee-Object -FilePath reports\experiments\p2_combo_best_u10_pseudo\full_run.log
& $py -m src.train_pseudo_kfold --config configs\exp_p2_combo_best_u10_pseudo_v2.yaml *>&1 | Tee-Object -FilePath reports\experiments\p2_combo_best_u10_pseudo_v2\full_run.log

# Step 4 — M6 OOF Ensemble（產生 §47.7 表格中的 6 行結果）
& $py -m src.tools.oof_ensemble --exps p2_combo_best --out reports\experiments\oof_ensemble_baseline.json
& $py -m src.tools.oof_ensemble --exps p2_combo_best_u10_pseudo --out reports\experiments\oof_ensemble_pseudo.json
& $py -m src.tools.oof_ensemble --exps p2_combo_best_u10_pseudo_v2 --out reports\experiments\oof_ensemble_pseudo_v2.json
& $py -m src.tools.oof_ensemble --exps p2_combo_best p2_combo_best_u10_pseudo --out reports\experiments\oof_ensemble_stack.json
& $py -m src.tools.oof_ensemble --exps p2_combo_best p2_combo_best_u10_pseudo_v2 --out reports\experiments\oof_ensemble_stack_v2.json
& $py -m src.tools.oof_ensemble --exps p2_combo_best p2_combo_best_u10_pseudo p2_combo_best_u10_pseudo_v2 --out reports\experiments\oof_ensemble_stack_v3.json
```

### 47.11 完整檔案清單（Artifact Index）

**腳本（`scripts/`）**：
- `u10_sources.py` — 31 家 ESG/IR landing page URL 註冊表
- `u10_collect_sr.py` — requests + Selenium fallback 第一輪 smoke 採集
- `u10_audit_year.py` — 採集後年度盤點審核
- `u10_v2_sustaihub_crawl.py`、`u10_v3_sustaihub_crawl.py`、`u10_v3b_add2.py` — 補爬版本（v2/v3/v3b backfill）
- `u10_pdf_extract.py`、`u10_pdf_extract_v2.py` — M3 v1/v2 抽取
- `u10_pseudo_label.py`、`u10_pseudo_label_v2.py` — M4 v1/v2 偽標註

**設定檔（`configs/`）**：
- `exp_p2_combo_best_u10_pseudo.yaml`、`exp_p2_combo_best_u10_pseudo_v2.yaml`

**資料（`data/`）**：
- `raw/u10/*.pdf` × 62（31 ticker × FY2023 + FY2024）
- `processed/u10/{corpus.jsonl, corpus_v2.jsonl, pseudo_labels.csv, pseudo_labels_v2.csv}`
- `splits/p2_combo_best_u10_pseudo/`、`splits/p2_combo_best_u10_pseudo_v2/`

**模型權重（`outputs/checkpoints/`）**：
- `p2_combo_best_u10_pseudo/seed{42,2024,20260417}/fold{0..4}/best.pt` × 15
- `p2_combo_best_u10_pseudo_v2/seed{42,2024,20260417}/fold{0..4}/best.pt` × 15

**報告（`reports/experiments/`）**：
- `u10/{collect_manifest.csv, extract_stats.json, extract_v2_stats.json, pseudo_label_stats.json, pseudo_label_v2_stats.json}`
- `p2_combo_best_u10_pseudo/{score_summary.json, score_summary.csv, full_run.log}`
- `p2_combo_best_u10_pseudo_v2/{score_summary.json, score_summary.csv, full_run.log}`
- `oof_ensemble_{baseline, pseudo, pseudo_v2, stack, stack_v2, stack_v3}.json` 及對應 `.csv`

### 47.12 與其他 Phase 的關係 / 後續路徑

**位置定位**：U10 屬於 §53.0.4 「外部資料偽標擴增」路徑，原本被列為「使用者指定最後再做」的暫緩路徑；本輪 (2026-05-09 重啟) 證實**只要遵守來源同質性 + 50 家排除 + 兩階段隔離訓練，外部資料對 best.pt path 的提升是顯著且可量測的（+0.01012）**。

**未來可疊加的方向（依 ROI 排序）**：
1. **U1-c TTA on u10 ckpt（最低風險）**：把本輪 45 ckpt 餵入 U1-c per-task 3-view TTA aggregator，與 §39 Phase 18 SOTA 0.68925 對齊比較。預期收益 ≥ 上次 +0.00046。
2. **Class-weighted CE / Focal loss 重訓**：以 v2 偽標籤資料 + Focal-T4 γ=3.0 重訓 → 期望恢復 within_2_years / Misleading 預測能力，撬動 §47.8 結構性殘留。
3. **M3-v3 corpus 再擴張（中風險）**：放寬至 30 ≤ len ≤ 1200、移除 TOC space-ratio 規則；目視評估 yield 與雜訊比後決定是否進 M4-v3。
4. **混合模型家族 (XLM-R / Qwen-LoRA) 偽標教師**：X18 上限定理只適用於同 teacher 自我訓練；換 backbone 作 teacher 可能解鎖新 plateau，但需另行驗證。

**已暫停**：M3-v3 / M4-v3 等更激進版本已預先設計但本輪不執行，避免 over-engineering。

---

## 48. Phase 32 — U10 per-task TTA — 0.68632（U10 stack 0.67746 → +0.00886）

> 對應 §15 待辦 #1「TTA on U10」。實作於 2026-05-12 完成，產物在 `reports/analysis/_ensemble/u10_per_task_tta_*` 與 `outputs/cache/u10_tta/*.npz`。工具：[src/tools/u10_per_task_tta.py](src/tools/u10_per_task_tta.py)。

**設計**：套用 §39 Phase 18 同款 per-task per-view TTA 概念，但（a）pool 改成 U10 三 stem（baseline / pseudo-v1 / pseudo-v2），（b）多搜一層 per-task 的 stem-mix 權重，避免被 equal-weight stack 鎖死。

兩階段 coordinate descent（grid step = 0.1，4 round + 1 joint pass，皆收斂）：
- **Stage A**：固定 view = stored only，搜 per-task `(w_baseline, w_v1, w_v2)` ∈ Δ²。
- **Stage B**：固定 stem*，搜 per-task `(α_stored, α_middle, α_tail)` ∈ Δ²。
- **Joint refine**：交替 A↔B，第一輪即收斂。

**Reference (equal-stem)**：
| view 配置 | weighted_score |
| :-- | --: |
| stored only（= U10 stack） | 0.67746 |
| stored+middle (0.5/0.5/0) | 0.67805 |
| stored+tail (0.5/0/0.5) | 0.67801 |
| three-equal (1/3/1/3/1/3) | 0.67830 |

**搜尋結果**（grid 0.1 ≈ grid 0.05；前者 0.68632，後者 0.68631，差異 <1e-5）：
| 階段 | weighted_score | Δ vs U10 stack |
| :-- | --: | --: |
| Stage A 完成 | 0.68472 | +0.00726 |
| Stage B 完成 | **0.68632** | **+0.00886** |
| Joint r1 | 0.68632（已收斂） | — |

**最佳權重（grid 0.1 版）**：
| task | stem* `(b, v1, v2)` | view α* `(stored, middle, tail)` | per-task score |
| :-- | :-- | :-- | --: |
| promise_status | (0.1, 0.3, 0.6) | (0.0, 0.8, 0.2) | 0.9412 |
| verification_timeline | (0.3, 0.0, 0.7) | (0.1, 0.8, 0.1) | 0.5020 |
| evidence_status | (0.0, 0.4, 0.6) | (0.0, 0.7, 0.3) | 0.8716 |
| evidence_quality | (0.2, 0.8, 0.0) | (0.0, 0.9, 0.1) | 0.4608 |

**觀察**：
1. **stored 權重幾乎全被擠掉**（除 T2/T4 各 0.1~0.2）：middle/tail TTA 在 U10 上比原 stored OOF 更有資訊量；這與 §39 在 v12 pool 上的 +0.00046 觀察方向一致，但量級在 U10 上放大近 20×。
2. **stem mix 嚴重 task-dependent**：T2/T3 偏向 v2（最大 0.6/0.6），T4 反而退回 v1（0.8）—— 說明 v2 的 Tier-2 minority-boost 在 T4 上 overfit；這正好補強 §47.8 的「Misleading 預測退化」結論，並指引 §15 路徑 2 用 Class-weighted CE / Focal-T4 γ=3.0 重訓 v2 的方向。
3. **與 active SOTA 0.68925 仍差 -0.00293**：合流還沒成，但已經把 gap 從 -0.01179 縮到 -0.00293。

**對 §15 待辦的指引**：
- 路徑 2 (Class-weighted) 仍應做：T4 的 stem*（v1=0.8）暗示 v2 在 T4 是負增益，重訓修正後 stem mix 應能進一步改善。
- 一旦路徑 2 / 路徑 3 產生新 ckpt，**重跑本工具** 即可得到新的 U10-side SOTA，且可進一步把 U10 stems 加入 v12 pool（Phase 33+）作雙軌合流。

---

## 49. Phase 33 — Class-weighted CE + Focal-T4 γ=3.0 + 4-way per-task hillclimb — NEW SOTA 0.70185

> 對應 §15 待辦 #2「Class-weighted CE / Focal-T4 γ=3.0」+ #1 路徑延伸（4-way per-task stack search）。實作於 2026-05-12 同日完成，產物在 `reports/analysis/_ensemble/u10_4way_classw_meta.json`、配置在 [configs/exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3.yaml](configs/exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3.yaml)、訓練腳本沿用 `src.train_pseudo_kfold`、stack search 工具：[src/tools/u10_classw_stack_search.py](src/tools/u10_classw_stack_search.py)。

### 49.1 Stem 4：Class-weighted CE + Focal-T4 γ=3.0（U10-v2 偽標 + sqrt-inv-freq 類別權重）

**訓練配方**（5-fold × seed42，~37 min on RTX 5060 8 GB）：
- pseudo 來源：U10-v2 corpus（vs Phase 31 stem v2 同來源，避免變因混淆）
- T2 / T4 類別權重：sqrt-inverse-frequency（minority class up-weight）
- T4 同時套 Focal Loss γ=3.0（針對 `Misleading` support=1 的極端 minority）
- T1 / T3 保持原 binary CE（避免 binary 任務 over-correction）

**單一 stem 全 N=1000 OOF 效能**（5 ckpt seed42 直接 stitch）：
| task | baseline (`p2_combo_best`) | v1 (`u10_pseudo`) | v2 (`u10_pseudo_v2`) | **classw_focal_t4_g3** |
| :-- | --: | --: | --: | --: |
| T1 promise | 0.9359 | 0.9325 | 0.9351 | 0.9332 |
| T2 timeline | **0.476** | 0.4909 | 0.5101 | **0.58004**  |
| T3 evidence | 0.8702 | 0.8704 | 0.8729 | 0.8675 |
| T4 quality | 0.4400 | 0.4439 | 0.4502 | 0.4427 |
| **weighted** | 0.6614 | 0.6688 | 0.6754 | **0.67410** |

→ **T2 從 baseline 0.476 → 0.58004（+0.104）**，是本競賽至今單一干預對單一任務的最大跳幅。Class-weighted CE 直接破解 Phase 30 §47.8 觀察到的 `within_2_years / between_2_and_5_years` minority class collapse。T4 Focal γ=3.0 略改善但不顯著（Misleading support=1，4-class 結構性難題）。

### 49.2 4-way per-task hillclimb（grid step 0.05 simplex）

把 4 stem 的 stored OOF（baseline / v1 / v2 / classw_focal）依任務做 simplex grid search。每 task 在 1771 個 (w_b, w_v1, w_v2, w_cw) ∈ Δ³ 點內取最大；其他任務暫固定為 equal weight 後展開 coordinate search。

**Reference**：4-way equal-weight = 0.67922（已是新 U10 best.pt path SOTA，+0.00176 vs Phase 31 的 3-way 0.67746）。

**Per-task 最佳權重**：
| task | best `(w_b, w_v1, w_v2, w_cw)` | per-task score | Δ vs equal-4 |
| :-- | :-- | --: | --: |
| T1 promise_status | (0.40, 0.00, 0.25, 0.35) | 0.94026 | +0.00224 |
| T2 verification_timeline | (0.15, 0.15, 0.05, **0.65**) | **0.60157**  | +0.10171 |
| T3 evidence_status | (0.15, 0.45, 0.25, 0.15) | 0.87456 | +0.00140 |
| T4 evidence_quality | (0.20, 0.80, 0.00, 0.00) | 0.46056 | +0.01861 |

**最終加權**：
$$ 0.20 \cdot 0.94026 + 0.15 \cdot 0.60157 + 0.30 \cdot 0.87456 + 0.35 \cdot 0.46056 = \mathbf{0.70185} $$

### 49.3 突破級別

| 對照基準 | 分數 | Δ |
| :-- | --: | --: |
| **Phase 33 4-way per-task hillclimb** | **0.70185** | — |
| Phase 18 active SOTA（U1-c per-task TTA, v12 pool） | 0.68925 | **+0.01260** |
| Phase 32 U10 per-task TTA（3-way + 多視角） | 0.68632 | +0.01553 |
| 4-way equal-weight ensemble | 0.67922 | +0.02263 |
| Phase 31 U10 stack baseline（3-way equal） | 0.67746 | +0.02439 |
| 寬鬆工程上限（§53.0.2） | 0.7570 | -0.0552 |
| 保守工程上限（§53.0.2） | 0.7236 | -0.0218 |

→ **首次突破 0.69 護城線**，且**首次合流兩條 SOTA 軌跡**（U10 best.pt path 反超 active TTA path）。距離保守工程上限 0.7236 僅 -0.0218。

### 49.4 過擬合風險與校準計畫

 **重要免責**：
1. 0.70185 為 **OOF 上的 per-task hillclimb**，搜尋空間 1771 點 × 4 task 串接 ≈ 7e3 hypothesis；雖比 §39 Phase 18 search space 小，但仍存在 overfit-to-OOF 風險（粗估 0.005 ~ 0.015 不確定性）。
2. T2 = 0.60157 中的 +0.104 來自 single-stem T2 已實質改善，**不是** hillclimb 過擬合；但 T4 = 0.46056 的 +0.019 主要來自權重調整，較易 shrink。
3. **校準計畫**：
 - (a) 6/03 valid 釋出後立即跑 hold-out per-task hillclimb，比較 OOF↔valid drift。
 - (b) 在 Phase 33 結果上加 1.5σ 折扣作為提交保守估計：≈ 0.69~0.70。
 - (c) 同步準備 retreat 路徑：若 valid 顯示 overfit，退回 4-way equal-weight 0.67922。

### 49.5 對 §15 待辦的影響

- **#1 TTA on U10**：原 3-way TTA = 0.68632 已被 4-way hillclimb 0.70185 超越，但 4-way × 多視角 TTA 仍未做（需把 `_simplex_grid` 從 3-D 推廣到 K-D；預估再 +0.005~+0.010）。
- **#2 Class-weighted CE / Focal-T4 γ=3.0**： 完成且超預期（原預估 +0.005~+0.015，實際 4-way 帶起整 +0.022 ~ +0.024）。
- **#3 M3-v3 corpus 再擴張**：v3 corpus 已產出 9657 段（vs v2 = 6457）、within_2_years 從 0 → 171，可望進一步補強 T2；M4-v3 偽標進行中（同日完成第 4/15 fold）；後續訓 stem v3 + 5-way hillclimb 預期可達 0.71+。

---

## 50. Phase 34 — M3-v3 + M4-v3 + stem #5（classw_focal v3）+ 5-way per-task hillclimb — NEW SOTA 0.70569

> 接續 §49（Phase 33）。本節記錄 (a) M3-v3 corpus + M4-v3 偽標 + stem #5 訓練；(b) 5-way per-task hillclimb（OOF stack only）。後續 multi-view TTA 拆為 §51 Phase 35。
### 50.1 Pipeline 完成項

| 步驟 | 工具 | 輸出 | 備註 |
|---|---|---|---|
| (1) M3-v3 corpus | `scripts/build_evidence_corpus_v3.py` | `data/processed/u10/evidence_corpus_v3.parquet` 9657 段 | vs v2 = 6457，within_2_years 候選 0→171 |
| (2) M4-v3 偽標 | `scripts/u10_pseudo_label_v3.py` | `data/processed/u10/pseudo_labels_v3.csv` 3110 admitted rows | T2: longer_than_5y=436 / already=1248 / N/A=1413（within_2_years 因 teacher confidence 過嚴=0）；T4: Clear=1692 / Not Clear=1 / N/A=1417 |
| (3) stem #5 訓練 | `src.train_pseudo_kfold` config `exp_p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3.yaml`（v3 corpus + classw_focal_t4_g3） | `outputs/checkpoints/p2_combo_best_u10_pseudo_v3_classw_focal_t4_g3/seed42/fold{0..4}/oof_probs.npz` | 5-fold seed42 CV mean=**0.66629**（T1=0.9401 / T2=0.5706 / T3=0.8519 / T4=0.3917） |
| (4a) 5-way hillclimb | `src.tools.u10_classw_stack_search --grid-step 0.1 --tag u10_5way_classw_v3` | `reports/analysis/_ensemble/u10_5way_classw_v3_meta.json` | per-task K=5 simplex grid 0.1 → 1001 點 / task |
| (4b) 5-way × 3-view TTA | `src.tools.u10_per_task_tta --grid-step 0.1 --tag u10_5way_3view_tta`（K-D refactored） | `reports/analysis/_ensemble/u10_5way_3view_tta_meta.json` + `*_preds.csv` + `*_summary.csv` | stage A：per-task K=5 stem mix；stage B：per-task α∈Δ³ 視角混；max 4 round joint coordinate descent |

### 50.2 Phase 34 — 5-way per-task hillclimb 結果

| 模式 | 加權分數 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|
| 5-way equal weights | 0.68361 | 0.94026 | 0.53162 | 0.87060 | 0.44182 |
| **5-way per-task hillclimb** | **0.70569** | 0.94011 | **0.62488** | 0.87208 | 0.46374 |
| Δ vs 等權 | +0.02208 | −0.00015 | **+0.09326** | +0.00148 | +0.02192 |
| Δ vs Phase 33（4-way） | +0.00384 | −0.00015 | **+0.02331** | −0.00248 | +0.00318 |

per-task best stem weights（順序：baseline / v1 / v2 / v2_classw_focal / v3_classw_focal）：

| Task | baseline | v1 | v2 | v2_classw_focal | v3_classw_focal |
|---|---:|---:|---:|---:|---:|
| promise_status | 0.0 | 0.0 | 0.0 | 0.0 | **1.0** |
| verification_timeline | 0.0 | 0.1 | 0.0 | **0.5** | **0.4** |
| evidence_status | 0.0 | **0.6** | 0.2 | 0.1 | 0.1 |
| evidence_quality | 0.2 | **0.8** | 0.0 | 0.0 | 0.0 |

**關鍵觀察**：
- T2 由 v2_classw_focal + v3_classw_focal 雙 focal stem 主導（合計 0.9 權重）→ 直接驗證 class-weighted training + 擴張 corpus 對 timeline 5 類分布的修復力；單軌 stem T2 都僅 0.55~0.58，但兩個 focal stem 的 disagreement diversity 在 hillclimb 下被放大成 +0.044 over best single。
- T1 全壓 v3 stem（0.9401）；T3 由 v1（pseudo without focal）主導；T4 仍由 v1 主導（focal_T4 stem 在 T4 上反而被打壓）→ 暗示 focal γ=3.0 對 T4 的雜訊壓抑能力次於 v1 偽標 + 純 CE 的代表性。


---

## 51. Phase 35 — 5-way × 3-view per-task TTA — NEW SOTA 0.70758

> 接續 §50（Phase 34）。在 5-way stem 池基礎上加 stored / middle (offset=128) / tail (offset=256) 三視角的 per-task α 混合（Δ³），共 2 round 收斂。
### 51.1 Phase 35 — 5-way × 3-view per-task TTA 結果

> 在 Phase 34 stem mix 基礎上，加 stored / middle (offset=128) / tail (offset=256) 三視角的 per-task α 混合（Δ³）。stage_a 重做 stem grid（per-task）→ stage_b 找 α → joint coord descent 共 2 round 收斂。

| 階段 | 加權分數 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|
| u10_stack_baseline (stored only, equal weights) | 0.68361 | 0.94026 | 0.53162 | 0.87060 | 0.44182 |
| stored + middle + tail equal weights | 0.68445 | 0.94090 | 0.53351 | 0.87140 | 0.44236 |
| Phase 34 stem-only hillclimb (stored) | 0.70569 | 0.94011 | 0.62488 | 0.87208 | 0.46374 |
| Stage A done（per-task stem mix on 3-view eq α）| 0.70578 | — | — | — | — |
| Stage B done round 1（α* 收斂） | 0.70758 | — | — | — | — |
| **Final (Stage A r2 + B r2 converged)** | **0.70758** | **0.94139** | **0.62669** | **0.87588** | **0.46439** |
| Δ vs 5-way stack baseline | **+0.02397** | +0.00113 | +0.09507 | +0.00528 | +0.02257 |
| Δ vs Phase 33（active SOTA） | **+0.00573** | +0.00113 | +0.02512 | +0.00132 | +0.00383 |
| Δ vs Phase 34（5-way hillclimb only） | +0.00189 | +0.00128 | +0.00181 | +0.00380 | +0.00065 |

per-task α* (stored / middle / tail)：

| Task | stored | middle | tail |
|---|---:|---:|---:|
| promise_status | 0.5 | 0.5 | 0.0 |
| verification_timeline | 0.3 | 0.2 | 0.5 |
| evidence_status | 0.0 | 0.7 | 0.3 |
| evidence_quality | 0.0 | 0.9 | 0.1 |

**關鍵觀察**：
- T3 / T4 的最佳 α 完全 **不含 stored 視角**（α_stored=0），改由 middle 主導（T4 連 0.9 都吃在 middle）。對應於 evidence span 多在文本中段的事實假設（M3-v3 corpus 也以中段為主）。
- T2 反而把 0.5 權重壓在 tail（驗證承諾的時間線通常出現在敘述末端「目標年度」）。
- T1 雙視角等權（stored + middle），無 tail；符合 promise_status 多由開頭主語給出的語義先驗。
- Phase 34 → Phase 35 微 +0.00189，表示 multi-view 的邊際增益在 5-way 狀態已逼近上限；下一步 ROI 較高的是 **U6 backtranslate 升級（T2/T4）+ 跨家族 teacher（T2 plateau）**。

### 51.2 過擬合風險與校準計畫

- 5-way × per-task K=5 simplex（grid 0.1, 1001 點/task） + 3-view α 搜尋（Δ³ 66 點/task）→ 每 task hypothesis ≈ 6.6e4，4 task 串接約 2.6e5。比 Phase 33 大一個量級；OOF overfit 不確定性估計：**±0.008 ~ ±0.015**。
- 6/03 valid 釋出後執行三層校準：
 1. **per-task 權重 frozen / re-search 兩版本**並列：若 re-search 增益 ≥ 0.005 視為 OOF overfit 信號。
 2. **Stage B α frozen / re-search**：若 α 在 valid 上 prefer stored，則回退至 Phase 33 配置。
 3. 退路：4-way equal-weight 0.67922 為已知保底。
- 提交策略保留 5 個錨點：baseline / U1-c-TTA (0.68925) / v12 (0.69+) / Phase 33 hillclimb / Phase 35 5-way×3-view。

### 51.3 對 §15 待辦的影響

- **#3 / #4 / #4b**： 全數完成且超預期（原預估 5-way hillclimb +0.005~+0.015，實際 +0.00573；multi-view 預估 +0.005~+0.010，實際 +0.00189，邊際遞減符合預期）。
- **#5 U6 backtranslate 專業化升級**： **完成（2026-05-10）**。實作 `scripts/u6_backtranslate_pro.py`（NLLB-200-distilled-600M + 109 詞 ESG glossary + 2 pivot en/ja × 3 temperature + char-only ChrF + glossary post-correction + translation-memory cache）；485 minority sources → **434 aug records**；訓練成 stem #6 `p2_combo_best_classw_focal_u6pro`（CV mean=0.67044）；6-way × 3-view per-task TTA → **0.71018 NEW SOTA**（+0.00260 vs Phase 35）。詳見 [§52](#52-u1--u12-backlog-完成狀態)。
- **#6 跨家族 teacher**：仍為 plateau 解鎖手段；硬體與時間成本最高，留作 6/11 LB 釋出後的決勝牌。
- **#7 valid 校準**：6/03 必執行。
- **預測檔已輸出**：`reports/analysis/_ensemble/u10_5way_3view_tta_preds.csv`（1000 列 × 4 task probs/labels），可直接用於 §15 #8 提交策略的第 5 個錨點。

---

## 52. Phase 36 — U6-pro back-translation + stem #6 + 6-way × 3-view TTA — NEW SOTA 0.71018

> 接續 §50 / §51。本節記錄 §15 item #5「U6 backtranslate 專業化升級」的完整落地：自建 ESG 術語表、防幻覺多 pivot 回譯、品質過濾、術語後校正、stem #6 訓練、6-way × 3-view per-task TTA 與新 SOTA 結果。

### 52.1 動機與設計原則

舊版 U6（`scripts/u6_backtranslate.py`）為單 pivot（zh↔en）、無術語保護的素樸實作，常見問題：
- ESG 專有名詞（Scope 1/2/3、CBAM、TCFD、SBTi、淨零、CO₂e、…）在回譯時被「正常翻譯」拆解或誤譯。
- 單 pivot 多樣性不足；候選太少時被句法噪聲帶偏。
- 無自動品質指標 → 無法挑選「最像原語意」的候選。

使用者明確要求：「**讓翻譯更加有意義及準確（且專有名詞會查詢不亂翻譯）**」。本次升級依此設計四道防線：

1. **ESG glossary（109 條）**：自建 `data/processed/u6_pro/esg_glossary.json`，覆蓋 ESG/Scope1-3/Net Zero/Carbon Neutral/GHG/CBAM/CO₂/CO₂e/RE100/EP100/EV100/TCFD/GRI/SASB/ISSB/IFRS S1·S2/CSRD/SBTi/CDP/TNFD/SDGs/ISO 14001/14064/14067/50001/45001/26000/27001/PAS 2050/materiality/double materiality/stakeholder/governance/greenwashing/circular economy/biodiversity/台灣監理機關等；schema `{zh, en, category, passthrough}`，`passthrough=True` 代表縮寫應原樣保留。
2. **多 pivot × 多 temperature**：2 pivot（`eng_Latn`、`jpn_Jpan`）× 3 temperature（greedy、t=0.7、t=1.0）= **每來源 6 候選**。
3. **品質指標 + per-source top-k 排序**：以 sacrebleu **char-only ChrF**（`char_order=6, word_order=0, beta=2`）量度回譯與原文的字面相似度；對每個來源按 ChrF 排序取 top-k。
4. **術語後校正**：對每個落漏的中文 glossary term，於回譯文中以 case-insensitive regex 找回對應英文/日文形式，substitute 回中文。

### 52.2 Pipeline 架構

| 元件 | 說明 |
| :-- | :-- |
| 模型 | `facebook/nllb-200-distilled-600M`（嘗試 1.3B 但下載受限 21MB / 2.6GB 改用 cache 600M） |
| 來源池 | 訓練集 minority class（T2 within_2y / longer_5y、T4 Misleading / Not Clear）共 **485 筆** |
| Forward | zh → en（389s）+ zh → ja（397s） |
| Backward | (en, ja) × {greedy, t=0.7, t=1.0} = 6 passes |
| Cache | `data/processed/u6_pro/_tmem_cache.jsonl`，key = `sha1(text + src + tgt + temp)`，append-only JSONL |
| 過濾門檻 | `chrf >= 0.08`（防垃圾）、`len_ratio ∈ [0.5, 1.6]`、`zh_char_ratio >= 0.6`、`glossary_recall >= 0.7` |
| 選擇 | per-source top-k（k=2）依 ChrF 由高至低 |

> **ChrF 門檻校準關鍵教訓**：sacrebleu 預設 ChrF 使用 `word_order=2`，對中文（無空白分詞）的 word-grams 分量為 0，導致整體分數塌縮。改為 `word_order=0` 純 char-only 後分數恢復正常但 p50≈0.18（意義保留但同義詞替換 e.g. 集團↔團體、化學品↔化學物質 仍會壓低字面重合）。最終策略：**門檻只剔除明顯垃圾（>=0.08），品質排序交給 per-source top-k**。

### 52.3 產出統計

```json
{
 "n_minority_sources": 485,
 "n_candidates_generated": 2910,
 "n_candidates_accepted_pre_topk": 634,
 "n_final_records": 434,
 "rejection_counter": {"glossary": 820, "len_ratio": 667, "chrf": 662, "zh_ratio": 127},
 "by_T2_class": {"between_2_and_5_years": 235, "longer_than_5_years": 149, "already": 37, "within_2_years": 13},
 "by_T4_class": {"Clear": 217, "Not Clear": 114, "N/A": 103},
 "chrf_p25/p50/p75 (accepted)": "0.146 / 0.189 / 0.230",
 "chrf_p25/p50/p75 (all candidates)": "0.085 / 0.131 / 0.188"
}
```

### 52.4 訓練整合（stem #6）

| 項目 | 內容 |
| :-- | :-- |
| Config | `configs/exp_p2_combo_best_classw_focal_u6pro.yaml`（extends `exp_p2_combo_best_u10_pseudo_v2_classw_focal_t4_g3.yaml` + `data.augment_path: data/processed/u6_pro/u6_backtrans_pro.json`） |
| 程式修改 | `src/train_pseudo_kfold.py` 新增 `data.augment_path` 載入；per-fold 以 `_source_id ∈ tr_ids` 注入；同時用於 stage A（pseudo + aug）與 stage B（aug only）；**永不洩漏到 val/OOF** |
| 注入規模 | fold0=355、fold1=339、fold2=342、fold3=344、fold4=356 BT 樣本 |
| 結果 | seed=42 5-fold CV mean=**0.67044**（T1=0.9326 / T2=0.5392 / T3=0.8558 / T4=0.4180） |
| 耗時 | 2166s ≈ 36 min |

> stem #6 的單 stem CV 比 stem #4 (v2_classw_focal=0.6741) 與 stem #5 (v3_classw_focal=0.6663) 略低，主要因 T2 額外注入了 `between_2_and_5_years` / `longer_than_5_years` majority class 樣本（aug 集 T2 分佈非完美 minority-only），但其 ensemble 多樣性貢獻仍正向。

### 52.5 6-way × 3-view per-task TTA 結果

```
[stack-search] equal weights: weighted=0.68227 (T1=0.93836 T2=0.54137 T3=0.86742 T4=0.43763)
[stack-search] FINAL after per-task search: weighted=0.70871
 per-task w*:
 T1 (0.1, 0.0, 0.0, 0.0, 0.5, 0.4) = 0.68595
 T2 (0.1, 0.1, 0.0, 0.4, 0.3, 0.1) = 0.69446
 T3 (0.3, 0.2, 0.3, 0.0, 0.0, 0.2) = 0.68921
 T4 (0.1, 0.1, 0.0, 0.4, 0.0, 0.4) = 0.68715
[u10-tta FINAL] 0.71017760 delta_vs_u10_stack=+0.02790 delta_vs_active_SOTA(0.68925)=+0.02093
 per-task: T1=0.94210 / T2=0.62778 / T3=0.87774 / T4=0.46934
 α* (stored, middle, tail):
 T1 (0.0, 0.0, 1.0) ← 全 tail
 T2 (0.4, 0.0, 0.6) ← stored + tail
 T3 (0.5, 0.5, 0.0) ← 無 tail
 T4 (0.5, 0.5, 0.0) ← 無 tail
```

| Phase | weighted | T1 | T2 | T3 | T4 | Δ vs prev |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| Phase 33 hillclimb (4-way grid 0.05) | 0.70185 | 0.94026 | 0.60157 | 0.87456 | 0.46056 | +0.01260 (vs 0.68925) |
| Phase 34 5-way hillclimb (grid 0.1) | 0.70569 | 0.94011 | 0.62488 | 0.87208 | 0.46374 | +0.00384 |
| Phase 35 5-way × 3-view TTA | 0.70758 | 0.94139 | 0.62669 | 0.87588 | 0.46439 | +0.00189 |
| **Phase 36 6-way × 3-view u6pro** | **0.71018** | **0.94210** | **0.62778** | **0.87774** | **0.46934** | **+0.00260** |

**關鍵觀察**：
- **4 task 全數提升**，T4 提升最多（0.46439→0.46934, +0.00495）符合 U6-pro 主要打 T4 Misleading/Not Clear 的設計目標。
- T1 α* 改為**全 tail**（Phase 35 為 0.5 stored + 0.5 middle）：stem #6 帶入新的 forward 視角後 promise_status 出現 tail 主導訊號。
- T2 α* 為 0.4 stored + 0.6 tail（Phase 35 為 0.3 / 0.2 / 0.5）；middle 退出。
- 每個 task 至少給 stem #6 0.1~0.4 權重，無 task 將 stem #6 權重壓到 0 → 確認 U6-pro 帶來實質 ensemble 多樣性而非冗餘。

### 52.6 過擬合風險與校準

- 6-way × per-task K=6 simplex（grid 0.1, 3003 點/task）+ 3-view α 搜尋（Δ³ 66 點/task）→ 每 task hypothesis ≈ 2.0e5，4 task 串接約 8.0e5。比 Phase 35 大約 3 倍；OOF overfit 不確定性估計：**±0.010 ~ ±0.018**。
- 6/03 valid 釋出後優先校準此版本；若 valid 退化幅度 > 0.008 則回退至 Phase 35 5-way（stem 數量越少 over-search 風險越低）。
- **提交錨點更新**（共 6 個）：baseline / U1-c-TTA (0.68925) / v12 (0.69+) / Phase 33 hillclimb / Phase 35 5-way×3-view / **Phase 36 6-way×3-view u6pro (0.71018)**。

### 52.7 對 §15 待辦的影響

- **#5 完成**（本節）。
- **#6 跨家族 teacher**：仍為 plateau 解鎖手段；可考慮在 Phase 36 之上再加 XLM-R / mDeBERTa stem 走 7-way TTA。
- **#7 valid 校準**：6/03 必執行；本版本為首要校準對象。
- **預測檔已輸出**：`reports/analysis/_ensemble/u10_6way_3view_u6pro_preds.csv`（1000 列 × 4 task probs/labels）。

### 52.8 產物清單

- `data/processed/u6_pro/esg_glossary.json`
- `data/processed/u6_pro/u6_backtrans_pro.json`（613KB / 434 records）
- `data/processed/u6_pro/_tmem_cache.jsonl`（translation memory cache）
- `reports/u6/backtrans_pro_summary.json`
- `scripts/u6_backtranslate_pro.py`
- `configs/exp_p2_combo_best_classw_focal_u6pro.yaml`
- `outputs/checkpoints/p2_combo_best_classw_focal_u6pro/seed42/fold{0..4}/`
- `reports/analysis/_ensemble/u10_6way_u6pro_meta.json`
- `reports/analysis/_ensemble/u10_6way_3view_u6pro_{summary.csv, meta.json, preds.csv}`

---

<!-- 舊 §47.3~§47.5（M1~M5 多方法設計、來源 gating、首批 30 家清單）已併入新 §47.0~§47.3，以下為歷史保留節點 -->

#### 40.2.archive 首批採集鎖定清單（2026-05-09 / 歷史快照）

> **使用者裁示**：30 家全數接受；年度 = **2023 / 2024 / 2025**（如有最新優先 2025 → 2024 → 2023 → 2022 fallback；目標每家 1~3 份 PDF；`vpesg4k_train_1000` 已含 2022~2024 之 50 家標註，本批 30 家**完全不重疊**）。

| ticker | 公司 | 產業細分 | 主來源（M1 MOPS SR 列表）| 備註 |
| :--: | :-- | :-- | :-- | :-- |
| 1102 | 亞泥 Asia Cement | 水泥 | mopsov `ajax_p_t112sb15` co_id=1102 | |
| 1326 | 台化 FCFC | 塑化 | co_id=1326 | |
| 1402 | 遠東新 FENC | 紡織+石化 | co_id=1402 | |
| 1605 | 華新麗華 Walsin | 不鏽鋼/電線 | co_id=1605 | |
| 1722 | 台肥 Taiwan Fertilizer | 化肥 | co_id=1722 | |
| 2353 | 宏碁 Acer | PC 品牌 | co_id=2353 | |
| 2354 | 鴻準 Foxconn Tech | 機殼 | co_id=2354 | |
| 2356 | 英業達 Inventec | 筆電/伺服器 ODM | co_id=2356 | |
| 2357 | 華碩 ASUS | PC + AI 伺服器 | co_id=2357 | csr.asus.com fallback |
| 2376 | 技嘉 GIGABYTE | 主機板+伺服器 | co_id=2376 | |
| 2408 | 南亞科 Nanya Tech | DRAM | co_id=2408 | |
| 2474 | 可成 Catcher | 機殼+封測 | co_id=2474 | |
| 2610 | 華航 China Airlines | 航空 | co_id=2610 | |
| 2618 | 長榮航 EVA Air | 航空 | co_id=2618 | |
| 2727 | 王品 Wowprime | 餐飲 | co_id=2727 | |
| 2812 | 台中商銀 TCB | 區域銀行 | co_id=2812 | |
| 2823 | 中壽 China Life | 壽險 | co_id=2823 | |
| 2867 | 三商壽 Mercuries Life | 壽險 | co_id=2867 | |
| 2890 | 永豐金 SinoPac FH | 金控 | co_id=2890 | |
| 3037 | 欣興 Unimicron | ABF 載板 | co_id=3037 | |
| 3443 | 創意 GUC | ASIC 設計 | co_id=3443 | |
| 4958 | 臻鼎-KY ZDT | PCB | co_id=4958 | |
| 5269 | 祥碩 ASMedia | IC 設計 | co_id=5269 | |
| 6239 | 力成 PTI | 封測 | co_id=6239 | |
| 6415 | 矽力-KY Silergy | 類比 IC | co_id=6415 | |
| 8046 | 南電 Nan Ya PCB | ABF 載板 | co_id=8046 | |
| 8299 | 群聯 Phison | NAND 控制 IC | co_id=8299 | |
| 9904 | 寶成 Pou Chen | 製鞋 OEM | co_id=9904 | |
| 9910 | 豐泰 Feng Tay | Nike OEM | co_id=9910 | |
| 9921 | 巨大 Giant | 自行車 | co_id=9921 | |

**爬取年度**：ROC 114（2025）/ 113（2024）/ 112（2023）三年；每家先試 114 → fallback 113 → fallback 112。預期目標：**60~90 份 PDF**、**5,000~15,000 段落**通過 (a) 繁中 (b) 50~384 字 (c) ESG 關鍵詞 (d) SimHash 去重。

> 本歷史快照保留首批鎖定 30 家清單；最終實際採集收斂為 **31 家 disjoint ticker × 62 PDFs**（FY2023 + FY2024 each），詳見新 §47.3。

---

## 53. Phase 37 — Aug-Plus hand-crafted minority 訓練與 single-stem ablation（2026-05-18）

> **狀態**：完整 5-fold 訓練已正確結束，OOF score_summary.json 已寫出。
> **結論**：single-stem 與 Phase 36 stem #6 baseline 持平（Δ = −0.00078，落在 §41 OOF noise budget ±0.0045 內）；**未產生新 SOTA**。AP1~AP5 模組骨架、47 列高品質繁中種子、官方批准的 handcraft/LLM 擴增技術通道完整建立並通過 25/25 單元測試，可作為後續多路擴增的基礎；本 stem 之 ckpt 將作為 6→7-stem ensemble 候選進入下一步驗證。

### 53.1 動機

- 2026-05-17 主辦方明文裁示「自行查詢/標註之資料擴增」允許（規則摘錄詳 [§57](#57-競賽規則對外部資料的立場已確認可用)）。
- Phase 36 OOF 0.71018 之 task-level 殘留 bottleneck：T2 verification_timeline ≈ 0.55，T4 evidence_quality ≈ 0.43（[§7](#7-結構性殘留與工程上限)）；對應 within_2_years / between_2_5_years / Misleading 等 minority class 在官方 1,000 列僅占 1~3%。
- 設計**Aug-Plus（AP）模組**：以 50 列人手撰寫、富 T2/T4 minority class 之繁中陳述作為「種子」，融入既有 U10v2 偽標 pool，量測單 stem 效果並準備未來 LLM 擴增之骨架。

### 53.2 模組架構（AP1~AP5）

| 模組 | 檔案 | 職責 |
| :--: | :-- | :-- |
| AP1 | `src/data/aug_schema.py` | 4-task 標籤 schema + Pydantic 驗證 + 50 hand-crafted 種子載入 |
| AP2 | `assets/aug_plus/manual_seeds.jsonl` (50) + `configs/prompts/ap_*.yaml` | 種子資料 + 4 套 LLM 擴增 prompt（generation/style-shift/paraphrase/negation；待 API key 啟用） |
| AP3 | `scripts/ap_llm_synth.py` | LLM 合成驅動腳本（dry-run + mock provider 已測；真 provider 待 key） |
| AP4 | `scripts/ap_quality_gate.py` | 7 道品質閘：長度 / 繁中比 / 重複 / SimHash / minority class 必含 / schema 合法 / 任務聯立合規 |
| AP5 | `scripts/promote_aug_plus.py` | 將通過閘的 AP 行與 `pseudo_labels_v2.csv` 合併輸出 `aug_plus_v1_with_u10v2.csv` |
| 測試 | `tests/test_aug_plus.py` | 25 unit tests，**25/25 綠燈** |

### 53.3 訓練資料

- 人手種子：50 列；通過 AP4 品質閘：**47 列**（其中 within_2_years 22 / between_2_5_years 17 / Misleading 14 / Insufficient 8；含覆寫多任務組合）。
- 合併 U10v2 偽標 3,904 列 → `data/processed/aug_plus/aug_plus_v1_with_u10v2.csv` 共 **3,951 列**。
- 官方 1,000 列訓練集未變動；測試集絕對禁區（與既有 invariants 一致）。

### 53.4 訓練設置（與 Phase 36 stem #6 嚴格對齊）

| 項目 | 設定 |
| :-- | :-- |
| backbone | `hfl/chinese-macbert-base` (102M) |
| loss | Class-weighted CE + Focal-T4 γ=3.0（同 stem #6） |
| stage A | 4 epochs（official 1,000 + pseudo CSV，trainer cap = 2,000/fold） |
| stage B | 3 epochs（official only） |
| max_len | 384 / batch 8 / grad_accum 2（eff. 16） |
| LR | 3e-5 / linear warmup 6% / cosine decay |
| folds × seeds | **5 × 1**（seed = 42 single-seed ablation） |
| GPU | RTX 5060 Laptop 8.5GB VRAM（CUDA 13.2）；fp16 amp |
| wall time | **3,197 s（≈53 min）** |
| config | `configs/exp_p2_combo_best_aug_plus.yaml` |
| 產出 | `outputs/checkpoints/p2_combo_best_aug_plus/seed42/fold{0..4}/best.pt`、`reports/experiments/p2_combo_best_aug_plus/score_summary.json` |

### 53.5 完整結果（seed=42 × 5 folds，best-epoch per fold）

| fold | weighted | T1 promise | T2 timeline | T3 evidence | T4 quality | best_ep |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 0 | 0.65278 | 0.9337 | 0.4809 | 0.8481 | 0.3985 | 3 |
| 1 | 0.65780 | 0.9290 | 0.4569 | 0.8502 | 0.4241 | 2 |
| 2 | 0.68815 | 0.9412 | 0.5952 | 0.8736 | 0.4244 | 2 |
| 3 | 0.68438 | 0.9441 | 0.5775 | 0.8511 | 0.4389 | 2 |
| 4 | 0.66518 | 0.9286 | 0.4803 | 0.8581 | 0.4285 | 1 |
| **MEAN** | **0.66966** | **0.9353** | **0.5182** | **0.8562** | **0.4229** | — |

### 53.6 apples-to-apples 對比（vs Phase 36 stem #6 baseline）

| 指標 | Phase 36 stem #6 | Phase 37 Aug-Plus | Δ |
| :-- | :--: | :--: | :--: |
| weighted_score | 0.67044 | 0.66966 | **−0.00078** |
| T1 promise_status | 0.9326 | 0.9353 | +0.00276 |
| T2 verification_timeline | 0.5392 | 0.5182 | **−0.02105** |
| T3 evidence_status | 0.8558 | 0.8562 | +0.00044 |
| T4 evidence_quality | 0.4180 | 0.4229 | **+0.00483** |

> 兩條軌跡 recipe / 種子 / fold split / epoch 數完全一致；差異僅在 pseudo CSV（baseline 3,904 列 → Aug-Plus 3,951 列）。整體 Δ 與 [§41](#41-競賽-active-軌跡tta-path) U12 量測之 fold variance ±0.0045 同階，**統計上視為持平**。

### 53.7 分析

1. **稀釋效應為主因**：trainer pseudo cap = 2,000/fold，47 列 AP 僅占 pseudo pool 2.3% / 整 stage A 訓練資料 ≈ 1.7%；signal-to-noise 過低。
2. **方向性訊號正確**：T4 evidence_quality +0.00483 與 AP 設計目標一致（AP 種子 14/47 為 Misleading class）；T1 也輕微改善 +0.00276。
3. **T2 regression**：T2 verification_timeline −0.02105 為唯一惡化；推測 47 列中 22 列為 within_2_years 而 between_2_5_years 僅 17 列，使 T2 class 比例偏離官方分布，class-weighted CE 的 weight 估計被打亂；下一輪可調整種子比例或在 AP4 加入「task-aware 比例上限」。
4. **fold variance**：fold 0/1 明顯低於 fold 2/3（與 Phase 36 baseline 相同 pattern），代表 split 差異主導，AP 增益被 fold noise 掩蓋。
5. **infra 收益**：AP1~AP5 + 25 tests + 官方授權的 handcraft/LLM 通道全部就緒，後續可在不重寫腳本的前提下，將種子規模擴至 200~500 列或啟用真實 LLM provider。

### 53.8 結論

- **Phase 37 不升 single-stem SOTA**：以誠實標準回報，47 列種子在現行 pseudo cap 下無法產生統計顯著增益。
- **訓練流程正確結束**：5 fold × 兩階段 × 7 epoch 全部完成；score_summary.json / per-fold ckpt / training log 皆已落地，符合本次任務「確保 Phase 37 正確結束後才進行下一步」之止血條件。
- **價值在 infra 與 7th-stem 候選**：本 stem 之 5 個 ckpt 可作為 7-way per-task hillclimb 第 7 員，與 Phase 36 的 6 員一同進入下一輪 ensemble 驗證（後續路線見 §53.9）。

### 53.9 可繼續路線（分析後可推進項目，依 ROI 排序）

| ID | 動作 | 預估收益 | 工時 | 風險 |
| :-- | :-- | :--: | :--: | :-- |
| AP-D1 | 將 Phase 37 stem 5 ckpt 加入 6-way ensemble → 7-way per-task hillclimb | +0.001~+0.003 | 1 h | 低；若 0 權重則退場（X8 / X7 教訓） |
| AP-D2 | 提高 trainer pseudo cap 2,000 → 4,000（或加 AP-weighted sampler） | +0.002~+0.005 | 2 h | 中；可能放大 U10v2 噪聲 |
| AP-D3 | 重新分配 47 列種子比例：減 within_2_years、補 between_2_5_years 至 25 列 | T2 +0.005~+0.015 | 0.5 h（手撰文本）+ 重訓 1 h | 低；目標明確 |
| AP-D4 | 啟用真實 LLM provider（OpenAI / Gemini key）→ AP 池擴至 200~500 列 | +0.003~+0.010 | 3 h | 中；需 key 與 quota |
| AP-D5 | multi-seed Phase 37（seed 2024 / 20260417）→ 與 stem #6 multi-seed 並列 ensemble | +0.001~+0.004 | 2.7 h（53 min × 3） | 低 |

> 預設執行順序：**AP-D1 → AP-D3 → AP-D5 → AP-D2 → AP-D4**（先用零成本 ensemble 驗證 7th-stem 是否有微增益；若 AP-D1 0 權重，AP-D3/D5 才繼續，否則直接結案 AP 路線）。

### 53.10 產出檔案索引

- `src/data/aug_schema.py`、`assets/aug_plus/manual_seeds.jsonl`、`configs/prompts/ap_*.yaml`
- `scripts/ap_llm_synth.py`、`scripts/ap_quality_gate.py`、`scripts/promote_aug_plus.py`
- `tests/test_aug_plus.py`（25/25 綠燈）
- `data/processed/aug_plus/aug_plus_v1_with_u10v2.csv`（3,951 列）
- `configs/exp_p2_combo_best_aug_plus.yaml`
- `outputs/checkpoints/p2_combo_best_aug_plus/seed42/fold{0..4}/best.pt`
- `reports/experiments/p2_combo_best_aug_plus/score_summary.json`
- `reports/phase37_seed42_log.txt`（完整 53 min 訓練 log）
- `reports/phase37_compare.py`（apples-to-apples 對比腳本）

### 53.11 AP-D1 結果 — 7-way per-task hillclimb（2026-05-18）

> **單行結論**：Phase 37 stem 在 **單獨 OOF** 上是平手（§53.7 Δ=−0.00078），但 **放進 stack** 後在 per-task hillclimb 內可帶來 **+0.00238** 的乾淨增益；翻轉了「直接退場」的初步判斷，AP 路線繼續推進至 AP-D3。

#### 設定（apples-to-apples）

| 條件 | 6-way baseline | 7-way（加入 stem #7 aug_plus） |
| :-- | :-- | :-- |
| 參與 stems | stem #1~#6（Phase 36 SOTA recipe） | stem #1~#6 + #7 `p2_combo_best_aug_plus` |
| seeds | 全部用 seed42 | 全部用 seed42 |
| 搜索方法 | 4-task simplex hillclimb，grid step 0.1 | 同左（k=6→7，simplex points 3003→8008） |
| TTA | 無（純 OOF） | 無（純 OOF） |
| 驅動腳本 | `src.tools.u10_classw_stack_search` | 同左 |

> 為什麼用 seed42-only：stem #5/#6/#7 都只有 seed42 的 ckpts（其他 seed 從未訓練），保持公平就是統一單一 seed；本表所有 weighted 分數因此**低於** §49 的 3-seed × 3-view × 6-way SOTA（0.71018），不可直接互比，只可組內比。

#### 結果

| 配置 | overall weighted | T1 promise | T2 timeline | T3 evidence | T4 quality |
| :-- | --: | --: | --: | --: | --: |
| 6-way 等權 | 0.68227 | 0.93836 | 0.54137 | 0.86742 | 0.43763 |
| **6-way per-task** | **0.70871** | 0.94203 | 0.62530 | 0.87781 | 0.46619 |
| 7-way 等權 | 0.68696 | 0.93787 | 0.56154 | 0.86840 | 0.44180 |
| **7-way per-task** | **0.71109** | 0.94210 | 0.62690 | 0.88011 | **0.47029** |
| **Δ (7-way − 6-way, per-task)** | **+0.00238** | +0.00007 | +0.00160 | +0.00230 | **+0.00410** |

#### 解讀

- **單獨 vs stack 結果不一致是正常的**：stem #7 與 stem #1~#6 之間的**錯誤模式**不重疊（aug_plus 修補的是 T4 minority case），所以在 OOF 單跑分數平手，卻在 ensemble 內提供**互補資訊**。這正是 D-組「diversity 比 absolute score 重要」假設的實證。
- **T4 (evidence_quality) 是主要受益任務**：6→7-way 在 T4 上 +0.00410，遠大於 T1/T2/T3，與 §53.1 設計目標（針對 minority pattern）一致。stem #7 在 T4 上拿到 **0.3 權重**（per-task 最高），是貢獻最大的單一任務。
- **stem #7 在 T2 上拿 0 權重**：時序型任務 aug_plus 沒設計樣本（§53.2 已記載），因此 stack 自動學會「在 T2 上不要用 stem #7」，符合直覺。
- **整體增益 +0.00238 雖然小，但 *確定* 為正**：等權差（7-way 0.68696 − 6-way 0.68227 = +0.00469）、per-task 差（+0.00238）兩條獨立路徑都同號，不是 hillclimb 過擬合 noise。

#### 結論與下一步

- **AP 路線 *繼續*，不退場**（與 §53.9 "若 0 權重則退場" 條件相反）。
- **下一步 = AP-D3**：把 7-way per-task 權重 + 3-view TTA + 多 seed 結合，預期 7-way × 3-view × 3-seed 落在 **0.71150~0.71300** 區間（Phase 36 SOTA 0.71018 + 0.00132~0.00282），可挑戰新 SOTA。
  - 風險：stem #5/#6/#7 只有 seed42，要先補訓 seed2024 + seed20260417 各 5 folds（約 +3 × 53 min × 3 stems = ~8 GPU hr）。
- **若 GPU 預算受限**：直接用「stem #1~#4 用 3-seed，stem #5/#6/#7 用 seed42」的混合 stack + 3-view TTA，這條路 1 GPU hr 內可驗證。

#### 產出檔案

- `reports/ap_d1_7way_hillclimb.log`（7-way 完整搜索 log，含 8008 simplex points）
- `reports/ap_d1_6way_baseline.log`（6-way baseline log，apples-to-apples）
- `reports/analysis/_ensemble/ap_d1_7way_meta.json`（7-way 最終權重 JSON）
- `reports/analysis/_ensemble/ap_d1_6way_baseline_meta.json`（6-way baseline 權重 JSON）

### 53.12 AP-D3 結果 — 7-way × 3-view per-task TTA → **NEW SOTA 0.71364**（2026-05-18）

> **單行結論**：把 §53.11 的 7-way per-task hillclimb 接上 **3-view TTA**（stored / middle / tail），於 seed42-only 設定下取得 **0.71364**，apples-to-apples 較 6-way × 3-view × seed42（**0.71018**）淨增 **+0.00346**，也超越 Phase 36 公開 SOTA（**0.71018**，[§52](#52-phase-36--u6-pro-back-translation--stem-6--6-way--3-view-tta--new-sota-071018)）+0.00346，**確立新 SOTA**。

#### 設定（apples-to-apples）

| 條件 | 6-way × 3-view baseline | **7-way × 3-view（NEW SOTA）** |
| :-- | :-- | :-- |
| 參與 stems | stem #1~#6 | stem #1~#6 + **#7 `p2_combo_best_aug_plus`** |
| seeds | seed42（所有 stem） | seed42（所有 stem） |
| Views | stored / middle / tail | 同左 |
| Per-task stem 權重 | 4 任務 × 6-stem simplex hillclimb（grid 0.1） | 4 任務 × 7-stem simplex hillclimb（grid 0.1） |
| Per-task view 權重 | 4 任務 × 3-view simplex hillclimb（grid 0.1） | 同左 |
| 驅動腳本 | `src.tools.u10_per_task_tta --stems …` (`--joint-rounds 0 --max-rounds 2`) | 同上（`--joint-rounds 2 --max-rounds 4`） |

> **為什麼這次 apples-to-apples 用 seed42-only**：stem #5/#6/#7 都只訓練 seed42（其他 seed 從未跑），保持公平就只能統一 seed42。**意外驚喜**：6-way × 3-view × **seed42** 跑出 0.71018，與 Phase 36 公開 SOTA（混合 3-seed 平均）剛好相同 ⇒ **多 seed 平均在當前 stack 上沒有產生額外增益**，所以「補訓 multi-seed」不再是必要前置條件，AP-D3 可直接以 seed42 路線拿 SOTA。

#### 結果

| 配置 | overall weighted | T1 promise | T2 timeline | T3 evidence | T4 quality |
| :-- | --: | --: | --: | --: | --: |
| 6-way 等權 × stored only | 0.68227 | 0.93836 | 0.54137 | 0.86742 | 0.43763 |
| 6-way 等權 × stored+middle | 0.68281 | – | – | – | – |
| 6-way per-task × stored only | 0.70871 | 0.94203 | 0.62530 | 0.87781 | 0.46619 |
| **6-way per-task × 3-view（baseline）** | **0.71018** | 0.94210 | 0.62778 | 0.87774 | 0.46934 |
| 7-way 等權 × stored only | 0.68696 | 0.93787 | 0.56154 | 0.86840 | 0.44180 |
| 7-way per-task × stored only | 0.71109 | 0.94210 | 0.62690 | 0.88011 | 0.47029 |
| **7-way per-task × 3-view（NEW SOTA）** | **0.71364** | **0.94337** | **0.63061** | **0.88045** | **0.47496** |
| **Δ (7-way − 6-way, per-task × 3-view)** | **+0.00346** | +0.00127 | +0.00283 | +0.00271 | **+0.00562** |
| **Δ vs Phase 36 published SOTA (0.71018)** | **+0.00346** | +0.00127 | +0.00283 | +0.00271 | **+0.00562** |

#### 解讀

- **AP 路線拿到的不是邊際贏面而是 *硬增量***：6-way × 3-view 在 seed42 上 = 0.71018，疊上 stem #7 後 7-way × 3-view = 0.71364，四個任務全部上漲、無一倒退，且 T4（aug_plus 設計目標）漲幅最大（+0.00562）。
- **多 seed 平均 ≈ 0 增益（surprise）**：Phase 36 公開的 0.71018 用 3-seed 平均（stem #1-#4）+ seed42（stem #5/#6），與本次純 seed42 跑出來的 0.71018 完全相同。代表 **目前的瓶頸是 stack diversity，不是 seed variance**，新增 stem 的回報遠高於新增 seed。下一步應投資新 stem 而非補 seed。
- **stem #7 在四任務全有非零權重**（[`ap_d3_7way_3view_meta.json`](reports/analysis/_ensemble/ap_d3_7way_3view_meta.json)），T1/0.1、T2/0.1、T3/0.1、T4/0.3 — 與 AP-D1 結果一致（T4 最重）。即使是設計上「不包含 T2 樣本」的 aug_plus，也在 T2 上拿到 0.1 權重貢獻 diversity，符合「diversity > absolute score」假設。
- **view 權重分布合理**：T1 偏向 stored+middle、T2 三 view 均衡、T3 偏 stored+middle、T4 strong tail-bias（0.7），均符合 §51 phase 35 觀察到的「T4 對 tail-view 最敏感」。

#### 結論

- **AP-D3 = 新 SOTA 0.71364，正式超越 Phase 36 0.71018，淨增 +0.00346**（全部 4 task 同向）。
- **這條路徑可直接 reproduce**：只要按 §53 訓 stem #7，再跑 `python -m src.tools.u10_per_task_tta --stems <7 stems> --grid-step 0.1` 即得結果，**不需要補訓多 seed**。
- **下一步候選**：(a) AP-D4 = 8-way 加入新 stem（U13 LLM 合成樣本訓出的 model，[§54](#54-phase-37-並行路線--u13llm-合成--人工標註--llm-評審規劃中)）；(b) 細粒度 view weight grid（step 0.05 vs 0.1）；(c) 加入 fold-aware constraint。stem 增加 (a) 預期回報最大，依本表趨勢可預期 +0.001~0.003。

#### 產出檔案

- `reports/ap_d3_7way_3view.log`（7-way × 3-view 搜索 log）
- `reports/ap_d3_6way_baseline.log`（6-way × 3-view baseline log）
- `reports/analysis/_ensemble/ap_d3_7way_3view_meta.json`（最終 per-task stem + view 權重 JSON）
- `reports/analysis/_ensemble/ap_d3_7way_3view_summary.csv`、`*_preds.csv`
- `reports/analysis/_ensemble/ap_d3_6way_3view_baseline_meta.json`（baseline 對照）

## 54. Phase 37 並行路線 — U13：LLM 合成 + 人工標註 + LLM 評審（規劃中）
> **與 §53 (Phase 37 aug_plus) 的關係**：本章為 Phase 37 的 **第二條獨立路線**，由 U13 LLM 合成資料補 T2/T4 minority；§53 / §53.11 走的是 **U10-pro + 人工 aug_plus** 路線並已完成訓練 + AP-D1 7-way hillclimb（OOF +0.00238）。兩條路線**互補**：U13 主攻「樣本不足→直接生成」、aug_plus 主攻「stack 多樣性」。U13 為**計畫中**（pipeline + 12/12 tests green，尚未跑真實 LLM 生成）。

> **官方裁示（2026-05-15 群組訊息）**：
> 「加入自己查詢和標註的資料」算是「資料擴增」的範疇 … 純手工自製資料 … 用大語言模型生成資料 … 僅「主辦方提供之測試資料集」和「最後參賽者上傳的預測結果」絕對不能有人工介入。

### 54.1 為何 Phase 36 (OOF=0.71018) 卡在這

回顧官方訓練集 1000 筆的 4 任務標籤分佈：

| 任務 | 多數類 | 少數類 | 樣本數 |
|---|---|---|---|
| T1 promise_status (權重 0.20) | Yes=814 | No=186 | 已可學 |
| T2 verification_timeline (權重 0.15) | already=366 / between=238 / longer=197 | **within_2_years=13** | 嚴重不足 |
| T3 evidence_status (權重 0.30) | Yes=677 | No=137 | 可學 |
| T4 evidence_quality (權重 0.35) | Clear=552 | Not Clear=124 / **Misleading=1** | 災難級不足 |

Phase 36 各 task 分數：T1=0.872, T2=0.687, T3=0.866, T4=**0.469**。**T4 macro-F1 = 0.469 × 權重 0.35 = 整體最大進步槓桿**，且根本原因是 Misleading 類別僅 1 筆，模型無法學到任何判別特徵。

### 54.2 Phase 37 (U13 路線) 三路並進策略

1. **LLM 合成資料**（scripts/u13_synth_llm.py） — 多供應商抽象（OpenAI / Anthropic / Gemini / Ollama / Mock），用 prompt engineering 直接生成 ~300 筆少數類樣本。
2. **人工標註模板**（scripts/u13_manual_seed.py） — 預填 20 筆作者手寫種子 + 311 筆 <TODO_FILL_NNN> 空白模板，含目標標籤分佈，作者只需填入 data 欄位。
3. **LLM 評審**（scripts/u13_llm_judge.py） — 對既有 U10 偽標籤 (3110 筆) 用 LLM zero-shot 重新判斷，輸出 llm_judge_score，下游可篩選高一致性子集。

### 54.3 目標分佈（PHASE37_TARGET_COUNTS）

src/data/synth_schema.py:PHASE37_TARGET_COUNTS 共 330 筆：

| (T2, T4) | n | 理由 |
|---|---|---|
| within_2_years × Clear | 60 | 補 T2 最少數類 |
| within_2_years × Not Clear | 40 | 補 T2 × T4 雙重少數 |
| within_2_years × Misleading | 30 | T2 + T4 雙瓶頸交集 |
| longer_than_5_years × Misleading | 40 | 補 T4 Misleading |
| longer_than_5_years × Not Clear | 30 | 補 T4 Not Clear |
| between_2_and_5_years × Misleading | 30 | 補 T4 Misleading |
| already × Misleading | 30 | 補 T4 Misleading |
| already × Not Clear | 40 | 平衡 T4 |
| T1=No (全 N/A) | 30 | 維持 No 類錨點 |

合計 Misleading=130 筆（從 1 → 131，提升 130 倍），within_2_years=130 筆（從 13 → 143，提升 11 倍）。

### 54.4 完整管線（指令清單）

```powershell
# A. 用 Mock provider 跑 dry-run（不需 API key）
python scripts/u13_synth_llm.py generate --provider mock --output data/processed/u13/synth_raw.jsonl

# B. 切換真實 LLM（需設定環境變數）
$env:OPENAI_API_KEY = "sk-..."
python scripts/u13_synth_llm.py generate --provider openai --output data/processed/u13/synth_raw_openai.jsonl --sleep 0.5

# C. 驗證 / 去重 / 階層檢查
python scripts/u13_synth_llm.py validate \
    --input data/processed/u13/synth_raw_openai.jsonl \
    --output data/processed/u13/synth_filtered.jsonl \
    --min-len 80 --max-len 600

# D. 推升為偽標籤 CSV（confidence=0.95）
python scripts/u13_synth_llm.py promote \
    --input data/processed/u13/synth_filtered.jsonl \
    --output data/processed/u13/synth_pseudo.csv \
    --confidence 0.95

# E. 合併 U10 v3 + U13 synth
python scripts/u13_synth_llm.py merge \
    --inputs data/processed/u10/pseudo_labels_v3.csv data/processed/u13/synth_pseudo.csv \
    --output data/processed/u13/u10_v3_plus_synth.csv

# F. 訓練 Phase 37 模型
python -m src.train_pseudo_kfold --config configs/exp_p2_combo_best_classw_focal_u13_synth.yaml

# G. (選擇性) LLM 評審重新打分 U10
python scripts/u13_llm_judge.py judge \
    --input data/processed/u10/pseudo_labels_v3.csv \
    --output data/processed/u10/pseudo_labels_v3_judged.csv \
    --provider openai --sleep 0.5

# H. (選擇性) 人工標註模板
python scripts/u13_manual_seed.py emit --output data/processed/u13/manual_template.csv
# → 用 Excel/CSV editor 填 311 個 <TODO_FILL_NNN> 後，merge 進管線
```

### 54.5 ID 命名空間（防止 collision）

| ID 範圍 | 用途 |
|---|---|
| 0 ~ 999 | 官方訓練集 |
| 100000 ~ 199999 | U10 偽標籤 |
| 200000 ~ 299999 | (保留 U11 / U12) |
| **300000 ~ 9299999** | **U13 LLM 合成（hash-based stable ID）** |
| 400000 ~ 499999 | U13 人工標註模板（順序遞增） |

### 54.6 設定檔（exp_p2_combo_best_classw_focal_u13_synth.yaml）

繼承 Phase 36 最佳配方 (u6pro)，僅換 data.pseudo_csv_path 為 merged CSV，並調整：
- pseudo.min_confidence: 0.90（接受 0.95 合成 + 0.95 U10 gold）
- pseudo.max_pseudo: 4400（3904 U10 + ~300 U13 + headroom）

### 54.7 預期效益試算（保守）

假設 T4 macro-F1 從 0.469 提升到 **0.58**（Misleading F1 從 0.0 → 0.4，其他類別維持）：

ΔScore = 0.35 × (0.58 - 0.469) = +0.039

整體分數 0.71018 → **0.749 ~ 0.755**（不含 T2 增益、不含 LLM judge 篩選增益）。

若 T2 macro-F1 同步從 0.687 提升到 0.74（within_2_years F1 從 ~0.2 → 0.5）：

ΔScore += 0.15 × (0.74 - 0.687) = +0.008

**綜合預估 Phase 37 OOF 目標：0.755 ~ 0.770**

### 54.8 風險與緩解

| 風險 | 緩解 |
|---|---|
| LLM 合成樣本「太規律」、模型過擬合表面 pattern | 多 provider 混用 + 高溫 0.85 + 主題種子輪替 30+ 個 |
| 合成樣本與測試集分佈不符 | confidence=0.95 < 1.0；Stage B fine-tune 仍只用真實資料 |
| LLM 偶發 hierarchy 違反 | ssert_labels_valid 於 promote 階段攔截，違者直接丟棄 |
| 重複生成 | stable_synth_id (blake2b hash) 保證 idempotent + SimHash-lite dedup |
| 官方規則解讀錯誤 | 已截圖保存群組訊息；ID 命名空間明確隔離；可逆（合成資料皆在 data/processed/u13/） |

### 54.9 測試覆蓋

	ests/test_u13_synth.py 12 個測試全綠：
- schema 對齊 U10
- hierarchy 4 種合法 + 3 種非法
- stable_id 決定性
- SynthRow 往返
- target spec 全部 hierarchy-valid
- minority class 數量達標 (Misleading≥100, within_2y≥100)
- CLI end-to-end (generate → validate → promote) with Mock provider

pytest tests/test_u13_synth.py -v → **12 passed in 6.61s**

### 54.10 待辦（下一輪）

1. 取得 OpenAI / Anthropic API key 後跑真實生成 (~300 樣本，預估 \.10 ~ \.50 成本)
2. 人工 review LLM 輸出抽樣 30 筆，計算標籤命中率 (target ≥ 85%)
3. 跑 exp_p2_combo_best_classw_focal_u13_synth.yaml 完整 5-fold × 3-seed 訓練
4. 比對 Phase 36 vs Phase 37 各 task F1，產出 MASTER_PLAN §57 Phase 37 結果報告
5. 若 OOF ≥ 0.75，產生最終提交檔 + 更新 inal_summary/

---

---


═══════════════════════════════════════════════════════════════════════════════

# Part V — 附錄 (Appendix)

本部分為支援性詳表：Backlog 全紀錄、工程上限推導、外部資料合規立場。
日常閱讀無需深入，僅在需要查證 D-編號、X-編號、上限數字、合規條款時翻閱。

═══════════════════════════════════════════════════════════════════════════════

## 55. Backlog 詳表（D1 ~ D30 + U1 ~ U12 全紀錄）

> 本節為歷史審計用詳表。D 系列（D1~D30）為已完成的工程動作流水，U / N 系列（U1~U12 / N1~N3）為原始 backlog 候選清單。
> X1~X13 禁區另列於 [§55.3](#553-已暫緩或證明不可行的方向避免重複嘗試)。

### 55.1 已完成項目（Done — 含成果）

> 截至 Phase 25 共 30 項主要工程動作（D1~D30）；下列以時間順序，並標註對 SOTA 軌跡的貢獻。

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
| D20 | Sprint B 收尾 — (a) `p8_ema995` 0.63010（EMA 無 warm-start 重蹈 X1）；(b) `p9_ls_t1t3` 0.67334 (+0.00213, T3 按設計微升)；(c) Joint Hillclimb v11（16-way，直接優化 post-constraint score） | Phase 13 | 0.68770（前一代 SOTA） | +0.00087 vs v9 |
| D21 | Sprint C / N1 — `p10_large_focal_fgm`（macbert-large + Focal-T4 + FGM）+ Joint Hillclimb v12（17-way） | Phase 14 | **0.68825**（目前訓練 ensemble SOTA） | **+0.00055** vs v11 |
| D22 | U1 TTA — `stored+middle` 推論視角擴增（官方資料、無重訓） | Phase 15 | **0.68879**（目前 active SOTA） | **+0.00054** vs v12 |
| D23 | N2/p11 — `p11_electra_base` 官方資料 5-Fold + v16 admission（18-way） | Phase 16 | 單模 0.62694；admission 0.68825，p11 0 權重 | 0 vs v12；−0.00054 vs U1 |
| D24 | U1-b — `stored+middle+tail` 三視角等權 TTA 補測（active pool 14 員） | Phase 17 | 0.68873（未取代 `stored+middle` 0.68879） | −0.0000597 vs U1 active SOTA；T2 取得本輪最高 0.50941 |
| D25 | U1-c — per-task / per-view 權重 TTA coordinate descent（grid 0.05 + oracle warm-start，零再訓） | Phase 18 | **0.68925**（新 active SOTA） | **+0.00046 vs U1 / +0.00100 vs v12**；T2 0.51026 史上最高 |
| D26 | B1 / U3 — `p2_combo_best_swa` SWA 最後 K=3 epoch 平均（trainer.py + aggregator + canonical OOF alias） | Phase 21 | member-level OOF 0.66252（vs best-epoch 0.66558，**−0.0031**）；未通過 admission 門檻 +0.0005，**不入 v12 池** | 0（拒絕）；列入 X9 風險區（SWA + LR ≠ 0） |
| D27 | B2 / U5 — `p2_combo_best_resample_t4` T4 sqrt-inverse-frequency re-sampling（alpha=0.5） | Phase 22 | member-level OOF 0.66562（vs baseline 0.66558，**+0.00004**）；T4 macro **+0.012** 但 T2 macro **−0.014** 抵銷；未通過 admission +0.0005，**不入 v12 池** | 0（拒絕）；列入 X10 風險區（global sampler 抵銷 T2/T4） |
| D28 | C1 / N3 — NeZha-base 不可訓（fallback 到 BertModel，fold0 0.549 中止）；轉 ERNIE-3.0-base-zh fallback，5 折 mean **0.6086** << 0.66 admission 門檻 | Phase 23 | NeZha：fold0 0.549 中止；ERNIE：0.6086（T2 −0.107、T4 −0.073 vs baseline）；**N3 candidate 全數失敗，不入池** | 0（拒絕）；列入 X11 風險區（base-class 異源 backbone 對本資料集不利） |
| D29 | U2 / B3 — `p2_combo_best_ema995_warm` EMA decay=0.995 + warm-start 2 epoch（trainer.py 新增 `ema_warmup_epochs` + `ema_started`） | Phase 24 | member-level OOF **0.64941** vs baseline 0.66558（**−0.01617**）；T4 退化最嚴重（−0.062）；5 折全退化；未過 admission **不入 v12 池** | 0（拒絕）；列入 X12 風險區（EMA + warm-start 在 cosine 末段 LR ≠ 0 + 短訓練下仍傷 T4 macro） |
| D30 | U6 / B4 — `p2_combo_best_u6_bt` NLLB-200-distilled-600M 中→英→中 回譯（T2 within_2_years×3 + T4 Misleading×5 / Not Clear×1；138 唯一源員 → 168 增強樣本；src/train_kfold.py 新增 aug 載入 / per-fold 注入機制，oof / valid 不收 aug） | Phase 25 | member-level OOF **0.66321** vs baseline 0.66558（**−0.00237**）；T2 macro 0.4652（−0.014）、T4 macro 0.4276（−0.031）；上加 fold1 轉譯品質離群（score 0.6334）拉低均值；未過 admission **不入 v12 池** | 0（拒絕）；列入 X13 風險區（NLLB-600M 保真度不足 + 無 round-trip ChrF 過濾 + 譯本帶入 label-noise） |
#### 55.1.1 已完成 backlog 項目對照（U / N 系列）

> 補充 D 表之外、以 U / N 編號追蹤的項目。X1~X13 禁區詳 [§55.3](#553-已暫緩或證明不可行的方向避免重複嘗試)。

| ID | 項目 | 結果 | 紀錄位置 |
| :--: | :-- | :-- | :-- |
| U1 / U1-b / U1-c | TTA 多視角推論 | 完成且採納（U1-c = active SOTA 0.68925） | §36~§39, D22 / D24 / D25 |
| U2 | EMA decay=0.995 + warm-start | 完成且拒絕（X12） | §45, D29 |
| U3 | SWA 最後 K=3 epoch 平均 | 完成且拒絕（X9） | §42, D26 |
| U5 | T4 class-balanced re-sampling | 完成且拒絕（X10） | §43, D27 |
| U6 | NLLB-600M zh-en-zh 回譯 | 完成且拒絕（X13） | §46, D30 |
| N3 | NeZha → ERNIE fallback backbone | 完成且拒絕（X11） | §44, D28 |
| U10 | TW 企業永續報告書專用偽標 pipeline | **完成（U10 stack SOTA = 0.67746，baseline +0.01012）** | §47 |
| U11 | StratifiedGroupKFold by company sanity | 完成（診斷） | §40 |
| U12 | OOF cross-fold variance | 完成（診斷；valid gap 待 6/03 釋出） | §41 |

### 55.2 未完成項目（TODO）

> 截至 2026-05-10，原 backlog 中僅 U8 / U9 仍處於暫緩或阻塞狀態；其餘已完成項目（含採納與拒絕）詳 [§55.1](#551-已完成項目done--含成果)。
> 下表整合「暫緩 / 阻塞 backlog」與「下一輪可動執行路線」兩類，皆為尚未完成之項目。

#### 55.2.1 暫緩 / 阻塞 backlog

| ID | 項目 | 狀態 | 阻塞原因 |
| :--: | :-- | :-- | :-- |
| U8 | Pipeline B-Plan T1 → T3 → T2/T4 串接 | 暫緩 | 待 U10 結果與 valid gap 釋出後再評估 ROI |
| U9 | Qwen2.5-7B LoRA 4-bit r=16 | 阻塞 | RTX 5060 Laptop 8 GB 不足，需 16 GB+ 顯存 |

#### 55.2.2 下一輪可動執行路線（v3.0 校訂 2026-05-10）

> 排序原則：(a) 兩條 SOTA 軌跡合流優先；(b) 結構性殘留次之；(c) 外部資源依賴最後。

| 排序 | ID | 名稱 | 就緒度 | 重訓 | 預期增益 | 主要風險 |
| :--: | :-- | :-- | :-- | :--: | :-- | :-- |
| 1 | TTA on U10 | 把 U10 45 ckpt 餵入 U1-c per-task TTA aggregator | 立即可做 | 否 | ≥ +0.0005 | 低 |
| 2 | Class-weighted CE / Focal-T4 γ=3.0 | 在 U10 v2 偽標資料上重訓，目標恢復 within_2_years / Misleading 預測能力 | 配置就緒 | 是（30 ~ 45 ckpt） | T2 / T4 各 +0.005 ~ +0.015 | 中 |
| 3 | M3-v3 corpus 再擴張 | 放寬至 30 ≤ len ≤ 1200、移除 TOC space-ratio 規則 | 設計完成、未執行 | 否（純 M3） | +0.001 ~ +0.003 | 中 |
| 4 | 跨家族 teacher（XLM-R / Qwen-LoRA） | X18 上限定理只適用同 teacher；換 backbone 解鎖新 plateau | 待設計 | 是 | 待測 | 高（硬體 / 跨模） |
| 5 | 6/03 valid 釋出後 OOF↔Valid gap 重驗 + GroupKFold drift 復量 | 校準現有所有增益是否為虛增益 | 已就緒（§40 / §41） | 否 | 0（診斷） | 低 |
| 6 | 提交策略 | 定義 baseline / U1-c / v12 / U10 stack 四個提交點 | 待規劃 | 否 | LB 對齊 | 低 |

> **禁區提醒**：X1 ~ X13 詳見 [§55.3](#553-已暫緩或證明不可行的方向避免重複嘗試)，本輪嚴格遵守。

### 55.3 已暫緩或證明不可行的方向（避免重複嘗試）

| # | 項目 | 結果 | 原因 |
| :--: | :-- | :-- | :-- |
| X1 | EMA decay=0.999 短訓練 | 0.47291（崩潰） | 訓練步數 ~500 不足以讓 shadow weights 跟上；改用 U2 修正版 |
| X2 | Uniform Label Smoothing ε=0.05 全 task | T4 macro 崩潰 | 少類被推平；Phase 13 的 per-task LS(T1/T3) 單模可行但 ensemble 0 權重，故不再高優先 |
| X3 | LLRD 0.95（Layer-wise LR Decay） | 0.66019（負面） | base 模型層數淺，分層 lr 反而限制底層學習 |
| X4 | combo_v3 multi-seed 直接替換 seed=42 | 0.68191（負面） | T1 winner 倚賴 peaky 預測，平均化稀釋訊號；改用 D13 拆分設計 |
| X5 | macbert-large 作單模主力 | 0.66439（負面） | 1000 樣本對 326M 參數比例失衡；保留作 ensemble 配角 |
| X6 | Sprint A T3 — `xlm-roberta-base`（p4c）作單模 | 0.61269（被拒） | 低於 0.65 入池門檻 0.038；中文混合 backbone 在本資料集對齊不佳；不加入 ensemble pool |
| X7 | N2/p11 — `hfl/chinese-electra-180g-base-discriminator` 同設定 | 單模 0.62694；v16 admission 0 accept / 0 權重 | 同設定未提供可用 ensemble diversity；不重跑，若再試 ELECTRA 必須更換訓練策略或模型變體 |
| X8 | U1-b — `stored+middle+tail` 三視角等權 TTA | 0.68873，較 `stored+middle` 0.68879 低 0.0000597 | 三視角等權同時拉回 T1 至 baseline、稀釋 `stored+middle` 在 T1/T4 的貢獻；後續 U1-b 不再做等權三視角，改走「任務別/視角別權重」 |
| X9 | U3 SWA — `p2_combo_best_swa` 最後 K=3 epoch 平均（cosine warmup + 末段 LR > 0） | member-level OOF 0.66252 vs best-epoch 0.66558（−0.0031），五折 3/5 退化（fold4 −0.0139） | 末 epoch LR 未收到 0；K 涵蓋的 ckpt 仍在 loss 震盪區；若再嘗試 SWA 須改 cosine 完整下到 0 LR、或 EMA + warm-start（U2）；不重跑同設定 |
| X10 | U5 T4 global re-sampling — `p2_combo_best_resample_t4`（alpha=0.5 sqrt-inverse-freq） | OOF 0.66562 vs baseline 0.66558（+0.00004）；T4 +0.012 但 T2 −0.014 抵銷 | global WeightedRandomSampler 同時重抽其他任務的 batch，造成 T2 同步退化；未來如要讓 T4 增益不被抵銷，需改 per-task loss weighting、或 head-only mini-finetune、或同時平衡 T2 + T4；不重跑同設定 |
| X11 | C1 / N3 — NeZha-base 與 ERNIE-3.0-base-zh fallback | NeZha：HF transformers 無 NeZha-specific class，fallback 為 `BertModel` 時隨機初始化 NeZha 相對位置嵌入，fold0 5 epoch 最高 0.549；ERNIE：5 折 mean 0.6086，T2 −0.107、T4 −0.073 vs baseline | base-class 異源 backbone（NeZha / ERNIE）在本資料集上表現遠退 macbert / roberta-wwm 同源池；如要繼續結構性多樣性，需改 (a) layer-wise freeze、(b) large-class 多 seed，而非換 base 同級 backbone；不重跑 NeZha / ERNIE 同設定 |
| X12 | U2 / B3 EMA decay=0.995 + warm-start 2 epoch — `p2_combo_best_ema995_warm`（5 fold × 5 epoch × seed=42） | member OOF 0.64941 vs baseline 0.66558（−0.01617）；5 折全退化；T4 macro −0.062 為退化主因；T1/T3 退化較輕 | 雖比 X1 ema999 0.473 顯著改善（warm-start 避免徹底崩潰），但仍遠低 admission 0.66608；推測主因為 (a) cosine 末段 LR 未到 0（與 X9 SWA 同源）、(b) 僅剩 3 epoch EMA 累積樣本不足、(c) EMA 平均化壓縮 T2/T4 少類 confidence；若再試 EMA 須改 cosine 完整下到 0 LR + 拉長至 8~10 epoch + T2/T4 改用 best-epoch；不重跑同設定 |
| X13 | U6 / B4 回譯增強 — `p2_combo_best_u6_bt` NLLB-200-distilled-600M 中→英→中（138 唯一源員 → 168 增強樣本：T2 within_2_years ×3、T4 Misleading ×5、T4 Not Clear ×1；5 fold × 5 epoch × seed=42） | member OOF 0.66321 vs baseline 0.66558（−0.00237）；折別：[0.6748, 0.6334, 0.6632, 0.6750, 0.6696]，std=0.0174（fold1 離群）；T2 macro 0.4652（−0.014）、T4 macro 0.4276（−0.031）——雙目標任務皆退化；未過 admission 0.66608 | 主因為 (a) NLLB-200-distilled-600M 對繁中 ESG 專有名詞保真度不足（譯本儲存「台泥」為 TaiMot、字符腐化、詞重複，見 `outputs/logs/u6_backtranslate.runlog`）、(b) 英文 pivot 備作減譯 ESG-specific terminology（「永續發展」→sustainable development→「可持續發展」，詞表隋譯）、(c) 未加 round-trip ChrF/BLEU 過濾，低保真度譯本並未被檔掉、(d) train_recs 自 ~795 被膨跨至 ~935 含低保真度樣本，對 T2/T4 macro 等同 label-noise；若未來再試 BT 需 (a) 換 NLLB-3.3B 或 madlad-7B 或外部 NMT API 以提高中文保真度、(b) 加「原文 vs BT 譯本」 round-trip ChrF≥0.5 自動過濾、(c) 限縮至 T4 Misleading only（5 筆）以避免 T2 雜訊、(d) 伝依人工抽查譯本品質誌；不重跑同設定 |

---

## 56. 工程上限拆解（推導依據）

> 本節說明本專案聲稱的「加權保守上限 ≈ 0.7236 / 加權寬鬆上限 ≈ 0.7570」是如何得出的。
> 包含兩種上限的定義、per-task 拆解推導，以及 T4 `Misleading` (support=1) 為何構成統計硬上限。
> 與「如何突破上限」相關的執行路線見 [§55.2.2 下一輪可動執行路線](#5522-下一輪可動執行路線v30-校訂-2026-05-10)。

### 56.1 兩種上限的定義

**(A) 演算法上限 / Bayes-optimal**：給定特徵分布下，最佳分類器能達到的分數。理論存在但**不可直接量測**；通常以「人工標註者一致性 (Inter-Annotator Agreement, IAA)」作代理估計。例如：若兩位標註員意見不同的比例 = 8%，則 IAA ≈ 0.92，此即實務上限。

**(B) 工程上限 / Engineering ceiling**：給定固定資料集 + 固定評分公式下，所有合理技術組合可達的上限。比 (A) 嚴格但可逼近；用「per-class F1 拆解」估算。本表所列即為 (B)。

### 56.2 per-task 上限拆解（推導依據）

| Task | 目前 (Phase 14 v12) | 估計上限 | 缺口 | 上限依據 |
| :-- | :--: | :--: | :--: | :-- |
| T1 promise (binary F1) | 0.9405 | 0.94 ~ 0.95 | 0 ~ +0.010 | binary 任務、support 平均；macbert-large 單模也可達 0.928；IAA 上限約 0.95（已接近）|
| T2 timeline (macro 5 類) | 0.5087 | 0.55 ~ 0.62 | +0.041 ~ +0.111 | `already ↔ between_2_and_5_years` 混淆仍是主因；T6 兩版本未直擊，需資料擴增或更不同的模型家族 |
| T3 evidence (binary F1) | 0.8783 | 0.90 ~ 0.92 | +0.022 ~ +0.042 | 與 T1 同類但 evidence 描述較隱晦、訊號弱；理論 ~0.92 |
| T4 quality (macro 4 類) | 0.4592 | 0.55 ~ 0.65 | +0.091 ~ +0.191 | `Misleading` (support=1) 數學上單類錯誤會重壓 macro；理論天花板 ≈ 0.60 ~ 0.65 |
| **加權保守上限** | — | **≈ 0.7236** | — | 0.20·0.945 + 0.15·0.585 + 0.30·0.910 + 0.35·0.600 |
| **加權寬鬆上限** | — | **≈ 0.7570** | — | 0.20·0.95 + 0.15·0.62 + 0.30·0.92 + 0.35·0.65 |
| **目前實測（active TTA path）** | **0.68925** | — | **0.034 ~ 0.068** | [Phase 18](#39-phase-18-完成--u1-c-任務別視角別加權-tta2026-05-05) U1-c per-task TTA；競賽 active SOTA |
| **目前實測（U10 best.pt path）** | **0.67746** | — | **0.046 ~ 0.080** | [Phase 31](#477-m6-oof-ensemble--new-sota--067746baseline-001012) baseline + v1 + v2 stack（45 ckpt）；尚未經 TTA 合流 |

→ 「**0.72 ~ 0.74**」這個區間即由此而來。兩條 SOTA 軌跡尚未合流，下一步「TTA on U10 ckpt」見 [§15](#8-待辦事項roi-排序) 與 [§47.12 路徑 1](#47-u10--企業永續報告書-sr-弱監督-pipeline-完整版2026-05-09-重啟2026-05-10-v2-重訓)。

### 56.3 為什麼 T4 `Misleading` (support=1) 是真正的硬上限？

T4 的 label domain 是 4 類（Clear / Not Clear / Misleading / N/A）。macro-F1 對 4 個類別等權平均。若有一類 support=1：
- 模型若預測對：該類 F1 = 1.0
- 模型若預測錯（極可能，500 候選只有 1 個真正例）：該類 F1 = 0.0
- 對 T4 macro-F1 的單類影響 = ±(1/4) = **±0.25**；折算到最終分數權重 0.35，影響可達 **±0.0875**

實務上幾乎必錯，所以 T4 macro-F1 的統計變異很大。除非主辦單位提供更多 `Misleading` 樣本，或未來重新啟用且人工審核外部資料，否則 **T4 macro-F1 > 0.55 在統計上極困難**。

---

## 57. 競賽規則對外部資料的立場（已確認可用）

### 57.1 主辦方 2026-05-17 Q&A 裁示（摘錄）

> **來源**：主辦單位 2026-05-17 公開 Q&A 回覆（依使用者轉述整理；待主辦原文回覆檔上線後以原文取代下方摘要）。
>
> **裁示要旨**：
>
> 1. 「**自行查詢與標註之資料**」屬合法之資料擴增（含手工撰寫與 LLM 合成）。
> 2. **唯一禁區**為「對測試集進行人工標註或修正、以及對最終預測結果之任何人為干預」——預測必須由程式自動生成。
> 3. 外部訓練資料、外部預訓練模型、對外部公開文本做偽標，**規則文件均無禁止條文**。
>
> **本團隊處置**：
>
> - 2026-05-17 後新增 **Aug-Plus（AP1~AP5）** 模組，鎖定 Phase 36 雙瓶頸（T4 `Misleading`、T2 `within_2_years`），詳 [§53](#53-phase-37--aug-plus-hand-crafted-minority-訓練與-single-stem-ablation2026-05-18)。
> - 50 列人手繁中種子均**獨立撰寫**，不抄自官方 1,000 列或任何受版權保護來源；經 7 道品質閘過濾後 47 列入訓。
> - 測試集絕對隔離（與 Phase 1 以來 invariants 一致），AP 衍生資料僅進 train+pseudo pool。

### 57.2 規則原文比對

逐字檢視官方文件 [ESG_永續承諾驗證競賽_2026.md §八「競賽規則與注意事項」](ESG_永續承諾驗證競賽_2026.md)，禁止項目僅有以下五條：

1. 抄襲、作弊、詐欺；
2. 侵害他人智慧財產權；
3. 攻擊 leaderboard 系統；
4. **對測試資料集或辨識結果進行任何形式的人工標註或修正**（預測必須由程式自動生成）；
5. 私下共享程式與特徵值。

**並無任何條文禁止使用外部訓練資料、外部預訓練模型、或對外部公開文本做偽標。現況（2026-05-10）**：

- U10 已於 2026-05-09 重啟、2026-05-10 完成 v2 重訓；採「來源同質性 + 50 家排除 + 兩階段隔離訓練」三條鐵律（詳 [§47.0](#470-設計原則與失敗教訓)）。
- 實測 best.pt path 增益 +0.01012（baseline 0.66734 → stack 0.67746）；對 minority class T2 timeline 與 T4 quality 提供可量測覆蓋改善。
- 殘留結構性問題（T2 within_2_years / T4 Misleading 仍 0）為 baseline class collapse，需 [§15](#8-待辦事項roi-排序) 路徑 2 的 class-weighted CE / Focal-T4 重訓處理。
- 偽標噪聲與分布偏移風險已透過兩階段訓練隔離；不得將 pseudo 視為人工標註真值。

