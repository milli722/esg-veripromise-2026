# VeriPromiseESG 2026 模型分析與訓練完整指南（零基礎可直接上手版）

> 文件定位：教學手冊 + 競賽實戰計畫書  
> 適用對象：第一次參賽到進階追分隊伍  
> 核心目標：可重現、可解釋、可持續迭代

---

## 目錄
- [0. 這份文件要幫你做到什麼](#0-這份文件要幫你做到什麼)
- [1. 先把任務看懂（完全白話）](#1-先把任務看懂完全白話)
- [2. 第一次訓練前的必備觀念與環境準備](#2-第一次訓練前的必備觀念與環境準備)
- [3. 資料探索與前處理策略（EDA and Preprocessing）](#3-資料探索與前處理策略eda-and-preprocessing)
- [4. 基準模型分析（Baseline Analysis）](#4-基準模型分析baseline-analysis)
- [5. 進階模型選擇與訓練規劃](#5-進階模型選擇與訓練規劃)
- [6. 驗證機制與評估指標](#6-驗證機制與評估指標)
- [7. 從零開始的行動清單（可直接照抄到 Notebook）](#7-從零開始的行動清單可直接照抄到-notebook)
- [8. 常見錯誤與排查手冊](#8-常見錯誤與排查手冊)
- [9. 計畫書完成標準（Definition of Done）](#9-計畫書完成標準definition-of-done)
- [10. 外部文獻與方法依據](#10-外部文獻與方法依據)

---

## 0. 這份文件要幫你做到什麼

如果你是第一次接觸模型訓練，這份文件的目標不是讓你背名詞，而是讓你完成三件具體事情：

1. 你能把官方 baseline 跑起來，拿到第一個可重現分數。
2. 你能清楚說明每個步驟在做什麼、為什麼要做、不做會有什麼後果。
3. 你能在有限時間內，用有紀律的方法穩定追分，而不是亂試參數。

> 本文件是「教學 + 實戰」結合版：先白話理解，再給你可執行的實作骨架。

### 閱讀方式建議

1. 第一次閱讀：先看第 1 章到第 2 章，建立完整任務觀。
2. 開始實作：直接照第 7 章跑出最小可行版本。
3. 要追分時：回到第 5 章與第 6 章做迭代與驗證。

---

## 1. 先把任務看懂（完全白話）

### 1.1 你在解決什麼問題

給你一段企業 ESG 文字，模型要同時回答 4 個問題：

1. 這段話有沒有承諾（promise_status）？
2. 這個承諾大概什麼時候可以驗證（verification_timeline）？
3. 這段話有沒有給具體證據（evidence_status）？
4. 證據清不清楚（evidence_quality）？

### 1.2 為什麼這不是一般文字分類

這個任務難在「邏輯關係」：

1. 如果沒有承諾，後面很多欄位通常要是 N/A。
2. 如果沒有證據，證據品質應該也要是 N/A。
3. 評分是四個任務加權，不是單一準確率。

也就是說，你不只要分對，還要分得「合理」。

### 1.3 評分是怎麼算的

四個任務權重如下：

- promise_status：0.20
- verification_timeline：0.15
- evidence_status：0.30
- evidence_quality：0.35

公式：

$$
Score = 0.2F1_{promise} + 0.15F1_{timeline} + 0.3F1_{evidence} + 0.35F1_{quality}
$$

> 重點：evidence_quality 權重最高，通常是勝負關鍵。

---

## 2. 第一次訓練前的必備觀念與環境準備

### 2.1 你只需要先懂這 8 個詞

| 名詞 | 白話解釋 | 你要怎麼用 |
|---|---|---|
| Baseline | 第一個可重現起點 | 用它當比較基準 |
| Epoch | 全部訓練資料跑完一次 | 看 loss 是否下降 |
| Batch Size | 一次餵給模型幾筆資料 | 影響速度與顯存 |
| Learning Rate | 每次更新參數步伐大小 | 太大會震盪，太小學很慢 |
| Macro-F1 | 各類別同權重平均 | 不平衡資料最重要 |
| CV（交叉驗證） | 輪流分訓練與驗證多次 | 降低分數運氣成分 |
| Overfitting | 記住訓練資料但不會泛化 | 要用早停與驗證控管 |
| OOF | 每筆資料由「沒看過它」的模型預測 | 真實泛化估計核心材料 |

### 2.2 環境最小可用配置

1. Python 3.10 以上。
2. 建議有 GPU；沒有 GPU 也可跑，但會慢很多。
3. 套件：torch、transformers、pandas、scikit-learn、numpy、matplotlib、seaborn。

建議先執行一次安裝（Notebook 第一格）：

```bash
pip install -U torch transformers pandas scikit-learn numpy matplotlib seaborn
```

如果你使用的是全新環境，建議加上：

```bash
pip install -U jupyter ipykernel
```

### 2.3 你應該先建立的專案資料夾

```text
project/
  data/
  notebooks/
  src/
  outputs/
    logs/
    checkpoints/
    reports/
```

### 2.4 第一次訓練前的 5 個檢查

1. 資料檔可正常讀取。
2. 目標欄位名稱和標籤拼字完全一致。
3. 隨機種子已固定。
4. 評分函數已先用假資料測過可執行。
5. 輸出路徑已建立，避免訓練到一半存檔失敗。

補充（Windows 路徑常見問題）：

1. 檔名若含 `[` `]`（例如基準 Notebook），在某些指令會被當 wildcard。
2. PowerShell 檢查路徑時請優先用 `-LiteralPath`。

> 建議：先完成這 5 項檢查再啟動訓練，可顯著降低中途中斷與重跑成本。

---

## 3. 資料探索與前處理策略（EDA and Preprocessing）

### 3.0 這一章在做什麼

你要先確認資料是可靠的，再進入訓練。

不做 EDA 的後果：

1. 標籤有問題卻沒發現，模型再強也學錯。
2. 類別嚴重不平衡但未處理，Macro-F1 會卡住。
3. 你不知道模型錯在哪，無法有效改進。

### 3.1 CSV 與 JSON 雙格式整合

已知資料狀況：

- CSV 行數：1000
- JSON id 計數：1000

建議作法：

1. 主要以 CSV 訓練（欄位整理方便）。
2. 用 JSON 做一致性檢查（避免編碼與跳脫字元問題）。
3. 以 id 為主鍵比對兩份資料。

```python
import pandas as pd
import json

df_csv = pd.read_csv("vpesg4k_train_1000 V1.csv")
with open("vpesg4k_train_1000 V1.json", "r", encoding="utf-8") as f:
    df_json = pd.DataFrame(json.load(f))

assert len(df_csv) == len(df_json)
assert set(df_csv["id"]) == set(df_json["id"])
```

### 3.2 標註規則一致性檢查（一定要做）

你必查三條規則：

1. `promise_status=No` 時，其他三欄應合理落在 N/A 邏輯。
2. `evidence_status=No` 時，`evidence_quality` 應為 N/A。
3. `promise_status=Yes` 時，`promise_string` 不應為空。

這一步的目的：確保你學到的是「可用規則」，不是雜訊。

### 3.3 類別分佈（你要知道哪裡最難）

#### promise_status

| 類別 | 筆數 | 比例 |
|---|---:|---:|
| Yes | 814 | 81.4% |
| No | 186 | 18.6% |

#### verification_timeline

| 類別 | 筆數 | 比例 |
|---|---:|---:|
| already | 366 | 36.6% |
| between_2_and_5_years | 238 | 23.8% |
| longer_than_5_years | 197 | 19.7% |
| N/A | 186 | 18.6% |
| within_2_years | 13 | 1.3% |

#### evidence_status

| 類別 | 筆數 | 比例 |
|---|---:|---:|
| Yes | 677 | 67.7% |
| N/A | 186 | 18.6% |
| No | 137 | 13.7% |

#### evidence_quality

| 類別 | 筆數 | 比例 |
|---|---:|---:|
| Clear | 552 | 55.2% |
| N/A | 323 | 32.3% |
| Not Clear | 124 | 12.4% |
| Misleading | 1 | 0.1% |

> 解讀：`Misleading` 極端少數，這會直接傷害 Macro-F1，需要特別策略。

### 3.4 文本清洗與特徵工程（先簡單再進階）

#### 最小必要清洗

1. 去除異常空白與多餘換行。
2. 壓縮重複標點。
3. 不要過度清理，避免語意流失。

#### 可先做的輕量特徵

1. `text_len`
2. `digit_ratio`
3. `year_count`
4. 關鍵詞旗標（如：將、預計、持續、達成）

```python
import re

def basic_clean(text):
    text = str(text).replace("\u3000", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([。；，、！？])\1+", r"\1", text)
    return text
```

### 3.5 統計檢定（幫你判斷特徵是否有價值）

你可用 ANOVA 或 Kruskal-Wallis 檢定：

1. ANOVA：適合分佈比較接近常態。
2. Kruskal：更穩健，對偏態資料較友善。

用途：檢查 `text_len` 等特徵是否對 `timeline` 或 `quality` 有區辨力。

---

## 4. 基準模型分析（Baseline Analysis）

### 4.0 這一章在做什麼

這一章要讓你建立「可信起跑線」。

如果你跳過：

1. 你會不知道模型升分是能力還是運氣。
2. 你會無法對團隊解釋為什麼這版要提交。

### 4.1 官方 baseline 流程（你真正要理解的版本）

官方 baseline 核心：

1. 中文 RoBERTa 當共享編碼器。
2. 四個任務各自接一個分類頭。
3. 四個任務 loss 相加一起學。
4. 用 Macro-F1 計算各任務，再加權成總分。

公式：

$$
\mathcal{L}_{total}=\mathcal{L}_{promise}+\mathcal{L}_{timeline}+\mathcal{L}_{evidence}+\mathcal{L}_{quality}
$$

### 4.2 baseline 優點（為什麼它值得保留）

1. 可重現：第一次上手最重要。
2. 可擴充：加 class weight、規則後處理都容易。
3. 可解釋：任務分開看，方便定位問題。

### 4.3 baseline 主要失分點

| 問題 | 為什麼失分 |
|---|---|
| 單次切分 | 1000 筆資料對切分非常敏感 |
| 類別不平衡 | 少數類別幾乎學不到 |
| 沒加任務邏輯 | 容易出現規則衝突預測 |
| 長文本截斷 | 關鍵證據可能被切掉 |

### 4.4 baseline++ 最小升級順序

請照這個順序做，不要跳著改：

1. 單次切分改 5-fold CV。
2. 加 task loss 權重（對齊比賽權重）。
3. 加規則後處理（No 對應 N/A）。
4. 再考慮 focal loss 與集成。

### 4.5 消融實驗（Ablation）規劃

最少做 6 組：

1. E0：官方 baseline
2. E1：E0 + 5-fold CV
3. E2：E1 + task loss weight
4. E3：E2 + class imbalance 技巧
5. E4：E3 + 規則後處理
6. E5：E4 + 多 seed ensemble

這樣你才能有說服力地回答：哪一個改動真的有效。

---

## 5. 進階模型選擇與訓練規劃

### 5.0 這一章在做什麼

幫你在有限資源下做正確選擇，而不是盲目追新模型。

### 5.1 模型路線圖（先易後難）

1. 傳統 ML（對照組）
2. Encoder Transformer（主力）
3. LoRA（進階追分）
4. RAG（有上下文資料才做）

### 5.2 模型比較矩陣

| 路線 | 推薦程度（新手） | 主要優勢 | 主要成本 |
|---|---|---|---|
| TF-IDF + LR/SVM | 高 | 快、可解釋 | 上限較低 |
| RoBERTa/DeBERTa | 很高 | 分數與穩定性最佳平衡 | 需 GPU |
| LLM + LoRA | 中 | 高潛力、參數效率高 | 設定較複雜 |
| RAG | 中~低 | 可解釋性強 | 檢索管線成本高 |

### 5.2.1 這些模型到底在做什麼（先用白話）

1. TF-IDF + LR/SVM：
    - 把文字拆成詞，計算每個詞的重要性分數（TF-IDF），再交給傳統分類器判斷標籤。
    - 可以想成「先把文章轉成一大串數字特徵，再做分類」。
2. RoBERTa/DeBERTa：
    - 模型會看詞與詞之間的上下文關係，學到句子語意後再分類。
    - 可以想成「不是只看關鍵字，而是看語境」。
3. LLM + LoRA：
    - 在大型語言模型上只訓練一小部分可調參數，讓模型適應你的任務。
    - 可以想成「給大模型裝一個任務專用的小插件」。
4. RAG：
    - 先去資料庫找相關段落，再把找到的內容一起給模型判斷。
    - 可以想成「先查資料再作答」，不是只靠模型記憶。

### 5.2.2 各模型的運行流程（訓練時）

#### A. TF-IDF + LR/SVM（傳統 ML）

1. 輸入文字。
2. tokenizer 拆詞（字詞切分）。
3. 轉成 TF-IDF 向量。
4. 用 LR 或 SVM 學習「向量 -> 標籤」對應。
5. 輸出每個欄位的分類模型。

這條路線快、穩定，適合當第一個對照組。

#### B. RoBERTa/DeBERTa（Transformer 主力）

1. 輸入文字。
2. tokenizer 轉成 token ids。
3. encoder 計算上下文語意向量。
4. 接分類頭（單任務或多任務）輸出 logits。
5. 與真實標籤計算 loss，反向傳播更新參數。

這條路線通常是競賽主力，因為語意理解能力明顯較好。

#### C. LLM + LoRA（參數高效率微調）

1. 載入 base LLM（主幹凍結）。
2. 在 attention 模組掛 LoRA adapter。
3. 僅更新 adapter 參數。
4. 以分類格式或 JSON 格式學習輸出。
5. 訓練後可保留 adapter，或合併回 base model。

這條路線適合有進階追分需求但 GPU 資源有限的場景。

#### D. RAG（檢索增強）

1. 把可用文件切塊並向量化建索引。
2. 對每筆樣本先檢索 top-k 相關段落。
3. 組合 query + contexts 給分類器或 LLM。
4. 輸出預測標籤。

這條路線重點不是模型變大，而是讓輸入資訊更完整。

### 5.2.3 各模型的運行流程（推論時）

1. TF-IDF + LR/SVM：文字 -> TF-IDF 向量 -> 類別機率 -> 標籤。
2. RoBERTa/DeBERTa：文字 -> token ids -> encoder -> logits -> 標籤。
3. LoRA 模型：文字 -> base model + adapter -> 輸出 JSON/標籤 -> 規則後處理。
4. RAG：文字 -> 檢索 context -> 模型判斷 -> 規則後處理。

共同最後一步都建議加上「規則後處理」，確保邏輯一致。

### 5.2.4 新手最容易問的兩個問題

1. 為什麼不是一開始就用最大模型？
    - 因為你先需要可重現與可診斷的流程。模型越大，錯誤定位越難、成本越高。
2. 為什麼要保留傳統 ML？
    - 因為它是穩定對照組。若深度模型沒有明顯超過它，通常表示流程有問題。

### 5.3 為什麼我們需要 LoRA？它到底怎麼運作的？（官方原理與文獻深度解析）

為了讓非資訊背景的團員也能徹底理解，又能讓實作人員清楚底層原理，我們根據微軟研究團隊（Hu et al., 2021）的論獻以及 Hugging Face PEFT 官方文檔進行以下深入又白話的剖析。

#### A. 白話概念：大模型的「外掛記憶卡」
如果我們要把一個幾十 GB 的大型語言模型（比如 LLaMA-3 8B）重新訓練（Full Fine-Tuning），會需要極端昂貴的伺服器與顯卡。
**LoRA (Low-Rank Adaptation)** 的發明完全改變了這件事：
- 它**不**修改原本龐大模型內部的記憶（將原本的模型凍結 Frozen）。
- 它在模型旁邊加上了幾個非常小的「額外學習矩陣」，就像是插上一張「專門學習 ESG 賽題規定的外掛記憶卡」。
- 訓練時，我們只更新這張小小記憶卡的數值，因此顯存需求（VRAM）和運算量直接巨幅下降，甚至能減少 10,000 倍的訓練參數！

#### B. 學術原理深度解析（從公式看為什麼它強大）
根據 Hugging Face `peft` 官方套件文檔介紹，LoRA 巧妙利用了**低秩分解（Low-Rank Decomposition）**的數學特性：
1. **矩陣拆分：** 模型本來要學習的權重更新量我們假設為 $\Delta W$。LoRA 把這個大矩陣拆成兩個極小的矩陣（$A$ 與 $B$）相乘。所以新的知識就是 $A \times B$。
2. **初始化設計（Identity Transform）：** 矩陣 $A$ 在一開始會給予微小的隨機亂數（Kaiming-uniform），但矩陣 $B$ 會**全部被填為 0**。這導致訓練的第 0 步時，$A \times B = 0$。也就是說，套上 LoRA 記憶卡的第一秒，整個模型的表現跟原模型「一模一樣」，沒有任何流失。接著在訓練中透過反向傳播，才慢慢把 ESG 知識寫入 $A$ 與 $B$ 之中。
3. **無痛推理零延遲 (Zero Inference Latency)：** 訓練完成後，我們透過官方提供的 `merge_and_unload()` 函數，可以直接把這張記憶卡裡面辛苦學到的數值（$A \times B$）**直接塞回（加回）原本的大腦矩陣 $W$ 裡面**。所以部署推論時，其實只是在跑一個與原模型同樣大小的模型，毫不拖泥帶水，速度極快！

#### C. `LoraConfig` 核心參數設定實戰指南
在實作這場競賽時，我們必須調整這幾個核心決定成敗的參數：
1. **`r` (Rank 秩)：** 
   - **意義：** 決定外掛記憶卡的「容量」大小。
   - **實戰：** 如果設太小，模型學不到複雜的 ESG 邏輯；設太大，會造成過擬合且浪費資源。這場比賽要同時判斷承諾、時間、證據品質，建議從 `r = 16` 或 `r = 32` 起步。
2. **`lora_alpha` (縮放因子)：** 
   - **意義：** 決定了我們這張記憶卡對大腦的「說話音量」（影響力度）。
   - **實戰：** 傳統做法是設為 `r` 的 2 倍。但我們強烈建議開啟文獻最新的 **Rank-Stabilized LoRA 參數 (`use_rslora=True`)**！這能大大穩定訓練過程，讓我們不怕擴大 rank 導致崩潰。
3. **`target_modules` (作用目標層)：** 
   - **意義：** 這張補丁要貼在大腦的哪些神經元上？
   - **實戰：** 語言模型最核心的機制是「注意力機制 (Attention)」。官方文獻指出，與其只貼在一小部分，不如盡可能將所有 Attention 矩陣（`q_proj`, `v_proj`, `k_proj`, `o_proj`）與 MLP 模塊都貼上補丁，效果常常會更好超越預期。
4. **`lora_dropout` (防過擬合護城河)：**
   - **意義：** 訓練時故意隨機關掉一些神經元連接。
   - **實戰：** 設為 `0.05` 到 `0.1` 即可，這能強迫模型不要死背這 1000 筆資料，而是真正學會判斷邏輯。

這就是我們把大模型縮小到能在普通顯示卡中訓練，而且效能還能匹敵全參數微調（Full Fine-tuning）的最強武器！

### 5.4 RAG 什麼時候值得做

你有以下條件才建議做：

1. 你手上有完整可檢索上下文（不是只有單段文字）。
2. 你想提升 evidence 判讀可解釋性。
3. 比賽規則允許該資料來源。

沒有以上條件，先不要做 RAG，先把分類主線做好。

RAG 實際運行拆解：

1. 離線階段：
    - 文件切塊 -> 產生向量 -> 建索引。
2. 線上推論階段：
    - 文字轉查詢向量 -> 檢索 top-k -> 拼接 context -> 模型輸出。

RAG 的成敗關鍵不只在模型，而在「檢索到的內容是否真的相關」。

### 5.5 進階追分配方（建議順序）

1. Class-balanced loss
2. Dynamic sampling
3. Focal loss（優先用於 quality 任務）
4. Pseudo labeling（高信心才收）
5. Ensemble（最終提交前）

每個配方在訓練流程中的位置：

1. Class-balanced loss：替換 loss 計算方式。
2. Dynamic sampling：替換 DataLoader 抽樣策略。
3. Focal loss：針對難例提高學習權重。
4. Pseudo labeling：在正式訓練前先擴增資料。
5. Ensemble：在多模型訓練完成後，做結果融合。

---

## 6. 驗證機制與評估指標

### 6.0 這一章在做什麼

這一章是你的「防翻車系統」。

很多隊伍不是模型不夠好，而是驗證設計不嚴謹。

### 6.1 資料切分策略

建議雙軌檢查：

1. 主流程：StratifiedKFold（貼近 baseline 與比賽提交節奏）。
2. 風險檢查：GroupKFold（company 分組）看是否有語氣洩漏。

### 6.2 交叉驗證設定建議

1. `n_splits=5`
2. `shuffle=True, random_state=42`
3. 分層鍵：`promise_status + evidence_status`
4. 每 fold 內保留 early stopping 機制
5. 保存 OOF 預測

### 6.3 必看指標與補充指標

主指標：

1. 每任務 Macro-F1
2. 最終加權分數

補充指標：

1. 每類別 Recall / Precision
2. confusion matrix
3. 規則一致率

規則一致率：

$$
Consistency = 1 - \frac{N_{violations}}{N_{samples}}
$$

### 6.4 規則後處理（務必加入）

```python
def apply_constraints(pred):
    p = pred.copy()
    if p["promise_status"] == "No":
        p["verification_timeline"] = "N/A"
        p["evidence_status"] = "N/A"
        p["evidence_quality"] = "N/A"
    elif p["evidence_status"] == "No":
        p["evidence_quality"] = "N/A"
    return p
```

這一步不是作弊，而是把任務本身邏輯放回預測流程。

### 6.5 提交策略（避免 public LB 迷思）

1. 每次只改一個主變數。
2. 每次提交都要有對應實驗紀錄。
3. 以 CV 穩定度作主決策，public LB 只做參考。

---

## 7. 從零開始的行動清單（可直接照抄到 Notebook）

### 7.1 一天內最小可行版本

1. 載入資料並做一致性檢查。
2. 跑官方 baseline 一次。
3. 改成 5-fold。
4. 加規則後處理。
5. 輸出第一版實驗報告。

### 7.2 7 天上手節奏

| 天數 | 目標 | 成果物 |
|---|---|---|
| Day 1 | 跑通 baseline | `baseline_run_report.md` |
| Day 2 | 建 5-fold + OOF | `fold_assignments.csv`, `oof.parquet` |
| Day 3 | 加權 loss + 後處理 | `cv_scores_v2.csv` |
| Day 4 | 錯誤分析 | `error_cases_top80.xlsx` |
| Day 5 | 進階配方 A/B | `ablation_v3.csv` |
| Day 6 | 集成與提交模擬 | `submission_plan.md` |
| Day 7 | 整理最終報告 | `final_model_card.md` |

### 7.3 你可以直接用的程式骨架

```python
# 0) 套件檢查（可選）
# !pip install -U torch transformers pandas scikit-learn numpy matplotlib seaborn

# 1) 固定隨機種子
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# 2) 讀資料
import pandas as pd
from pathlib import Path

ROOT = Path(".")
CSV_PATH = ROOT / "vpesg4k_train_1000 V1.csv"
JSON_PATH = ROOT / "vpesg4k_train_1000 V1.json"
BASELINE_NB = ROOT / "[External]_VeriPromiseESG_2026_ESG_Promise_Verification_Competition_Baseline_Code_ZH.ipynb"

assert CSV_PATH.exists(), f"缺少檔案: {CSV_PATH}"
assert JSON_PATH.exists(), f"缺少檔案: {JSON_PATH}"
assert BASELINE_NB.exists(), f"缺少檔案: {BASELINE_NB}"

df = pd.read_csv(CSV_PATH)

# 3) 分層 key
df["stratify_key"] = df["promise_status"] + "_" + df["evidence_status"]

# 4) 建立 fold
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df["fold"] = -1
for f, (_, va_idx) in enumerate(skf.split(df, df["stratify_key"])):
    df.loc[va_idx, "fold"] = f

# 5) 後續在每個 fold 內做訓練與驗證
```

---

## 8. 常見錯誤與排查手冊

### 8.1 常見錯誤

1. 只看單次分數，不看 CV 平均與方差。
2. 一次改太多東西，無法知道有效因子。
3. 忘記固定 seed，結果不可重現。
4. 沒看混淆矩陣，導致少數類別持續崩壞。
5. 沒做規則後處理，產生不合理預測。

### 8.2 你應該先檢查哪裡

1. 分數突然暴跌：先看資料切分與標籤映射是否一致。
2. 訓練 loss 降但驗證不升：大機率過擬合，啟用早停或增強正規化。
3. 某類別 F1 永遠 0：先看該類樣本量，再上 class weight 或重採樣。

---

## 9. 計畫書完成標準（Definition of Done）

當你符合以下條件，代表計畫書已經可實戰：

1. 有可重現 baseline（固定 seed、可重跑同趨勢）。
2. 有至少 3 組可比較實驗（含改動點與結果）。
3. 每版都有四任務 Macro-F1、加權分數、規則一致率。
4. 有錯誤案例報告（至少 50 筆高信心錯誤）。
5. 能對外說明「下一版要改什麼、為什麼改、成功標準是什麼」。

---

## 10. 外部文獻與方法依據

本計畫的設計依據以下公開資料做校準：

1. SemEval-2025 Task 6: Promise Verification
   - https://aclanthology.org/2025.semeval-1.321/
2. ML-Promise: A Multilingual Dataset for Corporate Promise Verification
   - https://aclanthology.org/2025.emnlp-main.1028/
3. Hugging Face PEFT LoRA 文件
   - https://huggingface.co/docs/peft/main/en/conceptual_guides/lora

> 方法原則：先建立可重現基線，再做可解釋且可驗證的追分迭代。