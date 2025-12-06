# Ensemble Heart Failure Prediction System
## Model Architecture
![MLP_final](https://hackmd.io/_uploads/ByaT_zGzZe.png)

## 1. Inductive Biases
1.  **Temporal Locality (The 6-Month Assumption):** We assume that the predictive signal for a specific admission outcome is concentrated in the immediate history. Events occurring prior to $t_{leave} - 6\text{ months}$ are treated as noise rather than signal.
2.  **Patient Independence (I.I.D. Assumption):** We assume distinct patients are Independent and Identically Distributed. There is no latent state sharing or information leakage between Patient A and Patient B.
3.  **Sequential Dependency:** Unlike bag-of-words approaches, the *order* of medical events within the observation window is critical. An examination followed by a surgery implies a different trajectory than a surgery followed by an examination.
4.  **Distributional Semantics:** We assume that medical codes (ICD, procedures) sharing similar contexts (appearing in similar patient histories) share semantic meaning, justifying the use of Skip-gram pre-training.



## 2. Data Pipeline & Logic (`preprocess.py`)
This module handles the transformation of raw medical records into sequence-ready tensors, strictly enforcing the biases defined above.

### 2.1 Data Splitting Strategy
To prevent data leakage we perform patient-level split:
* The dataset is partitioned into Train, Validation, and Test sets based on **Patient ID**, rather than individual admission records.
* A single patient's data (across multiple admissions) appears in only one of the three sets to prevent data leakage.

### 2.2 Observation Window Implementation
We enforce the **Temporal Locality** bias directly into feature generation:
* **Observation Window:** We define the relevant timeframe as the interval $[t_{leave} - 6\text{ months}, t_{leave}]$.
* **Bias Implementation:**
    1.  **Relevance:** Medical examinations or surgeries occurring outside this 6-month window are **excluded**, semantically meaning they are irrelevant to the current admission.
    2.  **Independence:** Each admission is treated as an independent event sequence bounded by this window.

### 2.3 Sequence Construction
* **Variable Lengths:** While sequence lengths vary between patients, the input tensor size for the LSTM is fixed via **padding**.
* **Padding Logic:** Sequences shorter than the maximum definition are padded with zero-vectors at the end. These padded regions are marked for exclusion during the gradient calculation (see `model.py`).

## 3. Model Architecture (`model.py`)
This file defines the LSTM architecture, handling variable-length sequences and dimension management.

### 3.1 Handling Variable Lengths: `pack_padded_sequence()`
To adhere to the requirement of **skipping training on padded parts**, we utilize PyTorch's packing mechanism:

* **Mechanism:** The `torch.nn.utils.rnn.pack_padded_sequence()` function collapses the padded tensor into a contiguous segment based on the actual length of each sequence.
* **Benefit:** The LSTM processes only valid time steps. This prevents the model from learning "padding patterns" and saves computational resources.

### 3.2 `process_packed_input`
* **process_packed_input:** This internal method handles the flow:
    1.  Accepts the padded input batch.
    2.  Packs the sequence.
    3.  Passes it through the LSTM layer.
    4.  Unpacks (pads) the output back to the original tensor shape for downstream fully connected layers.

## 4. Training Loop (`train.py`)
This script orchestrates the optimization process and enforces state independence.

### 4.1 Training Configuration
* **Optimizer:** AdamW
* **Criterion:** Binary Cross-entropy (on multi-label output).
* **Gradient Clipping:** Implemented to prevent exploding gradients, a common issue in RNNs/LSTMs, ensuring stability during backpropagation.

### 4.2 State Management (Patient Independence)
To satisfy the inductive bias that **different patients are independent**:
* **State Reset:** At the start of each new batch (brand new temporal state), the hidden state ($h_0$) and cell state ($c_0$) are explicitly reset to zero.
* **Constraint:** The time state inside the LSTM is reset, preventing context from Patient $i$ influencing the prediction for Patient $j$.

## 5. Embedding Pre-training (`train_skipgram.py`)
This module trains the vector representations for medical codes (ICD codes, surgery codes) before feeding them into the LSTM.

### 5.1 Skip-gram Mode
* **Objective:** Learns contextual relationships between medical codes.
* **Training Scope Strategy:**
    * *Option A (Strict):* Train embeddings only on the **Training Set** to maintain absolute isolation of the test data.
    * *Option B (Transductive):* Train embeddings on the **Whole Dataset** (Train + Val + Test). This is often permissible in unsupervised feature learning to handle rare codes appearing in the test set, provided labels are not leaked.

