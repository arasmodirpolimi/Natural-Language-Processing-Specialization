# Natural Language Processing Specialization (Educational Labs)

This repository collects hands-on Python labs and assignments covering core and advanced topics in Natural Language Processing (NLP). The material is organized by course themes (Classification & Vector Spaces, Probabilistic Models, Sequence Models, Attention & Transformers). Each week introduces focused concepts through small, self-contained scripts or notebooks you can run and extend.

> NOTE: The code is adapted from educational exercises. It is provided here for personal learning and experimentation. Do not redistribute proprietary course content verbatim. All original academic paper references remain the property of their respective authors.

---
## 📚 Topics Covered

### 1. Classification and Vector Spaces
- Text preprocessing (tokenization, normalization, stopword handling)
- Word frequency analysis and feature extraction
- Logistic regression for sentiment and topic classification
- Naïve Bayes intuition with visualization
- Manipulating pretrained word embeddings (analogies, distances, projection)
- Dimensionality reduction with PCA
- Hashing tricks and multi-plane locality sensitive hashing

### 2. Probabilistic Models
- Vocabulary construction and token statistics
- Edit distance–based candidate generation for spelling correction
- N-gram language modeling and smoothing intuition
- Part-of-speech tagging with probabilistic features
- Corpus preprocessing workflows (regex, normalization, handling OOV)
- Word embedding training (CBOW principles, step-by-step walkthrough)

### 3. Sequence Models
- Recurrent neural networks (RNN fundamentals, hidden state dynamics)
- Perplexity and evaluation of generative models
- TensorFlow basics for sequence tasks (tokenization, losses, layers)
- Vanishing gradients exploration and visualization
- Practical exercises building custom training loops
- Siamese / Triplet architectures for similarity learning

### 4. Attention and Transformer Models
- Basic additive attention (Bahdanau) implemented in NumPy
- Scaled dot-product (QKV) attention mechanics
- Positional encoding and sequence masking
- Multi-step Transformer utilities (tokenization, batching, training helpers)
- BLEU score implementation from first principles and validation with SacreBLEU
- Subword tokenization (SentencePiece, BPE concepts and frequency merges)
- Hugging Face pipelines and fine-tuning workflows for QA (BERT)

---
## 🗂 Repository Structure
```
Natural Language Processing with Classification and Vector Spaces/
  week1 ... week4
Natural Language Processing with Probabilistic Models/
  week1 ... week4
Natural Language Processing with Sequence Models/
  week1 ... week3
Natural Language Processing with Attention Models/
  week1 ... week3
```
Each `weekX` directory contains matching `.ipynb` (interactive) and `.py` (script) versions for most labs. Some helper files (`utils`, `utils_nb`, `transformer_utils`, `w*_unittest`) are referenced by assignments to abstract away boilerplate.

---
## 🧪 Representative Implementations
Below are brief summaries of selected core scripts to guide exploration:

### `C4W1_Basic_Attention.py`
Implements Bahdanau-style additive attention manually:
- Alignment `e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)` via concatenation + 2-layer MLP
- Softmax normalization to produce attention weights
- Weighted encoder state aggregation into a context vector
Useful for understanding how attention redistributes focus across sequence positions without training overhead.

### `C4W1_QKV_Attention.py`
Scaled dot-product attention with pretrained bilingual embeddings:
- Tokenization + embedding lookups for EN/FR sentences
- Weight computation: `softmax( (QK^T) / sqrt(d_k) )`
- Application of weights to values matrix to derive contextualized representations
- Visualization of alignment matrix for cross-lingual mapping
Great for grasping geometric similarity and alignment interpretation.

### `C4W1_Bleu_Score.py`
Custom BLEU metric implementation:
- Brevity penalty and clipped n-gram precision (1–4 grams)
- Geometric mean aggregation and comparison to SacreBLEU
- Corpus-level evaluation sample (English→German)
Highlights reproducibility and evaluation rigor for MT systems.

### `C1_W3_lecture_nb_02_manipulating_word_embeddings.py`
Explores intrinsic embedding evaluation:
- Vector arithmetic for analogies (`France - Paris + Madrid ≈ Spain`)
- Distance and direction inspection with 2D projections
- Closest-word search via L2 norms in embedding space
Foundation for semantic similarity and analogy reasoning.

### Transformer Utilities & Assignments
Files like `C4W3_Assignment.py` showcase integration of:
- TensorFlow / TensorFlow Text tokenization
- Positional encoding and masking modules
- Test harnesses (`w3_unittest`) for incremental validation
- JSON/config handling plus colored terminal diagnostics

---
## 🧩 Key Concepts Reinforced
- Vector space semantics: cosine similarity, analogies, PCA
- Probabilistic reasoning: smoothing, frequency counts, candidate generation
- Sequence modeling: hidden state evolution, gradient pathologies
- Evaluation metrics: BLEU, perplexity, accuracy variants
- Attention: additive vs. multiplicative (scaled dot-product) mechanics
- Subword modeling: frequency-based merges, vocabulary compactness

---
## 🛠 Dependencies
Collected from import scanning (you may not need all for every week):
- Core: `numpy`, `pandas`, `matplotlib`, `pickle`, `string`, `re`, `collections`, `random`, `math`, `itertools`, `json`, `time`, `traceback`
- NLP: `nltk`, `emoji`, `sacrebleu`, `sentencepiece`, `tensorflow`, `tensorflow_text`, `termcolor`, `datasets`, `transformers`, `sklearn`
- Jupyter / Interactive: `ipywidgets`
- Others (helper modules inside repo): `utils`, `utils2`, `utils_nb`, `transformer_utils`, `w*_unittest`

### Installation
Create a virtual environment (recommended) then install essentials incrementally to keep it lightweight:
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas matplotlib nltk sacrebleu scikit-learn emoji sentencepiece termcolor ipywidgets transformers datasets tensorflow==2.* tensorflow-text
```
Optionally omit heavy packages (TensorFlow/HF) if exploring only NumPy-based labs.

Download NLTK tokenizers if prompted:
```python
import nltk
nltk.download('punkt')
```

---
## ▶️ How to Run
### Jupyter Notebooks
Open any `.ipynb` in VS Code or Jupyter Lab and execute cells sequentially.

### Python Scripts
Most scripts are linear educational walkthroughs:
```powershell
python "Natural Language Processing with Attention Models/week1/C4W1_Basic_Attention.py"
```
If a script expects local `./data` assets (e.g., embeddings or aligned vectors), ensure the directory structure is intact before running.

### Hugging Face QA Example
For `C4W3_HF_Lab1_QA_BERT.py` (pipeline usage):
- Requires `transformers` and possibly internet access for model download.
- You can set `HF_HOME` to cache models locally.

### BLEU Corpus Evaluation
Ensure files exist under `data/` (`wmt19_src.txt`, `wmt19_ref.txt`, `wmt19_can.txt`) before executing the BLEU script.

---
## 🧪 Extending the Labs (Ideas)
- Swap additive attention with dot-product for performance comparison
- Add masking to QKV attention to simulate causal decoding
- Implement unigram vs. subword tokenization evaluation (SentencePiece vs. BPE)
- Integrate a small RNN language model trained on a subset of corpus text
- Build unit tests around BLEU subtasks (brevity penalty edge cases, clipped precision correctness)

---
## ✅ Testing & Validation
Several assignment scripts import `w*_unittest` modules. These lightweight test harnesses:
- Validate shapes and intermediate values (e.g. attention score normalization)
- Catch silent numerical mistakes early
Run an assignment script directly; if a test fails, inspect the printed assertion context.

---
## 🧹 Data & Assets
The `data/` folders contain small subsets of embeddings, vocabulary pickles, and corpus snippets. Large pretrained models are intentionally excluded. If a file is missing:
1. Check expected filename inside the script.
2. Replace with your own miniature sample (same format) to reproduce logic.
3. For aligned fastText vectors, see: https://fasttext.cc/docs/en/aligned-vectors.html

---
## 💡 Design Principles of the Code
- Pedagogical clarity over performance (explicit loops, repeated definitions)
- Minimal abstraction: functions defined inline to emphasize concept locality
- Deterministic seeds where randomness is used
- Visual diagnostics (matplotlib, heatmaps, attention maps)

---
## 🔒 License & Attribution
This repository is for personal educational use. Source code patterns reference publicly available academic papers:
- Bahdanau et al. (2014) - Neural Machine Translation by Jointly Learning to Align and Translate
- Vaswani et al. (2017) - Attention Is All You Need
- Papineni et al. (2002) - BLEU Metric

Please cite original authors if you publish derivative work. Avoid distributing proprietary course datasets or solutions wholesale.

---
## 🤝 Contributing
Small improvements welcome (typo fixes, dependency pinning, lightweight tests). Suggested workflow:
1. Fork & create a feature branch
2. Add or adjust a single concept per PR (keep pedagogical tone)
3. Include a short README subsection update if concept coverage changes

---
## 🐛 Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| ModuleNotFoundError (utils) | Running notebook cell out of repo root | `cd` to repo root or adjust `PYTHONPATH` |
| OOM when importing TensorFlow | Limited RAM | Use only NumPy labs or set `TF_FORCE_GPU_ALLOW_GROWTH=true` |
| Slow HF model download | Network throttling | Pre-download models; use offline cache (`transformers-cli download`) |
| NLTK tokenizer missing | Resource not downloaded | `nltk.download('punkt')` |

---
## 📈 Next Steps (Suggested Enhancements)
- Add `requirements.txt` with minimal and full profiles
- Introduce `pytest` tests for BLEU, attention, and embedding analogy functions
- Provide a lightweight Dockerfile for reproducible setup
- Add a benchmarking script comparing attention mechanisms on synthetic data

---
## ✅ Quick Start Minimal Install
If you only want to explore core NumPy concepts:
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install numpy matplotlib nltk sacrebleu
```
Then run: `python "Natural Language Processing with Attention Models/week1/C4W1_Basic_Attention.py"`

Enjoy exploring the building blocks of modern NLP!
