# Syntactic Annotation Toolkit for Child–Adult InTeractions (CAIT)

## 📖 Description
In this repository you can find:

- A dependency parser trained with **SuPar** on the golden annotations of  
  [UD_English-CHILDES](https://github.com/UniversalDependencies/UD_English-CHILDES)
- A **PoS tagger** trained with Stanza on the golden
- A **construction tagger** based on this in-domain parser

The parser model is available on Hugging Face:  
👉 https://huggingface.co/fpadovani/biaffine_roberta_large_childes_10

---

## ⚙️ Environment Setup

To use this project, you need to create a Python environment with the correct dependencies.

### 1. Create and activate a virtual environment

```bash
python3.12 -m venv cait_env
source cait_env/bin/activate

pip install torch==2.5.1
pip install -U git+https://github.com/Yu-val-weiss/supar-parser
```