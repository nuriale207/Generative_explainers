# Generative Explainers

Pipeline for further pretraining, fine-tuning, and explainability evaluation of clinical language models on Electronic Health Records (EHRs).

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_MaskedLM_model_train.py` | Further pretrain GatorTron with MLM on clinical text, expanding its vocabulary with ICD codes |
| 2 | `2_Next_code_model_train.py` | Fine-tune the pretrained model for multi-label ICD-10 diagnosis prediction |
| 3 | `3_Summarize_EHR_texts.py` | Summarize raw EHR texts using a quantized LLM (Mixtral-8x7B) |
| 4 | `4_Extract_prediction_explanations_from_label.py` | Generate natural language explanations for each predicted diagnosis using a generative model |
| 5 | `5_Evaluate_explainability.py` | Evaluate explanation quality using DoXPy (Degree of Explainability) |
| 6 | `6_Extract_textual_prediction_explanations_robustness.py` | Re-generate explanations to assess robustness and consistency |

## Installation

```bash
pip install -r requirements.txt

# Install local DoXPy package
pip install -e ./doxpy

# Download Spanish SpaCy model (required for step 5)
python -m spacy download es_dep_news_trf
```

## Environment variables

The following environment variables must be set before running the pipeline:

```bash
export HF_TOKEN=<your_huggingface_token>       # steps 1, 2, 3, 4, 6
export GROQ_API_KEY=<your_groq_api_key>        # only if using Groq-based functions
```

## Usage

### Step 1 — Further pretrain GatorTron

```bash
python 1_MaskedLM_model_train.py \
    --training_data_dir data/train.tsv \
    --eval_data_dir data/val.tsv \
    --output_dir output \
    --experiment_name gatortron_further_pretrain
```

### Step 2 — Fine-tune for diagnosis prediction

```bash
python 2_Next_code_model_train.py \
    --training_data_dir data/train.tsv \
    --eval_data_dir data/val.tsv \
    --test_data_dir data/test.tsv \
    --model_path output/gatortron_further_pretrain/model \
    --output_dir output \
    --experiment_name gatortron_diagnosis_finetune
```

### Step 3 — Summarize EHRs

```bash
python 3_Summarize_EHR_texts.py \
    --data_path data/patients.csv \
    --ehr_column TEXT \
    --output_path data/patients_summarized.csv
```

### Step 4 — Extract explanations

```bash
python 4_Extract_prediction_explanations_from_label.py \
    --data_path data/predictions.csv \
    --label_column Label \
    --text_column summarized_text_codes \
    --output output/explanations.csv
```

### Step 5 — Evaluate explainability (DoXPy)

```bash
python 5_Evaluate_explainability.py \
    --data data/explanations.csv \
    --explanation_column Explanation \
    --label_column Label \
    --language english \
    --output output/dox_scores.csv
```

### Step 6 — Robustness evaluation

```bash
python 6_Extract_textual_prediction_explanations_robustness.py \
    --data_path data/explanations.csv \
    --label_column Label \
    --text_column summarized_text_codes \
    --output output/explanations_robustness.csv
```

## Project structure

```
.
├── 1_MaskedLM_model_train.py
├── 2_Next_code_model_train.py
├── 3_Summarize_EHR_texts.py
├── 4_Extract_prediction_explanations_from_label.py
├── 5_Evaluate_explainability.py
├── 6_Extract_textual_prediction_explanations_robustness.py
├── utils/
│   ├── data_handler.py          # Dataset loading and tokenization
│   ├── model_handler.py         # Model loading and tokenizer utilities
│   ├── model_loader.py          # Quantized model loading
│   ├── metrics.py               # Multi-label classification metrics
│   └── website_functions_v2.py  # Prediction and summarization helpers
├── explainability_utils/
│   ├── explainability_utils.py  # Explanation generation with LLMs
│   ├── evaluation_utils.py      # Factuality and robustness metrics (USE-based)
│   └── word_analysis_utils.py   # Word-level attribution utilities
└── doxpy/                       # Local DoXPy package for DoX estimation
```
