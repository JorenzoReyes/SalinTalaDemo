# SalinTala Demo

A Streamlit application that compares baseline mBART50 with a fine-tuned LoRA model for Filipino/Taglish to English translation. The app provides side-by-side translation comparisons with confidence scores and evaluation metrics.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SalinTalaDemo
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
See `requirements.txt` for complete dependency list.

3. **Run the application**:
   ```bash
   python salintala.py
   ```

   The app will automatically open in your browser at `http://localhost:8501`

## Model Information

- **Base Model**: `facebook/mbart-large-50-many-to-many-mmt`
- **Fine-tuned Model**: LoRA-adapted version stored in `mbart50-finetuned-LoRA-best/`
- **Language Pair**: Filipino/Taglish (tl_XX) → English (en_XX)


## Notes

- The fine-tuned model directory (`mbart50-finetuned-LoRA-best/`) is gitignored due to large file sizes
- First run may take longer as models are downloaded and cached

## Model Directory

Download the fine-tuned model files from: [https://tinyurl.com/salintalademo](https://tinyurl.com/salintalademo)

Extract the downloaded files to the `mbart50-finetuned-LoRA-best/` directory in your project root.
