# SalinTala Demo

A Streamlit application that compares baseline mBART50 with a fine-tuned LoRA model for Filipino/Taglish to English translation. The app provides side-by-side translation comparisons with confidence scores and evaluation metrics.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SalinTalaDemo
   ```

2. **Create and activate a virtual environment**:

   **For macOS/Linux**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   **For Windows**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
See `requirements.txt` for complete dependency list.

4. **Run the application**:
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

Download the fine-tuned model folder from: [https://tinyurl.com/salintalademo](https://tinyurl.com/salintalademo)

Insert the downloaded folder to the local repo directory.
