#!/usr/bin/env python3
"""
Streamlit app: Compare vanilla mBART50 vs fine-tuned baseline (LoRA) for TL→EN.

- Left: input Filipino/Taglish text
- Right: translations from both models
- Shows a simple confidence score (avg token probability) per output

Run:
  streamlit run streamlit_translate.py
"""

import os
import torch
import streamlit as st
import pandas as pd
import requests
import urllib.parse
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel
from sacrebleu import BLEU, CHRF
from difflib import SequenceMatcher


BASE_MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
FINETUNED_DIR = "mbart50-finetuned-LoRA-best"  # folder in repo


@st.cache_resource(show_spinner=False)
def load_vanilla():
    model = MBartForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
    tok = MBart50Tokenizer.from_pretrained(BASE_MODEL_NAME)
    tok.src_lang = "tl_XX"
    tok.tgt_lang = "en_XX"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), tok, device


@st.cache_resource(show_spinner=False)
def load_finetuned():
    base = MBartForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
    try:
        model = PeftModel.from_pretrained(base, FINETUNED_DIR)
    except Exception:
        model = base
    tok = MBart50Tokenizer.from_pretrained(BASE_MODEL_NAME)
    tok.src_lang = "tl_XX"
    tok.tgt_lang = "en_XX"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), tok, device


def translate_with_confidence(model, tok, device, text: str):
    enc = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    enc = {k: v.to(device) for k, v in enc.items()}
    bos = tok.lang_code_to_id.get("en_XX", tok.eos_token_id)

    with torch.no_grad():
        out = model.generate(
            **enc,
            forced_bos_token_id=bos,
            max_length=128,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            length_penalty=0.8,
            repetition_penalty=1.2,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    seq = out.sequences[0]
    text_out = tok.decode(seq, skip_special_tokens=True).strip()

    # Compute average token probability as a simple confidence
    # Use transition scores util from HF to align scores with generated tokens
    try:
        transition_scores = model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        )[0]  # (seq_len-1,)
        # Use sigmoid over log-prob? Scores are log softmax; convert to probs
        token_probs = transition_scores.exp()
        # Confidence as average prob over generated tokens (ignore first token)
        conf = float(token_probs.mean().item())
    except Exception:
        conf = 0.0

    return text_out, conf


@st.cache_data
def load_reference_dataset():
    """Load the reference dataset for automatic reference matching."""
    try:
        df = pd.read_csv('annotated_tweets.csv')
        return df
    except Exception as e:
        st.warning(f"Could not load reference dataset: {e}")
        return None

def find_best_reference(input_text, dataset):
    """Find the best matching reference translation from the dataset."""
    if dataset is None or len(dataset) == 0:
        return None, 0.0
    
    best_match = None
    best_score = 0.0
    
    # First check for exact matches (100% similarity)
    input_lower = input_text.lower().strip()
    for _, row in dataset.iterrows():
        src_text = str(row['src']).lower().strip()
        
        # Check for exact match first
        if input_lower == src_text:
            return str(row['tgt']).strip(), 1.0
        
        # Calculate similarity for non-exact matches
        similarity = SequenceMatcher(None, input_lower, src_text).ratio()
        
        if similarity > best_score:
            best_score = similarity
            best_match = str(row['tgt']).strip()
    
    # Return best match if similarity is above threshold
    if best_score > 0.3:  # 30% similarity threshold
        return best_match, best_score
    
    return None, 0.0

def get_automatic_reference(input_text):
    """Automatically get reference translation - use 100% dataset match, else Google Translate."""
    # Step 1: Check dataset for exact 100% match only
    dataset = load_reference_dataset()
    if dataset is not None:
        reference, similarity = find_best_reference(input_text, dataset)
        if reference and similarity == 1.0:
            return reference, f"dataset_{similarity:.1%}"
    
    # Step 2: Use Google Translate if no good dataset match
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=tl&tl=en&dt=t&q={urllib.parse.quote(input_text)}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result and len(result) > 0 and result[0]:
                translation = result[0][0][0]
                if translation and translation.strip():
                    return translation.strip(), "google"
    except Exception as e:
        st.warning(f"Google Translate failed: {e}")
    
    return None, "none"

def calculate_metrics(prediction, reference):
    """Calculate BLEU and chrF scores between prediction and reference."""
    try:
        # Calculate BLEU score with effective_order for better sentence-level scoring
        bleu = BLEU(effective_order=True)
        bleu_score = bleu.sentence_score(prediction, [reference])
        bleu_percentage = bleu_score.score
        
        # Calculate chrF score
        chrf = CHRF()
        chrf_score = chrf.sentence_score(prediction, [reference])
        chrf_percentage = chrf_score.score
        
        return bleu_percentage, chrf_percentage
    except Exception as e:
        st.warning(f"Error calculating metrics: {e}")
        return 0.0, 0.0


def main():
    st.set_page_config(page_title="SalinTala: Demo", page_icon="🌐", layout="wide")
    
    # Apply dark mode CSS
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .stApp {
        background-color: #0e1117;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #404040;
        font-size: 30px;
        line-height: 1.4;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #ff4b4b;
        box-shadow: 0 0 0 1px #ff4b4b;
    }
    
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background-color: #ff6b6b;
        color: white;
    }
    
    .stSlider > div > div > div > div {
        background-color: #ff4b4b;
    }
    
    .stCheckbox > div > label > div {
        background-color: #262730;
        border: 1px solid #404040;
    }
    
    .stCheckbox > div > label > div[data-checked="true"] {
        background-color: #ff4b4b;
        border-color: #ff4b4b;
    }
    
    .stProgress > div > div > div > div {
        background-color: #ff4b4b;
    }
    
    .stSidebar {
        background-color: #262730;
    }
    
    .stSidebar .stText {
        color: #fafafa;
    }
    
    .stCaption {
        color: #8b949e;
    }
    
    .stSubheader {
        color: #fafafa;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #fafafa;
    }
    
    .stMarkdown p {
        color: #fafafa;
    }
    
    .stAlert {
        background-color: #262730;
        border: 1px solid #404040;
        color: #fafafa;
    }
    
    .stWarning {
        background-color: #262730;
        border: 1px solid #ffa500;
        color: #fafafa;
    }
    
    /* Custom styling for BLEU and chrF metrics */
    .bleu-metric {
        color: #3b82f6 !important;
        font-weight: bold;
        font-size: 20px;
    }
    
    .chrf-metric {
        color: #f59e0b !important;
        font-weight: bold;
        font-size: 20px;
    }
    
    /* Custom styling for confidence scores */
    .confidence-text {
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("SalinTala: Demo")
    st.markdown("### TL→EN Translation Comparison")
    st.caption("Vanilla mBART50 vs Fine-tuned mBART50")

    text = st.text_area("Enter Filipino/Taglish text", height=140, placeholder="Hal.: Good evening! Maganda sales namin these days …", help="Enter your Filipino or Taglish text here for translation")

    # Initialize session state for reference
    if "auto_reference" not in st.session_state:
        st.session_state.auto_reference = None
    if "reference_source" not in st.session_state:
        st.session_state.reference_source = "none"

    col1, col2 = st.columns(2)
    
    # Set fixed decoding parameters (optimized for Filipino→English translation)
    decode_params = {
        "beams": 4,        # Good quality/speed balance
        "ngram": 3,        # Prevents loops while allowing natural repetition
        "len_pen": 1.0,    # Neutral length bias (Filipino→English often needs more words)
        "rep_pen": 1.3,    # Slightly stronger repetition control for English
        "max_len": 128     # Reasonable length for most translations
    }

    if st.button("Translate", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
            return

        # Automatically get reference translation
        with st.spinner("🔍 Finding reference translation..."):
            reference, source = get_automatic_reference(text)
            st.session_state.auto_reference = reference
            st.session_state.reference_source = source
            
            if reference:
                if source.startswith("dataset"):
                    st.success(f"✅ Found reference in dataset ({source.split('_')[1]})")
                elif source == "google":
                    st.success("✅ Generated reference using Google Translate")
                else:
                    st.success("✅ Reference found")
            else:
                st.warning("⚠️ No reference translation available - metrics will be skipped")

        # Load models
        v_model, v_tok, device = load_vanilla()
        f_model, f_tok, _ = load_finetuned()

        # Override decoding per sidebar (patch via closure)
        def _english_fraction(s: str) -> float:
            if not s:
                return 0.0
            # Count basic English letters, digits, space and common punctuation as "English"
            eng_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.?!'\"-:;()[]{}\n\t")
            eng = sum((ch in eng_chars) for ch in s)
            return min(1.0, max(0.0, eng / max(1, len(s))))

        def run(model, tok):
            enc = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            enc = {k: v.to(device) for k, v in enc.items()}
            bos = tok.lang_code_to_id.get("en_XX", tok.eos_token_id)
            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    forced_bos_token_id=bos,
                    max_length=decode_params["max_len"],
                    num_beams=decode_params["beams"],
                    do_sample=False,
                    no_repeat_ngram_size=decode_params["ngram"],
                    length_penalty=decode_params["len_pen"],
                    repetition_penalty=decode_params["rep_pen"],
                    early_stopping=True,
                )
            seq = out_ids[0]
            text_out = tok.decode(seq, skip_special_tokens=True).strip()

            # Confidence: entropy-based certainty under teacher forcing (lower entropy → higher confidence)
            try:
                with torch.no_grad():
                    labels = seq.unsqueeze(0)
                    labels[labels == tok.pad_token_id] = -100
                    outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
                    logits = outputs.logits  # (1, T, V)
                    # Align with labels (ignore -100)
                    mask = (labels != -100)
                    T = int(mask.sum().item())
                    if T == 0:
                        raise RuntimeError("no valid tokens for confidence")
                    sel_logits = logits[mask]  # (T, V)
                    probs = torch.softmax(sel_logits, dim=-1)
                    # token-level entropy H = -sum p log p
                    ent = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)  # (T,)
                    # Normalize by log(V) to get 0..1
                    V = sel_logits.size(-1)
                    ent_norm = ent / float(torch.log(torch.tensor(V, dtype=ent.dtype, device=ent.device)))
                    entropy_mean = float(ent_norm.mean().item())  # 0..1
                    prob_core = max(0.0, min(1.0, 1.0 - entropy_mean))
                # English content ratio to penalize wrong-script outputs
                lang_factor = _english_fraction(text_out)
                # Very short outputs are unreliable → downweight
                length_factor = min(1.0, max(0.2, len(text_out) / 40.0))
                conf = max(0.0, min(1.0, prob_core * lang_factor * length_factor))
            except Exception:
                conf = 0.0
            return text_out, conf

        t_vanilla, c_vanilla = run(v_model, v_tok)
        t_finetuned, c_finetuned = run(f_model, f_tok)

        with col1:
            st.subheader("mBART50 (Baseline)")
            st.markdown(f"<div style='font-size: 30px; padding: 15px; background-color: #262730; color: #fafafa; border: 1px solid #404040; border-radius: 8px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.3);'>{t_vanilla}</div>", unsafe_allow_html=True)
            st.progress(min(max(c_vanilla, 0.0), 1.0))
            st.markdown(f'<div class="confidence-text">Confidence: {c_vanilla*100:.0f}%</div>', unsafe_allow_html=True)
            
            # Calculate and display metrics if reference is available
            if st.session_state.auto_reference:
                bleu_vanilla, chrf_vanilla = calculate_metrics(t_vanilla, st.session_state.auto_reference)
                col1a, col1b = st.columns(2)
                with col1a:
                    st.markdown(f'<div class="bleu-metric">BLEU: {bleu_vanilla:.1f}%</div>', unsafe_allow_html=True)
                with col1b:
                    st.markdown(f'<div class="chrf-metric">chrF: {chrf_vanilla:.1f}%</div>', unsafe_allow_html=True)
                
                # Show reference source
                if st.session_state.reference_source.startswith("dataset"):
                    st.caption(f"📊 Dataset reference ({st.session_state.reference_source.split('_')[1]})")
                elif st.session_state.reference_source == "google":
                    st.caption("🌐 Google Translate reference")
                else:
                    st.caption("📊 Reference translation")

        with col2:
            st.subheader("Fine-tuned mBART50 (LoRA Fine-tuned)")
            st.markdown(f"<div style='font-size: 30px; padding: 15px; background-color: #262730; color: #fafafa; border: 1px solid #404040; border-radius: 8px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.3);'>{t_finetuned}</div>", unsafe_allow_html=True)
            st.progress(min(max(c_finetuned, 0.0), 1.0))
            st.markdown(f'<div class="confidence-text">Confidence: {c_finetuned*100:.0f}%</div>', unsafe_allow_html=True)
            
            # Calculate and display metrics if reference is available
            if st.session_state.auto_reference:
                bleu_finetuned, chrf_finetuned = calculate_metrics(t_finetuned, st.session_state.auto_reference)
                col2a, col2b = st.columns(2)
                with col2a:
                    st.markdown(f'<div class="bleu-metric">BLEU: {bleu_finetuned:.1f}%</div>', unsafe_allow_html=True)
                with col2b:
                    st.markdown(f'<div class="chrf-metric">chrF: {chrf_finetuned:.1f}%</div>', unsafe_allow_html=True)
                
                # Show reference source
                if st.session_state.reference_source.startswith("dataset"):
                    st.caption(f"📊 Dataset reference ({st.session_state.reference_source.split('_')[1]})")
                elif st.session_state.reference_source == "google":
                    st.caption("🌐 Google Translate reference")
                else:
                    st.caption("📊 Reference translation")


if __name__ == "__main__":
    main()


