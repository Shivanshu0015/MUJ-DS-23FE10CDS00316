import re
import sys
import subprocess
import spacy
import streamlit as st

def sanitize_gloss_input(raw_gloss: str) -> str:
    """
    Cleans raw gloss input: forces uppercase, removes non-alphanumeric noise,
    and strips redundant whitespace.
    """
    if not raw_gloss:
        return ""
    cleaned = re.sub(r'[^a-zA-Z0-9\s\-_]', '', raw_gloss)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip().upper()
    return cleaned

@st.cache_resource(show_spinner=False)
def load_spacy_pipeline():
    """
    Loads the spaCy English language model.
    Downloads en_core_web_sm automatically if not locally available.
    """
    model_name = "en_core_web_sm"
    try:
        nlp = spacy.load(model_name)
    except OSError:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        nlp = spacy.load(model_name)
    return nlp