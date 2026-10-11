import streamlit as st
import json
from config import Config
from src.utils import sanitize_gloss_input
from src.nlp_processor import TraditionalNLPProcessor
from src.llm_client import GlossLLMTranslator

st.set_page_config(
    page_title="SignLingua-AI: Sign Gloss NLP & Translation System",
    page_icon="🤟",
    layout="wide"
)

# Header
st.title("🤟 SignLingua-AI: Sign Gloss NLP Pipeline & LLM Translator")
st.caption(
    "A hybrid NLP translation engine converting non-inflected Sign Language Gloss sequences "
    "into fluent natural text using local linguistic tokenization and LLM semantic reconstruction."
)

# Sidebar System Health
with st.sidebar:
    st.header("⚙️ System Status")
    if Config.OPENAI_API_KEY:
        st.success("API Key Active")
    else:
        st.error("Missing API Key")
        st.info("Configure OPENAI_API_KEY in your local .env file.")

    st.markdown("---")
    st.markdown("**Architecture Specs:**")
    st.markdown("- **Local NLP:** spaCy (`en_core_web_sm`), TextBlob")
    st.markdown(f"- **LLM Translation Engine:** OpenAI `{Config.LLM_MODEL}`")
    st.markdown("- **Zero-Shot Prompt Framework:** External JSON Schema")

# Preset Samples
PRESET_GLOSSES = {
    "Select a preset sample...": "",
    "Emergency Care: ME CHEST PAIN STRONG NOW HOSPITAL GO NEED": "ME CHEST PAIN STRONG NOW HOSPITAL GO NEED",
    "Office Update: TOMORROW MEETING CANCEL PROJECT FINISH NOT": "TOMORROW MEETING CANCEL PROJECT FINISH NOT",
    "Social Plan: WEEKEND MOVIE GO TOGETHER WANT YOU": "WEEKEND MOVIE GO TOGETHER WANT YOU",
    "Travel Question: TRAIN STATION WHERE TICKET HOW-MUCH BUY": "TRAIN STATION WHERE TICKET HOW-MUCH BUY"
}

selected_preset = st.selectbox("Choose a sample Sign Language Gloss sequence:", list(PRESET_GLOSSES.keys()))

input_gloss = st.text_input(
    "Or type custom Sign Language Gloss tokens (all-caps sign words):",
    value=PRESET_GLOSSES[selected_preset] if selected_preset != "Select a preset sample..." else "",
    placeholder="e.g. YESTERDAY ME SISTER GIFT BIRTHDAY GIVE"
)

col_run, col_clear = st.columns([2, 4])
with col_run:
    run_button = st.button("🚀 Process NLP Pipeline", type="primary", use_container_width=True)

if run_button:
    cleaned_gloss = sanitize_gloss_input(input_gloss)

    if not cleaned_gloss:
        st.warning("Please enter or select a valid gloss sequence before running the pipeline.")
    elif not Config.OPENAI_API_KEY:
        st.error("Cannot query the LLM without a valid OPENAI_API_KEY. Add it to .env.")
    else:
        # Step 1: Traditional Local NLP Analysis
        with st.spinner("Executing Local spaCy Morphological Analysis..."):
            nlp_proc = TraditionalNLPProcessor()
            local_gloss_metrics = nlp_proc.analyze_gloss(cleaned_gloss)

        # Step 2: LLM Translation Call
        with st.spinner("Invoking LLM for Semantic & Syntactic Reconstruction..."):
            translator = GlossLLMTranslator()
            llm_result = translator.translate(cleaned_gloss)

        if llm_result["status"] == "success":
            data = llm_result["data"]
            generated_analysis = nlp_proc.analyze_generated_text(data.get("standard_translation", ""))

            st.success("NLP Pipeline Execution Completed Successfully!")

            tab1, tab2, tab3 = st.tabs([
                "💬 Spoken English Translations",
                "🔬 Linguistic & Local NLP Metrics",
                "📄 Full Export & Diagnostics"
            ])

            with tab1:
                st.markdown("### 🗣️ Reconstructed Spoken English")
                st.info(f"**Standard:** {data.get('standard_translation', 'N/A')}")

                col_formal, col_casual = st.columns(2)
                with col_formal:
                    st.markdown("#### 🏢 Professional Register")
                    st.write(data.get("formal_translation", "N/A"))
                with col_casual:
                    st.markdown("#### ☕ Casual / Colloquial")
                    st.write(data.get("casual_translation", "N/A"))

                st.markdown("---")
                col_tense, col_conf = st.columns(2)
                with col_tense:
                    st.metric("Grammatical Tense Inferred", data.get("tense_identified", "Present"))
                with col_conf:
                    st.metric("LLM Confidence Score", f"{data.get('confidence_score', 0.95) * 100:.1f}%")

                st.markdown("#### 🧩 Inferred Missing Syntax & Prepositions")
                for item in data.get("inferred_elements", []):
                    st.markdown(f"- *Added functional element:* **{item}**")

            with tab2:
                st.markdown("### 📊 Local NLP Lexical Metrics (spaCy)")
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                metric_col1.metric("Input Gloss Tokens", local_gloss_metrics["token_count"])
                metric_col2.metric("Unique Gloss Tokens", local_gloss_metrics["unique_token_count"])
                metric_col3.metric("Type-Token Ratio (TTR)", local_gloss_metrics["type_token_ratio"])
                metric_col4.metric("Output Word Count", generated_analysis["word_count"])

                st.markdown("#### 🏷️ Gloss Token-Level Part-of-Speech Tagging")
                pos_display = [
                    f"**{item['token']}** `[{item['pos']}]` ({item['dep']})"
                    for item in local_gloss_metrics["pos_breakdown"]
                ]
                st.write(" → ".join(pos_display))

                st.markdown("#### 📈 Output Sentiment & Subjectivity")
                sent_col1, sent_col2 = st.columns(2)
                with sent_col1:
                    st.progress((generated_analysis["sentiment_polarity"] + 1) / 2)
                    st.caption(f"Polarity: {generated_analysis['sentiment_polarity']} (-1.0 Negative to +1.0 Positive)")
                with sent_col2:
                    st.progress(generated_analysis["sentiment_subjectivity"])
                    st.caption(f"Subjectivity: {generated_analysis['sentiment_subjectivity']} (0.0 Objective to 1.0 Subjective)")

            with tab3:
                st.markdown("### 📋 Structured JSON Output")
                export_payload = {
                    "input_gloss": cleaned_gloss,
                    "local_gloss_nlp": local_gloss_metrics,
                    "llm_translation_output": data,
                    "generated_text_nlp": generated_analysis
                }
                st.json(export_payload)

                st.download_button(
                    label="📥 Download Submission JSON Report",
                    data=json.dumps(export_payload, indent=2),
                    file_name="signlingua_translation_report.json",
                    mime="application/json"
                )
        else:
            st.error(llm_result["message"])