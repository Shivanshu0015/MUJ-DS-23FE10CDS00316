from textblob import TextBlob
from src.utils import load_spacy_pipeline

class TraditionalNLPProcessor:
    """
    Executes rule-based and statistical NLP analysis locally without API dependencies.
    Evaluates lexical density, parts of speech, and syntactic characteristics.
    """
    def __init__(self):
        self.nlp = load_spacy_pipeline()

    def analyze_gloss(self, gloss_text: str) -> dict:
        """Analyzes the raw input gloss tokens."""
        doc = self.nlp(gloss_text)
        tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
        unique_tokens = set(tokens)

        # Type-Token Ratio (TTR): classic lexical diversity metric
        ttr = round(len(unique_tokens) / len(tokens), 3) if tokens else 0.0

        pos_tags = [{"token": token.text, "pos": token.pos_, "dep": token.dep_} for token in doc]

        return {
            "token_count": len(tokens),
            "unique_token_count": len(unique_tokens),
            "type_token_ratio": ttr,
            "pos_breakdown": pos_tags
        }

    def analyze_generated_text(self, translated_text: str) -> dict:
        """Performs sentiment and grammatical complexity checks on the translated output."""
        doc = self.nlp(translated_text)
        blob = TextBlob(translated_text)

        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        sentences = list(doc.sents)

        avg_sentence_len = round(len(words) / len(sentences), 2) if sentences else 0.0

        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": avg_sentence_len,
            "sentiment_polarity": round(blob.sentiment.polarity, 3),
            "sentiment_subjectivity": round(blob.sentiment.subjectivity, 3),
            "noun_chunks": [chunk.text for chunk in doc.noun_chunks]
        }