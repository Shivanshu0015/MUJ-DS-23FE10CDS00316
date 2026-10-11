# NLP Final Project Viva - 2-Minute Script

**Introduction (0:00 - 0:30)**
"Hello Professor. For my final NLP project, I developed SignLingua-AI. It solves a specific linguistic problem: translating non-inflected Sign Language Glosses—which lack prepositions, grammatical tense, and conjugations—into natural spoken English. I chose a hybrid approach, combining traditional rule-based NLP with a modern Large Language Model."

**Demonstration & Architecture (0:30 - 1:15)**
*(Click run on a sample in the app)*
"Under the hood, the pipeline has two stages. First, I use local NLP libraries, specifically spaCy and TextBlob, to perform morphological analysis, part-of-speech tagging, and calculate the Type-Token Ratio. Second, I integrated the OpenAI API using a strict zero-shot JSON schema. This prompts the LLM to reconstruct the missing syntax and provide standard, formal, and casual conversational variants."

**Code Structure & Security (1:15 - 1:45)**
"To make this production-ready, I built a modular architecture separating my source code, prompt templates, and testing suites. I also implemented environment variables using `python-dotenv` so that the API keys are completely secured and never hardcoded into the repository."

**Conclusion (1:45 - 2:00)**
"As you can see in the metrics tab, the system successfully bridges rule-based token tracking with AI semantic inference. The entire codebase is available on my GitHub. I'm happy to answer any questions about my implementation!"