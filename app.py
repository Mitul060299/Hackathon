import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    AutoModelForSequenceClassification,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)
from sentence_transformers import SentenceTransformer, util
import faiss
import pandas as pd
import numpy as np
import re
import torch.nn.functional as F

st.set_page_config(page_title="AI Mental Health Chatbot (Prototype)", layout="wide")

# -----------------------------------------------------------
# Helper: choose lightweight defaults for Spaces (avoid huge models)
# -----------------------------------------------------------
LLM_PRIMARY = "tiiuae/falcon-rw-1b"        # smaller than phi-3 mini; safer for Spaces
LLM_FALLBACK = "microsoft/DialoGPT-small" # tiny fallback
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
ABSA_MODEL = "yangheng/deberta-v3-base-absa-v1.1"
SAFETY_MODEL = "unitary/toxic-bert"
EMBED_MODEL = "all-MiniLM-L6-v2"

# -----------------------------------------------------------
# Caching & model loading helpers
# -----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_llm(primary=LLM_PRIMARY, fallback=LLM_FALLBACK):
    """Load tokenizer+model. Use fallback if primary fails."""
    try:
        tok = AutoTokenizer.from_pretrained(primary)
        model = AutoModelForCausalLM.from_pretrained(primary, device_map="auto", torch_dtype=torch.float16)
        return tok, model, primary
    except Exception as e:
        st.warning(f"Primary LLM load failed: {e}. Falling back to {fallback}")
        tok = AutoTokenizer.from_pretrained(fallback)
        model = AutoModelForCausalLM.from_pretrained(fallback, device_map="auto", torch_dtype=torch.float16)
        return tok, model, fallback

@st.cache_resource(show_spinner=False)
def load_pipelines():
    """Load HF pipelines for sentiment, emotion, toxicity and embedding model."""
    # sentiment / emotion / safety pipelines using cpu/gpu whichever available
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=device)
    emotion_pipe = pipeline("text-classification", model=EMOTION_MODEL, return_all_scores=True, device=device)
    absa_pipe = pipeline("text-classification", model=ABSA_MODEL, device=device)
    safety_tokenizer = None
    safety_model = None
    try:
        safety_tokenizer = AutoTokenizer.from_pretrained(SAFETY_MODEL)
        safety_model = AutoModelForSequenceClassification.from_pretrained(SAFETY_MODEL).to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        st.info("Toxicity model failed to load — safety will use keyword embedding fallback.")
    embed_model = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    return {
        "sentiment": sentiment_pipe,
        "emotion": emotion_pipe,
        "absa": absa_pipe,
        "safety_tokenizer": safety_tokenizer,
        "safety_model": safety_model,
        "embed": embed_model,
    }

# -----------------------------------------------------------
# Utilities: detection & RAG
# -----------------------------------------------------------
def detect_sentiment(pipe, text):
    try:
        r = pipe(text)[0]
        label = r["label"].lower()
        score = float(r["score"])
        if "pos" in label: label = "positive"
        elif "neg" in label: label = "negative"
        else: label = "neutral"
        return label, score
    except Exception:
        return "neutral", 0.0

def detect_text_emotion(pipe, text, min_conf=0.35):
    try:
        res = pipe(text)
        if isinstance(res, list) and res and isinstance(res[0], list):
            scores = res[0]
        else:
            scores = res
        top = max(scores, key=lambda x: x["score"])
        label = top["label"].lower()
        score = float(top["score"])
        if score < min_conf: return "neutral", 0.0
        return label, score
    except Exception:
        return "neutral", 0.0

_ASPECT_KEYWORDS = {
    'girlfriend','boyfriend','partner','husband','wife','relationship','marriage','heartbreak','breakup','divorce',
    'family','mother','father','parent','sibling','friend',
    'job','career','work','boss','manager','colleague','layoff','termination','unemployment','job loss',
    'study','school','college','university','exam','test','marks','grades','education',
    'depression','depressed','anxiety','stressed','stress','fear','worry','lonely','isolation',
    'sad','sadness','grief','loss','trauma','hopeless','confused',
    'angry','anger','frustrated','irritated',
    'health','illness','sick','tired','fatigue','disease','mental health','therapy','counseling',
    'change','moving','transition'
}

def extract_aspects(text: str):
    """Extract candidate aspects using keyword search."""
    t = text.lower()
    aspects = []
    for kw in _ASPECT_KEYWORDS:
        if kw in t:
            aspects.append(kw)
    # Handle special phrases
    if "break up" in t: aspects.append("breakup")
    if "lost job" in t: aspects.append("job loss")
    if not aspects:
        aspects = ["situation"]
    return list(dict.fromkeys(aspects))  # deduplicate

def detect_absa(absa_pipe, text: str, min_confidence=0.4):
    """Run ABSA for extracted aspects and return sentiments with confidence scores."""
    try:
        aspects = extract_aspects(text)
        results = []

        for a in aspects:
            absa_input = f"[CLS] {a} [SEP] {text} [SEP]"
            out = absa_pipe(absa_input)

            if isinstance(out, list) and out and isinstance(out[0], dict):
                label = out[0].get("label", "neutral").lower()
                score = float(out[0].get("score", 0.0))

                if score >= min_confidence:
                    results.append({
                        "aspect": a,
                        "sentiment": label,
                        "confidence": round(score, 4)
                    })

        if not results:
            return [{"aspect": "situation", "sentiment": "neutral", "confidence": 0.0}]

        # Sort by confidence, highest first
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    except Exception as e:
        return [{"aspect": "situation", "sentiment": "neutral", "confidence": 0.0}]

def detect_toxicity(safety_tokenizer, safety_model, text, threshold=0.6):
    """Return True if input text is considered toxic/offensive."""
    if safety_model is None or safety_tokenizer is None:
        return False
    try:
        inputs = safety_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(safety_model.device)
        with torch.no_grad():
            logits = safety_model(**inputs).logits
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        max_prob = float(np.max(probs))
        return max_prob >= threshold
    except Exception:
        return False

def is_unsafe_message(embed_model, text, threshold=0.65):
    """Detect unsafe or crisis-related content using embeddings + toxicity classifier."""
    try:
        # Embedding-based similarity check
        unsafe_keywords = [
            "suicide","kill myself","self harm","hurt myself","end my life","overdose","cutting","hang myself",
            "can't go on","want to die","give up on life","life is pointless","i see no future","end it all"
        ]
        unsafe_emb = embed_model.encode(unsafe_keywords, convert_to_tensor=True)
        emb = embed_model.encode(text, convert_to_tensor=True)
        sims = F.cosine_similarity(emb, unsafe_emb)
        max_sim = torch.max(sims).item()

        # Toxicity classification check
        toxic_flag = detect_toxicity(pipes["safety_tokenizer"], pipes["safety_model"], text)

        return (max_sim >= threshold) or toxic_flag
    except Exception:
        return False

# Enhanced intent detection function
def detect_intent(text):
    """Detect user intent from text to provide more appropriate responses."""
    text_lower = text.lower()
    
    # Gratitude/Thanks patterns
    gratitude_patterns = [
        "thank", "thanks", "grateful", "appreciate", "helped", "better", 
        "positive advice", "good advice", "feel better", "that helps",
        "you're right", "makes sense", "i understand"
    ]
    
    # Question/Advice seeking patterns
    advice_patterns = [
        "what should i do", "should i", "any advice", "help me", "how do i",
        "what can i", "how can i", "suggestions", "recommend", "guidance"
    ]
    
    # Emotional support seeking patterns
    support_patterns = [
        "feeling", "feel", "sad", "depressed", "anxious", "worried", "scared",
        "upset", "angry", "frustrated", "lonely", "overwhelmed", "stressed"
    ]
    
    # Closure/Ending patterns
    closure_patterns = [
        "goodbye", "bye", "see you", "talk later", "that's all", "i'm done",
        "nothing else", "i'm good", "i'm okay now"
    ]
        
    # Check patterns in order of priority
    if any(pattern in text_lower for pattern in gratitude_patterns):
        return "gratitude"
    elif any(pattern in text_lower for pattern in advice_patterns):
        return "advice_seeking"
    elif any(pattern in text_lower for pattern in closure_patterns):
        return "closure"
    elif any(pattern in text_lower for pattern in support_patterns):
        return "emotional_support"
    else:
        return "general"

# Enhanced response generation based on intent
def generate_contextual_response(intent, text_emotion, user_text, aspects):
    """Generate more contextually appropriate responses based on intent."""
    
    if intent == "gratitude":
        gratitude_responses = [
            "You're very welcome! I'm glad I could help.",
            "I'm so happy that was helpful for you.",
            "You're welcome! Remember, I'm here if you need more support.",
            "I'm glad you found that useful. Take care of yourself!",
            "You're welcome! It's wonderful to hear you're feeling a bit better.",
        ]
        return gratitude_responses[hash(user_text) % len(gratitude_responses)]
    
    elif intent == "advice_seeking":
        if "exam" in user_text.lower() or "test" in user_text.lower() or "marks" in user_text.lower():
            return "Consider reviewing what went well and what you can improve for next time. Remember, each test is a learning opportunity."
        elif "relationship" in user_text.lower() or "friend" in user_text.lower():
            return "Communication is key in relationships. Consider having an open, honest conversation about how you're feeling."
        else:
            return "It might help to break down the situation and consider your options. What feels most important to you right now?"
    
    elif intent == "closure":
        return "Take care of yourself, and remember I'm here if you need support again."
    
    elif intent == "emotional_support":
        if text_emotion in ["sad", "sadness"]:
            return "I can see you're going through a difficult time. Your feelings are completely valid."
        elif text_emotion in ["angry", "anger"]:
            return "It's understandable to feel frustrated. Would you like to talk about what's causing these feelings?"
        elif text_emotion in ["fear", "anxiety"]:
            return "Feeling anxious can be really overwhelming. Remember to take things one step at a time."
        else:
            return "I'm here to listen and support you through whatever you're experiencing."
    
    else:  # general
        return "I'm here to support you. How are you feeling right now?"

# Updated build_prompt function with better context awareness
def build_prompt_enhanced(
    user_text,
    prev_user_messages,
    text_emotion,
    text_emotion_score,
    sentiment,
    sentiment_score,
    aspects,
    intent,
    rag_docs=None
):
    """Enhanced prompt building with intent awareness."""
    
    # Get a contextual base response
    contextual_response = generate_contextual_response(intent, text_emotion, user_text, aspects)
    
    # Build conversation history
    history = ""
    if prev_user_messages:
        history = "\n".join([f"User: {m}" for m in prev_user_messages[-2:]]) + "\n"
    
    # RAG context
    rag_context = ""
    if rag_docs:
        rag_context = "\n".join([f"- {h['doc']}" for h in rag_docs])
    
    # Create context-aware prompt
    if intent == "gratitude":
        prompt = (
            "You are a supportive mental health assistant. The user is expressing gratitude. "
            "Respond warmly and encouragingly. Keep response to 1 sentence.\n\n"
            f"Recent conversation:\n{history}"
            f"User: {user_text}\n"
            f"Suggested response style: {contextual_response}\n"
            "Assistant:"
        )
    elif intent == "advice_seeking":
        prompt = (
            "You are a supportive mental health assistant. The user is seeking advice. "
            "Provide helpful, practical guidance. Keep response to 1-2 sentences.\n\n"
            f"Context: User emotion is {text_emotion}, sentiment is {sentiment}\n"
            f"Relevant knowledge:\n{rag_context}\n"
            f"Recent conversation:\n{history}"
            f"User: {user_text}\n"
            "Assistant:"
        )
    else:
        prompt = (
            "You are a supportive mental health assistant. "
            "Provide empathetic support. Keep response to 1-2 sentences.\n\n"
            f"Context: User emotion is {text_emotion}, intent is {intent}\n"
            f"Relevant knowledge:\n{rag_context}\n"
            f"Recent conversation:\n{history}"
            f"User: {user_text}\n"
            "Assistant:"
        )
    
    return prompt

# RAG helpers
@st.cache_resource(show_spinner=False)
def build_faiss_index(_embed_model, docs):
    emb = _embed_model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, emb

def retrieve_rag(embed_model, index, docs, query, top_k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)
    results = []
    for i, score in zip(I[0], D[0]):
        if i < len(docs):
            results.append({"id": str(i), "doc": docs[i], "score": float(score)})
    return results

_previous_responses = []

def soft_duplicate_filter(embed_model, reply, sim_threshold=0.92):
    """Avoid repeating same response verbatim or semantically."""
    global _previous_responses
    
    original_reply = reply
    
    if _previous_responses:
        # Exact match check
        if reply.strip() == _previous_responses[-1].strip():
            # Instead of adding text, generate a slight variation
            variations = [
                "I hear you and want you to know that I'm here for you.",
                "Your feelings are valid, and I'm here to listen.",
                "I understand this is difficult, and you don't have to go through it alone.",
                "It's okay to feel this way, and I'm here to support you."
            ]
            # Pick a variation that wasn't used recently
            for var in variations:
                if var not in _previous_responses[-3:]:
                    reply = var
                    break
        else:               
            # Semantic similarity check
            try:
                current_emb = embed_model.encode(reply, convert_to_tensor=True)
                prev_embs = embed_model.encode(_previous_responses, convert_to_tensor=True)
                sims = F.cosine_similarity(current_emb, prev_embs)
                if torch.max(sims).item() >= sim_threshold:
                    # Generate a variation instead of adding text
                    if "sad" in original_reply.lower():
                        reply = "I can see you're going through a tough time right now."
                    elif "understand" in original_reply.lower():
                        reply = "Your feelings matter, and it's okay to experience them."
                    else:
                        reply = "I'm here to listen and support you through this."
            except Exception:
                pass  # If embedding fails, keep original reply
    
    # Update history
    _previous_responses.append(reply)
    if len(_previous_responses) > 5:
        _previous_responses.pop(0)
    
    return reply

def generate_from_model(tokenizer, model, prompt, max_new_tokens=150, temperature=0.7, top_p=0.9):
    """Generate response from active model with safety + deduplication."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3
    )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Strip the prompt prefix if included
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    
    # More comprehensive text cleaning
    text = re.sub(r'Assistant:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Bot:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Response:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?\].*', '', text, flags=re.IGNORECASE | re.DOTALL)  # Remove anything starting with [
    text = re.sub(r'User:\s*.*', '', text, flags=re.IGNORECASE | re.DOTALL)  # Remove User: and everything after
    text = re.sub(r'\*\*.*?\*\*', '', text, flags=re.DOTALL)  # Remove **bold** text
    text = re.sub(r'A:\s*.*', '', text, flags=re.IGNORECASE | re.DOTALL)  # Remove A: responses
    text = re.sub(r'Solution \d+:.*', '', text, flags=re.IGNORECASE | re.DOTALL)  # Remove Solution patterns
    text = re.sub(r'Instruction \d+:.*', '', text, flags=re.IGNORECASE | re.DOTALL)  # Remove Instruction patterns
    text = re.sub(r'---.*', '', text, flags=re.DOTALL)  # Remove everything after ---
    
    # Remove quotes at the beginning and end if they exist
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    elif text.startswith('"'):
        text = text[1:].strip()
    elif text.endswith('"'):
        text = text[:-1].strip()
    
    # Split by newlines and take only the first meaningful response
    lines = text.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Stop if we encounter patterns that indicate meta-text
        if any(pattern in line.lower() for pattern in ['solution', 'instruction', 'user:', 'bot:', 'assistant:', '[', 'a:']):
            break
        clean_lines.append(line)
    
    # Join the clean lines
    text = ' '.join(clean_lines).strip()
    
    # If text is too long, truncate at sentence boundary
    if len(text) > 300:
        sentences = text.split('.')
        truncated = []
        char_count = 0
        for sentence in sentences:
            if char_count + len(sentence) > 300:
                break
            truncated.append(sentence)
            char_count += len(sentence) + 1
        text = '.'.join(truncated).strip()
        if text and not text.endswith('.'):
            text += '.'
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = text.strip()
    
    # Fallback if text is empty or too short
    if not text or len(text) < 5:
        text = "I understand how you're feeling. I'm here to support you."
    
    return text

CRISIS_MESSAGE = (
    "💛 I’m concerned about your safety. I can’t assist with that here. Please contact local emergency services "
    "or a crisis helpline right now.\n\nIf in India: AASRA +91-9820466726\nUS: 988\nUK: Samaritans 116 123"
)

def generate_response_pipeline_enhanced(pipes, tokenizer, llm_model, embed_model, user_text, prev_user_messages):
    """Enhanced response pipeline with intent detection."""
    
    # Safety check on raw user text first
    if is_unsafe_message(embed_model, user_text):
        return CRISIS_MESSAGE, "distressed", 1.0, "negative", [{"aspect":"safety","sentiment":"negative","confidence":1.0}]

    # Enhanced analysis
    text_emotion, text_emotion_score = detect_text_emotion(pipes["emotion"], user_text)
    sentiment_label, sentiment_score = detect_sentiment(pipes["sentiment"], user_text)
    aspects = detect_absa(pipes["absa"], user_text)
    intent = detect_intent(user_text)  # New intent detection

    # RAG for advice-seeking intents
    rag_docs = None
    if intent == "advice_seeking":
        rag_docs = retrieve_rag(embed_model, index, docs, user_text, top_k=3)

    # For gratitude, use a direct contextual response
    if intent == "gratitude":
        final_reply = generate_contextual_response(intent, text_emotion, user_text, aspects)
        return final_reply, text_emotion, text_emotion_score, sentiment_label, aspects

    # Build enhanced prompt
    prompt = build_prompt_enhanced(
        user_text,
        prev_user_messages,
        text_emotion,
        text_emotion_score,
        sentiment_label,
        sentiment_score,
        aspects,
        intent,
        rag_docs=rag_docs
    )

    # Generate response
    try:
        generated = generate_from_model(tokenizer, llm_model, prompt, max_new_tokens=150, temperature=0.7, top_p=0.9)
    except Exception as e:
        return "Sorry, something went wrong generating a reply.", text_emotion, text_emotion_score, sentiment_label, aspects

    # Safety check on generated text
    if is_unsafe_message(embed_model, generated):
        return CRISIS_MESSAGE, text_emotion, text_emotion_score, sentiment_label, aspects

    # Apply duplicate filter
    final_reply = soft_duplicate_filter(embed_model, generated)
    
    return final_reply, text_emotion, text_emotion_score, sentiment_label, aspects

# -----------------------------------------------------------
# Load default models / pipelines
# -----------------------------------------------------------
with st.spinner("Loading models (may take ~1min)..."):
    tokenizer, llm_model, active_llm = load_llm()
    pipes = load_pipelines()

st.sidebar.title("Prototype Controls")
st.sidebar.write(f"Active LLM: **{active_llm}**")
st.sidebar.write(f"Embed model: **{EMBED_MODEL}**")
st.sidebar.write("Tip: Large LLMs might be slow on free Spaces — use smaller models or HF Inference API for production.")

# Upload or use fallback RAG docs
st.sidebar.header("RAG Knowledge Base")
uploaded = st.sidebar.file_uploader("Upload CSV/XLSX with 'Knowledge Entry' column (optional)", type=["csv","xlsx"])
if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df_rag = pd.read_csv(uploaded)
        else:
            df_rag = pd.read_excel(uploaded)
        if "Knowledge Entry" not in df_rag.columns:
            st.sidebar.error("File must have 'Knowledge Entry' column")
            docs = ["If you feel overwhelmed, try slow breathing: inhale 4s, hold 2s, exhale 6s."]
        else:
            docs = df_rag["Knowledge Entry"].dropna().astype(str).tolist()
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")
        docs = ["If you feel overwhelmed, try slow breathing: inhale 4s, hold 2s, exhale 6s."]
else:
    # fallback minimal KB
    docs = [
        "If you feel overwhelmed, try slow breathing: inhale 4s, hold 2s, exhale 6s.",
        "For exam stress, break tasks into 25-minute focus blocks (Pomodoro).",
        "Reach out to a friend or counselor when you feel isolated."
    ]

# Build FAISS index
index, _ = build_faiss_index(pipes["embed"], docs)

# -----------------------------------------------------------
# Main UI layout
# -----------------------------------------------------------
col1, col2 = st.columns((2,3))

with col1:
    st.header("Chat with the Prototype")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Type how you're feeling or ask for advice..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Analyzing..."):
            prev_user_messages = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
            reply, te, tes, sent, aspects = generate_response_pipeline_enhanced(
                pipes, tokenizer, llm_model, pipes["embed"], prompt, prev_user_messages
            )

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(reply)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": reply})

            with st.expander("Analysis (sentiment / emotion / ABSA / RAG)"):
                st.write("Sentiment:", sent, f"({tes:.2f})")
                st.write("Emotion:", te, f"({tes:.2f})")
                st.write("ABSA:", aspects)
                rag_hits = retrieve_rag(pipes["embed"], index, docs, prompt, top_k=3)
                st.write("RAG hits:", rag_hits)

with col2:
    st.header("Prototype Overview")
    st.markdown("""
    **What this demo shows:**  
    - Multimodal-aware pipeline: sentiment, emotion, ABSA, safety checks, RAG retrieval + LLM generation.  
    - Small knowledge base is used for grounding responses; upload your own CSV/XLSX with `Knowledge Entry` column to test.  
    """)
    st.subheader("Tech stack (visible to judges)")
    st.write("- Python, Streamlit, PyTorch")
    st.write("- HuggingFace Transformers (LLMs & pipelines)")
    st.write("- SentenceTransformers + FAISS (RAG)")
    st.write("- Deployed as Hugging Face Space (Streamlit)")

    st.subheader("Notes & Warnings")
    st.info("""
    • This demo uses smaller LLM defaults to fit the free Space environment.
    • For heavy models (phi-3, gemma), prefer using the Hugging Face Inference API or a GPU-backed paid plan.
    • The system is **not** a substitute for professional help. In real deployments, integrate verified helpline flows and human-in-the-loop review.
    """)

    st.subheader("Quick checklist for judges")
    st.write("1. Type a message in the chat input and send.")
    st.write("2. Expand the Analysis panel to see sentiment/emotion/ABSA and RAG hits.")
    st.write("3. Upload an Excel/CSV with `Knowledge Entry` to test RAG grounding.")

# Footer
st.write("---")
st.caption("Prototype for hackathon submission. Built by Mitul Srivastava — AI for mental health demo.")