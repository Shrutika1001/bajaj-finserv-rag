import os, json, re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import gradio as gr

# Be gentle on free CPU envs
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ------------------ Lazy globals ------------------
e5 = None
qa_extractive = None
gen_model = None
gen_tok = None
emb_matrix = None
index_data: List[Dict[str, Any]] = None

# Smaller, light models (decent accuracy on CPU)
E5_MODEL = "intfloat/e5-small-v2"
EXTRACTIVE_MODEL = "distilbert-base-uncased-distilled-squad"
GEN_MODEL = "google/flan-t5-base"

# ------------------ Regexes ------------------
# currency: ₹ 1,234.56 ; Rs. ; crore/Cr
CURR_RE = re.compile(r'(₹\s?[0-9][0-9,\.]*\s*(?:crore|cr)?)|(rs\.?\s?[0-9][0-9,\.]*\s*(?:crore|cr)?)', re.IGNORECASE)
CRORE_RE = re.compile(r'([0-9][0-9,\.]*)\s*(?:crore|cr)\b', re.IGNORECASE)

# percent (robust: %, percent, per cent)
PCT_RE = re.compile(
    r'(?<![A-Za-z0-9])(?:\d{1,3}(?:[.,]\d+)?\s*%|\d{1,3}(?:[.,]\d+)?\s*per\s*cent|\d{1,3}(?:[.,]\d+)?\s*percent)',
    re.IGNORECASE
)

# generic number
NUM_RE = re.compile(r'\b\d{1,3}(?:[,\d]{0,3})*(?:\.\d+)?\b')

# ------------------ Sentence splitting ------------------
def split_into_sentences(text: str) -> List[str]:
    """
    Use blingfire if available for robust, fast sentence splitting.
    Fallback to a simple regex otherwise.
    """
    try:
        import blingfire as bf
        s = bf.text_to_sentences(text or "")
        return [t.strip() for t in s.split("\n") if t.strip()]
    except Exception:
        # crude but serviceable fallback
        parts = re.split(r'(?<=[\.\?\!])\s+', text or "")
        return [p.strip() for p in parts if p.strip()]

# ------------------ Tiny utilities ------------------
def _norm_num_str(s: str) -> str:
    return re.sub(r'\s+', '', (s or '').lower())

def _dedupe_ordered(vals: List[str]) -> List[str]:
    seen = set()
    out = []
    for v in vals:
        k = _norm_num_str(v)
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out

def same_unit(a: str, b: str) -> bool:
    a_pct = '%' in a or 'percent' in a.lower() or 'per cent' in a.lower()
    b_pct = '%' in b or 'percent' in b.lower() or 'per cent' in b.lower()
    if a_pct and b_pct:
        return True
    a_cur = ('₹' in a) or ('crore' in a.lower()) or (' cr' in a.lower()) or ('rs' in a.lower())
    b_cur = ('₹' in b) or ('crore' in b.lower()) or (' cr' in b.lower()) or ('rs' in b.lower())
    return a_cur and b_cur

def find_percents(text: str) -> List[str]:
    vals = [m.group(0) for m in PCT_RE.finditer(text or "")]
    # normalize wording variants to use '%'
    vals = [v.replace("percent", "%").replace("per cent", "%") for v in vals]
    return _dedupe_ordered(vals)

def find_currency(text: str) -> List[str]:
    out = []
    out += [m.group(0) for m in CURR_RE.finditer(text or "")]
    out += [m.group(0) + " crore" if not m.group(0).lower().endswith("crore") else m.group(0)
            for m in CRORE_RE.finditer(text or "")]
    return _dedupe_ordered(out)

def _percent_numbers(vals: List[str]) -> List[float]:
    out = []
    for v in vals:
        if '%' in v or 'percent' in v.lower() or 'per cent' in v.lower():
            num = re.sub(r'[^0-9.,]', '', v).replace(',', '')
            if num:
                try:
                    out.append(float(num))
                except:
                    pass
    return out

def guess_preferred_unit(query: str) -> str | None:
    q = (query or "").lower()
    percent_cues = [
        "gnpa", "gross npa", "npa", "net npa", "npl",
        "ratio", "roe", "roa", "nim",
        "combined ratio", "loss ratio", "expense ratio",
        "margin", "vnb margin", "nbm", "%"
    ]
    currency_cues = [
        "inr", "₹", "rs", "crore", "cr",
        "revenue", "income", "pat", "profit", "ebitda", "pbt",
        "gwp", "premium", "aum", "assets under management"
    ]
    if any(c in q for c in percent_cues):
        return "percent"
    if any(c in q for c in currency_cues):
        return "currency"
    return None

# ------------------ Models (lazy) ------------------
def lazy_import_models():
    global e5, qa_extractive, gen_model, gen_tok
    if e5 is None or qa_extractive is None or gen_model is None:
        from sentence_transformers import SentenceTransformer
        from transformers import pipeline, AutoTokenizer
        import torch
        torch.set_num_threads(1)

        if e5 is None:
            e5 = SentenceTransformer(E5_MODEL)
        if qa_extractive is None:
            qa_extractive = pipeline("question-answering", model=EXTRACTIVE_MODEL, framework="pt", device=-1)
        if gen_model is None:
            gen_model = pipeline("text2text-generation", model=GEN_MODEL, framework="pt", device=-1)
            gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL)

def e5_embed_passages(texts: List[str]) -> np.ndarray:
    lazy_import_models()
    return e5.encode([f"passage: {t}" for t in texts], convert_to_numpy=True, normalize_embeddings=True)

def e5_embed_query(q: str) -> np.ndarray:
    lazy_import_models()
    return e5.encode([f"query: {q}"], convert_to_numpy=True, normalize_embeddings=True)[0]

# ------------------ Load index (once) ------------------
INDEX_PATH = Path("index/index.json")
assert INDEX_PATH.exists(), "index/index.json is missing. Upload it in a folder named 'index'."

with open(INDEX_PATH, "r", encoding="utf-8") as f:
    index_data = json.load(f)

def get_emb_matrix() -> np.ndarray:
    global emb_matrix
    if emb_matrix is None:
        passage_texts = [rec["text"] for rec in index_data]
        emb_matrix = e5_embed_passages(passage_texts)
    return emb_matrix

# ------------------ Retrieval ------------------
def cosine_sim_batch(qv: np.ndarray, M: np.ndarray) -> np.ndarray:
    return M @ qv  # normalized vectors → dot = cosine

def retrieve_top_passages(query: str, pool: int = 80, k: int = 8) -> List[Dict[str, Any]]:
    qv = e5_embed_query(query)
    M = get_emb_matrix()
    scores = cosine_sim_batch(qv, M)
    idxs = np.argpartition(scores, -pool)[-pool:]
    idxs = idxs[np.argsort(scores[idxs])[::-1]]
    out = []
    for i in idxs[:pool]:
        rec = index_data[i]
        out.append({
            "score": float(scores[i]),
            "source": rec.get("source", ""),
            "page": rec.get("page", -1),
            "text": rec.get("text", ""),
            "idx": int(i)
        })
    return out[:k]

# ------------------ Context construction ------------------
def refine_with_sentences(query: str, passages: List[Dict[str, Any]], top_sents: int = 16) -> List[str]:
    """
    Build a list of candidate sentences from top passages.
    Heuristic: prefer sentences that (a) contain query tokens and (b) contain a number/percent/currency.
    """
    q = (query or "").lower()
    # soft keyword tokens
    kw_tokens = re.findall(r"[a-zA-Z]{4,}", q)
    out = []

    for p in passages:
        sents = split_into_sentences(p["text"])
        for s in sents:
            sl = s.lower()
            has_kw = any(t in sl for t in kw_tokens) if kw_tokens else False
            has_num = bool(PCT_RE.search(s) or CURR_RE.search(s) or CRORE_RE.search(s) or NUM_RE.search(s))
            # score: keywords + numbers
            score = (2.0 if has_kw else 0.0) + (1.0 if has_num else 0.0)
            # tiny bonus if shortish sentence (typically charts/captions)
            if len(s) < 180:
                score += 0.2
            out.append((score, s))

    out.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in out[:top_sents]]

# ------------------ Trend builders ------------------
def build_trend_from_nearby(sentences: List[str], preferred_unit: str | None = None) -> str:
    def vals_in(s: str) -> List[str]:
        if preferred_unit == "percent":
            v = find_percents(s)
            if v: return v
        if preferred_unit == "currency":
            v = find_currency(s)
            if v: return v
        # generic
        return find_percents(s) + find_currency(s)

    trend_words = ["up from", "down from", "vs", "versus", "compared to", "compared with", "from"]

    # within one sentence
    for s in sentences:
        vals = vals_in(s)
        if len(vals) >= 2:
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    if same_unit(vals[i], vals[j]):
                        sl = s.lower()
                        if any(w in sl for w in trend_words):
                            if "up from" in sl or "down from" in sl or "from" in sl:
                                cur, prev = vals[-1], vals[0]
                                direction = "up from" if "up from" in sl else ("down from" if "down from" in sl else "from")
                                return f"{cur} {direction} {prev}"
                            for k in ["vs", "versus", "compared to", "compared with"]:
                                if k in sl:
                                    return f"{vals[0]} {k} {vals[1]}"
                            return f"{vals[0]} vs {vals[1]}"

    # across adjacent sentences
    for i in range(len(sentences) - 1):
        s1, s2 = sentences[i], sentences[i + 1]
        vals1, vals2 = vals_in(s1), vals_in(s2)
        if vals1 and vals2:
            for a in vals1:
                for b in vals2:
                    if same_unit(a, b):
                        joined = (s1 + " " + s2).lower()
                        if "up from" in joined or "down from" in joined or "from" in joined:
                            cur, prev = b, a
                            direction = "up from" if "up from" in joined else ("down from" if "down from" in joined else "from")
                            return f"{cur} {direction} {prev}"
                        for k in ["vs", "versus", "compared to", "compared with"]:
                            if k in joined:
                                return f"{a} {k} {b}"
                        return f"{a} vs {b}"

    # single-value fallbacks honoring preferred unit
    if preferred_unit == "percent":
        for s in sentences:
            v = find_percents(s)
            if v: return v[0]
    if preferred_unit == "currency":
        for s in sentences:
            v = find_currency(s)
            if v: return v[0]

    # generic
    for s in sentences:
        v = find_percents(s)
        if v: return v[0]
    for s in sentences:
        v = find_currency(s)
        if v: return v[0]
    return ""

def coerce_percent_answer(query: str, sentences: List[str]) -> str | None:
    """When a ratio-type question accidentally picks ₹, force a percent-only answer."""
    vals = []
    for s in sentences:
        vals.extend(find_percents(s))
    vals = _dedupe_ordered(vals)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    # choose last as current, previous as prior
    cur, prev = vals[-1], vals[-2]
    nums = _percent_numbers([prev, cur])
    if len(nums) == 2:
        direction = "up from" if nums[1] > nums[0] else ("down from" if nums[1] < nums[0] else "from")
        return f"{cur} {direction} {prev}"
    return f"{cur} vs {prev}"

# ------------------ QA helpers ------------------
def answer_extractive(question: str, contexts: List[str], max_chars: int = 1600) -> str | None:
    if not contexts:
        return None
    lazy_import_models()
    ctx = " ".join(contexts)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars]
    try:
        out = qa_extractive({"question": question, "context": ctx})
        ans = (out.get("answer") or "").strip()
        return ans or None
    except Exception:
        return None

def answer_generative(question: str, contexts: List[str], max_input_tokens: int = 460, max_words: int = 90) -> str:
    lazy_import_models()
    ctx = " ".join(contexts)
    # build short prompt
    prompt = f"Answer the question using only the context.\n\nQuestion: {question}\n\nContext:\n{ctx}\n\nAnswer:"
    out = gen_model(prompt, max_new_tokens=180, do_sample=False)[0]["generated_text"]
    # trim words
    words = out.split()
    if len(words) > max_words:
        out = " ".join(words[:max_words]) + "..."
    return out.strip()

def format_nice_sentence(query: str, span: str) -> str:
    q = (query or "").strip().rstrip("?")
    if not span:
        return "Not available in the provided documents."
    # Simple subjectizer for nicer reads
    subj = None
    ql = q.lower()
    if "gnpa" in ql or "gross npa" in ql or "npa" in ql:
        subj = "GNPAs for BFL for the latest quarter are"
    elif "combined ratio" in ql:
        subj = "Combined ratio for the latest quarter is"
    elif "aum" in ql:
        subj = "AUM is"
    elif "pat" in ql:
        subj = "Profit after tax is"
    if subj:
        return f"{subj} {span}"
    return f"{q}: {span}"

# ------------------ RAG answer ------------------
def rag_answer(query: str) -> Dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return {"answer": "Please ask a question.", "sources": []}

    # 1) retrieve
    passages = retrieve_top_passages(query, pool=80, k=8)
    if not passages:
        return {"answer": "Not available in the provided documents.", "sources": []}

    # 2) build candidate sentences
    contexts_raw = refine_with_sentences(query, passages, top_sents=18)

    # 2b) focus sentences to contain query tokens (esp. GNPA/NPA/ratio)
    kw = query.lower()
    query_terms: List[str] = []
    if any(k in kw for k in ["gnpa", "gross npa", "npa"]):
        query_terms = ["gnpa", "gross npa", "npa"]
    elif "combined ratio" in kw:
        query_terms = ["combined ratio", "ratio"]
    if not query_terms:
        query_terms = [t for t in re.findall(r"[a-zA-Z]{4,}", kw)]
    focused_contexts = [s for s in contexts_raw if any(t in s.lower() for t in query_terms)] or contexts_raw

    # 3) decide unit
    preferred_unit = guess_preferred_unit(query)

    # 4) trend builder (unit-aware + keyword-focused)
    trend = build_trend_from_nearby(focused_contexts[:12], preferred_unit=preferred_unit)

    # 5) If ratio-type but trend ended up currency, force percent
    def _looks_currency(s: str) -> bool:
        return bool(re.search(r'(₹|rs\.?|crore|cr\b)', (s or '').lower()))

    if preferred_unit == "percent" and trend and _looks_currency(trend):
        forced = coerce_percent_answer(query, focused_contexts[:12])
        if forced:
            if any(x in kw for x in ["gnpa", "gross npa", "npa"]):
                trend = f"GNPAs for BFL for the latest quarter are {forced}."
            else:
                trend = forced

    # 6) extractive fallback (unit-aware)
    if not trend:
        span = answer_extractive(query, focused_contexts[:12])
        if span:
            if preferred_unit == "percent" and not PCT_RE.search(span):
                alt = coerce_percent_answer(query, focused_contexts[:12])
                if alt:
                    span = alt
            if preferred_unit == "currency" and not (re.search(r'(₹|rs\.?|crore|cr\b)', (span or '').lower())):
                for s in focused_contexts[:12]:
                    v = find_currency(s)
                    if v:
                        span = v[0]; break
            looks_value = bool(PCT_RE.search(span) or re.search(r'(₹|rs\.?|crore|cr\b)', (span or '').lower()))
            final = format_nice_sentence(query, span) if looks_value else span
            srcs = _sources_from_passages(passages, max_n=3)
            return {"answer": final, "sources": srcs}

    if trend:
        final = format_nice_sentence(query, trend)
        srcs = _sources_from_passages(passages, max_n=3)
        return {"answer": final, "sources": srcs}

    # 7) generative backup
    gen = answer_generative(query, focused_contexts[:14], max_input_tokens=460, max_words=90)
    srcs = _sources_from_passages(passages, max_n=3)
    return {"answer": gen, "sources": srcs}

def _sources_from_passages(passages: List[Dict[str, Any]], max_n: int = 3) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for p in passages:
        key = (p.get("source",""), p.get("page",-1))
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": p.get("source",""), "page": p.get("page",-1)})
        if len(out) >= max_n:
            break
    return out

# ------------------ Gradio UI ------------------
EXAMPLE_QUESTIONS = [
    "What is Gross NPAs for Bajaj Finance?",
    "State the combined ratio reported by BAGIC in Q1 FY26.",
    "What is BALIC's new business margin?",
    "List the AUM for BALIC in Q1 FY26.",
    "What was Bajaj Finserv's consolidated profit after tax growth in Q1 FY26?",
    "Summarize Bajaj Finance performance in Q1 FY26."
]

def _ask(user_q: str):
    out = rag_answer((user_q or "").strip())
    ans = out.get("answer","")
    srcs = out.get("sources",[])
    # Pretty-print sources
    if srcs:
        src_txt = "\n".join([f"— {s.get('source','')} (p.{s.get('page','?')})" for s in srcs])
        return f"{ans}\n\n{src_txt}"
    return ans

with gr.Blocks(title="Bajaj Finserv RAG Chatbot") as demo:
    gr.Markdown(
        "# 📊 Bajaj Finserv Q&A (RAG)\n"
        "Ask questions about the **Q1 FY26 Earnings Call Transcript** and **Investor Presentation**.\n"
        "This runs fully on free CPU models."
    )
    with gr.Row():
        q = gr.Textbox(lines=2, label="Your question")
    with gr.Row():
        ask_btn = gr.Button("Ask", variant="primary")
    out = gr.Textbox(lines=8, label="Answer", show_copy_button=True)
    gr.Examples(
        examples=[[e] for e in EXAMPLE_QUESTIONS],
        inputs=[q],
        examples_per_page=6
    )
    ask_btn.click(_ask, inputs=[q], outputs=[out])
    q.submit(_ask, inputs=[q], outputs=[out])



# ===================== Start the Gradio server on HF Spaces =====================
# Using launch() is the simplest, most reliable way on free CPU Spaces.
# - queue() helps avoid timeouts while models load lazily.
# - The SPACE_ID env var exists on Hugging Face Spaces, so we use it to choose ports cleanly.
if "SPACE_ID" in os.environ:
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
else:
    # local dev
    demo.launch()


