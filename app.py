import os, json, re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import gradio as gr

os.environ["TRANSFORMERS_NO_TF"] = "1"  # avoid TF/Keras import on CPU

# ===================== Lazy globals =====================
e5 = None
qa_extractive = None
gen_model = None
gen_tok = None
emb_matrix = None
index_data = None

# Small, RAM-friendly models for free CPU Spaces
E5_MODEL = "intfloat/e5-small-v2"                          # embeddings
EXTRACTIVE_MODEL = "distilbert-base-uncased-distilled-squad"  # span extractor
GEN_MODEL = "google/flan-t5-base"                          # generative fallback

# ===================== Lazy model import =====================
def lazy_import_models():
    """Load models only once, when first needed."""
    global e5, qa_extractive, gen_model, gen_tok
    if e5 is None:
        from sentence_transformers import SentenceTransformer
        from transformers import pipeline, AutoTokenizer
        import torch
        torch.set_num_threads(1)  # be gentle on shared CPU

        e5 = SentenceTransformer(E5_MODEL)
        qa_extractive = pipeline("question-answering", model=EXTRACTIVE_MODEL, framework="pt", device=-1)
        gen_model = pipeline("text2text-generation", model=GEN_MODEL, framework="pt", device=-1)
        gen_tok   = AutoTokenizer.from_pretrained(GEN_MODEL)

def e5_embed_passages(texts: List[str]) -> np.ndarray:
    return e5.encode([f"passage: {t}" for t in texts], convert_to_numpy=True, normalize_embeddings=True)

def e5_embed_query(q: str) -> np.ndarray:
    return e5.encode([f"query: {q}"], convert_to_numpy=True, normalize_embeddings=True)[0]

# ===================== Load index =====================
INDEX_PATH = Path("index/index.json")
assert INDEX_PATH.exists(), "index/index.json is missing. Upload it in a folder named 'index'."

with open(INDEX_PATH, "r", encoding="utf-8") as f:
    index_data: List[Dict[str, Any]] = json.load(f)

def get_emb_matrix() -> np.ndarray:
    """Build embeddings lazily to save startup RAM."""
    global emb_matrix
    if emb_matrix is None:
        lazy_import_models()
        passage_texts = [rec["text"] for rec in index_data]
        emb_matrix = e5_embed_passages(passage_texts)
    return emb_matrix

# ===================== Retrieval helpers =====================
def cosine_sim_batch(qv: np.ndarray, M: np.ndarray) -> np.ndarray:
    return M @ qv

def retrieve_top_passages(query: str, pool: int = 80, k: int = 8):
    """E5 dense retrieval -> top pool, then keep top k."""
    lazy_import_models()
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
            "source": rec["source"],
            "page": rec["page"],
            "text": rec["text"],
            "idx": i
        })
    return out[:k], qv

# ===================== Sentence splitting =====================
try:
    import blingfire as bf
    def sent_split(text: str):
        return [s.strip() for s in bf.text_to_sentences(text or "").split("\n") if s.strip()]
except Exception:
    _bullet_pat = re.compile(r"[•▪●\-\u2013\u2014]\s+")
    _sent_pat   = re.compile(r"(?<=[\.\!\?])\s+")
    def sent_split(text: str):
        if not text: return []
        t = text.replace("\r", "\n")
        t = _bullet_pat.sub(". ", t)
        parts = []
        for line in t.split("\n"):
            line = line.strip()
            if not line: continue
            parts.extend(_sent_pat.split(line))
        return [s.strip() for s in parts if len(s.strip()) >= 6]

# ===================== Context refinement =====================
_NUM_PAT = re.compile(r"(\d{1,3}(?:[\d,]{0,3})+(?:\.\d+)?|\d{1,3}\.\d+)")
_HAS_PCT = re.compile(r"\d{1,3}(\.\d+)?\s*%")
_HAS_CUR = re.compile(r"(₹)\s?[\d,]+(?:\.\d+)?(?:\s*(cr|crore|crores))?", re.I)

def is_numeric_sentence(s: str) -> bool:
    s2 = (s or "").replace("INR", "₹")
    return bool(_HAS_PCT.search(s2) or _HAS_CUR.search(s2) or _NUM_PAT.search(s2))

def tokset(s: str):
    return set(re.findall(r"[a-zA-Z0-9%₹]+", (s or "").lower()))

def _window_join(sent_list, idx, radius=1, max_chars=900):
    start = max(0, idx - radius)
    end   = min(len(sent_list), idx + radius + 1)
    return " ".join(sent_list[start:end])[:max_chars]

def refine_with_sentences(query: str, passages, qv, top_sents: int = 14):
    """Re-split to sentences and score each by sim + lexical + numeric signal."""
    all_sent_groups = []
    for p in passages:
        sents = [s.strip() for s in sent_split(p["text"]) if len(s.strip()) >= 6]
        if not sents: continue
        all_sent_groups.append({"sents": sents, "source": p["source"], "page": p["page"]})
    if not all_sent_groups:
        return passages

    cand_chunks = []
    qtok = tokset(query)
    # embed per-sentence with e5 (fast for small K)
    for grp in all_sent_groups:
        sents = grp["sents"]
        vecs  = e5_embed_passages(sents)
        sims  = vecs @ qv
        for i, s in enumerate(sents):
            lex = len(tokset(s) & qtok)
            is_num = 1.0 if is_numeric_sentence(s) else 0.0
            score = 1.0 * sims[i] + 0.12 * lex + 0.15 * is_num
            chunk = _window_join(sents, i, radius=1)
            cand_chunks.append({
                "text": chunk,
                "source": grp["source"],
                "page": grp["page"],
                "score": float(score)
            })

    # de-dup & keep top N
    seen = set(); uniq=[]
    for c in sorted(cand_chunks, key=lambda x: x["score"], reverse=True):
        key = (c["source"], c["page"], c["text"])
        if key in seen: continue
        seen.add(key); uniq.append(c)
        if len(uniq) >= top_sents: break
    return uniq[:4]

def re_rank(query: str, contexts, top_m: int = 4):
    """Lightweight passthrough (no cross-encoder to save RAM)."""
    return contexts[:top_m] if contexts else contexts

# ===================== QA + postprocess =====================
def expects_percent(q: str):
    ql = (q or "").lower()
    return any(tok in ql for tok in ["%", "percent", "percentage", "ratio", "roe", "margin", "growth", "yoy", "cor", "combined ratio"])

def expects_currency(q: str):
    ql = (q or "").lower()
    return any(tok in ql for tok in ["₹", "inr", "crore", "cr", "aum", "profit", "pat", "revenue", "income"])

def normalize_num(ans: str):
    if not ans: return ""
    a = ans.strip().replace("INR", "₹")
    return re.sub(r"\s+", " ", a)

def looks_like_percent(s: str):
    return bool(re.search(r"\d{1,3}(?:\.\d+)?\s*%", s or ""))

def looks_like_currency(s: str):
    return bool(re.search(r"(?:₹)\s?[\d,]+(?:\.\d+)?(?:\s*(?:cr|crore|crores))?", s or "", flags=re.I))

def pick_single_value(ans: str, want_pct: bool, want_cur: bool) -> str:
    if not ans: return ""
    s = ans.replace("INR", "₹")
    if want_pct:
        m = re.findall(r"\d{1,3}(?:\.\d+)?\s*%", s)
        if m:
            # choose median to avoid outliers
            vals = sorted((float(x.replace("%","").strip()), x) for x in m)
            return vals[len(vals)//2][1]
    if want_cur:
        m = re.findall(r"(?:₹)\s?[\d,]+(?:\.\d+)?(?:\s*(?:cr|crore|crores))?", s, flags=re.I)
        if m:
            m.sort(key=len, reverse=True)
            return m[0]
    m = re.search(r"\d{1,3}(?:\.\d+)?\s*%", s)
    if m: return m.group(0)
    m = re.search(r"(?:₹)\s?[\d,]+(?:\.\d+)?(?:\s*(?:cr|crore|crores))?", s, flags=re.I)
    if m: return m.group(0)
    return s.strip()

def consensus_qa(question: str, contexts, min_conf: float = 0.18):
    """Try extractive QA across contexts; vote + type filter (percent/currency) if needed."""
    candidates = []
    for ctx in contexts[:3]:
        res = qa_extractive(question=question, context=ctx["text"])
        ans = normalize_num(res.get("answer",""))
        score = float(res.get("score",0))
        if ans and score >= min_conf:
            candidates.append((ans, score))
    if not candidates:
        return None
    want_pct = expects_percent(question)
    want_cur = expects_currency(question)

    # type filter
    typed = candidates
    if want_pct:
        only_pct = [(a,s) for a,s in candidates if looks_like_percent(a)]
        if only_pct: typed = only_pct
    if want_cur and typed:
        only_cur = [(a,s) for a,s in typed if looks_like_currency(a)]
        if only_cur: typed = only_cur

    from collections import Counter
    counts = Counter(a for a,_ in typed)
    top_ans, _ = counts.most_common(1)[0]
    return pick_single_value(top_ans, want_pct, want_cur)

def answer_generative(question: str, contexts, max_input_tokens: int = 420, max_words: int = 90) -> str:
    """Short abstractive answer if extractive fails."""
    context = "\n\n".join([c["text"] for c in contexts[:2]])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer clearly in one or two sentences using exact figures:"
    enc = gen_tok(prompt, truncation=True, max_length=max_input_tokens, return_tensors="pt")
    out = gen_model.model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=180,
        do_sample=False
    )
    ans = gen_tok.decode(out[0], skip_special_tokens=True).strip()
    words = ans.split()
    return (" ".join(words[:max_words]) + "...") if len(words) > max_words else ans

def rag_answer(query: str):
    lazy_import_models()
    passages, qv = retrieve_top_passages(query, pool=80, k=8)
    if not passages:
        return {"answer": "Not available in the provided documents.", "sources": []}
    contexts = refine_with_sentences(query, passages, qv, top_sents=14)
    contexts = re_rank(query, contexts, top_m=4)
    ans = consensus_qa(query, contexts, min_conf=0.18)
    if not ans:
        ans = answer_generative(query, contexts, max_input_tokens=420, max_words=90)
    sources = [{"source": c["source"], "page": c["page"]} for c in contexts]
    return {"answer": ans, "sources": sources}

# ===================== UI =====================
with gr.Blocks(title="Bajaj Finserv RAG Chatbot") as demo:
    gr.Markdown("# 💬 Bajaj Finserv Doc Chatbot")
    gr.Markdown("Ask about Q1 FY26 KPIs from the Investor Presentation and Earnings Call.")
    with gr.Row():
        q = gr.Textbox(label="Your question", placeholder="e.g., State the combined ratio reported by BAGIC in Q1 FY26.", scale=4)
        btn = gr.Button("Ask", variant="primary", scale=1)
    answer = gr.Textbox(label="Answer")
    sources = gr.JSON(label="Sources (debug)")
    def _ask(user_q):
        out = rag_answer((user_q or "").strip())
        return out["answer"], out["sources"]
    btn.click(_ask, inputs=[q], outputs=[answer, sources])

    # Optional programmatic endpoint
    api = gr.Interface(fn=lambda x: rag_answer(x), inputs=gr.Textbox(label="query"), outputs=gr.JSON(), title="API")

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