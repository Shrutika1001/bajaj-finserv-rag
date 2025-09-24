**Bajaj Finserv RAG Chatbot (Q1 FY26)**

**Live Demo:** https://huggingface.co/spaces/Shrutika01/bajaj-chatbot

Ask questions about **Bajaj Finserv’s Q1 FY26** using a lightweight **RAG** pipeline that runs on free CPU (Hugging Face Spaces).

---
# Features

* Answers grounded in the **two PDFs** only (no internet).
* Dense retrieval (E5-small), extractive QA, generative fallback (FLAN-T5).
* Memory-friendly: lazy loading, sentence-level ranking, no vector DB.

---

## Tech Stack

* Python, PyTorch
* sentence-transformers, transformers
* Gradio (UI + API)
* Hugging Face Spaces (deployment)

---

## Example Questions

* State the combined ratio reported by BAGIC in Q1 FY26.
* What was Bajaj Finserv’s consolidated PAT growth in Q1 FY26?
* What is BALIC’s NBM in Q1 FY26?
* Report BALIC AUM in Q1 FY26.
* Summarize Bajaj Finance performance in Q1 FY26.

---

## How It Works

1. Retrieve top passages with E5-small.
2. Refine at sentence level (similarity + lexical + numeric cues).
3. Answer with extractive QA; if uncertain, fallback to FLAN-T5 for a concise summary.
4. Return the answer (+ debug sources in UI).

