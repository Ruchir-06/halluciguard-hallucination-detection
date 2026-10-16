# HalluciGuard

RAG pipeline with hallucination detection: retrieval (ChromaDB) → MI pruning → local Ollama generation → sentence-level NLI verification (DeBERTa) → attention attribution → hierarchical trust score. FastAPI backend, React dashboard, SQLite storage. CPU only, Python 3.11.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama serve &            # local LLM server
ollama pull qwen2.5:3b
cp .env.example .env      # optional overrides (OLLAMA_MODEL, OLLAMA_HOST)
python scripts/seed_corpus.py
uvicorn app.main:app --port 8000
cd frontend && npm install && npm run dev    # in a second terminal, then open http://localhost:5173
cd ..
python scripts/run_adversarial_eval.py       # optional: populate the Robustness tab
pytest
```

Generation runs locally through Ollama (`OLLAMA_MODEL=qwen2.5:3b`, `OLLAMA_HOST=http://localhost:11434`). No cloud LLM or API key is used.

## LLM cache and logprobs

- Ollama returns per-token logprobs, so `logprobs_source` is `tokens` and the token-probability trust component is used. If a model returns none, the component is null and the weights renormalize.
- `LLM_CACHE` (default `true`, env override): `generate()` caches responses in `cache/llm/<sha256(model+prompt)>.json` and logs cache hit/miss plus the count of real API calls.
- `POST /evaluate` and `python scripts/run_eval.py` accept:
  - `retrieval_only` / `--retrieval-only`: retrieval and pruning metrics only, no generation.
  - `rescore` / `--rescore`: reuse the latest stored answer per question from SQLite and rerun NLI, localization and trust only.
  - `limit` / `--limit N`: first N questions.
  The response includes `api_calls` (real Ollama requests made).

## Marking the wrong parts

Besides the sentence-level NLI label, each sentence is checked word-by-word: numbers (and years) that are not in the passage the sentence is about, and names that appear in no retrieved passage, are marked bright red inside the sentence, and the sentence counts as flagged. Caveat: a legitimate number that comes from a *different* retrieved passage than the sentence's main topic can be marked.

## Speed

The API preloads the embedder, NLI model and Ollama model at startup (`WARMUP=0` disables it) and asks Ollama to keep the model loaded (`OLLAMA_KEEP_ALIVE`, default 30m). Expect roughly 8 seconds per `/query` with `qwen2.5:3b` on CPU (about 5s generation, 3s NLI).

## Demo: provoking hallucinations

Set `PROMPT_MODE=loose` (default `strict`) to let the generator add outside knowledge, so unsupported sentences show up red in the dashboard. Do not mix loose-mode runs with the strict-mode evaluation numbers below.

Faulty mode: tick "Faulty mode" in the Query tab (or send `"inject_faults": true` to `/query`) to deliberately corrupt 1-2 sentences of the real answer and watch the checker catch them. These responses are tagged and excluded from evaluation.

## Final results

Corpus: 6 renewable-energy documents (12 chunks). Eval set: 22 QA pairs (`data/eval/qa_eval_set.json`). Strict prompt, no faults injected.

Benchmark (`python scripts/compare_generators.py`, saved in `outputs/generator_comparison.json`; cache off, models pre-warmed):

| Model | Hallucination rate | Avg trust | Avg token-prob | Avg generation latency |
| --- | --- | --- | --- | --- |
| Ollama `qwen2.5:3b` | 8.7% (2 of 23 claim sentences) | 0.90 | 0.94 | 2.5 s |

The Ollama row uses the current checker (wrong-span flagging, filler sentences excluded from the counts).

Retrieval: precision@6 0.33 (capped: each doc has only 2 chunks, so at most 2 of 6 can be relevant), precision after MI pruning 0.74, average prune ratio 62%.

Adversarial robustness (corrupted sentence vs the original clean chunk, 120 tests), measured twice: with NLI alone and with the full pipeline (claim splitting + number/name check + filler):

| Attack | NLI alone | Full pipeline |
| --- | --- | --- |
| Negation | 100% | 100% |
| Stale fact | 100% | 100% |
| Noise | 100% | 100% |
| LLM contradiction (`qwen2.5:3b` rewrites) | 20.8% | 29.2% |
| Control false-positive rate | 16.7% | 20.8% |

Attention check (`python scripts/run_attention_check.py`): for the 69 sentences NLI flagged in the labelled set, deleting the top-5 attended source words and re-running NLI shifted the label probabilities by 0.24 on average, against 0.11 for deleting 5 random words, but the top words beat the random control (by a 0.05 margin) in only 35% of sentences. Attention words are a useful hint on average, not a reliable per-sentence explanation.

The low LLM-contradiction rate is partly a generator weakness: the 3B model often returns near-paraphrases instead of a factual change. Separately, NLI is weak on exact numbers: a corrupted sentence with a wrong figure can still be labelled "entailment" (seen in faulty mode).

The full project guide (concepts, how each step works, results, limitations, likely questions) is in `docs/guide/HalluciGuard_Complete_Guide.pdf`; rebuild it with `python3 docs/guide/build_guide.py` (needs Google Chrome). Exact rules and values are in `FACTS.md`.

## Detection quality (labeled set + ablation)

`python scripts/run_ablation.py` (outputs `outputs/ablation.json`). Test set: 176 sentences from 92 real answers of `qwen2.5:3b` (strict and loose prompt), 55 unsupported and 121 supported (filler and ambiguous sentences excluded). **The labels were made by an AI assistant reading each sentence against the source documents, not by a human**; spot-check `data/eval/labeled_sentences.json` before citing. "Positive" = unsupported sentence.

| Variant | Precision | Recall | F1 (95% bootstrap CI) | False-positive rate |
| --- | --- | --- | --- | --- |
| NLI only (sentence level) | 0.68 | 0.82 | 0.74 (0.65-0.83) | 0.17 |
| Span check only (numbers/names) | 0.57 | 0.22 | 0.32 (0.17-0.46) | 0.07 |
| NLI + span check | 0.65 | 0.84 | 0.73 (0.64-0.81) | 0.21 |
| NLI + claim splitting | 0.65 | 0.82 | 0.73 (0.63-0.81) | 0.20 |
| Full pipeline (claims + span + filler) | 0.62 | 0.84 | 0.71 (0.62-0.80) | 0.23 |
| Full pipeline without MI pruning (6 chunks) | 0.63 | 0.67 | 0.65 (0.54-0.74) | 0.18 |

What this shows:
- **MI pruning helps:** recall 0.84 vs 0.67 and F1 0.71 vs 0.65 (feeding all 6 chunks to NLI dilutes the evidence).
- **The span check and claim splitting did not measurably improve F1** on this set: all NLI variants are within each other's confidence intervals (0.71-0.74). They trade a little precision for a little recall. They remain useful for explaining *which words* are wrong, and the span check catches exact-number corruptions NLI misses in faulty mode, but this labeled set (mostly commentary and invented details) does not reward them.
- **Recall by error type (full pipeline):** invented specifics 0.89 (n=19), unsupported commentary 0.93 (n=27), contradictions/miscopied facts only 0.44 (n=9, small sample), the hardest case.
- **Entailment threshold:** 0.3 to 0.9 all give F1 about 0.71-0.72, so 0.5 is not critical; higher thresholds trade precision for recall.
- Caveat: sentences from the same answer are correlated, so the intervals are optimistic.

Answer-level trust (83 answers, 33 contain an unsupported sentence), AUROC of each component for spotting bad answers:

| Component | AUROC |
| --- | --- |
| Token probability (logprobs) | 0.98 |
| Sentence (NLI) | 0.88 |
| Final trust score | 0.89 |
| Entity | 0.65 |
| Document (all authorities equal) | 0.46 (uninformative) |

Caveat: loose-prompt answers are both less confident and more often unsupported, so the strong token-probability result partly measures the prompt mode, not hallucination alone. The document component carries no information while every document has the same authority.

..

## Notes and limitations

- **Logprobs**: Ollama (`qwen2.5:3b`) returns per-token logprobs, so the token-probability component is used.
- Attention output is described as "source words the verifier attended to most" (self-attention over premise+hypothesis), not proof of causation.
- Sentences that rely on a pronoun ("It was completed in 2012") may be flagged because the chunk lacks the antecedent.
- Numbers vary slightly between runs because LLM output is not fully deterministic.
- Endpoints: `POST /documents`, `GET /documents`, `POST /query`, `GET /response/{id}`, `POST /evaluate`, `POST /adversarial/run`, `GET /adversarial/results`, `GET /health`.
