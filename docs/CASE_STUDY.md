# Case Study — Contexto: A Grounded Document Q&A Service

**Role:** Sole engineer — architecture, backend, frontend, testing, deployment
**Type:** Self-initiated product build
**Stack:** Python · FastAPI · LangChain · ChromaDB · fastembed (ONNX) · Vanilla JS · Docker
**Repository:** https://github.com/ArihaNoor/Contexto

---

## The problem

Anyone who works with long PDFs — contracts, research papers, technical
specifications, policy documents — loses hours to the same task: finding the one
paragraph that answers a specific question.

Pasting the document into a general-purpose chatbot fails in two ways that
matter. First, the model fills gaps with plausible invention; ask about a clause
that isn't there and you get a confident answer anyway. Second, there is no way
to check: no page number, no quoted passage, nothing to verify against. For any
document where being wrong has a cost, an unverifiable answer is worse than no
answer.

I set out to build the opposite: a service where **every answer is constrained
to the uploaded document and carries the evidence for itself.**

---

## What I built

Contexto is a Retrieval-Augmented Generation service with a REST API and a web
front end. Upload a PDF, ask questions, get answers that stream in token by
token with page-level citations attached.

**The pipeline**

1. **Ingest** — PyPDF extracts text and page metadata. A recursive character
   splitter produces overlapping ~1000-character chunks that respect paragraph
   and sentence boundaries rather than cutting mid-thought.
2. **Embed** — Each chunk becomes a 384-dimension vector via
   `BAAI/bge-small-en-v1.5`, running locally on CPU through fastembed's ONNX
   runtime. No GPU, no torch, no per-token embedding cost, no network hop.
3. **Index** — Vectors land in a ChromaDB collection named for that upload's
   session id.
4. **Retrieve** — A question is embedded and matched by cosine similarity; the
   four closest chunks become the grounding context.
5. **Generate** — Context, chat history, and a strict guardrail prompt go to the
   language model, which streams its answer back over server-sent events.

**The grounding contract**

> "Answer the user's question strictly using the provided context blocks. If the
> answer is not present in the context, explicitly state that the document does
> not contain enough information."

Paired with citations that carry the page number and the exact retrieved
excerpt, this means a reader can verify any claim in a single click — and when
the document genuinely doesn't cover something, the service says so instead of
inventing an answer.

---

## Engineering decisions that mattered

### Session isolation enforced by structure, not by a filter

The conventional approach is one shared vector collection with a `session_id`
metadata filter on every query. It works — until a filter is omitted somewhere,
and one user's document surfaces inside another user's answer.

I gave each upload its own ChromaDB collection instead. Cross-document leakage
stops being something to remember and becomes structurally impossible: a query
is scoped to a collection that only ever contained one document. Cleanup
simplifies too — `DELETE /clear` is a single `delete_collection` call rather
than a filtered mass-delete.

### A provider-agnostic LLM layer, forced by a real outage

The first build bound directly to the `huggingface_hub` SDK. Two failures
changed that.

The configured model — `Qwen/Qwen2.5-7B-Instruct` — turned out not to be served
by any inference provider enabled on the account, so **every single query
returned a 502**. Then, while I was benchmarking, the account's monthly
inference credits ran out and the service went down completely.

Both failures had the same root cause: a single hard dependency on one vendor's
SDK and one vendor's quota. A portfolio project that only works while a free
tier holds out isn't finished.

Every provider worth using — Hugging Face, OpenAI, Gemini, Ollama, vLLM,
LM Studio, OpenRouter — speaks the same OpenAI `/chat/completions` protocol. So
I replaced the SDK binding with a small client that speaks that protocol
directly over httpx, plus named presets for each provider.

```ini
# From a dead hosted provider to a free local model:
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
```

Two lines. No code change. The service now runs entirely offline at zero
marginal cost, and switching to OpenAI or Gemini for production is the same
two-line edit. The whole abstraction is one module that nothing else imports.

### Streaming, because the real bottleneck wasn't where the spec assumed

The original spec set a 3-second target for query responses. Measured against a
hosted provider, full answers were averaging **6.2s** — roughly double.

Rather than guess, I instrumented the two halves separately. Retrieval —
embedding the question and running the vector search — completes in
milliseconds. Essentially all the latency was the model writing its answer, and
no amount of retrieval tuning was going to fix that.

So I changed what the user waits for instead. The service now streams over
server-sent events, and because retrieval finishes long before generation does,
**citations are sent as the very first event** — the sources are on screen and
readable before the model has written a word. The user gets something to engage
with in about a second rather than staring at a spinner for six.

### Measuring honestly

My first streaming benchmark reported that all 286 tokens arrived in the same
instant — a 0.00s spread, which would have meant the streaming implementation
did nothing.

Before changing any code, I tested the layers separately. The provider client
streamed correctly in isolation (first token at 1.43s, spread 6.67s). The
culprit was FastAPI's `TestClient`, which buffers `text/event-stream` responses
entirely before returning them. Against a real uvicorn process, streaming
behaved exactly as designed: first token at 4.9s, tokens arriving steadily over
the following 18.8s.

The fix was to the measurement, not the product — the benchmark harness now
starts a real server process, because a number produced by the wrong instrument
is worse than no number.

### Errors that tell you what to do

Early on, provider failures were caught broadly and returned to the client as
`f"LLM generation failed: {exc}"` — leaking internals to users while telling
them nothing useful.

Now each failure mode maps to an actionable message: a 402 says the account is
out of credits *and* suggests switching to a local provider; a connection
refused against localhost asks whether `ollama serve` is running; an unsupported
model names the environment variable to change. Anything genuinely unexpected is
logged with a full traceback server-side and returned as a generic 500 that
leaks nothing — a behaviour that is itself covered by a test.

---

## Results

<!--BENCHMARKS-->

**Quality and reliability**

- **52 automated tests**, running fully offline in under 6 seconds — the
  provider is stubbed with `httpx.MockTransport`, embeddings run locally, and
  ChromaDB is redirected to a temp directory. No API key needed to run the
  suite.
- Coverage spans the full ingest → query → clear lifecycle, 1-based page-number
  correctness, cross-session isolation, multi-turn memory eviction, SSE event
  ordering, and every provider failure mode: 401, 402, 429, unsupported model,
  unreachable server, malformed and empty responses.
- Containerised with Docker, with the embedding model baked into the image so
  containers start warm instead of pulling ~130 MB on first request.

---

## What I would build next

The limitations are documented rather than hidden, because knowing where a
system stops is part of engineering it:

- **Persistent chat memory.** Conversation history currently lives in an
  in-process dict, so it doesn't survive a restart or scale past one worker.
  Redis-backed history is the swap.
- **OCR for scanned PDFs.** Image-only documents return "no extractable text".
  A Tesseract fallback would widen coverage significantly.
- **Hybrid retrieval.** Dense vector search alone underperforms on keyword-heavy
  queries — exact clause numbers, part codes, defined terms. BM25 combined with
  a reranking pass would close that gap.
- **Authentication.** Session ids are unguessable 128-bit values, which is
  adequate for single-user use but not for multi-tenant deployment.

---

## Summary

Contexto turns a static PDF into something you can interrogate — and, crucially,
verify. The engineering that matters isn't the RAG pipeline itself, which is
well-trodden; it's the decisions around it: isolation that can't be forgotten,
a vendor dependency that survived its own vendor going down, latency work aimed
at the bottleneck that actually existed, and a test suite that runs without a
credit card.
