"""Drive the real UI in a browser and capture the screenshots used in the README.

Usage: ./venv/bin/python scripts/screenshots.py [--url http://localhost:8077]
       [--pdf FILE] [--out docs/images]

Requires a running Contexto server plus a configured LLM provider, and the
dev-only ``playwright`` package. Screenshots are taken against the live app, so
what lands in the README is what the app actually renders.
"""

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.conftest import build_pdf  # noqa: E402

CHROME_CANDIDATES = [
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]

DEMO_PAGES = [
    "Contexto - Technical Overview. Contexto is a retrieval augmented "
    "generation service. Uploaded PDFs are split into overlapping chunks and "
    "embedded locally with the BAAI/bge-small-en-v1.5 model, which runs on CPU "
    "through the fastembed ONNX runtime and needs no GPU and no API key.",
    "Storage and retrieval. Each upload creates its own ChromaDB collection, "
    "keyed by session id, so two documents can never leak context into one "
    "another. Queries embed the question and run a cosine similarity search, "
    "returning the four closest chunks as grounding context.",
    "Answering and citations. The retrieved blocks are injected into a strict "
    "system prompt that forbids answering outside the supplied context. Every "
    "response ships with page level citations, and answers stream token by "
    "token over server sent events so reading can begin immediately.",
]

QUESTION = "How are the embeddings generated, and do they need a GPU?"


def find_chrome() -> str | None:
    for path in CHROME_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8077")
    parser.add_argument("--pdf", help="PDF to upload (default: synthetic demo doc)")
    parser.add_argument("--out", default="docs/images")
    parser.add_argument("--timeout", type=int, default=180, help="answer wait (s)")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        pdf_path = Path(args.pdf)
        pdf_bytes = pdf_path.read_bytes()
        pdf_name = pdf_path.name
    else:
        pdf_bytes = build_pdf(DEMO_PAGES)
        pdf_name = "contexto-technical-overview.pdf"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=find_chrome(), args=["--no-sandbox"]
        )
        page = browser.new_page(viewport={"width": 1280, "height": 860})
        page.goto(args.url, wait_until="networkidle")

        page.screenshot(path=out_dir / "01-upload.png")
        print("captured 01-upload.png")

        page.set_input_files("#file-input", files=[
            {"name": pdf_name, "mimeType": "application/pdf", "buffer": pdf_bytes}
        ])
        page.wait_for_selector("#chat-view:not(.hidden)", timeout=120_000)
        print("document ingested")

        page.fill("#chat-input", QUESTION)
        page.click("#send-btn")

        # Catch the answer mid-flight to prove streaming is real.
        page.wait_for_selector(".msg.bot.streaming .answer-body", timeout=120_000)
        page.wait_for_function(
            "document.querySelector('.msg.bot.streaming .answer-body')"
            "?.textContent.length > 40",
            timeout=120_000,
        )
        page.screenshot(path=out_dir / "02-streaming.png")
        print("captured 02-streaming.png")

        page.wait_for_selector(".msg.bot:not(.streaming)", timeout=args.timeout * 1000)
        page.click(".sources summary")
        page.wait_for_timeout(400)
        page.screenshot(path=out_dir / "03-answer-with-citations.png")
        print("captured 03-answer-with-citations.png")

        browser.close()

    print(f"\nScreenshots written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
