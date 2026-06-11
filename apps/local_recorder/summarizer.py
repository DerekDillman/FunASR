"""Transcript summarization with a fast local LLM.

Primary path: Ollama (https://ollama.com) running on the same laptop — small
instruct models like ``llama3.2:3b`` or ``qwen2.5:3b-instruct`` summarize a
meeting transcript in seconds on CPU. Long transcripts are summarized
map-reduce style so they fit small context windows.

Fallback path: if Ollama is unreachable, a dependency-free extractive
summarizer picks the most informative sentences so the feature still works.
"""

import json
import re
import time
import urllib.error
import urllib.request

from config import Config

STYLE_PROMPTS = {
    "meeting": (
        "You are summarizing a meeting transcript. Produce:\n"
        "## Summary\nA short paragraph of what the meeting was about.\n"
        "## Key points\nBullet list of the main discussion points.\n"
        "## Decisions\nBullet list of decisions made (or 'None recorded').\n"
        "## Action items\nBullet list of tasks/owners/deadlines mentioned (or 'None recorded')."
    ),
    "training": (
        "You are summarizing a training/lecture transcript. Produce:\n"
        "## Topic\nOne sentence on what was taught.\n"
        "## Key concepts\nBullet list of the concepts covered, each with a one-line explanation.\n"
        "## Practical takeaways\nBullet list of things the learner should do or remember."
    ),
    "instructions": (
        "You are summarizing spoken instructions. Produce:\n"
        "## Goal\nOne sentence describing the end result.\n"
        "## Steps\nA numbered, ordered list of the steps as instructed.\n"
        "## Warnings & tips\nBullet list of cautions or tips mentioned (or 'None')."
    ),
    "general": (
        "Summarize the following transcript faithfully and concisely. Use\n"
        "'## Summary' followed by a paragraph, then '## Highlights' with bullets."
    ),
}


class Summarizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ---- status ----

    def status(self):
        """Report whether Ollama is reachable and the model is pulled."""
        try:
            with urllib.request.urlopen(f"{self.cfg.ollama_url}/api/tags", timeout=2) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in tags.get("models", [])]
            wanted = self.cfg.ollama_model
            has_model = any(m == wanted or m.split(":")[0] == wanted for m in models)
            return {
                "available": True,
                "model": wanted,
                "model_pulled": has_model,
                "models": models,
                "fallback": "extractive",
            }
        except Exception as e:
            return {
                "available": False,
                "model": self.cfg.ollama_model,
                "error": str(e),
                "hint": "Install Ollama and run: ollama pull " + self.cfg.ollama_model,
                "fallback": "extractive",
            }

    # ---- summarization ----

    def summarize(self, transcript: str, style: str = "meeting") -> dict:
        """Blocking; run in a worker thread. Returns a summary record."""
        style = style if style in STYLE_PROMPTS else "general"
        transcript = transcript.strip()
        if not transcript:
            raise ValueError("transcript is empty")

        started = time.time()
        try:
            text = self._summarize_llm(transcript, style)
            method = f"ollama:{self.cfg.ollama_model}"
        except Exception as e:
            text = _extractive_summary(transcript)
            method = "extractive-fallback"
            text = (
                f"> Local LLM unavailable ({e}); showing an extractive summary. "
                f"Install [Ollama](https://ollama.com) and run "
                f"`ollama pull {self.cfg.ollama_model}` for full summaries.\n\n" + text
            )
        return {
            "style": style,
            "text": text,
            "method": method,
            "created_at": time.time(),
            "elapsed_s": round(time.time() - started, 1),
        }

    def _summarize_llm(self, transcript: str, style: str) -> str:
        chunks = _split_chunks(transcript, self.cfg.summarize_chunk_chars)
        if len(chunks) == 1:
            return self._ollama_chat(STYLE_PROMPTS[style], chunks[0])
        # Map-reduce: condense each chunk, then summarize the condensations.
        partials = []
        for i, chunk in enumerate(chunks):
            partials.append(
                self._ollama_chat(
                    "Condense this portion of a longer transcript into dense notes, "
                    "keeping every fact, decision, name, number and task.",
                    f"(part {i + 1} of {len(chunks)})\n\n{chunk}",
                )
            )
        return self._ollama_chat(
            STYLE_PROMPTS[style] + "\nThe input below is condensed notes from the full transcript.",
            "\n\n".join(partials),
        )

    def _ollama_chat(self, system: str, user: str) -> str:
        payload = json.dumps(
            {
                "model": self.cfg.ollama_model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": 0.2},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.cfg.ollama_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = (data.get("message") or {}).get("content", "").strip()
        if not content:
            raise RuntimeError("empty response from Ollama")
        return content


def _split_chunks(text: str, max_chars: int):
    if len(text) <= max_chars:
        return [text]
    lines = text.split("\n")
    chunks, cur, cur_len = [], [], 0
    for line in lines:
        if cur and cur_len + len(line) > max_chars:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _extractive_summary(transcript: str, max_sentences: int = 8) -> str:
    """Tiny frequency-based extractive summarizer (no dependencies)."""
    # Strip "[hh:mm:ss - hh:mm:ss]" timestamps before scoring.
    clean = re.sub(r"\[\d{2}:\d{2}:\d{2} - \d{2}:\d{2}:\d{2}\]\s*", "", transcript)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+|\n+", clean) if len(s.strip()) > 12]
    if not sentences:
        return clean[:1000]

    words = re.findall(r"[\w']+", clean.lower())
    stop = set(
        "the a an and or but if then so to of in on at for with is are was were be been being "
        "i you he she it we they this that these those there here have has had do does did not "
        "no yes my your our their his her its as by from about into over after before just like "
        "really very also can could would should will going gonna want know think right okay um uh".split()
    )
    freq = {}
    for w in words:
        if w not in stop and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1

    def score(s):
        toks = re.findall(r"[\w']+", s.lower())
        if not toks:
            return 0.0
        return sum(freq.get(t, 0) for t in toks) / (len(toks) ** 0.5)

    ranked = sorted(range(len(sentences)), key=lambda i: score(sentences[i]), reverse=True)
    keep = sorted(ranked[:max_sentences])
    bullets = "\n".join(f"- {sentences[i]}" for i in keep)
    return f"## Highlights (extractive)\n{bullets}"
