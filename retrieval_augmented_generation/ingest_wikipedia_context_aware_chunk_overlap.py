import argparse
import os
import time
import logging
import hashlib
import re
import statistics
from typing import Any, Dict, Iterable, List, Optional

import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI
from supabase import Client, create_client

load_dotenv()

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

MAX_TOKENS_PER_CHUNK = 256
CHUNK_OVERLAP = 24
PREV_CONTEXT_SENTENCES = 2

MIN_CHUNK_TOKENS = 40
MIN_CHUNK_CHARS = 180
MIN_ALPHA_RATIO = 0.55
LIST_MIN_ALPHA_RATIO = 0.35

EMBEDDING_BATCH_SIZE = 64
SUPABASE_INSERT_BATCH_SIZE = 100
MAX_RETRIES = 5

MIN_TEXT_CHARS = 500
TIMELINE_GROUP_SIZE = 8
BIRTHS_DEATHS_GROUP_SIZE = 10

MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
MONTH_PATTERN = "|".join(MONTH_NAMES)

MONTH_LINE_RE = re.compile(
    rf"^(?:{MONTH_PATTERN})(?: \d{{1,2}})?$|^\d{{1,2}} (?:{MONTH_PATTERN})$"
)
TIMELINE_EVENT_RE = re.compile(
    rf"^(?:{MONTH_PATTERN}) \d{{1,2}}|^\d{{1,2}} (?:{MONTH_PATTERN})"
)
LIST_ITEM_RE = re.compile(
    r"^(?:[-*•]|(?:\(?\d+\)?|\(?[A-Za-z]\)|[A-Za-z]\.|\d+[.)]|[ivxlcdm]+[.)]))\s+",
    re.IGNORECASE,
)
LEADING_FRAGMENT_RE = re.compile(
    r"^(?:and|or|but|of|the|a|an|to|in|on|for|from|with|by|into|after|before)\b",
    re.IGNORECASE,
)

IGNORE_SECTION_TITLES = {
    "references",
    "external links",
    "other websites",
    "related pages",
    "see also",
    "notes",
    "bibliography",
    "gallery",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

openai_client: Optional[OpenAI] = None
supabase: Optional[Client] = None

encoder = tiktoken.get_encoding("cl100k_base")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_openai_client() -> OpenAI:
    global openai_client

    if openai_client is None:
        openai_api_key = os.environ["OPENAI_API_KEY"]
        openai_client = OpenAI(api_key=openai_api_key)

    return openai_client


def get_supabase_client() -> Client:
    global supabase

    if supabase is None:
        supabase_url = os.environ["SUPABASE_URL"]
        supabase_service_role_key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        supabase = create_client(supabase_url, supabase_service_role_key)

    return supabase


def with_retry(fn, description: str, max_retries: int = MAX_RETRIES):
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            sleep_seconds = min(2 ** attempt, 30)

            logging.warning(
                "%s failed. attempt=%s/%s error=%s sleeping=%ss",
                description,
                attempt,
                max_retries,
                repr(e),
                sleep_seconds,
            )

            time.sleep(sleep_seconds)

    raise RuntimeError(f"{description} failed after {max_retries} retries") from last_error


def batched(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def token_count(text: str) -> int:
    return len(encoder.encode(text))


def decode_tokens(tokens: List[int]) -> str:
    return encoder.decode(tokens).strip()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def alpha_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    alpha_chars = sum(1 for ch in chars if ch.isalpha())
    return alpha_chars / len(chars)


def numeric_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    numeric_chars = sum(1 for ch in chars if ch.isdigit())
    return numeric_chars / len(chars)


def heading_case(line: str) -> bool:
    words = [word for word in re.split(r"\s+", line) if word]

    if not words:
        return False

    minor_words = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }

    for index, word in enumerate(words):
        normalized = word.strip("()[]{}\"'")
        if not normalized:
            continue
        if normalized.isupper():
            continue
        if index > 0 and normalized.lower() in minor_words:
            continue
        if normalized[0].isupper():
            continue
        return False

    return True


def is_probable_heading(line: str) -> bool:
    line = line.strip()

    if not line:
        return False

    lowered = line.lower()
    if lowered in IGNORE_SECTION_TITLES:
        return True

    if len(line) > 80:
        return False

    if line.endswith((".", ",", ";", ":")):
        return False

    if token_count(line) > 12:
        return False

    if LIST_ITEM_RE.match(line):
        return False

    if MONTH_LINE_RE.match(line) or TIMELINE_EVENT_RE.match(line):
        return False

    return heading_case(line) or line.isupper()


def detect_document_type(text: str) -> str:
    lines = [line.strip() for line in normalize_text(text).split("\n") if line.strip()]
    paragraphs = split_section_into_paragraphs(text)

    if not lines:
        return "article"

    timeline_lines = sum(
        1 for line in lines if MONTH_LINE_RE.match(line) or TIMELINE_EVENT_RE.match(line)
    )
    list_lines = sum(1 for line in lines if LIST_ITEM_RE.match(line))
    prose_paragraphs = sum(
        1
        for paragraph in paragraphs
        if len(paragraph) >= MIN_CHUNK_CHARS and alpha_ratio(paragraph) >= MIN_ALPHA_RATIO
    )

    if timeline_lines / len(lines) >= 0.18:
        return "timeline"

    if (
        list_lines >= 10
        and list_lines / len(lines) >= 0.35
        and prose_paragraphs <= max(4, len(paragraphs) // 3)
    ):
        return "list"

    return "article"


def split_into_sections(title: str, text: str) -> List[Dict[str, str]]:
    text = normalize_text(text)
    lines = text.split("\n")

    sections: List[Dict[str, str]] = []
    current_section_title = "Introduction"
    current_lines: List[str] = []

    def flush_section():
        nonlocal current_lines, current_section_title

        section_text = normalize_text("\n".join(current_lines))
        section_key = current_section_title.strip().lower()

        if section_text and section_key not in IGNORE_SECTION_TITLES:
            sections.append(
                {
                    "section_title": current_section_title,
                    "section_path": f"{title} > {current_section_title}",
                    "text": section_text,
                }
            )

        current_lines = []

    for line in lines:
        stripped = line.strip()

        if is_probable_heading(stripped):
            flush_section()
            current_section_title = stripped
        else:
            current_lines.append(line)

    flush_section()

    if not sections:
        sections.append(
            {
                "section_title": "Full Article",
                "section_path": title,
                "text": text,
            }
        )

    return sections


def split_section_into_paragraphs(section_text: str) -> List[str]:
    section_text = normalize_text(section_text)

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", section_text)
        if paragraph.strip()
    ]

    if len(paragraphs) <= 1:
        paragraphs = [
            paragraph.strip()
            for paragraph in section_text.split("\n")
            if paragraph.strip()
        ]

    return paragraphs


def split_text_into_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    flat_text = re.sub(r"\s*\n\s*", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[\"'(A-Z0-9])", flat_text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return sentences or [flat_text]


def split_large_text_by_tokens(text: str, max_tokens: int) -> List[str]:
    tokens = encoder.encode(text)
    pieces = []

    start = 0
    while start < len(tokens):
        end = start + max_tokens
        piece = decode_tokens(tokens[start:end])

        if piece:
            pieces.append(piece)

        if end >= len(tokens):
            break

        start = end

    return pieces


def split_large_text_by_sentences(text: str, max_tokens: int) -> List[str]:
    sentences = split_text_into_sentences(text)
    pieces: List[str] = []
    current_sentences: List[str] = []
    current_tokens = 0

    def flush_current():
        nonlocal current_sentences, current_tokens
        if not current_sentences:
            return
        pieces.append(normalize_text(" ".join(current_sentences)))
        current_sentences = []
        current_tokens = 0

    for sentence in sentences:
        sentence_tokens = token_count(sentence)

        if sentence_tokens > max_tokens:
            flush_current()
            pieces.extend(split_large_text_by_tokens(sentence, max_tokens))
            continue

        if current_sentences and current_tokens + sentence_tokens > max_tokens:
            flush_current()

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    flush_current()

    pieces = [piece for piece in pieces if piece]

    if len(pieces) >= 2:
        previous_sentences = split_text_into_sentences(pieces[-2])
        last_sentences = split_text_into_sentences(pieces[-1])

        while previous_sentences and last_sentences:
            last_piece = normalize_text(" ".join(last_sentences))
            previous_piece = normalize_text(" ".join(previous_sentences))

            if (
                token_count(last_piece) >= MIN_CHUNK_TOKENS
                and len(last_piece) >= MIN_CHUNK_CHARS
            ):
                break

            candidate_last_sentences = [previous_sentences[-1]] + last_sentences
            candidate_previous_sentences = previous_sentences[:-1]

            if not candidate_previous_sentences:
                break

            candidate_last = normalize_text(" ".join(candidate_last_sentences))
            candidate_previous = normalize_text(" ".join(candidate_previous_sentences))

            if token_count(candidate_last) > max_tokens:
                break

            if token_count(candidate_previous) < MIN_CHUNK_TOKENS:
                break

            previous_sentences = candidate_previous_sentences
            last_sentences = candidate_last_sentences
            pieces[-2] = candidate_previous
            pieces[-1] = candidate_last

    return [piece for piece in pieces if piece]


def get_overlap_text(text: str, overlap_tokens: int) -> str:
    tokens = encoder.encode(text)
    if not tokens:
        return ""
    overlap = tokens[-overlap_tokens:]
    return decode_tokens(overlap)


def get_prev_context(text: str, max_sentences: int = PREV_CONTEXT_SENTENCES) -> str:
    sentences = split_text_into_sentences(text)
    if not sentences:
        return ""
    return normalize_text(" ".join(sentences[-max_sentences:]))


def build_embedding_text(
    title: str,
    section_path: str,
    content: str,
    prev_context: str = "",
) -> str:
    parts = [
        f"Title: {title}",
        f"Section: {section_path}",
    ]

    if prev_context:
        parts.append(f"Previous context: {prev_context}")

    parts.append(content)
    return "\n\n".join(parts).strip()


def looks_like_fragment(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return True

    if stripped[0] in ",;:.)]}":
        return True

    first_alpha = next((ch for ch in stripped if ch.isalpha()), "")
    if first_alpha and first_alpha.islower():
        return True

    if LEADING_FRAGMENT_RE.match(stripped):
        return True

    if "\n" not in stripped and token_count(stripped) < 60 and not re.search(r"[.!?]$", stripped):
        return True

    return False


def looks_like_list_or_label_only(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return True

    if len(lines) == 1:
        line = lines[0]

        if line.endswith(":"):
            return True

        if token_count(line) <= 6 and not re.search(r"[.!?]$", line):
            return True

        if re.fullmatch(r"\(?[a-z0-9]+\)?(?:[\s\-][A-Za-z0-9]+){0,5}", line, re.IGNORECASE):
            return True

    if len(text) < MIN_CHUNK_CHARS:
        list_like_lines = sum(
            1
            for line in lines
            if LIST_ITEM_RE.match(line) or (token_count(line) <= 6 and not re.search(r"[.!?]$", line))
        )
        if list_like_lines == len(lines):
            return True

    return False


def is_orphan_chunk(content: str, chunk_strategy: str) -> bool:
    min_alpha_ratio = LIST_MIN_ALPHA_RATIO if chunk_strategy in {"timeline_grouped", "list_grouped"} else MIN_ALPHA_RATIO

    if token_count(content) < MIN_CHUNK_TOKENS:
        return True

    if len(content) < MIN_CHUNK_CHARS:
        return True

    if alpha_ratio(content) < min_alpha_ratio:
        return True

    if looks_like_fragment(content):
        return True

    if looks_like_list_or_label_only(content):
        return True

    return False


def merge_chunk_pair(
    left: Dict[str, Any],
    right: Dict[str, Any],
    prefer_right_metadata: bool = False,
) -> Dict[str, Any]:
    merged = dict(right if prefer_right_metadata else left)
    merged["content"] = normalize_text(f'{left["content"]}\n\n{right["content"]}')
    merged["source_token_count"] = token_count(merged["content"])
    merged["is_orphan_repaired"] = True
    return merged


def repair_orphan_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    repaired: List[Dict[str, Any]] = []

    for chunk in chunks:
        chunk = dict(chunk)
        chunk.setdefault("is_orphan_repaired", False)

        if is_orphan_chunk(chunk["content"], chunk["chunk_strategy"]):
            if repaired:
                repaired[-1] = merge_chunk_pair(repaired[-1], chunk)
            else:
                chunk["_needs_merge_with_next"] = True
                chunk["is_orphan_repaired"] = True
                repaired.append(chunk)
            continue

        if repaired and repaired[-1].get("_needs_merge_with_next"):
            pending = dict(repaired.pop())
            pending.pop("_needs_merge_with_next", None)
            pending = merge_chunk_pair(
                pending,
                chunk,
                prefer_right_metadata=True,
            )
            repaired.append(pending)
            continue

        repaired.append(chunk)

    if repaired and repaired[-1].get("_needs_merge_with_next"):
        repaired[-1].pop("_needs_merge_with_next", None)

    return repaired


def split_oversized_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    resized: List[Dict[str, Any]] = []

    for chunk in chunks:
        if token_count(chunk["content"]) <= MAX_TOKENS_PER_CHUNK:
            resized.append(chunk)
            continue

        parts = split_large_text_by_sentences(chunk["content"], MAX_TOKENS_PER_CHUNK)

        if len(parts) <= 1:
            resized.append(chunk)
            continue

        for part in parts:
            piece = dict(chunk)
            piece["content"] = normalize_text(part)
            piece["source_token_count"] = token_count(piece["content"])
            resized.append(piece)

    return resized


def build_article_units(section_text: str) -> List[str]:
    units: List[str] = []

    for paragraph in split_section_into_paragraphs(section_text):
        paragraph = normalize_text(paragraph)
        if not paragraph:
            continue

        if token_count(paragraph) <= MAX_TOKENS_PER_CHUNK:
            units.append(paragraph)
        else:
            units.extend(split_large_text_by_sentences(paragraph, MAX_TOKENS_PER_CHUNK))

    return units


def join_lines_to_blocks(lines: List[str], group_size: int) -> List[str]:
    blocks: List[str] = []
    current_lines: List[str] = []
    current_tokens = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        line_tokens = token_count(line)
        if current_lines and (
            current_tokens + line_tokens > MAX_TOKENS_PER_CHUNK or len(current_lines) >= group_size
        ):
            blocks.append(normalize_text("\n".join(current_lines)))
            current_lines = []
            current_tokens = 0

        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        blocks.append(normalize_text("\n".join(current_lines)))

    return blocks


def build_timeline_units(section_title: str, section_text: str) -> List[str]:
    lines = [line.strip() for line in section_text.split("\n") if line.strip()]
    section_key = section_title.lower()

    if section_key in {"births", "deaths"}:
        return join_lines_to_blocks(lines, BIRTHS_DEATHS_GROUP_SIZE)

    blocks: List[str] = []
    current_lines: List[str] = []
    current_tokens = 0

    for line in lines:
        if MONTH_LINE_RE.match(line):
            if current_lines:
                blocks.append(normalize_text("\n".join(current_lines)))
            current_lines = [line]
            current_tokens = token_count(line)
            continue

        line_tokens = token_count(line)
        if current_lines and (
            current_tokens + line_tokens > MAX_TOKENS_PER_CHUNK or len(current_lines) >= TIMELINE_GROUP_SIZE
        ):
            blocks.append(normalize_text("\n".join(current_lines)))
            current_lines = [line]
            current_tokens = line_tokens
            continue

        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        blocks.append(normalize_text("\n".join(current_lines)))

    return blocks


def build_list_units(section_text: str) -> List[str]:
    lines = [line.strip() for line in section_text.split("\n") if line.strip()]
    return join_lines_to_blocks(lines, TIMELINE_GROUP_SIZE)


def build_section_units(
    document_type: str,
    section_title: str,
    section_text: str,
) -> tuple[str, List[str]]:
    if document_type == "timeline":
        return "timeline_grouped", build_timeline_units(section_title, section_text)

    if document_type == "list":
        return "list_grouped", build_list_units(section_text)

    return "section_sentence_aware", build_article_units(section_text)


def chunk_section(
    section_index: int,
    section_title: str,
    section_path: str,
    units: List[str],
    chunk_strategy: str,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    current_parts: List[str] = []
    current_token_count = 0

    def flush_current_parts():
        nonlocal current_parts, current_token_count

        if not current_parts:
            return

        content = normalize_text("\n\n".join(current_parts))
        if not content:
            current_parts = []
            current_token_count = 0
            return

        chunks.append(
            {
                "content": content,
                "section_index": section_index,
                "section_title": section_title,
                "section_path": section_path,
                "chunk_strategy": chunk_strategy,
                "source_token_count": token_count(content),
                "is_orphan_repaired": False,
            }
        )

        current_parts = []
        current_token_count = 0

    for unit in units:
        unit = normalize_text(unit)
        if not unit:
            continue

        unit_pieces = (
            split_large_text_by_sentences(unit, MAX_TOKENS_PER_CHUNK)
            if token_count(unit) > MAX_TOKENS_PER_CHUNK
            else [unit]
        )

        for piece in unit_pieces:
            piece_tokens = token_count(piece)

            if current_parts and current_token_count + piece_tokens > MAX_TOKENS_PER_CHUNK:
                flush_current_parts()

            current_parts.append(piece)
            current_token_count += piece_tokens

    flush_current_parts()

    return chunks


def finalize_chunks(title: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    finalized: List[Dict[str, Any]] = []

    for chunk_index, chunk in enumerate(chunks):
        item = dict(chunk)
        item["chunk_index"] = chunk_index
        item["chunk_id"] = stable_hash(
            f'{title}|{item["section_path"]}|{chunk_index}|{item["content"]}'
        )[:16]
        finalized.append(item)

    for idx, chunk in enumerate(finalized):
        prev_chunk = finalized[idx - 1] if idx > 0 else None
        next_chunk = finalized[idx + 1] if idx + 1 < len(finalized) else None

        prev_context = ""
        if prev_chunk and prev_chunk["section_path"] == chunk["section_path"]:
            prev_context = get_prev_context(prev_chunk["content"], PREV_CONTEXT_SENTENCES)

        chunk["prev_context"] = prev_context
        chunk["embedding_text"] = build_embedding_text(
            title=title,
            section_path=chunk["section_path"],
            content=chunk["content"],
            prev_context=prev_context,
        )
        chunk["token_count"] = token_count(chunk["content"])
        chunk["source_token_count"] = chunk["token_count"]
        chunk["embedding_token_count"] = token_count(chunk["embedding_text"])
        chunk["prev_chunk_id"] = prev_chunk["chunk_id"] if prev_chunk else None
        chunk["next_chunk_id"] = next_chunk["chunk_id"] if next_chunk else None

    return finalized


def validate_chunks(title: str, chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        logging.warning("Chunk diagnostics title=%s total_chunks=0", title)
        return

    token_values = [chunk["token_count"] for chunk in chunks]
    chunks_under_min_tokens = [
        chunk for chunk in chunks if chunk["token_count"] < MIN_CHUNK_TOKENS
    ]
    lowercase_starts = [
        chunk
        for chunk in chunks
        if chunk["content"] and next((ch for ch in chunk["content"] if ch.isalpha()), "").islower()
    ]
    punctuation_starts = [
        chunk
        for chunk in chunks
        if chunk["content"] and chunk["content"].lstrip()[:1] in ",;:.)]}"
    ]
    high_numeric_ratio = [
        chunk for chunk in chunks if numeric_ratio(chunk["content"]) > 0.20
    ]
    shortest_chunks = sorted(chunks, key=lambda chunk: chunk["token_count"])[:20]

    logging.info(
        (
            "Chunk diagnostics title=%s total_chunks=%s avg_tokens=%.1f min_tokens=%s "
            "chunks_under_40_tokens=%s chunks_starting_with_lowercase=%s "
            "chunks_starting_with_punctuation=%s chunks_with_high_numeric_ratio=%s"
        ),
        title,
        len(chunks),
        statistics.mean(token_values),
        min(token_values),
        len(chunks_under_min_tokens),
        len(lowercase_starts),
        len(punctuation_starts),
        len(high_numeric_ratio),
    )

    if chunks_under_min_tokens or lowercase_starts or punctuation_starts:
        logging.warning(
            "Top shortest chunks title=%s shortest=%s",
            title,
            [
                {
                    "chunk_index": chunk["chunk_index"],
                    "tokens": chunk["token_count"],
                    "preview": chunk["content"][:120],
                }
                for chunk in shortest_chunks
            ],
        )


def chunk_text(title: str, text: str) -> tuple[str, List[Dict[str, Any]]]:
    document_type = detect_document_type(text)
    sections = split_into_sections(title, text)

    all_chunks: List[Dict[str, Any]] = []

    for section_index, section in enumerate(sections):
        chunk_strategy, units = build_section_units(
            document_type=document_type,
            section_title=section["section_title"],
            section_text=section["text"],
        )
        section_chunks = chunk_section(
            section_index=section_index,
            section_title=section["section_title"],
            section_path=section["section_path"],
            units=units,
            chunk_strategy=chunk_strategy,
        )
        all_chunks.extend(section_chunks)

    repaired_chunks = repair_orphan_chunks(all_chunks)
    resized_chunks = split_oversized_chunks(repaired_chunks)
    finalized_chunks = finalize_chunks(title, resized_chunks)

    return document_type, finalized_chunks


def embed_batch(texts: List[str]) -> List[List[float]]:
    def call_openai():
        response = get_openai_client().embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]

    return with_retry(call_openai, f"OpenAI embedding batch size={len(texts)}")


def insert_document(title: str, source_url: str | None, metadata: Dict[str, Any]) -> str:
    def call_supabase():
        response = (
            get_supabase_client().table("documents")
            .insert(
                {
                    "title": title,
                    "source_url": source_url,
                    "metadata": metadata,
                }
            )
            .execute()
        )
        return response.data[0]["id"]

    return with_retry(call_supabase, f"insert document title={title}")


def insert_chunks(rows: List[Dict[str, Any]]) -> None:
    def call_supabase():
        return get_supabase_client().table("document_chunks").insert(rows).execute()

    with_retry(call_supabase, f"insert chunks batch size={len(rows)}")


def ingest_wikipedia(max_articles: int = 10):
    logging.info("Loading Simple Wikipedia dataset")

    dataset = load_dataset(
        "wikipedia",
        "20220301.simple",
        split=f"train[:{max_articles}]",
        trust_remote_code=True,
    )

    logging.info("Loaded articles=%s", len(dataset))

    total_documents = 0
    total_chunks = 0

    for item in tqdm(dataset, desc="Ingesting Wikipedia articles"):
        title = item.get("title", "").strip()
        text = item.get("text", "").strip()
        url = item.get("url")

        if not title or not text:
            continue

        if len(text) < MIN_TEXT_CHARS:
            continue

        article_hash = stable_hash(text)
        document_type, chunks = chunk_text(title, text)

        if not chunks:
            continue

        validate_chunks(title, chunks)

        document_id = insert_document(
            title=title,
            source_url=url,
            metadata={
                "source": "wikipedia",
                "dataset": "20220301.simple",
                "article_id": item.get("id"),
                "article_hash": article_hash,
                "text_chars": len(text),
                "chunk_count": len(chunks),
                "chunking_strategy": "section_aware_v2",
                "document_type": document_type,
                "max_tokens_per_chunk": MAX_TOKENS_PER_CHUNK,
                "chunk_overlap": CHUNK_OVERLAP,
            },
        )

        all_rows = []

        for chunk_batch_start, chunk_batch in enumerate(batched(chunks, EMBEDDING_BATCH_SIZE)):
            embedding_inputs = [chunk["embedding_text"] for chunk in chunk_batch]
            embeddings = embed_batch(embedding_inputs)

            for local_idx, (chunk, embedding) in enumerate(zip(chunk_batch, embeddings)):
                global_chunk_index = chunk_batch_start * EMBEDDING_BATCH_SIZE + local_idx

                all_rows.append(
                    {
                        "document_id": document_id,
                        "chunk_index": global_chunk_index,
                        "content": chunk["content"],
                        "token_count": chunk["token_count"],
                        "metadata": {
                            "source": "wikipedia",
                            "dataset": "20220301.simple",
                            "article_title": title,
                            "article_url": url,
                            "article_hash": article_hash,
                            "chunk_id": chunk["chunk_id"],
                            "chunk_index": global_chunk_index,
                            "section_index": chunk["section_index"],
                            "section_title": chunk["section_title"],
                            "section_path": chunk["section_path"],
                            "document_type": document_type,
                            "chunk_strategy": chunk["chunk_strategy"],
                            "is_orphan_repaired": chunk["is_orphan_repaired"],
                            "source_token_count": chunk["source_token_count"],
                            "embedding_token_count": chunk["embedding_token_count"],
                            "prev_chunk_id": chunk["prev_chunk_id"],
                            "next_chunk_id": chunk["next_chunk_id"],
                            "prev_context": chunk["prev_context"],
                            "chunking_strategy": "section_aware_v2",
                            "embedded_with_context": True,
                        },
                        "embedding": embedding,
                    }
                )

        for rows_batch in batched(all_rows, SUPABASE_INSERT_BATCH_SIZE):
            insert_chunks(rows_batch)

        total_documents += 1
        total_chunks += len(all_rows)

        logging.info(
            "Inserted title=%s document_type=%s chunks=%s total_documents=%s total_chunks=%s",
            title,
            document_type,
            len(all_rows),
            total_documents,
            total_chunks,
        )

    logging.info(
        "DONE. total_documents=%s total_chunks=%s",
        total_documents,
        total_chunks,
    )


def preview_file_chunks(input_path: str, title: str | None = None, max_preview_chunks: int = 12) -> None:
    with open(input_path, "r") as f:
        text = f.read()

    preview_title = title or os.path.splitext(os.path.basename(input_path))[0]
    document_type, chunks = chunk_text(preview_title, text)
    validate_chunks(preview_title, chunks)

    logging.info(
        "Preview title=%s document_type=%s chunk_count=%s",
        preview_title,
        document_type,
        len(chunks),
    )

    for chunk in chunks[:max_preview_chunks]:
        logging.info(
            (
                "chunk_index=%s section_index=%s strategy=%s repaired=%s "
                "tokens=%s section=%s preview=%r"
            ),
            chunk["chunk_index"],
            chunk["section_index"],
            chunk["chunk_strategy"],
            chunk["is_orphan_repaired"],
            chunk["token_count"],
            chunk["section_title"],
            chunk["content"][:240],
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Wikipedia with context-aware chunking or preview chunking on a local file."
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=20000,
        help="Number of Wikipedia articles to ingest.",
    )
    parser.add_argument(
        "--preview-file",
        help="Local text file to chunk and inspect without embedding or Supabase writes.",
    )
    parser.add_argument(
        "--preview-title",
        help="Optional title override used during local preview chunking.",
    )
    parser.add_argument(
        "--max-preview-chunks",
        type=int,
        default=12,
        help="Number of preview chunks to print.",
    )
    args = parser.parse_args()

    if args.preview_file:
        preview_file_chunks(
            input_path=args.preview_file,
            title=args.preview_title,
            max_preview_chunks=args.max_preview_chunks,
        )
        return

    ingest_wikipedia(max_articles=args.max_articles)


if __name__ == "__main__":
    main()
