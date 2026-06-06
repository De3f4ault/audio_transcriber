"""Semantic chunking pipeline.

Implements text cleaning, speaker diarization splitting, and semantic boundary detection
using a sliding window over sentence groups.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.memory.singletons import get_boundary_embedder

try:
    import nltk  # type: ignore[import-untyped]

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize  # type: ignore[import-untyped]
except ImportError:
    # Fallback if nltk is somehow missing
    def sent_tokenize(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


logger = get_logger("memory.chunking")

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class SpeakerBlock:
    speaker: str
    text: str


@dataclass
class Chunk:
    content: str
    uuid: str
    tier: int
    speaker: str | None = None


@dataclass
class ParentGroup:
    parent_text: str
    children: list[Chunk]


# ---------------------------------------------------------------------------
# 4.1 Text Cleaning
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Clean Whisper hallucinations, null bytes, and repetition loops."""
    # Strip NUL bytes
    text = text.replace("\x00", "")

    # Strip Whisper non-speech tags
    tags_to_remove = [
        r"\[BLANK_AUDIO\]",
        r"\[MUSIC\]",
        r"\[music\]",
        r"\[Music\]",
        r"\[Silence\]",
        r"\[silence\]",
        r"\(silence\)",
        r"\(Music\)",
        r"\[Applause\]",
        r"\(Applause\)",
        r"\[Laughter\]",
        r"\(Laughter\)",
    ]
    for tag in tags_to_remove:
        text = re.sub(tag, "", text)

    # Collapse repeated word runs (e.g., "the the the" -> "the")
    # This is a naive regex for exact word repetition
    text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Detect trailing repetition loops (e.g. model gets stuck repeating the last sentence)
    sentences = sent_tokenize(text)
    if len(sentences) >= 3:
        # Check last 3 sentences. If they are identical, trim them and assume it's a loop.
        if sentences[-1].lower() == sentences[-2].lower() == sentences[-3].lower():
            # Find the start of the loop (first sentence that is identical to the last)
            loop_sentence = sentences[-1].lower()
            trim_idx = len(sentences)
            for i in range(len(sentences) - 1, -1, -1):
                if sentences[i].lower() == loop_sentence:
                    trim_idx = i
                else:
                    break
            text = " ".join(sentences[: trim_idx + 1])  # Keep one instance

    return text


# ---------------------------------------------------------------------------
# 4.2 AdvancedSemanticChunker
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class AdvancedSemanticChunker:
    """Chunks text using sliding window cosine similarity on sentence groups."""

    def __init__(self, breakpoint_percentile: float, max_tokens: int, sentence_group_size: int):
        self.breakpoint_percentile = breakpoint_percentile
        self.max_tokens = max_tokens
        self.sentence_group_size = sentence_group_size

    def chunk(self, text: str, speaker: str | None = None) -> list[Chunk]:
        sentences = sent_tokenize(text)
        if not sentences:
            return []

        # If it's too short to form two groups, return as one chunk
        if len(sentences) <= self.sentence_group_size:
            return [Chunk(content=text, uuid=str(uuid.uuid4()), tier=3, speaker=speaker)]

        # Group sentences
        groups = []
        for i in range(0, len(sentences), self.sentence_group_size):
            group_text = " ".join(sentences[i : i + self.sentence_group_size])
            groups.append(group_text)

        def _stable_uuid(txt: str) -> str:
            return str(uuid.uuid5(uuid.NAMESPACE_OID, txt))

        if len(groups) < 2:
            return [Chunk(content=text, uuid=_stable_uuid(text), tier=3, speaker=speaker)]

        # Embed groups
        embedder = get_boundary_embedder()
        # SentenceTransformers returns a list of tensors or numpy arrays
        embeddings = embedder.encode(groups, convert_to_numpy=True)

        # Calculate cosine distances (1 - cosine_similarity) between adjacent groups
        distances = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i], embeddings[i + 1])
            distances.append(1.0 - sim)

        # Determine threshold
        if not distances:
            return [Chunk(content=text, uuid=_stable_uuid(text), tier=3, speaker=speaker)]

        threshold = np.percentile(distances, self.breakpoint_percentile)

        # Split groups based on threshold
        chunks = []
        current_chunk_sentences: list[str] = []

        # We need to map groups back to original sentences for precise chunking
        group_idx = 0
        sentence_idx = 0

        while sentence_idx < len(sentences):
            current_chunk_sentences.append(sentences[sentence_idx])

            # If this sentence is the end of a group, check if we need to split
            # A group ends when we've added `sentence_group_size` sentences, or we reach the end
            if (sentence_idx + 1) % self.sentence_group_size == 0 or (sentence_idx + 1) == len(
                sentences
            ):
                # If there's a distance measured for the current group
                if group_idx < len(distances):
                    if distances[group_idx] > threshold:
                        # Split here
                        chunk_text = " ".join(current_chunk_sentences)
                        chunks.append(
                            Chunk(
                                content=chunk_text,
                                uuid=_stable_uuid(chunk_text),
                                tier=3,
                                speaker=speaker,
                            )
                        )
                        current_chunk_sentences = []
                group_idx += 1

            sentence_idx += 1

        # Add remaining sentences
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(
                Chunk(content=chunk_text, uuid=_stable_uuid(chunk_text), tier=3, speaker=speaker)
            )

        # Enforce max_tokens safeguard (SentenceSplitter fallback)
        final_chunks: list[Chunk] = []
        try:
            from llama_index.core.node_parser import SentenceSplitter

            splitter = SentenceSplitter(chunk_size=self.max_tokens, chunk_overlap=20)
        except ImportError:
            splitter = None

        for c in chunks:
            if splitter is not None and len(c.content.split()) > self.max_tokens:
                # Approximation of tokens by words, if it's too long, use LLamaIndex's splitter
                sub_texts = splitter.split_text(c.content)
                for st in sub_texts:
                    final_chunks.append(
                        Chunk(content=st, uuid=_stable_uuid(st), tier=3, speaker=speaker)
                    )
            else:
                final_chunks.append(c)

        return final_chunks


# ---------------------------------------------------------------------------
# 4.3 / 4.4 Content-Aware Router and Speaker Splitting
# ---------------------------------------------------------------------------


def speaker_turn_splitter(diarized_segments: list[dict[str, Any]]) -> list[SpeakerBlock]:
    """Splits transcript on speaker turn boundaries.

    Expects segments with 'speaker' and 'text' keys.
    """
    blocks = []
    current_speaker = None
    current_text: list[str] = []

    for seg in diarized_segments:
        spk = seg.get("speaker", "UNKNOWN")
        txt = seg.get("text", "").strip()

        if not txt:
            continue

        if spk != current_speaker:
            if current_speaker is not None and current_text:
                blocks.append(SpeakerBlock(speaker=current_speaker, text=" ".join(current_text)))
            current_speaker = spk
            current_text = [txt]
        else:
            current_text.append(txt)

    if current_speaker is not None and current_text:
        blocks.append(SpeakerBlock(speaker=current_speaker, text=" ".join(current_text)))

    return blocks


def content_aware_router(
    text: str, diarized_segments: list[dict[str, Any]] | None = None
) -> list[Chunk]:
    """Route text to appropriate chunking strategy based on length and diarization."""
    settings = get_settings()
    cleaned_text = _clean_text(text)

    def _stable_uuid(txt: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, txt))

    # 1. Short text -> single chunk
    if len(cleaned_text) < settings.chunk_short_threshold:
        return [Chunk(content=cleaned_text, uuid=_stable_uuid(cleaned_text), tier=3)]

    chunker = AdvancedSemanticChunker(
        breakpoint_percentile=settings.chunk_breakpoint_percentile,
        max_tokens=settings.chunk_max_tokens,
        sentence_group_size=settings.chunk_sentence_group_size,
    )

    chunks = []

    # 2. Diarized text -> split by speaker, then chunk
    if diarized_segments:
        # We need to clean the text inside the segments too
        clean_segments = []
        for seg in diarized_segments:
            clean_segments.append(
                {"speaker": seg.get("speaker", "UNKNOWN"), "text": _clean_text(seg.get("text", ""))}
            )

        blocks = speaker_turn_splitter(clean_segments)
        for block in blocks:
            # If a block is very short, don't semantic-chunk it further
            if len(block.text) < settings.chunk_short_threshold:
                chunks.append(
                    Chunk(
                        content=block.text,
                        uuid=_stable_uuid(block.text),
                        tier=3,
                        speaker=block.speaker,
                    )
                )
            else:
                chunks.extend(chunker.chunk(block.text, speaker=block.speaker))
        return chunks

    # 3. Long text -> AdvancedSemanticChunker
    # Long threshold check (if very long, might need different percentile or max_tokens handling, handled in Chunker)
    if len(cleaned_text) >= settings.chunk_long_threshold:
        pass  # The chunker logic handles this

    return chunker.chunk(cleaned_text)


# ---------------------------------------------------------------------------
# 4.5 Parent-Child Grouper
# ---------------------------------------------------------------------------


def parent_child_grouper(chunks: list[Chunk]) -> list[ParentGroup]:
    """Group sentence-level chunks into topic-level parents.

    For now, this simply groups adjacent chunks by speaker. If no speaker,
    it groups into fixed sizes or uses semantic similarity.
    Implementation here is a basic placeholder for Phase 4.5
    """
    if not chunks:
        return []

    groups = []
    current_children: list[Chunk] = []
    current_speaker = chunks[0].speaker

    for chunk in chunks:
        # Basic grouping strategy: break on speaker change, or every 5 chunks if no speaker
        if chunk.speaker != current_speaker or len(current_children) >= 5:
            if current_children:
                parent_text = " ".join([c.content for c in current_children])
                groups.append(ParentGroup(parent_text=parent_text, children=current_children))
            current_children = [chunk]
            current_speaker = chunk.speaker
        else:
            current_children.append(chunk)

    if current_children:
        parent_text = " ".join([c.content for c in current_children])
        groups.append(ParentGroup(parent_text=parent_text, children=current_children))

    return groups
