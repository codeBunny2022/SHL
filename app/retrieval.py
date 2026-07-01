import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import get_close_matches

import numpy as np
from google import genai

from app.config import (
    CATALOG_PATH,
    DATA_DIR,
    EMBEDDINGS_PATH,
    GEMINI_API_KEY,
    GEMINI_EMBED_MODEL,
)

# Common shorthand -> catalog name (for compare / refine)
ALIASES: dict[str, str] = {
    "opq": "Occupational Personality Questionnaire OPQ32r",
    "opq32r": "Occupational Personality Questionnaire OPQ32r",
    "gsa": "Global Skills Assessment",
    "dsi": "Dependability and Safety Instrument (DSI)",
    "verify g+": "SHL Verify Interactive G+",
    "verify g plus": "SHL Verify Interactive G+",
}


@dataclass
class CatalogItem:
    entity_id: str
    name: str
    url: str
    test_type: str
    keys: list[str]
    description: str
    job_levels: list[str]
    languages: list[str]
    duration: str
    remote: str
    adaptive: str
    embed_text: str


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9+#]+", text.lower())
    return {t for t in tokens if len(t) > 2}


class CatalogIndex:
    def __init__(self) -> None:
        with open(CATALOG_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        self.items = [CatalogItem(**row) for row in raw]
        self.by_name = {item.name: item for item in self.items}
        self.embeddings = np.load(EMBEDDINGS_PATH)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-8)
        self._client: genai.Client | None = None
        self._token_sets = [_tokenize(item.embed_text) for item in self.items]
        meta_path = DATA_DIR / "index_meta.json"
        self._use_tfidf_query = False
        self._vocab: list[str] = []
        self._vocab_index: dict[str, int] = {}
        self._df: dict[str, int] = {}
        self._n_docs = len(self.items)
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("type") == "tfidf":
                self._use_tfidf_query = True
                self._vocab = meta["vocab"]
                self._vocab_index = {w: i for i, w in enumerate(self._vocab)}
                self._df = {k: int(v) for k, v in meta["df"].items()}
                self._n_docs = int(meta["n_docs"])
        elif not GEMINI_API_KEY:
            self._use_tfidf_query = True
            self._build_tfidf_vocab()

    def _build_tfidf_vocab(self) -> None:
        df: Counter[str] = Counter()
        for tokens in self._token_sets:
            df.update(set(tokens))
        self._vocab = sorted(df)
        self._vocab_index = {w: i for i, w in enumerate(self._vocab)}
        self._df = dict(df)
        self._n_docs = len(self.items)

    def _tfidf_vector(self, text: str) -> np.ndarray:
        tokens = _tokenize(text)
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        tf = Counter(tokens)
        for term, count in tf.items():
            if term not in self._vocab_index:
                continue
            idf = np.log((1 + self._n_docs) / (1 + self._df[term])) + 1
            vec[self._vocab_index[term]] = count * idf
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-8)

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
        return self._client

    def embed_query(self, query: str) -> np.ndarray:
        if self._use_tfidf_query:
            return self._tfidf_vector(query)
        result = self.client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=[query],
        )
        vec = np.array(result.embeddings[0].values, dtype=np.float32)
        vec /= max(np.linalg.norm(vec), 1e-8)
        return vec

    def _lexical_scores(self, query: str) -> np.ndarray:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return np.zeros(len(self.items), dtype=np.float32)
        scores = np.zeros(len(self.items), dtype=np.float32)
        for i, tokens in enumerate(self._token_sets):
            if not tokens:
                continue
            overlap = len(q_tokens & tokens) / len(q_tokens | tokens)
            scores[i] = overlap
        return scores

    def search(self, query: str, k: int = 20) -> list[CatalogItem]:
        if not query.strip():
            return self.items[:k]
        q_vec = self.embed_query(query)
        semantic = self.embeddings @ q_vec
        lexical = self._lexical_scores(query)
        combined = 0.75 * semantic + 0.25 * lexical
        top_idx = np.argsort(combined)[::-1][:k]
        return [self.items[i] for i in top_idx]

    def resolve_name(self, name: str) -> str | None:
        """Resolve alias or fuzzy name to exact catalog name."""
        cleaned = name.strip()
        if cleaned in self.by_name:
            return cleaned
        alias = ALIASES.get(cleaned.lower())
        if alias and alias in self.by_name:
            return alias
        matches = get_close_matches(cleaned, self.by_name.keys(), n=1, cutoff=0.75)
        if matches:
            return matches[0]
        # substring match for short names like "OPQ" in longer catalog names
        lower = cleaned.lower()
        for catalog_name in self.by_name:
            if lower in catalog_name.lower() and len(lower) >= 3:
                return catalog_name
        return None

    def get_by_name(self, name: str) -> CatalogItem | None:
        resolved = self.resolve_name(name)
        return self.by_name.get(resolved) if resolved else None

    def get_by_names(self, names: list[str]) -> list[CatalogItem]:
        found: list[CatalogItem] = []
        seen: set[str] = set()
        for name in names:
            item = self.get_by_name(name)
            if item and item.name not in seen:
                found.append(item)
                seen.add(item.name)
        return found

    def format_candidates(self, items: list[CatalogItem]) -> str:
        lines = []
        for item in items:
            levels = ", ".join(item.job_levels[:6])
            langs = ", ".join(item.languages[:4])
            lines.append(
                f"- {item.name} | test_type={item.test_type} | keys={', '.join(item.keys)} | "
                f"duration={item.duration or 'n/a'} | job_levels={levels or 'n/a'} | "
                f"languages={langs or 'n/a'} | {item.description[:220]}"
            )
        return "\n".join(lines)
