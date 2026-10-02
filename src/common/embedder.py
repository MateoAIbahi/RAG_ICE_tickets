"""
Modèle d'embedding partagé par l'API et l'ingestion.

Un seul modèle chargé une seule fois par processus (singleton), inférence
sérialisée par un verrou pour éviter les pics de RAM quand plusieurs requêtes
arrivent en même temps sur l'API.

Variables d'environnement :
  EMBED_MODEL         modèle HF (défaut BAAI/bge-m3, ~2,3 Go en float32)
  EMBED_DIM           dimension attendue en base (défaut 1024) ; contrôlée au chargement
  EMBED_POOLING       cls (bge-m3) | mean (famille e5)
  EMBED_QUERY_PREFIX  préfixe ajouté aux questions (ex. "query: " pour e5)
  EMBED_DOC_PREFIX    préfixe ajouté aux passages (ex. "passage: " pour e5)
  EMBED_MAX_TOKENS    taille max d'un chunk, en tokens (défaut 512)
  EMBED_OVERLAP       recouvrement entre chunks, en tokens (défaut 64)
  EMBED_BATCH_SIZE    taille de batch d'inférence (défaut 8 ; baisser si RAM juste)
  EMBED_THREADS       threads CPU torch (défaut : choix de torch)
  EMBED_DTYPE         float32 (défaut) | bfloat16 (RAM / 2, mais lent sur vieux CPU)
  EMBED_BACKEND       hf (défaut) | fake (tests uniquement, sans modèle)
"""
import hashlib
import math
import os
import re
import threading

_LOCK = threading.Lock()
_INSTANCE = None


def get_embedder():
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                backend = os.getenv("EMBED_BACKEND", "hf")
                _INSTANCE = _FakeEmbedder() if backend == "fake" else _HFEmbedder()
    return _INSTANCE


class _BaseEmbedder:
    max_tokens: int
    overlap: int
    query_prefix: str
    doc_prefix: str

    def _token_spans(self, text: str) -> list[tuple[int, int]]:
        raise NotImplementedError

    def _encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return len(self._token_spans(text))

    def chunk(self, text: str, header: str = "") -> list[str]:
        """
        Découpe `text` en morceaux qui tiennent dans le modèle UNE FOIS le header
        ajouté. Les découpes se font sur des frontières de tokens et on recopie
        le texte d'origine (pas de decode), donc rien n'est altéré ni perdu.
        """
        text = (text or "").strip()
        if not text:
            return []
        budget = self.max_tokens - self.count_tokens(header + self.doc_prefix) - 8
        budget = max(budget, 64)
        spans = self._token_spans(text)
        if len(spans) <= budget:
            return [text]

        step = max(budget - self.overlap, 1)
        chunks = []
        for start in range(0, len(spans), step):
            window = spans[start:start + budget]
            chunks.append(text[window[0][0]:window[-1][1]].strip())
            if start + budget >= len(spans):
                break
        return [c for c in chunks if c]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]


class _HFEmbedder(_BaseEmbedder):
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.name = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
        self.pooling = os.getenv("EMBED_POOLING", "cls")
        self.query_prefix = os.getenv("EMBED_QUERY_PREFIX", "")
        self.doc_prefix = os.getenv("EMBED_DOC_PREFIX", "")
        self.max_tokens = int(os.getenv("EMBED_MAX_TOKENS", "512"))
        self.overlap = int(os.getenv("EMBED_OVERLAP", "64"))
        self.batch_size = int(os.getenv("EMBED_BATCH_SIZE", "8"))
        threads = os.getenv("EMBED_THREADS")
        if threads:
            torch.set_num_threads(int(threads))
        dtype = torch.bfloat16 if os.getenv("EMBED_DTYPE") == "bfloat16" else torch.float32

        print(f"[EMBED] Chargement {self.name} ({dtype}, pooling={self.pooling})")
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModel.from_pretrained(
            self.name, torch_dtype=dtype, low_cpu_mem_usage=True
        ).eval()

        dim = self.model.config.hidden_size
        expected = int(os.getenv("EMBED_DIM", "1024"))
        if dim != expected:
            raise RuntimeError(
                f"Le modèle {self.name} produit des vecteurs de dimension {dim} "
                f"mais EMBED_DIM={expected}. Il faut une migration de la colonne "
                f"`embedding` + une réindexation complète avant de changer de modèle."
            )
        self.dim = dim

    def _token_spans(self, text):
        enc = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )
        return [s for s in enc["offset_mapping"] if s[1] > s[0]]

    def _encode(self, texts):
        torch = self.torch
        out = []
        with _LOCK, torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_tokens, return_tensors="pt",
                )
                hidden = self.model(**enc).last_hidden_state
                if self.pooling == "mean":
                    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                else:
                    vec = hidden[:, 0]
                vec = torch.nn.functional.normalize(vec.float(), dim=-1)
                out.extend(vec.tolist())
        return out


class _FakeEmbedder(_BaseEmbedder):
    """Embedding par hachage de mots : déterministe, sans modèle. Tests uniquement."""

    _WORD = re.compile(r"\w+", re.UNICODE)

    def __init__(self):
        self.dim = int(os.getenv("EMBED_DIM", "1024"))
        self.max_tokens = int(os.getenv("EMBED_MAX_TOKENS", "512"))
        self.overlap = int(os.getenv("EMBED_OVERLAP", "64"))
        self.query_prefix = ""
        self.doc_prefix = ""

    def _token_spans(self, text):
        return [m.span() for m in self._WORD.finditer(text)]

    def _encode(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in self._WORD.findall(t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out
