"""
Anlam bilgili verici eşleştirmesi — ``DonorIndex.by_sense`` adaylarını süzer (plan X3).

Sorun: ``by_sense`` anlamın 2 harften uzun ilk 6 sözcüğünden HERHANGİ biri
verici maddesinin anlamında geçince adayı kabul eder (FTS ``OR``). Ölçüldü
(Türk dilleri arası altın, AYAR bölümü, n=1.379): verici yakınlığı rampası
(SCA 0,35–0,60, şans denetimsiz) mirasların %55,7'sinde, alıntıların
%16,9'unda ateşleniyor; ateşlenen mirasların %97,5'inde eşleşme TEK
sözcükten. Elle bakılan 50 yanlış pozitifin 20'si anlam uyumsuz (``make``,
``word``, ``all``, ``much`` gibi genel sözcükler, özel adlar, yan anlamlar),
28'i anlam uyumlu ama biçim tesadüfü, 2'si gerçek ortak alıntı / etiket
hatası. Bkz. ``data/cache/work/xtr/PREREG_sense.md``.

Süzgeçler (ön-kayıtlı adaylar; biri seçilmeden üretimde KAPALI):

``s1:<τ>``
    e5-small (``intfloat/multilingual-e5-small``) kosinüsü: sorgu anlamı ile
    verici anlamı arasındaki benzerlik ≥ τ.
``s2:<k>``
    Çift yönlü eşleşme: ileri yön ``by_sense``in sözcük eşleşmesi; geri
    yönde verici anlamı başvuru kavram havuzunda (yerel CLDF
    ``parameters.csv`` adları, Concepticon hizalı) aranır ve sorgu anlamı
    ilk ``k`` içinde olmalı.
``s3:strict`` / ``s3:fallback``
    Tam kavram eşitliği: iki anlam da Concepticon kavramlarına (yerel CLDF
    ``parameters.csv``: NorthEuraLex, WOLD, savelyev …) eşlenir ve kesişim
    boş olmamalı. ``fallback``: sorgu anlamı hiçbir kavrama eşlenmiyorsa
    süzgeç uygulanmaz; ``strict``: o zaman aday kalmaz.

Bayrak: ``ETY_DONOR_SENSE_FILTER`` — boş/``0``/``off`` kapalı; ``1``/``on``
kabul edilen tanım (:data:`ACCEPTED_SPEC`; yoksa kapalı); ya da açık bir
tanım (``s1:0.86``).

⚠️ Şans denetimi ``donor_proximity.nearest_donor`` içinde SÜZÜLMÜŞ havuzla
kurulur: süzgeç havuzu küçültür, null da aynı havuza karşı ölçülmelidir.
"""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from engine.config import CLDF_DIR
from engine.logging_setup import get_logger

logger = get_logger(__name__)

#: e5 modeli (~0,5 GB bellek; yalnız S1/S2 açıkken yüklenir).
E5_MODEL = "intfloat/multilingual-e5-small"

#: Kavram eşlemesi ve geri arama havuzu için yerel CLDF kaynakları.
CONCEPT_SOURCES = (
    "northeuralex", "wold", "savelyevturkic", "robbeetstriangulation",
    "hruschkaturkic", "ronataswestoldturkic", "starostinaltaic",
)

#: Ön-kayıtlı R1 sınamasında kabul edilen tanım; ``None`` = hiçbiri kabul
#: edilmedi, ``ETY_DONOR_SENSE_FILTER=1`` süzgeci AÇMAZ.
ACCEPTED_SPEC: str | None = None

#: Anlam metni bu uzunlukta kırpılır (kaikki anlamları çok uzun olabiliyor).
MAX_GLOSS_CHARS = 200


@dataclass(frozen=True)
class SenseFilter:
    """Tek bir süzgeç tanımı."""

    kind: str  # "s1" | "s2" | "s3"
    tau: float = 0.0
    k: int = 0
    strict: bool = True

    @property
    def spec(self) -> str:
        if self.kind == "s1":
            return f"s1:{self.tau:.2f}"
        if self.kind == "s2":
            return f"s2:{self.k}"
        return "s3:" + ("strict" if self.strict else "fallback")

    def filter(self, sense: str, rows: Sequence[Any]) -> list[Any]:
        """``rows`` içinden anlamı sorguyla uyumlu olanları döndürür (sıra korunur)."""
        if not rows:
            return []
        glosses = [_gloss_of(row) for row in rows]
        if self.kind == "s1":
            sims = cosine_to(sense, glosses)
            return [row for row, s in zip(rows, sims, strict=True) if s >= self.tau]
        if self.kind == "s2":
            keep = reverse_rank_ok(sense, glosses, self.k)
            return [row for row, ok in zip(rows, keep, strict=True) if ok]
        query = concepts_of(sense)
        if not query:
            return [] if self.strict else list(rows)
        return [row for row, gloss in zip(rows, glosses, strict=True) if concepts_of(gloss) & query]


def parse_spec(spec: str) -> SenseFilter:
    kind, _, arg = spec.strip().partition(":")
    if kind == "s1":
        return SenseFilter("s1", tau=float(arg))
    if kind == "s2":
        return SenseFilter("s2", k=int(arg))
    if kind == "s3" and arg in ("", "strict", "fallback"):
        return SenseFilter("s3", strict=arg != "fallback")
    raise ValueError(f"bilinmeyen anlam süzgeci: {spec!r}")


@lru_cache(maxsize=8)
def _from_spec(spec: str) -> SenseFilter | None:
    value = spec.strip().lower()
    if value in ("", "0", "off", "false", "no"):
        return None
    if value in ("1", "on", "true", "yes"):
        return parse_spec(ACCEPTED_SPEC) if ACCEPTED_SPEC else None
    return parse_spec(value)


def filter_from_env() -> SenseFilter | None:
    """``ETY_DONOR_SENSE_FILTER`` bayrağının süzgeci (varsayılan kapalı)."""
    return _from_spec(os.environ.get("ETY_DONOR_SENSE_FILTER", ""))


def _gloss_of(row: Any) -> str:
    try:
        return row["gloss"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


# --- e5 ----------------------------------------------------------------------


@lru_cache(maxsize=1)
def _model() -> Any:
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(E5_MODEL, local_files_only=True)
    except Exception:
        return SentenceTransformer(E5_MODEL)


_EMBED_CACHE: dict[str, Any] = {}


def embed(texts: Iterable[str]) -> Any:
    """Normalize e5 gömmeleri (``query: `` önekiyle, simetrik benzerlik)."""
    import numpy as np

    texts = [(t or "")[:MAX_GLOSS_CHARS] for t in texts]
    missing = list(dict.fromkeys(t for t in texts if t not in _EMBED_CACHE))
    if missing:
        vectors = _model().encode([f"query: {t}" for t in missing], batch_size=128,
                                  normalize_embeddings=True, show_progress_bar=False)
        for text, vector in zip(missing, vectors, strict=True):
            _EMBED_CACHE[text] = vector.astype("float32")
        if len(_EMBED_CACHE) > 400_000:  # bellek sınırı
            _EMBED_CACHE.clear()
            return embed(texts)
    return np.stack([_EMBED_CACHE[t] for t in texts]) if texts else np.zeros((0, 384), "float32")


def cosine_to(sense: str, glosses: Sequence[str]) -> list[float]:
    if not sense or not glosses:
        return [0.0] * len(glosses)
    query = embed([sense])[0]
    return [float(x) for x in embed(glosses) @ query]


# --- kavram havuzu (Concepticon hizalı yerel CLDF) ----------------------------


def normalize_gloss(text: str) -> str:
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]|“[^”]*”|\"[^\"]*\"", " ", (text or "").lower())
    text = re.sub(r"^\s*(to|the|a|an)\s+", "", text.strip())
    return re.sub(r"\s+", " ", re.sub(r"[^a-zçğıöşü' -]+", " ", text)).strip()


def _parts(text: str) -> list[str]:
    """Anlamı alt anlamlara böler (``;`` ``,`` ``/``; ``X of Y:`` önekinden sonrası)."""
    text = (text or "")[:MAX_GLOSS_CHARS * 2]
    out = []
    for chunk in re.split(r"[;,/]", text):
        if ":" in chunk:
            chunk = chunk.rsplit(":", 1)[1]
        norm = normalize_gloss(chunk)
        if norm:
            out.append(norm)
    return out


@lru_cache(maxsize=1)
def concept_table() -> dict[str, frozenset[str]]:
    """Normalize ad -> Concepticon kimlikleri (yerel CLDF ``parameters.csv``)."""
    table: dict[str, set[str]] = {}
    for dataset in CONCEPT_SOURCES:
        path = CLDF_DIR / dataset / "parameters.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                cid = (row.get("Concepticon_ID") or "").strip()
                if not cid:
                    continue
                # ⚠️ NorthEuralex_Gloss Almanca ("Auge::N"): alınmaz.
                for name in (row.get("Name") or "", (row.get("Concepticon_Gloss") or "").lower()):
                    for part in _parts(name.replace(" OR ", ";")):
                        table.setdefault(part, set()).add(cid)
    return {k: frozenset(v) for k, v in table.items()}


@lru_cache(maxsize=200_000)
def concepts_of(text: str) -> frozenset[str]:
    table = concept_table()
    out: set[str] = set()
    for part in _parts(text):
        out |= table.get(part, frozenset())
    return frozenset(out)


@lru_cache(maxsize=1)
def reference_pool() -> tuple[tuple[str, ...], Any, Any]:
    """Geri arama havuzu: her Concepticon kavramı için bir ad + e5 gömmesi."""
    names: dict[str, str] = {}
    for dataset in CONCEPT_SOURCES:
        path = CLDF_DIR / dataset / "parameters.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                cid = (row.get("Concepticon_ID") or "").strip()
                gloss = (row.get("Concepticon_Gloss") or "").strip()
                if cid and gloss and cid not in names:
                    names[cid] = gloss.lower()
    import numpy as np

    texts = tuple(sorted(set(names.values())))
    return texts, np.array([normalize_gloss(t) for t in texts]), embed(texts)


def reverse_rank_ok(sense: str, glosses: Sequence[str], k: int) -> list[bool]:
    """Geri yön: verici anlamının en yakın ``k`` kavramı arasında sorgu anlamı var mı?

    Sorgu anlamı başvuru havuzuna eklenir (havuzda aynısı varsa o çıkarılır);
    sıra = havuzda verici anlamına sorgudan DAHA yakın kavram sayısı.
    """
    if not sense or not glosses:
        return [False] * len(glosses)
    _, normalized, pool = reference_pool()
    mask = normalized != normalize_gloss(sense)
    g = embed(glosses)
    q = embed([sense])[0]
    to_query = g @ q
    to_pool = g @ pool[mask].T
    return [int((row > s).sum()) < k for row, s in zip(to_pool, to_query, strict=True)]
