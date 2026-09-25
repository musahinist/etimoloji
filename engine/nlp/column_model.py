"""
Sütun modeli — hizalama sütunundan ata sesi (veya "bu sütun kökte yok")
seçen öğrenilmiş sınıflandırıcı.

Neden
-----
Öğrenilmiş örüntü tablosu (``proto_patterns``) her ``(dil, ses)`` çiftine
bağımsız oy verdirir ve sütunu HER ZAMAN bir sese çevirir. İki şeyi
göremez: sütunun bağlamını (konum, komşu sütunlar, hangi dillerin orada
bittiği) ve sütunun kökte hiç olmamasını (tanıklar altın kökten uzun:
yaχšï ~ *jak). Ölçüldü: tablo tahminlerinin %35'inde uzunluk bile tutmuyor.

Yöntem
------
Her bilgi taşıyan sütun için aday kümesi kurulur: sütundaki sesler, tablo
dağılımlarının önerdiği ata sesler, kural kararı, kural+tablo kararı ve
``∅`` (sütunu at). Her aday için seyrek özellikler çıkarılır (sütundaki
payı, arkaiklik ağırlıklı payı, tablo olasılığı, kural/tablo kararıyla
eşleşme, dil başına eşleşme, konum, ünlü uyumu; ∅ için sona uzaklık, boşluk
oranı, "tanık burada bitmiş" oranı, CV iskeleti). İkili lojistik regresyon
(L2, C=1) adayları puanlar; sütunda en yüksek puanlı aday seçilir. Bütün
sütunlar ∅ çıkarsa en güçlü ses adaylı sütun ses alır.

Eğitim etiketleri altın kök sütunlara DP ile hizalanarak üretilir (∅ =
kökte karşılığı yok); bu yüzden tablonun aksine uzunluğu tutmayan kümeler
de kullanılır. Ek eğitim verisi: Starling ``turcet`` kökleri (~1.750).

Özellikler kural/tablo kararlarını KULLANIR (yığınlama). Eğitim maddelerinin
tablo özellikleri, maddenin kendi katı hariç tutularak öğrenilmiş iç
tablolardan gelir; aksi hâlde tablo eğitim maddelerinde gerçekte olduğundan
iyi görünür ve model ona fazla güvenir.

Ölçüldü (``make eval-cv``, 5 kat, n=320; tanıksız kök yasağı dahil)::

    sistem           tam      NED      BCFS
    öğrenilmiş tablo 0,2719   0,3445   0,5437
    sütun modeli     0,2969   0,3303   0,5562
    fark (model−tablo): tam +0,025 GA[+0,006, +0,047] · NED −0,014
                        GA[−0,026, −0,003] · BCFS +0,013 GA[+0,001, +0,024]

Ön kayıtlı prototip (koruma YOK, eski taban 0,3426): NED −0,0205, Holm
düzeltmesi sonrası (tabloya karşı 4 karşılaştırma) p=0,008; tam doğruluk
Holm sonrası anlamlı değildi (p=0,19). Denenip katkı vermeyen: ünlü sütunu
silme yasağı + CV iskeleti + makullük kapısı paketi (NED +0,0006).

⚠️ Ön kayıt SONRASI eklenen tek değişiklik: küratörlü kararların korunması
(:data:`PROTECT_CURATED`). Korumasız model ``*teŋiŕ`` yerine ``*deniŕ``
üretiyor ve bilinen-biçim testini bozuyordu — ``proto_phonology`` adım 3
notunda tablo için ölçülmüş olan hatanın aynısı. Koruma altında model
tanısal/denklik sütunlarında yalnız ∅ diyebilir. Yukarıdaki ``eval-cv``
sayıları korumalı sürümündür. Dev (n=83) NED 0,3058 -> 0,3087: fark
gürültü düzeyinde, yön olumsuz.

⚠️ Sızıntı: Starling kökü, sınanan maddelerin veya TEST kavramlarının
Türkçe tanığıyla aynı TRK biçimini taşıyorsa ya da ata biçimi sınanan bir
altın kökle aynıysa eğitimden çıkarılır. Test bölümünden YALNIZ Türkçe
tanık biçimleri okunur; test altın kökleri hiçbir yerde okunmaz.

sklearn kullanılmaz: eğitim (L-BFGS) ve çıkarım bu dosyada, saf Python.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.config import PROJECT_ROOT
from engine.logging_setup import get_logger

logger = get_logger(__name__)

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "proto_column_model.json"
SCHEMA = "turkic-etymology-column-model/v1"

NULL = "∅"
#: L2 düzenlileştirme — ön kayıtta sabitlendi, CV katlarında ayarlanmadı.
C = 1.0
#: Tablo dağılımından aday sayılmak için gereken asgari olasılık.
TABLE_CANDIDATE_MIN = 0.05
K_INNER = 5
#: Kural katmanının küratörlü karar yolları (``pick_proto_sound`` adım 1–2).
CURATED_METHODS = frozenset({"tanisal", "denklik"})
PROTECT_CURATED = True

_VOWELS = set("aeıioöuüäâîû")
_FRONT = set("eiöüä")
KEY_LANGS = frozenset({
    "cv", "sah", "dlg", "otk", "klj", "tk", "az", "tr", "gag", "kk", "ky", "tt", "ba",
    "uz", "ug", "tyv", "alt", "khk", "kim", "qwm", "crh", "krc", "kum", "nog", "kaa",
    "slq", "ybe", "clw", "atv", "cjs", "bay", "kdr", "__anchor__", "wot",
})
_ANCHOR_PREFERENCE = ("tr", "az", "tk", "gag", "kk", "ky", "tt", "uz", "ug")


# --- ses yardımcıları -------------------------------------------------------
def norm(sound: str) -> str:
    from engine.utils.proto_notation import normalize_proto

    return normalize_proto(sound, strip_length=True)


def graphemes(text: str) -> list[str]:
    """Birleşik işaretleri harfine bağlı tutarak böler (``d̮`` tek birimdir)."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.combining(ch) and out:
            out[-1] += ch
        else:
            out.append(ch)
    return [unicodedata.normalize("NFC", g) for g in out]


def base(sound: str) -> str:
    from engine.utils.proto_notation import fold_transcription as _fold_transcription

    decomposed = unicodedata.normalize("NFD", _fold_transcription(sound.casefold()))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def is_vowel(sound: str) -> bool:
    return base(sound)[:1] in _VOWELS


# --- tablo ve kural kararları -----------------------------------------------
@contextmanager
def _pattern_table_as(table: Any) -> Iterator[None]:
    """``pick_proto_sound``u belirli bir tabloyla (veya tablosuz) koşturur.

    ⚠️ Modül düzeyi durumu geçici olarak değiştirir; iş parçacığı güvenli
    değildir (motorun geri kalanı gibi).
    """
    from engine.nlp import proto_phonology

    saved = (proto_phonology._PATTERN_TABLE, proto_phonology._PATTERN_TABLE_LOADED)
    proto_phonology._PATTERN_TABLE = table
    proto_phonology._PATTERN_TABLE_LOADED = True
    try:
        yield
    finally:
        proto_phonology._PATTERN_TABLE, proto_phonology._PATTERN_TABLE_LOADED = saved


def _table_dist(table: Any, present: dict[str, str]) -> dict[str, float]:
    from engine.nlp.proto_patterns import MIN_SUPPORT

    tally: Counter = Counter()
    if table is not None:
        for lang, sound in present.items():
            observed = table.counts.get((lang, sound))
            if not observed:
                continue
            total = sum(observed.values())
            if total < MIN_SUPPORT:
                continue
            for proto, count in observed.items():
                tally[norm(proto)] += count / total
    z = sum(tally.values())
    return {k: v / z for k, v in tally.items()} if z else {}


def column_context(informative: list[Any], table: Any) -> list[dict[str, Any]]:
    """Sütun başına: kural kararı, kural+tablo kararı, tablo dağılımı."""
    from engine.nlp.proto_phonology import pick_proto_sound

    n = len(informative)
    ctx = []
    for i, col in enumerate(informative):
        pos = "initial" if i == 0 else ("final" if i == n - 1 else "medial")
        with _pattern_table_as(None):
            rules = pick_proto_sound(col, pos)
        with _pattern_table_as(table):
            hybrid = pick_proto_sound(col, pos)
        ctx.append({
            "pos": pos,
            "rules": norm(rules.sound) if rules.sound else "",
            "hyb": norm(hybrid.sound) if hybrid.sound else "",
            "method": hybrid.method,
            "decision": hybrid,
            "tab": _table_dist(table, col.present),
        })
    return ctx


def candidates_for(column: Any, ctx: dict[str, Any]) -> list[str]:
    cands = {norm(s) for s in column.present.values()}
    cands |= {p for p, v in ctx["tab"].items() if v >= TABLE_CANDIDATE_MIN}
    cands |= {ctx["rules"], ctx["hyb"]}
    cands.discard("")
    return sorted(cands) + [NULL]


def features(informative: list[Any], ctx: list[dict[str, Any]], i: int, cand: str) -> dict[str, float]:
    from engine.nlp.multi_alignment import GAP
    from engine.nlp.proto_phonology import weight_for

    col, c = informative[i], ctx[i]
    n = len(informative)
    present = col.present
    langs = sorted(col.sounds)
    f: dict[str, float] = {}
    pos = c["pos"]
    if cand == NULL:
        ended = sum(
            1 for lang in langs
            if all(not informative[j].sounds.get(lang) or informative[j].sounds.get(lang) == GAP
                   for j in range(i, n))
        )
        skel = "".join("V" if is_vowel(x["hyb"]) else "C" for x in ctx[: i + 1])
        f["null"] = 1
        f[f"null&pos={pos}"] = 1
        f[f"null&dend={min(n - 1 - i, 3)}"] = 1
        f[f"null&idx={min(i, 6)}"] = 1
        f[f"null&n={min(n, 8)}"] = 1
        f["null&gap"] = col.gap_ratio
        f["null&ended"] = ended / max(1, len(langs))
        f[f"null&hyb={c['hyb']}"] = 1
        f[f"null&skel={skel[-4:]}"] = 1
        f[f"null&nv={skel.count('V')}"] = 1
        f["null&agree"] = max(Counter(norm(s) for s in present.values()).values()) / max(1, len(present))
        f[f"null&nlang={min(len(present), 10) // 3}"] = 1
        return f
    vals = [norm(s) for s in present.values()]
    share = sum(v == cand for v in vals) / max(1, len(vals))
    wtot = sum(weight_for(lang, s) for lang, s in present.items()) or 1
    wshare = sum(weight_for(lang, s) for lang, s in present.items() if norm(s) == cand) / wtot
    tab = c["tab"].get(cand, 0.0)
    v = is_vowel(cand)
    f["share"] = share
    f["wshare"] = wshare
    f["tab"] = tab
    f["absent"] = float(share == 0)
    f["tabmax"] = float(bool(c["tab"]) and max(c["tab"], key=c["tab"].get) == cand)
    f["is_rules"] = float(cand == c["rules"])
    f["is_hyb"] = float(cand == c["hyb"])
    f[f"is_hyb&{c['method']}"] = float(cand == c["hyb"])
    f[f"cand={cand}"] = 1
    f[f"v={v}&pos={pos}"] = 1
    f[f"v={v}&share"] = share
    f[f"v={v}&tab"] = tab
    f[f"hyb>{c['hyb']}>{cand}"] = 1
    if pos == "initial":
        f[f"init={cand}"] = 1
    for lang, s in present.items():
        if lang in KEY_LANGS and norm(s) == cand:
            f[f"m_{lang}&v={v}"] = 1
    if v:
        others = [x["hyb"] for j, x in enumerate(ctx) if j != i and is_vowel(x["hyb"])]
        if others:
            front_share = sum(base(o)[:1] in _FRONT for o in others) / len(others)
            f["harm"] = front_share if base(cand)[:1] in _FRONT else 1 - front_share
    return f


# --- etiketler: altın kök <-> sütunlar ---------------------------------------
def label_columns(gold_candidates: Iterable[str], columns: list[Any]) -> list[str] | None:
    """Altın kökü sütunlara hizalar: sütun başına ata ses veya ``∅``.

    Maliyetler: sütunu atmak 0,5; karşılıksız altın ses 0,7; eşleşme 0
    (aynı ses) / 0,2 (aynı temel harf) / 0,5 (aynı sınıf) / 1,0.
    Eşdeğer altın biçimlerden en ucuz hizalanan seçilir.
    """
    best: tuple[float, list[str]] | None = None
    colsets = [{norm(s) for s in c.present.values()} for c in columns]
    colbases = [{base(s) for s in c.present.values()} for c in columns]
    n = len(columns)
    for gold in gold_candidates:
        gs = graphemes(norm(gold))
        if not gs:
            continue
        m = len(gs)
        inf = float("inf")
        dist = [[inf] * (m + 1) for _ in range(n + 1)]
        back: list[list[tuple[str, int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
        dist[0][0] = 0.0
        for i in range(n + 1):
            for j in range(m + 1):
                here = dist[i][j]
                if here == inf:
                    continue
                if i < n and here + 0.5 < dist[i + 1][j]:
                    dist[i + 1][j], back[i + 1][j] = here + 0.5, ("drop", i, j)
                if j < m and here + 0.7 < dist[i][j + 1]:
                    dist[i][j + 1], back[i][j + 1] = here + 0.7, ("ins", i, j)
                if i < n and j < m:
                    p = gs[j]
                    if p in colsets[i]:
                        cost = 0.0
                    elif base(p) in colbases[i]:
                        cost = 0.2
                    elif any(is_vowel(s) == is_vowel(p) for s in colsets[i]):
                        cost = 0.5
                    else:
                        cost = 1.0
                    if here + cost < dist[i + 1][j + 1]:
                        dist[i + 1][j + 1], back[i + 1][j + 1] = here + cost, ("sub", i, j)
        labels = [NULL] * n
        i, j = n, m
        while (i, j) != (0, 0):
            op, pi, pj = back[i][j]  # type: ignore[misc]
            if op == "sub":
                labels[pi] = gs[pj]
            i, j = pi, pj
        if best is None or dist[n][m] < best[0]:
            best = (dist[n][m], labels)
    return best[1] if best else None


# --- model ----------------------------------------------------------------
def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class ColumnModel:
    weights: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def prob(self, feats: dict[str, float]) -> float:
        w = self.weights
        return _sigmoid(self.intercept + sum(v * w.get(k, 0.0) for k, v in feats.items()))

    def score_columns(self, informative: list[Any], table: Any) -> tuple[list[dict[str, Any]], list[list[tuple[float, str]]]]:
        ctx = column_context(informative, table)
        scored = []
        for i, col in enumerate(informative):
            cands = candidates_for(col, ctx[i])
            probs = [(self.prob(features(informative, ctx, i, cd)), cd) for cd in cands]
            scored.append(sorted(probs, key=lambda t: (-t[0], t[1])))
        return ctx, scored

    def decide(self, informative: list[Any], table: Any) -> list[Any]:
        """Sütun başına ``ColumnDecision``; ∅ için ses ``""`` döner."""
        from engine.nlp.proto_phonology import ColumnDecision

        ctx, scored = self.score_columns(informative, table)
        choice = [s[0][1] for s in scored]
        if choice and all(c == NULL for c in choice):
            # "En az bir ses" koruması: motor boş kök üretmez.
            best_i = max(range(len(scored)), key=lambda i: max(p for p, s in scored[i] if s != NULL))
            choice[best_i] = next(s for p, s in scored[best_i] if s != NULL)
        if PROTECT_CURATED:
            # Elle yazılmış tanısal/denklik kararları öğrenilmiş sese yenilmez
            # (``proto_phonology`` adım 3 notu: *teŋiŕ -> *teniŕ). Model yalnız
            # "sütun kökte yok" (∅) diyebilir.
            for i, c in enumerate(ctx):
                if c["method"] in CURATED_METHODS and choice[i] != NULL and c["hyb"]:
                    choice[i] = c["hyb"]
        decisions = []
        for i, sound in enumerate(choice):
            hybrid = ctx[i]["decision"]
            if sound == NULL:
                decisions.append(ColumnDecision(
                    "", "sütun modeli: sütun kökte yok (ek/eklenti)", "sutun_modeli_bos",
                    hybrid.agreement,
                ))
                continue
            alternatives = tuple((s, round(p, 4)) for p, s in scored[i] if s != NULL)
            prob = next(p for p, s in scored[i] if s == sound)
            decisions.append(ColumnDecision(
                sound,
                f"sütun modeli ({prob:.2f})",
                "sutun_modeli",
                hybrid.agreement,
                is_diagnostic=hybrid.is_diagnostic and norm(hybrid.sound) == sound,
                alternatives=alternatives,
            ))
        return decisions

    def as_dict(self) -> dict[str, Any]:
        return {
            "_schema": SCHEMA,
            **self.meta,
            "intercept": self.intercept,
            "weights": dict(sorted(self.weights.items())),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnModel:
        if data.get("_schema") != SCHEMA:
            raise ValueError(f"beklenmeyen şema: {data.get('_schema')}")
        meta = {k: v for k, v in data.items() if k not in ("_schema", "intercept", "weights")}
        return cls({k: float(v) for k, v in data["weights"].items()}, float(data["intercept"]), meta)


# --- yükleme (motor kancası) -------------------------------------------------
_MODEL: ColumnModel | None = None
_MODEL_LOADED = False


def load(path: Path | None = None) -> ColumnModel | None:
    source = path or MODEL_PATH
    if not source.exists():
        return None
    try:
        return ColumnModel.from_dict(json.loads(source.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        logger.warning("Sütun modeli okunamadı: %s", source, exc_info=True)
        return None


def active_model() -> ColumnModel | None:
    global _MODEL, _MODEL_LOADED
    if not _MODEL_LOADED:
        _MODEL = load()
        _MODEL_LOADED = True
    return _MODEL


def set_model(model: ColumnModel | None) -> None:
    """Ölçüm için etkin modeli değiştirir (``None`` = model kapalı)."""
    global _MODEL, _MODEL_LOADED
    _MODEL, _MODEL_LOADED = model, True


def reset_model_cache() -> None:
    global _MODEL, _MODEL_LOADED
    _MODEL, _MODEL_LOADED = None, False


def decide(informative: list[Any]) -> list[Any] | None:
    """Motor kancası: model yüklüyse sütun kararları, değilse ``None``."""
    model = active_model()
    if model is None or not informative:
        return None
    from engine.nlp.proto_phonology import _pattern_table

    return model.decide(informative, _pattern_table())


def column_features(informative: list[Any], decisions: list[Any]) -> dict[str, Any]:
    """Sütun kararlarından güven özellikleri — yalnız RAPORLANIR, hiçbir karar bunlarla verilmez.

    * ``column_margin_min`` / ``column_margin_mean``: sütunda ilk iki aday
      puanı arasındaki fark (tek aday varsa fark = ilk puan).
    * ``nbest_margin``: N-best listesinde ilk iki biçmin puan farkı.
    * ``table_vote_confidence_min`` / ``_mean``: öğrenilmiş tablonun sütun
      oyu güveni (tablo oy veremiyorsa 0).
    * ``agreement_min``: en düşük sütun uyumu.
    * ``method_shares``: karar yollarının payları.
    """
    from engine.nlp.nbest_reranking import generate
    from engine.nlp.proto_phonology import _pattern_table

    if not decisions:
        return {}
    margins = []
    for d in decisions:
        scores = sorted((s for _, s in (d.alternatives or ())), reverse=True)
        if not scores:
            scores = [1.0] if d.sound else [0.0]
        margins.append(scores[0] - (scores[1] if len(scores) > 1 else 0.0))
    nbest = generate(decisions)
    nbest_margin = (nbest[0][1] - nbest[1][1]) if len(nbest) > 1 else (nbest[0][1] if nbest else 0.0)
    table = _pattern_table()
    votes = [table.vote(c.present)[1] if table is not None else 0.0 for c in informative]
    methods = Counter(d.method for d in decisions)
    return {
        "column_margin_min": round(min(margins), 4),
        "column_margin_mean": round(sum(margins) / len(margins), 4),
        "nbest_margin": round(float(nbest_margin), 4),
        "table_vote_confidence_min": round(min(votes), 4) if votes else 0.0,
        "table_vote_confidence_mean": round(sum(votes) / len(votes), 4) if votes else 0.0,
        "agreement_min": round(min(d.agreement for d in decisions), 4),
        "method_shares": {m: round(c / len(decisions), 4) for m, c in sorted(methods.items())},
    }


# --- eğitim verisi -----------------------------------------------------------
def engine_forms(witnesses: list[dict[str, str]]) -> dict[str, str]:
    """Harness'in motora verdiği girdiden ``reconstruct``un hizaladığı biçimler.

    ``ComparativeReconstructor.reconstruct`` ile aynı kural: çapa dili girdiden
    çıkar, dil başına en kısa biçim (>=2 harf), çapa ayrı tanık olarak girer.
    """
    from engine.evaluation.harness import _anchor_for
    from engine.fetchers.base import TURKIC_LANGUAGES_MAP
    from engine.utils.orthography import to_comparison_form

    anchor_raw, anchor_lang = _anchor_for(witnesses)
    anchor = to_comparison_form(anchor_raw)
    by_lang: dict[str, str] = {}
    for w in witnesses:
        code = w["lang_code"]
        if code == anchor_lang or code not in TURKIC_LANGUAGES_MAP:
            continue
        form = to_comparison_form(w.get("word") or "")
        if len(form) >= 2 and (code not in by_lang or len(form) < len(by_lang[code])):
            by_lang[code] = form
    forms = dict(by_lang)
    if anchor and anchor not in by_lang.values():
        forms["__anchor__"] = anchor
    return forms


def informative_columns(forms: dict[str, str]) -> list[Any]:
    from engine.nlp.multi_alignment import align_forms

    if len(forms) < 2:
        return []
    return [c for c in align_forms(forms) if c.gap_ratio <= 0.5]


def table_observations(item: Any, mapping: dict[str, str]) -> list[tuple[str, str, str]]:
    """``crossval.learn_table`` / ``proto_patterns.learn`` ile aynı kuralla gözlemler."""
    from engine.nlp.multi_alignment import align_forms
    from engine.utils.orthography import to_comparison_form

    proto = to_comparison_form(item.gold_form)
    forms = {mapping[k]: to_comparison_form(v) for k, v in item.witnesses.items()
             if k in mapping and to_comparison_form(v)}
    if len(forms) < 2 or not proto:
        return []
    columns = [c for c in align_forms(forms) if c.gap_ratio <= 0.5]
    if len(columns) != len(proto):
        return []
    return [(lang, s, p) for c, p in zip(columns, proto, strict=True) for lang, s in c.present.items() if s]


def table_from(observations: Iterable[list[tuple[str, str, str]]]) -> Any:
    from engine.nlp.proto_patterns import ProtoPatternTable

    table = ProtoPatternTable(trained_on="column_model/inner")
    for obs in observations:
        if obs:
            table.n_sets += 1
        for lang, sound, proto in obs:
            table.n_columns += 1
            table.observe(lang, sound, proto)
    return table


@dataclass
class GoldSet:
    """Eğitim/ölçüm için hazırlanmış altın madde."""

    item: Any
    informative: list[Any]
    observations: list[tuple[str, str, str]]
    turkish: set[str]


def prepare_gold(items: list[Any], mapping: dict[str, str]) -> dict[str, GoldSet]:
    from engine.evaluation.harness import _witnesses_for

    out = {}
    for item in items:
        out[item.set_id] = GoldSet(
            item=item,
            informative=informative_columns(engine_forms(_witnesses_for(item, mapping))),
            observations=table_observations(item, mapping),
            turkish=gold_turkish(item, mapping),
        )
    return out


def gold_turkish(item: Any, mapping: dict[str, str]) -> set[str]:
    from engine.utils.orthography import to_comparison_form

    return {to_comparison_form(f) for lang, f in item.witnesses.items() if mapping.get(lang) == "tr" and f}


# --- Starling ----------------------------------------------------------------
def starling_first_form(text: str) -> str:
    """Starling alanından ilk biçim: anlamlar, (künyeler/isteğe bağlı parça), tireler atılır."""
    text = re.sub(r"'[^']*'", " ", text or "")
    text = re.sub(r"\([^)]*\)", " ", text)
    for token in re.split(r"[,;/]", text):
        token = token.strip()
        if not token:
            continue
        token = token.split()[0].strip(".0123456789").replace("-", "")
        if token:
            return token
    return ""


def starling_protos(proto: str) -> list[str]:
    """``*büŕ- / *bür-`` -> ``[*büŕ, *bür]``; isteğe bağlı parça ``(re)`` atılır."""
    out = []
    for part in re.split(r"[,/]", proto):
        part = re.sub(r"\([^)]*\)", "", part).replace("-", "").replace("*", "").strip()
        if part:
            out.append("*" + part)
    return out


def starling_form(raw: str) -> str:
    from engine.utils.orthography import to_comparison_form

    # Starling ``j`` = Türk imlası ``y`` (savelyevturkic çevriyazısı ``y`` kullanır).
    return to_comparison_form(raw.replace("j", "y"))


@dataclass
class StarlingSet:
    number: int
    protos: tuple[str, ...]
    turkish: set[str]
    informative: list[Any]


def prepare_starling() -> list[StarlingSet]:
    """Starling ``turcet`` köklerini harness girdisi gibi hizalar (kat bağımsız)."""
    from engine.db.starling import FIELD_LANGUAGES, _turkish_forms, load_turcet
    from engine.utils.orthography import to_comparison_form

    out = []
    for etym in load_turcet():
        protos = starling_protos(etym.proto)
        if not protos:
            continue
        by_lang = {}
        for fld, code in FIELD_LANGUAGES.items():
            form = starling_form(starling_first_form(etym.reflexes.get(fld, "")))
            if len(form) >= 2:
                by_lang[code] = form
        if len(by_lang) < 2:
            continue
        anchor_lang = next((c for c in _ANCHOR_PREFERENCE if c in by_lang), next(iter(by_lang)))
        anchor = by_lang.pop(anchor_lang)
        forms = dict(by_lang)
        if anchor not in forms.values():
            forms["__anchor__"] = anchor
        informative = informative_columns(forms)
        if not informative:
            continue
        turkish = {to_comparison_form(x.rstrip("-")) for x in _turkish_forms(etym.reflexes.get("TRK", ""))}
        out.append(StarlingSet(etym.number, tuple(protos), turkish, informative))
    return out


def starling_allowed(sets: list[StarlingSet], excluded_turkish: set[str], excluded_protos: set[str]) -> list[StarlingSet]:
    """Sızıntı süzgeci: sınanan/test Türkçe biçimi veya sınanan altın kökle çakışan kökler çıkar."""
    return [
        s for s in sets
        if not (s.turkish & excluded_turkish) and not ({norm(p) for p in s.protos} & excluded_protos)
    ]


# --- eğitim -----------------------------------------------------------------
Row = tuple[dict[str, float], int]


def _rows_for(informative: list[Any], gold_candidates: Iterable[str], table: Any) -> list[Row]:
    labels = label_columns(gold_candidates, informative)
    if labels is None:
        return []
    ctx = column_context(informative, table)
    rows = []
    for i in range(len(informative)):
        for cand in candidates_for(informative[i], ctx[i]):
            rows.append((features(informative, ctx, i, cand), int(cand == labels[i])))
    return rows


def fit_logistic(rows: list[Row], c: float = C, max_iter: int = 2000, tol: float = 1e-10) -> tuple[dict[str, float], float]:
    """L2 lojistik regresyon, L-BFGS, saf Python, deterministik.

    Amaç: ``Σ logloss + ||w||² / (2C)`` (sabit terim cezasız) — sklearn
    ``LogisticRegression(C=...)`` ile aynı amaç.
    """
    names = sorted({k for feats, _ in rows for k in feats})
    index = {k: i for i, k in enumerate(names)}
    data = [([index[k] for k in feats], list(feats.values()), y) for feats, y in rows]
    dim = len(names) + 1  # son eleman: sabit

    def evaluate(w: list[float]) -> tuple[float, list[float]]:
        grad = [0.0] * dim
        loss = 0.0
        b = w[-1]
        for idx, vals, y in data:
            z = b
            for j, v in zip(idx, vals, strict=True):
                z += w[j] * v
            loss += (math.log1p(math.exp(z)) if z < 30 else z) - y * z
            r = _sigmoid(z) - y
            for j, v in zip(idx, vals, strict=True):
                grad[j] += r * v
            grad[-1] += r
        for j in range(dim - 1):
            loss += w[j] * w[j] / (2 * c)
            grad[j] += w[j] / c
        return loss, grad

    w = [0.0] * dim
    f, g = evaluate(w)
    history: list[tuple[list[float], list[float], float]] = []
    for _ in range(max_iter):
        # iki döngülü özyineleme
        q = g[:]
        alphas = []
        for s, y, rho in reversed(history):
            a = rho * sum(si * qi for si, qi in zip(s, q, strict=True))
            alphas.append(a)
            q = [qi - a * yi for qi, yi in zip(q, y, strict=True)]
        if history:
            s, y, _ = history[-1]
            gamma = sum(si * yi for si, yi in zip(s, y, strict=True)) / sum(yi * yi for yi in y)
            q = [gamma * qi for qi in q]
        for (s, y, rho), a in zip(history, reversed(alphas), strict=True):
            bcoef = rho * sum(yi * qi for yi, qi in zip(y, q, strict=True))
            q = [qi + si * (a - bcoef) for qi, si in zip(q, s, strict=True)]
        direction = [-qi for qi in q]
        slope = sum(di * gi for di, gi in zip(direction, g, strict=True))
        if slope >= 0:
            direction, slope, history = [-gi for gi in g], -sum(gi * gi for gi in g), []
        step = 1.0
        while True:
            w_new = [wi + step * di for wi, di in zip(w, direction, strict=True)]
            f_new, g_new = evaluate(w_new)
            if f_new <= f + 1e-4 * step * slope or step < 1e-10:
                break
            step *= 0.5
        s = [a - b for a, b in zip(w_new, w, strict=True)]
        y = [a - b for a, b in zip(g_new, g, strict=True)]
        sy = sum(si * yi for si, yi in zip(s, y, strict=True))
        if sy > 1e-12:
            history.append((s, y, 1.0 / sy))
            history = history[-10:]
        converged = abs(f - f_new) <= tol * max(1.0, abs(f))
        w, f, g = w_new, f_new, g_new
        if converged or max(abs(x) for x in g) < 1e-5:
            break
    weights = {name: w[i] for i, name in enumerate(names) if abs(w[i]) > 1e-9}
    return weights, w[-1]


def train(
    gold: dict[str, GoldSet],
    train_ids: list[str],
    starling: list[StarlingSet],
    table: Any,
    *,
    fold_of: Any,
) -> tuple[ColumnModel, int]:
    """Modeli eğitir.

    :param train_ids: eğitim altın maddeleri.
    :param starling: sızıntı süzgecinden GEÇMİŞ Starling kökleri.
    :param table: sınanan maddelerde kullanılacak tablo (Starling özellikleri
        bununla çıkarılır — Starling kökleri tabloyu eğitmediği için içeriden
        bakış yoktur).
    :param fold_of: kavram -> iç kat; eğitim maddelerinin tablo özellikleri
        kendi iç katı hariç öğrenilmiş tablodan gelir.
    """
    folds = {sid: fold_of(gold[sid].item.concept) for sid in train_ids}
    inner = {
        g: table_from(gold[sid].observations for sid in train_ids if folds[sid] != g)
        for g in sorted(set(folds.values()))
    }
    rows: list[Row] = []
    for sid in train_ids:
        gs = gold[sid]
        if gs.informative:
            rows += _rows_for(gs.informative, gs.item.gold_candidates, inner[folds[sid]])
    for s in starling:
        rows += _rows_for(s.informative, s.protos, table)
    weights, intercept = fit_logistic(rows)
    model = ColumnModel(weights, intercept, {
        "C": C,
        "n_gold_sets": len(train_ids),
        "n_starling_sets": len(starling),
        "n_rows": len(rows),
    })
    return model, len(starling)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def train_release(dataset: str = "savelyevturkic") -> ColumnModel:
    """Yayın modeli: ``train`` bölümü + Starling (dev/test Türkçe biçimleri ve dev kökleri hariç).

    Tablo özellikleri için ``proto_patterns`` ile aynı şekilde yalnız
    ``train``den öğrenilmiş tablo kullanılır; ``make eval-baseline`` (dev)
    böylece temiz kalır.
    """
    from engine.db.cldf_wordlist import CldfWordlist
    from engine.db.language_mapping import build_mapping
    from engine.db.starling import STARLING_DIR
    from engine.evaluation.crossval import fold_of
    from engine.evaluation.gold import GoldStandard

    gold_std = GoldStandard.build(dataset)
    mapping = build_mapping(CldfWordlist.load(dataset))
    train_items = gold_std.split("train")
    dev_items = gold_std.split("dev")
    gold = prepare_gold(train_items, mapping)
    table = table_from(g.observations for g in gold.values())
    # ⚠️ Test bölümünden YALNIZ Türkçe tanık biçimi okunur (sızıntı süzgeci);
    # altın kök okunmaz. ``split('test')`` kilidi bilinçli olarak kullanılmıyor.
    excluded_turkish = set().union(
        *(gold_turkish(it, mapping) for it in gold_std.items if it.split in ("dev", "test"))
    )
    excluded_protos = {norm(g) for it in dev_items for g in it.gold_candidates}
    starling_all = prepare_starling()
    if not starling_all:
        raise SystemExit("Starling turcet yok: önce `make starling`.")
    starling = starling_allowed(starling_all, excluded_turkish, excluded_protos)
    model, _ = train(gold, list(gold), starling, table, fold_of=fold_of)
    model.meta.update({
        "trained_on": f"{dataset}/train + starling/turcet",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_starling_available": len(starling_all),
        "starling_sha256": {
            "turcet.dbf": _sha256(STARLING_DIR / "turcet.dbf"),
            "turcet.var": _sha256(STARLING_DIR / "turcet.var"),
        },
        "gold_source_ref": gold_std.source_ref,
    })
    return model


def save(model: ColumnModel, path: Path | None = None) -> Path:
    target = path or MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model.as_dict(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Sütun modeli (ata ses / ∅) eğitimi")
    ap.add_argument("--train", action="store_true", help="yayın modelini eğit ve kaydet")
    args = ap.parse_args()
    if not args.train:
        ap.print_help()
        return 1
    model = train_release()
    path = save(model)
    print(
        f"{model.meta['n_gold_sets']} altın küme · {model.meta['n_starling_sets']} Starling kökü "
        f"({model.meta['n_starling_available']} içinden) · {model.meta['n_rows']} satır · "
        f"{len(model.weights)} ağırlık · {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
