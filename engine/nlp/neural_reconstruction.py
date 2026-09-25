"""
Sinir ağı rekonstrüksiyon — sütun modeline ikinci üreteç adayı (D2, M7).

Neden
-----
Sütun modeli aday biçimleri aynı hizalama iskeletinde sütun sütun ses
değiştirerek üretir; uzunluk hatalarını (tahminlerin %35'i) ve ünlü
uyumu gibi bütünsel kısıtları göremez. Dizi-dönüştürücü (seq2seq) bu
kısıtları örtük öğrenir (Kim ve ark. 2023: Hóu'da SVM %15,5 -> %41).

Yöntem
------
Küçük kodlayıcı-çözücü karakter Transformer (d=128, 2+2 katman, ~0,7M
parametre). Girdi tek dizi: ``<src=sav|sta> <tgt=...> <anc> çapa <dil>
biçim <dil> biçim ...``. Kaynak belirteci (savelyev / Starling) iki yazım
geleneğini ayırır; çıkarım her zaman ``<src=sav>`` ile yapılır. Çok görevli:
``<tgt=proto>`` ata biçimi, ``<tgt=dil>`` o dilin refleksini (girdiden
çıkarılmış) tahmin eder — refleks görevi etiketsiz ön-eğitimin yerini
tutar (hazır ön-eğitimli Cognate Transformer ağırlığı yayımlanmamış;
depoda ``models/`` boş). Tanık düşürme artırması, ışın araması (5).

Ölçüldü — OLUMSUZ (``make eval-cv-neural``, 5 kat, n=320, 2026-09-25;
ön kayıt ``data/cache/work/neural/PREREG.md``)::

    sistem          tam      NED      BCFS
    sütun modeli    0,2969   0,3282   0,5566
    neural          0,2750   0,3394   0,5206
    neural − sütun: NED +0,011 GA[−0,014, +0,037] (Holm p=0,79) -> RED
                    tam −0,022 GA[−0,066, +0,022] · BCFS −0,036 GA[−0,061, −0,008]

İkinci aday (sütun top-1 ∪ ışın-5, sinir log-olasılığıyla seçim) sinirin
kendi top-1'iyle birebir aynı çıktı -> RED. Sinir modeli kendi ışınını
her zaman sütun adayından olası bulur; bu birleşim biçimi işe yaramaz.

Olumlu yan bulgu: **ışın-5 kapsamı 0,472** (doğru kök ilk 5 adayda);
sütun modelinin N-best tavanı 500 adayda 0,338. Uzunluk hatalarını da
kapsayan aday üretimi mümkün — ama top-1 seçimi sütun modelinden kötü.
Sonraki deneme: sinir ışınını ADAY ÜRETECİ, sütun/özellik tabanlı bir
sıralayıcıyı SEÇİCİ yapmak (ön kayda yeni aday olarak).

Sinir üretir, sütun modeli seçer (ön kayıt 2, ışın 10; kapsam 5'te 0,481,
10'da 0,553). Sütun modeline karşı (NED 0,3282)::

    B1 nsel_column  ışın içinden sütun log-olasılığıyla seçim  NED 0,3134  −0,015 GA[−0,036, +0,007]  RED
    B2 nsel_ranker  lojistik sıralayıcı (TRAIN, 3 iç kat)      NED 0,3100  −0,018 GA[−0,034, −0,003]
                    tam 0,3219 (+0,025 GA[+0,003, +0,050]) · BCFS +0,012 anlamlı değil
                    ham p=0,020, Holm (3 aday) p=0,059 -> ön kayıtlı ölçüt SAĞLANMADI, RED
    B3 nsel_vote    karşılıklı sıra füzyonu                    NED 0,3291  +0,001  RED

B2 umut verici ama kabul edilmedi; doğrulama için tek adaylı YENİ ön kayıt
gerekir. Kat başına ~600 sn (3 iç + 1 dış sinir eğitimi), tepe 2,8 GB.

Kat başına eğitim+çıkarım (yalnız dış sinir) ~150 sn (CPU, 4 iş parçacığı, 15 epok);
eval-cv sürecinin tepe belleği 2,9 GB. Üretime BAĞLANMADI.

torch isteğe bağlıdır; yalnız bu modül içe aktarır.
"""

from __future__ import annotations

import math
import random
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from engine.logging_setup import get_logger

logger = get_logger(__name__)

#: Hiperparametreler — ön kayıtta sabitlendi.
D_MODEL = 128
HEADS = 4
LAYERS = 2
FF = 256
DROPOUT = 0.25
BATCH = 32
LR = 1e-3
GOLD_WEIGHT = 3
WITNESS_DROP = 0.2
BEAM = 5
MAX_SRC = 256
MAX_TGT = 20
#: Epok sayısı — kat 0 TRAIN'inin iç ayrımında seçildi (bkz. ``choose_epochs``).
EPOCHS = 15
THREADS = 4

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SRC_SAV, SRC_STA = "<src=sav>", "<src=sta>"
TGT_PROTO = "<tgt=proto>"
ANC = "<anc>"
_ANCHOR_PREFERENCE = ("tr", "az", "tk", "gag", "kk", "ky", "tt", "uz", "ug")
_STRIP = set("-*?() .,;'’ʼ/")


def chars(text: str) -> list[str]:
    """Biçimi karakter dizisine çevirir; uzunluk işareti (makron) ``:`` olur.

    savelyev ``o:``, Starling ``ō`` yazar; ikisi de ``o :`` olur.
    """
    text = unicodedata.normalize("NFC", (text or "").strip().casefold())
    out: list[str] = []
    for ch in text:
        if ch in _STRIP:
            continue
        decomposed = unicodedata.normalize("NFD", ch)
        if "̄" in decomposed:
            base = unicodedata.normalize("NFC", decomposed.replace("̄", ""))
            out += [base, ":"]
        elif unicodedata.category(ch) == "Mn":
            if out:
                out[-1] = unicodedata.normalize("NFC", out[-1] + ch)
        else:
            out.append(ch)
    return out


def target_chars(proto: str) -> list[str]:
    """Altın ata biçim -> hedef karakterler (``normalize_proto`` geleneğinde)."""
    from engine.evaluation.metrics import normalize_proto

    return list(normalize_proto(proto))


@dataclass
class Example:
    source: str  # SRC_SAV | SRC_STA
    anchor: str
    forms: dict[str, str]  # dil -> biçim (çapa dili hariç)
    proto: str | None  # normalize edilmiş ata biçim (hedef) — yoksa yalnız refleks görevi
    weight: int = 1


def witness_input(word: str, entries: list[dict[str, str]]) -> tuple[str, dict[str, str]]:
    """Harness/motor girdisinden ``(çapa, {dil: en kısa biçim})``."""
    by_lang: dict[str, str] = {}
    for w in entries:
        code, form = w.get("lang_code") or "", w.get("word") or ""
        if not code or len(chars(form)) < 2:
            continue
        if code not in by_lang or len(chars(form)) < len(chars(by_lang[code])):
            by_lang[code] = form
    return word, by_lang


def gold_example(item: Any, mapping: dict[str, str]) -> Example | None:
    from engine.evaluation.harness import _anchor_for, _witnesses_for

    witnesses = _witnesses_for(item, mapping)
    anchor, anchor_lang = _anchor_for(witnesses)
    if not anchor:
        return None
    _, forms = witness_input(anchor, [w for w in witnesses if w["lang_code"] != anchor_lang])
    return Example(SRC_SAV, anchor, forms, "".join(target_chars(item.gold_form)), GOLD_WEIGHT)


@dataclass
class StarlingExample:
    example: Example
    protos: tuple[str, ...]
    turkish: set[str]


def prepare_starling() -> list[StarlingExample]:
    """Starling turcet köklerini ham biçimleriyle (hizalamasız) hazırlar."""
    from engine.db.starling import FIELD_LANGUAGES, _turkish_forms, load_turcet
    from engine.nlp.column_model import starling_first_form, starling_protos
    from engine.utils.orthography import to_comparison_form

    out = []
    for etym in load_turcet():
        protos = starling_protos(etym.proto)
        if not protos:
            continue
        by_lang = {}
        for fld, code in FIELD_LANGUAGES.items():
            raw = starling_first_form(etym.reflexes.get(fld, "")).replace("j", "y")
            if len(chars(raw)) >= 2:
                by_lang[code] = raw
        if len(by_lang) < 2:
            continue
        anchor_lang = next((c for c in _ANCHOR_PREFERENCE if c in by_lang), next(iter(by_lang)))
        anchor = by_lang.pop(anchor_lang)
        turkish = {to_comparison_form(x.rstrip("-")) for x in _turkish_forms(etym.reflexes.get("TRK", ""))}
        ex = Example(SRC_STA, anchor, by_lang, "".join(target_chars(protos[0])))
        out.append(StarlingExample(ex, tuple(protos), turkish))
    return out


def starling_allowed(sets: list[StarlingExample], excluded_turkish: set[str], excluded_protos: set[str]) -> list[Example]:
    """``column_model.starling_allowed`` ile AYNI sızıntı süzgeci."""
    from engine.nlp.column_model import norm

    return [
        s.example for s in sets
        if not (s.turkish & excluded_turkish) and not ({norm(p) for p in s.protos} & excluded_protos)
    ]


# --- model -------------------------------------------------------------------
@dataclass
class Vocab:
    itos: list[str] = field(default_factory=lambda: [PAD, BOS, EOS, UNK, SRC_SAV, SRC_STA, TGT_PROTO, ANC])

    def __post_init__(self) -> None:
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def add(self, token: str) -> None:
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    def ids(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, self.stoi[UNK]) for t in tokens]


def lang_token(code: str) -> str:
    return f"<{code}>"


def tgt_token(code: str) -> str:
    return f"<tgt={code}>"


def source_tokens(source: str, target: str, anchor: str, forms: dict[str, str]) -> list[str]:
    toks = [source, target, ANC] + chars(anchor)
    for code in sorted(forms):
        toks += [lang_token(code)] + chars(forms[code])
    return toks[: MAX_SRC - 1] + [EOS]


def _build_net(vocab_size: int) -> Any:
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab_size, D_MODEL, padding_idx=0)
            self.pos_src = nn.Embedding(MAX_SRC, D_MODEL)
            self.pos_tgt = nn.Embedding(MAX_TGT + 2, D_MODEL)
            self.tf = nn.Transformer(
                D_MODEL, HEADS, LAYERS, LAYERS, FF, DROPOUT, batch_first=True, norm_first=True
            )
            self.out = nn.Linear(D_MODEL, vocab_size)

        def encode(self, src: Any) -> tuple[Any, Any]:
            pos = torch.arange(src.size(1), device=src.device)
            mask = src == 0
            return self.tf.encoder(self.emb(src) + self.pos_src(pos), src_key_padding_mask=mask), mask

        def decode(self, memory: Any, mem_mask: Any, tgt: Any) -> Any:
            pos = torch.arange(tgt.size(1), device=tgt.device)
            causal = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
            h = self.tf.decoder(
                self.emb(tgt) + self.pos_tgt(pos), memory, tgt_mask=causal, tgt_is_causal=True,
                tgt_key_padding_mask=tgt == 0, memory_key_padding_mask=mem_mask,
            )
            return self.out(h)

    return Net()


class NeuralReconstructor:
    def __init__(self, vocab: Vocab, net: Any) -> None:
        self.vocab = vocab
        self.net = net
        self.net.eval()

    # -- çıkarım
    def _encode(self, word: str, entries: list[dict[str, str]]) -> tuple[Any, Any]:
        import torch

        anchor, forms = witness_input(word, entries)
        src = torch.tensor([self.vocab.ids(source_tokens(SRC_SAV, TGT_PROTO, anchor, forms))])
        return self.net.encode(src)

    def beam(self, word: str, entries: list[dict[str, str]], k: int = BEAM) -> list[tuple[float, str]]:
        """Işın araması: ``[(jeton başına ort. log-olasılık, biçim), ...]`` en iyiden."""
        import torch

        with torch.no_grad():
            memory, mask = self._encode(word, entries)
            beams: list[tuple[float, list[int]]] = [(0.0, [1])]
            done: list[tuple[float, list[int]]] = []
            eos = self.vocab.stoi[EOS]
            for _ in range(MAX_TGT):
                if not beams:
                    break
                tgt = torch.tensor([b[1] for b in beams])
                logp = torch.log_softmax(self.net.decode(memory.expand(len(beams), -1, -1),
                                                         mask.expand(len(beams), -1), tgt)[:, -1], -1)
                cand = []
                for bi, (score, seq) in enumerate(beams):
                    top = torch.topk(logp[bi], k)
                    for lp, ix in zip(top.values.tolist(), top.indices.tolist(), strict=True):
                        cand.append((score + lp, seq + [ix]))
                cand.sort(key=lambda c: -c[0])
                beams = []
                for score, seq in cand:
                    if seq[-1] == eos:
                        done.append((score / (len(seq) - 1), seq))
                    elif len(beams) < k:
                        beams.append((score, seq))
            done.sort(key=lambda d: -d[0])
            out, seen = [], set()
            for score, seq in done:
                text = "".join(self.vocab.itos[i] for i in seq[1:-1] if i > 7 or i == 3)
                if text and text not in seen:
                    seen.add(text)
                    out.append((score, text))
            return out[:k]

    def score(self, word: str, entries: list[dict[str, str]], proto: str) -> float:
        """Verilen ata biçimin jeton başına ort. log-olasılığı (``<src=sav>``)."""
        import torch

        from engine.evaluation.metrics import normalize_proto

        tgt_ids = [1] + self.vocab.ids(list(normalize_proto(proto))) + [self.vocab.stoi[EOS]]
        with torch.no_grad():
            memory, mask = self._encode(word, entries)
            logp = torch.log_softmax(self.net.decode(memory, mask, torch.tensor([tgt_ids[:-1]])), -1)[0]
            total = sum(logp[i, t].item() for i, t in enumerate(tgt_ids[1:]))
        return total / (len(tgt_ids) - 1)

    def reconstruct(self, word: str, entries: list[dict[str, str]]) -> dict[str, Any]:
        beam = self.beam(word, entries)
        if not beam:
            return {"reconstructed_root": "", "is_reconstructible": False, "confidence": 0.0}
        score, text = beam[0]
        return {
            "reconstructed_root": "*" + text,
            "is_reconstructible": True,
            "confidence": round(math.exp(score), 4),
            "candidates": ["*" + t for _, t in beam],
        }


# --- eğitim ------------------------------------------------------------------
def _instances(examples: list[Example], rng: random.Random) -> list[tuple[list[str], list[str]]]:
    """Bir epokun (girdi, hedef) çiftleri: ata görevi (ağırlıklı) + küme başına bir refleks görevi."""
    out = []
    for ex in examples:
        langs = list(ex.forms)
        if ex.proto:
            for _ in range(ex.weight):
                kept = {c: f for c, f in ex.forms.items() if rng.random() >= WITNESS_DROP} or ex.forms
                out.append((source_tokens(ex.source, TGT_PROTO, ex.anchor, kept), list(ex.proto)))
        if len(langs) >= 2:
            target = rng.choice(langs)
            rest = {c: f for c, f in ex.forms.items() if c != target and rng.random() >= WITNESS_DROP}
            out.append((source_tokens(ex.source, tgt_token(target), ex.anchor, rest), chars(ex.forms[target])))
    return out


def _vocab_for(examples: list[Example]) -> Vocab:
    vocab = Vocab()
    for ex in examples:
        for code in ex.forms:
            vocab.add(lang_token(code))
            vocab.add(tgt_token(code))
        for tok in chars(ex.anchor) + [c for f in ex.forms.values() for c in chars(f)] + list(ex.proto or ""):
            vocab.add(tok)
    return vocab


def train(
    examples: list[Example],
    *,
    epochs: int = EPOCHS,
    seed: int = 0,
    checkpoints: tuple[int, ...] = (),
    on_checkpoint: Any = None,
) -> NeuralReconstructor:
    """Modeli eğitir (CPU, ``THREADS`` iş parçacığı)."""
    import torch

    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    vocab = _vocab_for(examples)
    net = _build_net(len(vocab.itos))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(_instances(examples, random.Random(0))) / BATCH)
    total = steps_per_epoch * epochs
    warm = max(1, min(400, total // 10))
    # Isınmadan sonra sabit öğrenme hızı: ara denetim noktası, o epok sayısıyla
    # eğitilmiş modelin aynısıdır (epok seçimi için).
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    start = time.time()
    for epoch in range(1, epochs + 1):
        net.train()
        data = _instances(examples, rng)
        rng.shuffle(data)
        epoch_loss = 0.0
        for b in range(0, len(data), BATCH):
            batch = data[b : b + BATCH]
            src_len = max(len(s) for s, _ in batch)
            tgt_len = min(MAX_TGT, max(len(t) for _, t in batch)) + 1
            src = torch.zeros(len(batch), src_len, dtype=torch.long)
            tin = torch.zeros(len(batch), tgt_len, dtype=torch.long)
            tout = torch.zeros(len(batch), tgt_len, dtype=torch.long)
            for i, (s, t) in enumerate(batch):
                ids = vocab.ids(s)
                src[i, : len(ids)] = torch.tensor(ids)
                tids = vocab.ids(t[: tgt_len - 1])
                tin[i, : len(tids) + 1] = torch.tensor([1] + tids)
                tout[i, : len(tids) + 1] = torch.tensor(tids + [vocab.stoi[EOS]])
            memory, mask = net.encode(src)
            logits = net.decode(memory, mask, tin)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tout.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            epoch_loss += loss.item() * len(batch)
        if epoch % 10 == 0 or epoch == 1:
            logger.info("neural epok %d/%d kayıp %.4f (%.0f sn)", epoch, epochs, epoch_loss / len(data), time.time() - start)
        if epoch in checkpoints and on_checkpoint is not None:
            net.eval()
            on_checkpoint(epoch, NeuralReconstructor(vocab, net))
    return NeuralReconstructor(vocab, net)


def choose_epochs(dataset: str = "savelyevturkic", grid: tuple[int, ...] = (15, 30, 45, 60)) -> dict[int, float]:
    """Epok seçimi: YALNIZ çapraz doğrulama kat 0'ının TRAIN kısmı, iç ayrımla.

    Sınanan katların hiçbir maddesi ve test altın kökleri görülmez.
    """
    from engine.db.cldf_wordlist import CldfWordlist
    from engine.db.language_mapping import build_mapping
    from engine.evaluation import harness
    from engine.evaluation.crossval import fold_of
    from engine.evaluation.gold import GoldStandard
    from engine.nlp.column_model import gold_turkish
    from engine.nlp.column_model import norm as proto_norm

    gold = GoldStandard.build(dataset)
    mapping = build_mapping(CldfWordlist.load(dataset))
    items = [it for it in list(gold.split("train")) + list(gold.split("dev")) if fold_of(it.concept) != 0]
    inner_held = [it for it in items if fold_of(it.concept + "#inner") == 0]
    inner_train = [it for it in items if fold_of(it.concept + "#inner") != 0]
    test_turkish = set().union(*(gold_turkish(it, mapping) for it in gold.items if it.split == "test"))
    excluded_tr = set().union(*(gold_turkish(it, mapping) for it in items if fold_of(it.concept) == 0 or it in inner_held))
    fold0 = [it for it in list(gold.split("train")) + list(gold.split("dev")) if fold_of(it.concept) == 0]
    excluded_tr |= set().union(*(gold_turkish(it, mapping) for it in fold0)) | test_turkish
    excluded_protos = {proto_norm(g) for it in inner_held + fold0 for g in it.gold_candidates}
    examples = [e for e in (gold_example(it, mapping) for it in inner_train) if e]
    examples += starling_allowed(prepare_starling(), excluded_tr, excluded_protos)
    scores: dict[int, float] = {}

    def check(epoch: int, model: NeuralReconstructor) -> None:
        result = harness.run(model.reconstruct, inner_held, mapping=mapping)
        scores[epoch] = round(sum(result.item_ned) / len(result.item_ned), 4)
        print(f"epok {epoch}: iç NED {scores[epoch]} tam {sum(result.item_correct) / len(inner_held):.4f}", flush=True)

    train(examples, epochs=max(grid), checkpoints=grid, on_checkpoint=check)
    return scores


# --- sinir üretir, sütun modeli seçer (ön kayıt 2) --------------------------
SELECT_BEAM = 10


@dataclass
class Candidate:
    text: str  # "*..." biçiminde
    column: float = 0.0  # S(c): sütun modeli log-olasılığı
    unexplained: int = 0
    neural: float = 0.0  # jeton başına ort. log-olasılık
    rank: int = SELECT_BEAM  # sinir ışınındaki sıra (0'dan); ışın dışı = SELECT_BEAM
    in_beam: bool = False
    is_column: bool = False


def column_score(scored: list[list[tuple[float, str]]], informative: list[Any], candidate: str) -> tuple[float, int]:
    """Adayın sütun modeli log-olasılığı ve sütunlara oturmayan ses sayısı.

    Aday ``label_columns`` ile sütunlara hizalanır; sütun başına etiket
    olasılığı sütunun aday kümesinde normalize edilir, kümede olmayan etiket
    1e-4 alır; oturmayan her ses log(0,01) cezası.
    """
    from engine.nlp.column_model import NULL, graphemes, label_columns, norm

    sounds = len(graphemes(norm(candidate)))
    labels = label_columns([candidate], informative) if informative else None
    if labels is None:
        return sounds * math.log(0.01), sounds
    total = 0.0
    for label, column in zip(labels, scored, strict=True):
        z = sum(p for p, _ in column) or 1.0
        p = next((p for p, s in column if s == label), 0.0)
        total += math.log(p / z) if p > 0 else math.log(1e-4)
    unexplained = max(0, sounds - sum(label != NULL for label in labels))
    return total + unexplained * math.log(0.01), unexplained


def build_candidates(
    model: NeuralReconstructor,
    scored: list[list[tuple[float, str]]],
    informative: list[Any],
    word: str,
    entries: list[dict[str, str]],
    column_prediction: str | None,
) -> list[Candidate]:
    """Sinir ışını-10 ∪ {sütun top-1}, her aday puanlanmış."""
    from engine.evaluation.metrics import normalize_proto

    out: dict[str, Candidate] = {}
    for rank, (lp, text) in enumerate(model.beam(word, entries, k=SELECT_BEAM)):
        key = normalize_proto(text)
        if key and key not in out:
            out[key] = Candidate("*" + text, neural=lp, rank=rank, in_beam=True)
    if column_prediction:
        key = normalize_proto(column_prediction)
        if key in out:
            out[key].is_column = True
        elif key:
            out[key] = Candidate(column_prediction, neural=model.score(word, entries, column_prediction), is_column=True)
    for cand in out.values():
        cand.column, cand.unexplained = column_score(scored, informative, cand.text)
    return list(out.values())


def candidate_features(cand: Candidate, n_columns: int, word: str, entries: list[dict[str, str]]) -> dict[str, float]:
    from engine.evaluation.metrics import normalize_proto, normalized_edit_distance
    from engine.utils.orthography import to_comparison_form

    text = normalize_proto(cand.text)
    forms = [to_comparison_form(w.get("word") or "") for w in entries] + [to_comparison_form(word)]
    forms = [f for f in forms if f] or [text]
    lengths = sorted(len(f) for f in forms)
    return {
        "column": cand.column / 10.0,
        "column_mean": cand.column / max(1, n_columns),
        "unexplained": float(cand.unexplained),
        "neural": cand.neural,
        "rank": cand.rank / SELECT_BEAM,
        "in_beam": float(cand.in_beam),
        "is_column": float(cand.is_column),
        "len_anchor": abs(len(text) - len(to_comparison_form(word))) / 3.0,
        "len_median": (len(text) - lengths[len(lengths) // 2]) / 3.0,
        "ned_witness": sum(normalized_edit_distance(text, f) for f in forms) / len(forms),
    }


def select_column(cands: list[Candidate]) -> str | None:
    """B1: yalnız sinir ışını içinden, sütun skoru en yüksek."""
    beam = [c for c in cands if c.in_beam]
    return max(beam, key=lambda c: (c.column, -c.rank)).text if beam else None


def select_vote(cands: list[Candidate]) -> str | None:
    """B3: karşılıklı sıra füzyonu; eşitlikte sütun top-1."""
    if not cands:
        return None
    by_neural = sorted(cands, key=lambda c: -c.neural)
    by_column = sorted(cands, key=lambda c: (-c.column, not c.is_column))
    score = {id(c): 1 / (1 + by_neural.index(c)) + 1 / (1 + by_column.index(c)) for c in cands}
    return max(cands, key=lambda c: (round(score[id(c)], 9), c.is_column, -by_neural.index(c))).text


def select_ranker(cands: list[Candidate], weights: dict[str, float], intercept: float,
                  n_columns: int, word: str, entries: list[dict[str, str]]) -> str | None:
    """B2: öğrenilmiş lojistik sıralayıcı."""
    if not cands:
        return None

    def z(c: Candidate) -> float:
        feats = candidate_features(c, n_columns, word, entries)
        return intercept + sum(v * weights.get(k, 0.0) for k, v in feats.items())

    return max(cands, key=lambda c: (z(c), c.is_column)).text


if __name__ == "__main__":
    print(choose_epochs())
