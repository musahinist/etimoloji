"""
Yalnız-LLM alıntı taban çizgisi — ``make eval-llm``.

Soru şu: **motor, kelimeyi doğrudan bir LLM'e sormaktan daha iyi mi?**
Motor hiçbir aşamada bu modülü çağırmaz; burası yalnız ölçümdür.

Claude'a her madde için yalnız *kelime, dil ve anlam* verilir; altın
kümenin verici dili (``BorrowingCase.donor``) isteme **girmez**. Ölçüm,
``borrowing_eval.evaluate`` ile aynı RAPOR yarısında (tek indisli maddeler)
yapılır, böylece ``data/eval/borrowing.json``daki motor sayılarıyla
doğrudan karşılaştırılabilir.

⚠️ **Kirlenme.** Türkçe altın küme TDK + Nişanyan'dan kuruldu; WOLD ve
Wiktionary de kamuya açık. Bunların hepsi büyük olasılıkla modelin eğitim
verisinde. Yüksek bir LLM skoru "model etimoloji çıkarıyor" demek
değildir; ezber de olabilir. Bu yüzden sonuç bir **tavan/karşılaştırma
noktası** olarak okunmalı, motora kanıt olarak eklenmemeli.

⚠️ Cevaplar ``data/eval/llm_cache/<model>.jsonl``e yazılır. Yeniden koşmak
API'ye gitmez; önbellek silinmedikçe sonuç tekrarlanabilir.
"""

from __future__ import annotations

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from engine.evaluation.borrowing_eval import (
    PRF,
    BorrowingCase,
    always_borrowed,
    load_turkish_gold_cases,
    load_wold_cases,
    phonotactic_only,
    score_system,
)
from engine.evaluation.report import EVAL_DIR
from engine.evaluation.significance import mcnemar_test
from engine.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "claude-opus-5"
CACHE_DIR = EVAL_DIR / "llm_cache"

LANGUAGE_NAMES = {"tr": "Turkish", "sah": "Sakha (Yakut)"}

SYSTEM_PROMPT = (
    "You are a historical linguist specialising in the Turkic languages. "
    "For the given word, decide whether it is a loanword in the given "
    "language, i.e. borrowed from another language at any period, or "
    "inherited (from Proto-Turkic, or formed within the language from "
    "inherited material). Judge the word as a lexical item in that "
    "language; do not look anything up. If borrowed, name the immediate "
    "donor language in English; otherwise leave donor_language empty."
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "borrowed": {"type": "boolean"},
        "donor_language": {"type": "string"},
    },
    "required": ["borrowed", "donor_language"],
    "additionalProperties": False,
}


def _key(case: BorrowingCase) -> str:
    return f"{case.lang_code}\t{case.word}\t{case.sense}"


def _prompt(case: BorrowingCase) -> str:
    lines = [f"Language: {LANGUAGE_NAMES.get(case.lang_code, case.lang_code)}", f"Word: {case.word}"]
    if case.sense:
        lines.append(f"Meaning: {case.sense}")
    return "\n".join(lines)


class CachedClassifier:
    """Tek maddelik LLM sınıflandırıcısı; cevaplar diske önbelleklenir.

    Alt sınıf yalnız :meth:`_call`ı yazar: ``borrowed`` (``None`` = cevap
    yok), ``donor_language``, ``model``, ``stop_reason`` ve token sayıları.
    """

    def __init__(self, model: str):
        self.model = model
        self.cache_path = CACHE_DIR / f"{model.replace(':', '_')}.jsonl"
        self._lock = threading.Lock()
        self.cache: dict[str, dict[str, Any]] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row

    def _call(self, case: BorrowingCase) -> dict[str, Any]:
        raise NotImplementedError

    def classify(self, case: BorrowingCase) -> dict[str, Any]:
        key = _key(case)
        if key in self.cache:
            return self.cache[key]
        row = {"key": key, "borrowed": None, "donor_language": "", **self._call(case)}
        with self._lock:
            self.cache[key] = row
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row


class ClaudeClassifier(CachedClassifier):
    def __init__(self, model: str = DEFAULT_MODEL, *, effort: str = "low"):
        import anthropic

        super().__init__(model)
        self.client = anthropic.Anthropic()
        self.effort = effort

    def _call(self, case: BorrowingCase) -> dict[str, Any]:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _prompt(case)}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
            },
            # Politika reddinde istek sunucu tarafında yedek modele düşer.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        row: dict[str, Any] = {
            "model": response.model,
            "stop_reason": response.stop_reason,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        if response.stop_reason == "end_turn":
            text = next(b.text for b in response.content if b.type == "text")
            row.update(json.loads(text))
        return row


class OllamaClassifier(CachedClassifier):
    """Yerel Ollama (varsayılan ``config.OLLAMA_MODEL``); şema zorlamalı JSON."""

    def __init__(self, model: str | None = None):
        from engine import config

        super().__init__(model or config.OLLAMA_MODEL)
        self.url = config.OLLAMA_GENERATE_URL
        self.timeout = config.OLLAMA_TIMEOUT

    def _call(self, case: BorrowingCase) -> dict[str, Any]:
        from engine.utils.network import post_json

        result = post_json(
            self.url,
            {
                "model": self.model,
                "system": SYSTEM_PROMPT,
                "prompt": _prompt(case),
                "format": OUTPUT_SCHEMA,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 100},
            },
            timeout=self.timeout,
            allow_private=True,  # Ollama bilinçli olarak yerel bir servistir
        )
        row: dict[str, Any] = {
            "model": self.model,
            "stop_reason": (result or {}).get("done_reason", "error"),
            "input_tokens": (result or {}).get("prompt_eval_count", 0),
            "output_tokens": (result or {}).get("eval_count", 0),
        }
        try:
            row.update(json.loads((result or {}).get("response") or ""))
        except json.JSONDecodeError:
            logger.warning("Ollama cevabı JSON değil: %r", case.word)
        return row


def report_half(cases: list[BorrowingCase]) -> list[BorrowingCase]:
    """``borrowing_eval.evaluate`` ile AYNI rapor yarısı."""
    return [c for i, c in enumerate(cases) if i % 2 == 1]


def bootstrap_f(result: PRF, cases: list[BorrowingCase], predictions: list[bool],
                *, iterations: int = 5000, seed: int = 20240601) -> tuple[float, float]:
    """F için %95 bootstrap güven aralığı."""
    rng = random.Random(seed)
    size = len(cases)
    samples = []
    for _ in range(iterations):
        tp = fp = fn = 0
        for _ in range(size):
            i = rng.randrange(size)
            gold, pred = cases[i].is_borrowed, predictions[i]
            tp += pred and gold
            fp += pred and not gold
            fn += gold and not pred
        samples.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    samples.sort()
    return samples[int(0.025 * iterations)], samples[int(0.975 * iterations)]


def run_dataset(name: str, cases: list[BorrowingCase], classifier: CachedClassifier,
                *, limit: int | None, workers: int) -> dict[str, Any]:
    cases = report_half(cases)[:limit]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(classifier.classify, cases))
    # Reddedilen/ayrıştırılamayan madde "miras" sayılır ve ayrıca raporlanır.
    predictions = [bool(r["borrowed"]) for r in rows]
    answers = dict(zip([_key(c) for c in cases], predictions, strict=True))
    llm = score_system(lambda c: answers[_key(c)], cases)
    low, high = bootstrap_f(llm, cases, predictions)

    baselines = {
        "always_borrowed": score_system(always_borrowed, cases),
        "phonotactic_only": score_system(phonotactic_only, cases),
    }
    return {
        "n_report": len(cases),
        "no_answer": sum(r["borrowed"] is None for r in rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "llm": {**llm.as_dict(), "fscore_ci95": [round(low, 4), round(high, 4)]},
        "baselines": {
            key: {
                **prf.as_dict(),
                "mcnemar_vs_llm_p": round(mcnemar_test(llm.per_item, prf.per_item).p_value, 4),
            }
            for key, prf in baselines.items()
        },
    }


def _stored_engine_scores() -> dict[str, dict[str, float]]:
    path = EVAL_DIR / "borrowing.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for dataset in ("turkish_gold", "wold"):
        systems = (data.get(dataset) or {}).get("systems") or {}
        out[dataset] = {k: systems[k]["fscore"] for k in ("engine", "engine_trained") if k in systems}
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Yalnız-LLM alıntı taban çizgisi")
    ap.add_argument("--provider", choices=("claude", "ollama"), default="claude")
    ap.add_argument("--model", default=None, help="Varsayılan: claude-opus-5 / config.OLLAMA_MODEL")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--limit", type=int, default=None, help="Veri kümesi başına en çok madde")
    ap.add_argument("--workers", type=int, default=None, help="Varsayılan: claude 8, ollama 1")
    args = ap.parse_args()

    if args.provider == "ollama":
        classifier: CachedClassifier = OllamaClassifier(args.model)
    else:
        classifier = ClaudeClassifier(args.model or DEFAULT_MODEL, effort=args.effort)
    workers = args.workers or (1 if args.provider == "ollama" else 8)
    datasets = {
        "turkish_gold": load_turkish_gold_cases(with_witnesses=False),
        "wold": load_wold_cases(with_witnesses=False),
    }
    stored = _stored_engine_scores()
    payload: dict[str, Any] = {
        "_schema": "turkic-etymology-llm-borrowing/v1",
        "model": classifier.model,
        "effort": args.effort,
        "protocol": "borrowing_eval ile aynı rapor yarısı (tek indisler); verici dili isteme girmez.",
    }
    for name, cases in datasets.items():
        if not cases:
            print(f"{name}: veri yok, atlandı")
            continue
        result = run_dataset(name, cases, classifier, limit=args.limit, workers=workers)
        result["engine_stored_fscore"] = stored.get(name, {})
        payload[name] = result

        llm = result["llm"]
        print(f"\n=== {name} · rapor yarısı n={result['n_report']} · {classifier.model} ===")
        print(f"LLM                F={llm['fscore']:.4f} {llm['fscore_ci95']}  "
              f"P={llm['precision']:.4f} R={llm['recall']:.4f}  cevapsız={result['no_answer']}")
        for key, value in result["engine_stored_fscore"].items():
            print(f"{key + ' (kayıtlı)':18} F={value:.4f}")
        for key, row in result["baselines"].items():
            print(f"{key:18} F={row['fscore']:.4f}  McNemar p={row['mcnemar_vs_llm_p']}")
        print(f"token: girdi {result['input_tokens']}, çıktı {result['output_tokens']}")

    if args.limit is None:
        out = EVAL_DIR / f"llm_borrowing_{args.provider}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
