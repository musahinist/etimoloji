"""
Künye (provenance) dosyalarını gereksiz yere yeniden yazmamak.

Künyeler (``_provenance.json``, ``*.provenance.json``) repoya commit edilir,
veri dosyaları edilmez. Taze klonda ``make bootstrap`` veriyi indirir ve
künyeyi yeniden yazıyordu: veri birebir aynı (SHA-256 aynı) olsa bile
``retrieved_at`` değiştiği için çalışma ağacı kirleniyordu. Burada künye,
yalnız OYNAK alanlar (indirme zamanı) dışında bir şey değiştiyse yazılır.

Sıkıştırılmış dökümün SHA'sı ancak sıkıştırma belirlenimciyse anlamlıdır:
``gzip.open`` başlığa o anın zamanını ve dosya adını yazar, aynı içerik her
indirmede başka SHA verir. ``deterministic_gzip`` ikisini de sabitler.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any

#: Veriyi değil indirmenin/eğitimin ANINI anlatan alanlar.
VOLATILE_FIELDS: tuple[str, ...] = ("retrieved_at",)


def _strip(data: Any, volatile: Iterable[str]) -> Any:
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if k not in set(volatile)}
    return data


def write_if_changed(
    path: Path, data: dict[str, Any], text: str, *, volatile: Iterable[str] = VOLATILE_FIELDS
) -> dict[str, Any]:
    """``text``i (``data``nın serileştirilmiş hâli) ``path``e yazar; ama
    diskteki dosya oynak alanlar dışında ``data`` ile aynıysa DOKUNMAZ ve
    diskteki içeriği döndürür (künye aynı veriyi anlatıyor, commit'teki
    indirme tarihi korunur)."""
    volatile = tuple(volatile)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if isinstance(existing, dict) and _strip(existing, volatile) == _strip(
            json.loads(json.dumps(data, ensure_ascii=False)), volatile
        ):
            return existing
    path.write_text(text, encoding="utf-8")
    return data


class _OwnedGzipFile(gzip.GzipFile):
    """Verilen dosya tutamağını kapanışta kendisi de kapatan ``GzipFile``
    (``GzipFile`` dışarıdan verilen ``fileobj``i kapatmaz)."""

    def __init__(self, raw: IO[bytes]) -> None:
        self._raw = raw
        super().__init__(filename="", mode="wb", fileobj=raw, mtime=0)

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._raw.close()


def deterministic_gzip(path: Path) -> gzip.GzipFile:
    """Aynı içerik için bayt bayt aynı ``.gz`` (başlıkta zaman 0, ad yok)."""
    raw = path.open("wb")
    try:
        return _OwnedGzipFile(raw)
    except Exception:
        raw.close()
        raise
