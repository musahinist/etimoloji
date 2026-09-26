"""Anlam köprüsü tablosunun (9l S1) tekrar üretilebilirliği ve eksik tablo uyarısı."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
from pathlib import Path
from unittest import mock

import pytest

from engine.db import sense_bridge as sb

ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location("download_sense_bridge", ROOT / "scripts" / "download_sense_bridge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def dumps(tmp_path: Path) -> tuple[Path, Path]:
    tr = tmp_path / "raw" / "trwikt.jsonl.gz"
    en = tmp_path / "raw" / "en.jsonl.gz"
    tr.parent.mkdir()
    tr_records = [
        {"lang_code": "tr", "word": "bando", "pos": "noun",
         "senses": [{"glosses": ["Bir müzik topluluğu"], "translations": [{"lang_code": "en", "word": "band"}]}]},
        {"lang_code": "tr", "word": "kasaphane", "pos": "noun",
         "translations": [{"code": "en", "word": "slaughterhouse"}, {"code": "it", "word": "macello"}]},
        {"lang_code": "en", "word": "lamp", "pos": "noun", "senses": [{"glosses": ["lamba"]}]},
    ]
    en_records = [
        {"lang_code": "en", "word": "band", "pos": "noun", "translations": [{"code": "tr", "word": "bando"}]},
        {"lang_code": "en", "word": "slaughterhouse", "pos": "noun"},
    ]
    from engine.utils.provenance import deterministic_gzip

    for path, records in ((tr, tr_records), (en, en_records)):
        with deterministic_gzip(path) as handle:
            handle.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8"))
    return tr, en


def test_build_is_deterministic_and_records_digest(tmp_path: Path, dumps):
    tr, en = dumps
    a, b = tmp_path / "a" / "tr_en.db", tmp_path / "b" / "tr_en.db"
    meta_a = sb.build(tr, en, out=a)
    meta_b = sb.build(tr, en, out=b)
    assert meta_a["content_sha256"] == meta_b["content_sha256"] == sb.table_digest(a) == sb.table_digest(b)
    assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest()
    provenance = json.loads(sb.provenance_path(a).read_text(encoding="utf-8"))
    assert provenance["content_sha256"] == meta_a["content_sha256"]
    assert provenance["sources"]["trwiktionary"]["sha256"] == hashlib.sha256(tr.read_bytes()).hexdigest()
    # İngilizce dışı çeviri (it: macello) okunmaz.
    with mock.patch.object(sb, "BRIDGE_DB", a):
        sb._connection.cache_clear()
        sb.english_sense.cache_clear()
        try:
            assert sb.english_sense("bando") == "band"
            assert sb.english_sense("kasaphane") == "slaughterhouse"
            assert sb.english_sense("lamba") == "lamp"
        finally:
            sb._connection.cache_clear()
            sb.english_sense.cache_clear()


def test_rebuild_keeps_provenance_when_only_time_changes(tmp_path: Path, dumps):
    tr, en = dumps
    out = tmp_path / "tr_en.db"
    sb.build(tr, en, out=out)
    before = sb.provenance_path(out).read_text(encoding="utf-8")
    with mock.patch.object(sb, "datetime") as fake:
        fake.now.return_value.isoformat.return_value = "2099-01-01T00:00:00+00:00"
        sb.build(tr, en, out=out)
    assert sb.provenance_path(out).read_text(encoding="utf-8") == before


def test_missing_table_logs_warning(tmp_path: Path, caplog):
    with mock.patch.object(sb, "BRIDGE_DB", tmp_path / "yok.db"):
        sb._connection.cache_clear()
        try:
            with caplog.at_level(logging.WARNING, logger="engine"):
                logging.getLogger("engine").propagate = True
                assert sb._connection() is None
            assert any("make sense-bridge" in r.getMessage() for r in caplog.records)
        finally:
            sb._connection.cache_clear()


def test_script_uses_local_dumps_with_matching_sha_and_skips_when_current(tmp_path: Path, dumps, monkeypatch):
    tr, en = dumps
    script = _load_script()
    reference = tmp_path / "ref" / "tr_en.db"
    sb.build(tr, en, out=reference)
    out = tmp_path / "out" / "tr_en.db"
    out.parent.mkdir()
    sb.provenance_path(out).write_text(sb.provenance_path(reference).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(script, "RAW_DIR", tmp_path / "unused")
    monkeypatch.setattr(script, "_download", mock.Mock(side_effect=AssertionError("ağ kullanılmamalı")))
    assert script.main(["--raw-dir", str(tr.parent), "--out", str(out)]) == 0
    assert sb.table_digest(out) == sb.table_digest(reference)
    # Güncel tablo: hiçbir şey kurulmaz.
    with mock.patch.object(script, "build", side_effect=AssertionError("yeniden kurulmamalı")):
        assert script.main(["--out", str(out)]) == 0
