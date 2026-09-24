"""İndiricilerin atlama mantığı: künye var diye DEĞİL, veri dosyası var ve
SHA-256'sı künyeyle aynı diye atlanır.

Künyeler (``*provenance.json``) commit edilir, veri dosyaları git-ignore'dadır;
taze klonda künye var ama veri yoktur. Eski mantık bu durumda hiçbir şey
indirmiyordu ("[Salar] zaten var…").
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _NoNetwork:
    """``download`` ağa çıkarsa haber veren sahte oturum."""

    def __init__(self) -> None:
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("ağa çıkıldı")


# --- kaikki sözlükleri -------------------------------------------------------


@pytest.fixture
def lexicons(tmp_path, monkeypatch):
    module = _load("download_lexicons")
    monkeypatch.setattr(module, "LEXICON_DIR", tmp_path)
    return module


def _write_lexicon(directory: Path, *, sha: str | None = None) -> Path:
    data = directory / "slq.jsonl.gz"
    with gzip.open(data, "wb") as fh:
        fh.write(b'{"word": "su"}\n')
    prov = {"language": "Salar", "code": "slq", "sha256_stored": sha or _sha(data)}
    (directory / "slq.provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    return data


def test_lexicon_skips_when_file_matches(lexicons, tmp_path):
    _write_lexicon(tmp_path)
    session = _NoNetwork()
    prov = lexicons.download("Salar", session=session)
    assert prov["code"] == "slq"
    assert session.calls == 0


def test_lexicon_downloads_when_only_provenance_exists(lexicons, tmp_path):
    """Taze klon: künye commit edilmiş, veri dosyası yok."""
    _write_lexicon(tmp_path).unlink()
    session = _NoNetwork()
    with pytest.raises(RuntimeError, match="ağa çıkıldı"):
        lexicons.download("Salar", session=session)
    assert session.calls == 1


def test_lexicon_downloads_when_sha_mismatch(lexicons, tmp_path):
    _write_lexicon(tmp_path, sha="0" * 64)
    session = _NoNetwork()
    with pytest.raises(RuntimeError):
        lexicons.download("Salar", session=session)
    assert session.calls == 1


def test_lexicon_force_redownloads(lexicons, tmp_path):
    _write_lexicon(tmp_path)
    session = _NoNetwork()
    with pytest.raises(RuntimeError):
        lexicons.download("Salar", session=session, force=True)


def test_tr_edition_requires_all_files(lexicons, tmp_path):
    directory = tmp_path / lexicons.TR_SUBDIR
    directory.mkdir()
    shards = {}
    for code in ("tr", "az"):
        path = directory / f"{code}.jsonl.gz"
        with gzip.open(path, "wb") as fh:
            fh.write(b"{}\n")
        shards[code] = _sha(path)
    (directory / "_provenance.json").write_text(
        json.dumps({"entries": {"tr": 1, "az": 1}, "files": shards}), encoding="utf-8"
    )
    session = _NoNetwork()
    assert lexicons.download_tr_edition(session=session) == {"tr": 1, "az": 1}
    assert session.calls == 0

    (directory / "az.jsonl.gz").unlink()
    with pytest.raises(RuntimeError):
        lexicons.download_tr_edition(session=session)


# --- CLDF --------------------------------------------------------------------


@pytest.fixture
def cldf(tmp_path, monkeypatch):
    module = _load("download_cldf")
    monkeypatch.setattr(module, "CLDF_DIR", tmp_path)
    return module


def _write_cldf(directory: Path) -> Path:
    target = directory / "savelyevturkic"
    target.mkdir()
    forms = target / "forms.csv"
    forms.write_text("ID,Form\n1,su\n", encoding="utf-8")
    prov = {"dataset": "savelyevturkic", "files": {"forms.csv": {"sha256": _sha(forms), "rows": 1}}}
    (target / "_provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    return forms


def test_cldf_skips_when_files_match(cldf, tmp_path):
    _write_cldf(tmp_path)
    session = _NoNetwork()
    assert cldf.download("savelyevturkic", session=session)["dataset"] == "savelyevturkic"
    assert session.calls == 0


def test_cldf_downloads_when_only_provenance_exists(cldf, tmp_path, monkeypatch):
    _write_cldf(tmp_path).unlink()
    monkeypatch.setattr(cldf, "_resolve_ref", lambda *a, **k: ("v1", ""))
    session = _NoNetwork()
    with pytest.raises(RuntimeError, match="ağa çıkıldı"):
        cldf.download("savelyevturkic", session=session)


# --- Starling ve Apertium ------------------------------------------------------


def test_starling_and_apertium_is_current(tmp_path):
    for name, filename in (("download_starling", "turcet.dbf"), ("download_apertium", "chv-tur.dix")):
        module = _load(name)
        directory = tmp_path / name
        directory.mkdir()
        assert not module.is_current(directory)
        data = directory / filename
        data.write_bytes(b"veri")
        key = filename if name == "download_starling" else filename.removesuffix(".dix")
        (directory / "_provenance.json").write_text(
            json.dumps({"files": {key: {"sha256": _sha(data)}}}), encoding="utf-8"
        )
        assert module.is_current(directory)
        data.unlink()
        assert not module.is_current(directory)
