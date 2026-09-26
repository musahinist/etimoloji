"""Eren / Gülensoy ayrıştırıcısı — yapay (telifsiz) madde örnekleriyle."""

from engine.db import eren_gulensoy as eg


def _eren(text: str) -> dict:
    return eg.parse_eren_entry(text.strip().splitlines())


def test_eren_turkish_root_and_suffix():
    rec = _eren("""
çaput 'eski bez'. Ağızlarda 'bez' olarak geçer.
~ Tkm çabıt 'gömlek'.
~ OT çapğut 'paçavra'.
Clauson'a göre çap-
kökünden -gut ekiyle kurulmuştur.
Räsänen: V 99 b; Clauson: ED 396 a.
""")
    assert rec["lemma"] == "çaput" and rec["origin"] == "turkish"
    assert rec["root"] == "çap-" and rec["suffixes"] == ["-gut"]
    assert rec["ot_form"] == "çapğut"
    assert "Räsänen: V 99 b" in rec["refs"]


def test_eren_loan_gives_no_root():
    rec = _eren("""
meşe 'bir ağaç'.
< Far bişa 'orman'. Türkçe bir kökünden değildir.
""")
    assert rec["origin"] == "loan" and rec["donor"] == "Far" and rec["root"] == ""


def test_eren_rejected_view_not_taken_and_gloss_ignored():
    rec = _eren("""
engir yer. 'üzüm çubuklarının kökünden çıkan sürgün'.
Türkçe ör- kökünden getirilmesi yanlıştır.
Kökünü bilmiyoruz.
""")
    assert rec["root"] == "" and rec["origin"] == "unknown"


def test_eren_plus_derivation_and_redirect_and_verb_key():
    rec = _eren("""
çaylak 'bir kuş'
~ Tkm çay 'çaylak'.
Türetildiği açıktır : çay + -lak eki.
""")
    assert rec["root"] == "çay" and rec["suffixes"] == ["-lak"]
    assert _eren("sermin bk. selmin.")["redirect"] == "selmin"
    assert eg.verb_key("savurmak") == "savur" and eg.verb_key("kaz") == ""


def test_gulensoy_derivation_and_ocr_fixes():
    rec = eg.parse_gulensoy_entry(["uçarı “Ele avuca sığmaz”", "€ uç-ar-ı", "An.ağl.: uçarı"])
    assert rec["root"] == "uç-" and rec["suffixes"] == ["-ar", "-ı"]
    # OCR: söz başı t -> f (Eski Türkçede yerli f- yok)
    rec = eg.parse_gulensoy_entry(["tansık “Olağanüstü olay”", "— OT, tangsuk (DLT)", "x OT. fang “şaşacak”"])
    assert rec["ot_form"] == "tangsuk" and rec["root"] == "tang"
    # OCR: + işareti t okunmuş -> ortak önek kök
    rec = eg.parse_gulensoy_entry(["sağlayıcı “Sağlayan”", "€ sağtla-y-ıcı"])
    assert rec["root"] == "sağ"
    # ilk ses uyuşmuyorsa (OCR artığı) kök yok — tahmin yapılmaz
    rec = eg.parse_gulensoy_entry(["tasar “Plan”", "€ fastar"])
    assert rec["root"] == ""


def test_gulensoy_kurdish_headword_skipped():
    assert eg.parse_gulensoy_entry(["kerekış (Kürt.) “Karakış”", "< Tü. karakış"]) is None


def test_clean_root():
    assert eg.clean_root("ob-/op-") == "ob"
    assert eg.clean_root("*kapa") == "kapa"
