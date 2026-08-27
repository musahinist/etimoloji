"""
CLDF dil kimliklerini motorun dil kodlarına bağlar.

Her veri kümesi aynı dili başka türlü adlandırır::

    savelyevturkic   "Chuvash"   "Khalaj"   "Turkish"
    hruschkaturkic   "CHV"       "KHAL"     "TRK"
    starostinaltaic  "chuvash"   —          "turkish"

Bu yüzden eşleme **ada göre değil, Glottocode'a göre** yapılır. Glottolog
kimliği veri kümeleri arası tek kararlı anahtardır; ada göre eşleme
("Turkish" ~ "TRK") kırılgan ve sessizce yanlıştır.

⚠️ Eşlenemeyen dil **sessizce düşmez**, :func:`unmapped_languages` ile
raporlanır. Sessiz düşme bu projede ölçülmüş bir hata kaynağıydı: ünlü
uzunluğu tanıklarının %48'i tam bu yolla kayboluyordu.
"""

from __future__ import annotations

from engine.logging_setup import get_logger

logger = get_logger(__name__)

#: Glottocode -> motorun dil kodu (``TURKIC_LANGUAGES_MAP`` anahtarları).
GLOTTOCODE_TO_ENGINE: dict[str, str] = {
    # Oğuz
    "anat1259": "tr",
    "nucl1301": "tr",
    "nort2697": "az",
    "turk1304": "tk",
    "gaga1249": "gag",
    # Kıpçak
    "kaza1248": "kk",
    "kara1467": "kaa",
    "kirg1245": "ky",
    "noga1249": "nog",
    "kumy1244": "kum",
    "kara1465": "krc",
    "crim1257": "crh",
    "midd1325": "tt",
    "tata1255": "tt",
    "bash1264": "ba",
    "kara1464": "kdr",
    "bara1273": "bay",
    # Karluk
    "uzbe1247": "uz",
    "sout2699": "uz",
    "uigh1240": "ug",
    "sala1264": "slq",
    # Sibirya
    "yaku1245": "sah",
    "dolg1241": "dlg",
    "tuvi1240": "tyv",
    "tofa1248": "kim",
    "khak1248": "khk",
    "shor1247": "cjs",
    "sout2694": "alt",
    "nort2686": "atv",
    "midd1324": "clw",
    "west2402": "ybe",
    # Oğur
    "chuv1255": "cv",
    "bolg1249": "wot",  # Batı Eski Türkçe — geri kurulmuş, bkz. west_old_turkic
    # Arguca — hiçbir ana kola girmez
    "turk1303": "klj",
    # Tarihî
    "oldu1238": "otk",
}

#: Glottocode taşımayan kayıtlar için veri kümesi kimliğine göre yedek eşleme.
DATASET_ID_TO_ENGINE: dict[str, str] = {
    "codexcumanicus": "qwm",
    "cuman": "qwm",
    "sjg": "ybe",  # hruschkaturkic'te Sarı Yugur Glottocode taşımıyor
    "tof": "kim",  # hruschkaturkic'te Tofa Glottocode taşımıyor
    # `starostinaltaic` hiç Glottocode taşımıyor; kimlikleri düz dil adıdır.
    # Yalnız TÜRKİ altkümesi eşlenir — Korece/Japonca/Moğolca/Tunguzca dalları
    # bilinçli olarak dışarıda bırakılır (Altay hipotezi tartışmalıdır,
    # Vovin 2005; bu veri kümesi akrabalık kanıtına katılmaz).
    "turkish": "tr",
    "azerbaidzhan": "az",
    "turkmen": "tk",
    "gagauz": "gag",
    "tatar": "tt",
    "bashkir": "ba",
    "kirghiz": "ky",
    "karakalpak": "kaa",
    "noghai": "nog",
    "balkar": "krc",
    "karaim": "kdr",
    "uzbek": "uz",
    "uighur": "ug",
    "salar": "slq",
    "yakut": "sah",
    "tuva": "tyv",
    "khankassian": "khk",
    "saryyughur": "ybe",
    "sharyyoghur": "ybe",
    "chuvash": "cv",
    "oldturkic": "otk",
}


def to_engine_code(language_id: str, glottocode: str = "") -> str | None:
    """CLDF dil kimliğini motor koduna çevirir; eşleşmezse ``None``."""
    if glottocode:
        code = GLOTTOCODE_TO_ENGINE.get(glottocode.strip().lower())
        if code:
            return code
    normalized = "".join(ch for ch in language_id.lower() if ch.isalnum())
    return DATASET_ID_TO_ENGINE.get(normalized)


def build_mapping(wordlist) -> dict[str, str]:  # noqa: ANN001 — döngüsel import kaçınma
    """Bir :class:`~engine.db.cldf_wordlist.CldfWordlist` için tam eşleme."""
    mapping: dict[str, str] = {}
    for lang_id, info in wordlist.languages.items():
        code = to_engine_code(lang_id, info.glottocode)
        if code:
            mapping[lang_id] = code
    return mapping


def unmapped_languages(wordlist) -> list[str]:  # noqa: ANN001
    """Eşlenemeyen diller — sessizce düşmesinler diye açıkça raporlanır."""
    mapping = build_mapping(wordlist)
    missing = sorted(set(wordlist.languages) - set(mapping))
    if missing:
        logger.warning(
            "%s: %d dil motor koduna eşlenemedi ve değerlendirmeye girmeyecek: %s",
            getattr(wordlist, "dir", "?"),
            len(missing),
            ", ".join(missing),
        )
    return missing
