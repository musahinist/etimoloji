import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from engine.db.cldf_exporter import CldfExporter
from engine.db.database import DatabaseManager
from engine.logging_setup import get_logger, set_verbose
from engine.nlp.hypothesis_validation_protocol import HypothesisValidationProtocol
from engine.search_engine import SearchEngine

logger = get_logger(__name__)


def _stage_mark(stage: dict[str, Any], fail_label: str) -> str:
    """A-HVP aşama durumunu üç halli basar.

    ``is_valid`` ÜÇ durumludur: ``True`` geçti, ``False`` ölçüldü ve ihlal
    etti, ``None`` ölçülemedi (kanıt yok). Eskiden üçü de
    ``if stage.get("is_valid")`` ile ikiye indiriliyordu; bu yüzden
    ölçülemeyen aşama ihlal gibi raporlanıyordu (``bardak``: tarihli
    tanıklama yok -> ``is_valid`` None -> "❌ ANAKRONİZM"). Protokolün
    kendisi doğru davranıyordu — ``rejection_reasons`` boş, rozet
    🟢 DOĞRULANDI. Çelişki yalnızca bu gösterimdeydi.
    """
    is_valid = stage.get("is_valid")
    if is_valid is None:
        return "➖ ÖLÇÜLEMEDİ"
    return "✅ GEÇTİ" if is_valid else fail_label


def print_finding_formatted(finding: dict[str, Any]) -> None:
    query_word = finding.get("query_word", "")
    morphology = finding.get("morphology", "Yalın Kök")
    root = finding.get("root", {})
    timeline = finding.get("timeline", [])
    related_cognates = finding.get("related_cognates", [])
    turkic_languages = finding.get("turkic_languages", [])
    sources = finding.get("sources", [])
    from_cache = finding.get("from_cache", False)
    ai_enrichment = finding.get("ai_agent_enrichment", "")
    web_sources = finding.get("discovered_web_sources", [])
    nlp_analysis = finding.get("nlp_analysis", {})

    # 1. GENEL BAŞLIK VE MORFOLOJİ ÖZETİ
    print("\n" + "═" * 80)
    print(f" 🔍 ETIMOLOJI BULGUSU VE KÖKEN RAPORU: {query_word.upper()} {'(Veritabanı Önbelleği)' if from_cache else ''}")
    print("═" * 80)
    print(f" 🧩 Morfoloji & Yapı         : {morphology}")
    print(f" 📌 Ana Kök / Rekonstrüksiyon: {root.get('proto_turkic', 'Bilinmiyor')}")
    if root.get("provenance"):
        print(f" 🏷️  Kökün Kaynağı           : {root['provenance']}")
    protos = root.get("source_proto_forms") or []
    if protos:
        listed = ", ".join(f"{p['form']} ({p['count']} kayıt)" for p in protos)
        engine_root = str(root.get("proto_turkic") or "").strip("*-")
        # En sık verilen biçimle karşılaştır: `bitig` için bir kayıt fiil
        # kökünü (*biti-) de anıyor; "herhangi biriyle eşleşiyor" demek
        # asıl ayrışmayı (*bitig ~ *biti) gizliyordu.
        differs = bool(engine_root) and protos[0]["form"].strip("*-") != engine_root
        print(f" 📚 Kaynaklardaki Ata Biçim  : {listed}"
              + (f"  ⚠️ motorun rekonstrüksiyonu ({root.get('proto_turkic')}) bununla ayrışıyor" if differs else ""))
    for layer in root.get("origin_layers") or []:
        print(f" 🧭 Köken Katmanı            : {layer}")
    if root.get("root_note"):
        note = root["root_note"]
        print(f" 🌱 Kök Notu                 : {note[:300]}{'…' if len(note) > 300 else ''}")
    print(f" 📖 Anlam                     : {root.get('meaning', 'Bilinmiyor')}")
    if root.get("historical_meaning"):
        print(f" 📜 Tarihî Anlam              : {root['historical_meaning']}")
    for group in root.get("meanings") or []:
        print(f"    • {group['source']}")
        for m in group["meanings"]:
            print(f"        – {m}")
    print(f" 📚 Kaynak Portföyü           : {', '.join(sources)}")

    # 2. A-HVP (YAPAY ZEKA HİPOTEZ DOĞRULAMA VE HAKEMLİK PROTOKOLÜ) ÇIKTISI
    proven_hypo = nlp_analysis.get("proven_hypothesis") or {}
    val_report = proven_hypo.get("validation_report") or {}
    if val_report:
        print("\n" + "─" * 80)
        print(" ⚖️  A-HVP (AI HYPOTHESIS VALIDATION PROTOCOL) HAKEM RAPORU")
        print("─" * 80)
        print(f"  • Hakem Kararı & Rozet      : {val_report.get('badge')}")
        print(f"  • Genel Güven Skoru          : {val_report.get('score_percentage')} ({val_report.get('final_confidence_score')})")
        print(f"  • Hipotez Türü              : {proven_hypo.get('hypothesis_type')}")
        print(f"  • Kaynak Form / Ata Biçim    : {proven_hypo.get('origin_form')}")

        stages = val_report.get("stage_breakdown", {})
        s1 = stages.get("stage1_phonetic_chain", {})
        s2 = stages.get("stage2_time_lock", {})
        s3 = stages.get("stage3_semantic_drift", {})
        s4 = stages.get("stage4_cognate_triangulation", {})

        s2_reason = s2.get("reason") or s2.get("violation") or ""

        # ⚠️ AŞAMA SKORU GÖSTERİLMELİ. `✅ GEÇTİ` "ihlal yok" demektir,
        # "kanıt güçlü" demek DEĞİLDİR; rozet ise skor üzerinden verilir.
        # Skor gizlenince rozet okunamaz hâle geliyordu — ölçüldü:
        #     baş   fonetik skor 1.000  -> ✅   (rozet 🟢, stage_score 0.849)
        #     kalem fonetik skor 0.244  -> ✅   (rozet 🔴, stage_score 0.396)
        # İkisi ekranda birebir aynı görünüyor, kullanıcı "hepsi geçti ama
        # neden reddedildi" diye haklı olarak soruyordu.
        def _score(stage: dict) -> str:
            value = stage.get("score")
            return f" [skor {value:.2f}]" if isinstance(value, int | float) else ""

        print(f"  • 1. Fonetik Halka (IPA Kuralları): {_stage_mark(s1, '❌ İHLAL')}{_score(s1)} -> Eşleşen Ses Kuralları: {', '.join(s1.get('matched_rules', []))}")
        print(f"  • 2. Kronolojik Zaman Kilidi : {_stage_mark(s2, '❌ ANAKRONİZM')}{_score(s2)}" + (f" -> {s2_reason}" if s2_reason else ""))
        print(f"  • 3. Semantik Yörünge Sınırı : {_stage_mark(s3, '❌ İHLAL')}{_score(s3)} -> {s3.get('reason')}")
        print(f"  • 4. Akraba Dil Triangulation: {_stage_mark(s4, '❌ İHLAL')}{_score(s4)} -> Numune Akrabalar: {', '.join(s4.get('sample_cognates', [])[:4])}")
        print(f"  • Ölçülebilen kanıt kalitesi  : {val_report.get('stage_score')} (rozet bu sayıya göre verilir; eşikler 🟢 0.75 / 🟡 0.50)")
        missing = val_report.get("missing_evidence") or []
        if missing:
            print(f"  • Ölçülemeyen aşamalar       : {', '.join(missing)} (kapsam %{val_report.get('evidence_coverage', 0) * 100:.0f})")

        rejections = val_report.get("rejection_reasons", [])
        if rejections:
            print("\n  ⚠️  AKADEMİK HAKEM RED GEREKÇELERİ:")
            for rej in rejections:
                print(f"     ❌ {rej}")

    # 3. KÖKEN ZİNCİRİ ve TÜRKİ DİLLERDEKİ KARŞILIKLAR
    # Kaynak dil kayıtları (`donor`: Latince facies, İtalyanca faccia) Türki
    # dil değildir; eskiden "Türki dillerdeki karşılıklar" başlığı altında,
    # anlamsız bir "Ses Değişimi" etiketiyle basılıyordu.
    donor_entries = [e for e in turkic_languages if e.get("lang_code") == "donor"]
    turkic_languages = [e for e in turkic_languages if e.get("lang_code") not in ("donor", "ai")]
    if donor_entries:
        print("\n" + "─" * 80)
        print(f" 🧭 KÖKEN ZİNCİRİ — KAYNAK DİLLER ({len(donor_entries)} kayıt)")
        print("─" * 80)
        for entry in donor_entries:
            meaning = entry.get("meaning", "")
            print(f"  • {entry.get('lang_name', ''):<30} : {entry.get('word', ''):<24} [{'Anlam: ' + meaning if meaning else 'N/A'}]")

    print("\n" + "─" * 80)
    print(f" 🌍 TÜRKİ DİLLERDEKİ ANLAMLARI VE KARŞILIKLARI ({len(turkic_languages)} Dil/Katman)")
    print("─" * 80)

    if not turkic_languages:
        print("  ⚠️  Herhangi bir Türki dilde karşılık bulunamadı.")
    else:
        for entry in turkic_languages:
            lang_name = entry.get("lang_name", "")
            word = entry.get("word", "")
            meaning = entry.get("meaning", "")
            shift = entry.get("phonetic_shift", "")

            shift_info = f" [Ses Değişimi: {shift}]" if shift and shift != "Standart Lehçe Ses Uyumu" else ""
            # Runik/Arap yazılı biçimin yanında okunuşu (𐰋𐰃𐱅𐰏 bitig)
            if entry.get("comparison") and entry.get("script") not in (None, "Latin"):
                word = f"{word} {entry['comparison']}"
            print(f"  • {lang_name:<30} : {word:<24} [{'Anlam: ' + meaning if meaning else 'N/A'}]{shift_info}")
            if entry.get("formation"):
                print(f"      ↳ Yapı: {entry['formation']}")
            if entry.get("etymology"):
                note = entry["etymology"]
                print(f"      ↳ Kaynak notu: {note[:300]}{'…' if len(note) > 300 else ''}")
            if entry.get("source_cognates"):
                cogs = ", ".join(
                    f"{c['lang']} {c.get('reading') or c.get('form')}"
                    + (f" “{c['gloss']}”" if c.get("gloss") else "")
                    for c in entry["source_cognates"][:8]
                )
                print(f"      ↳ Kaynağın andığı akrabalar: {cogs}")

    mentions = finding.get("etymology_mentions") or {}
    if mentions.get("items"):
        print("\n" + "─" * 80)
        print(f" 🔁 ETİMOLOJİSİNDE BU KELİME GEÇEN KAYITLAR ({mentions['total']} kayıt; tanık sayılmaz)")
        print("─" * 80)
        for m in mentions["items"][:12]:
            form = m["word"] if m["comparison"] == m["word"] else f"{m['word']} {m['comparison']}"
            print(f"  • {m['lang_name']:<30} : {form:<24} [{m['gloss'] or 'N/A'}]")
            print(f"      ↳ {m['etymology']}")
            for source, text in (m.get("live") or {}).items():
                print(f"      ↳ {source}: {text[:200]}")
        for h in mentions.get("homonym_cognates") or []:
            print(f"  ⚠️  eşsesli, tanık sayılmadı: {h['lang_name']} {h['word']} “{h['meaning']}” (anlam benzerliği {h['similarity']})")
        if mentions["total"] > 12:
            print(f"  … ve {mentions['total'] - 12} kayıt daha (--json)")

    # 4. CANLI KEŞFEDİLEN WEB KAYNAKLARI VE MAKALE BAĞLANTILARI
    if web_sources:
        print("\n" + "─" * 80)
        print(" 🌐 CANLI KEŞFEDİLEN WEB KAYNAKLARI VE MAKALE BAĞLANTILARI")
        print("─" * 80)
        for s in web_sources:
            url = s.get("url", "")
            title = s.get("title", "")
            snip = s.get("snippet", "")
            print(f"  🔗 [{title}]")
            print(f"     URL  : {url}")
            if snip:
                print(f"     Özet : {snip[:120]}...")

    # 5. TARİHSEL ZAMAN ÇİZELGESİ VE KÖK AKRABA SÖZCÜK AĞI
    if timeline:
        print("\n" + "─" * 80)
        print(" ⏳ TARİHSEL ZAMAN ÇİZELGESİ (EVRİM KRONOLOJİSİ)")
        print("─" * 80)
        for step in timeline:
            print(f"  • {step}")

    if related_cognates:
        print("\n" + "─" * 80)
        print(" 🔗 KÖK AKRABA SÖZCÜK AĞI")
        print("─" * 80)
        print(f"  • Aynı kökten türeyen akraba kelimeler: {', '.join(related_cognates)}")

    # 6. HESAPLAMALI NLP ALINTI & REKONSTRÜKSİYON ANALİZİ
    if nlp_analysis:
        loan_eval = nlp_analysis.get("loanword_classification", {})
        cog_eval = nlp_analysis.get("cognate_distribution", {})
        recon_eval = nlp_analysis.get("reconstruction", {})

        print("\n" + "─" * 80)
        print(" 🧬 HESAPLAMALI NLP ALINTI & REKONSTRÜKSİYON ANALİZİ")
        print("─" * 80)
        print(f"  • Sınıflandırma              : {loan_eval.get('classification')}")
        if loan_eval.get("source_override"):
            print(f"       ↳ {loan_eval['source_override']}")
        probs = loan_eval.get('probabilities', {})
        p_native = probs.get('p_native_turkic', 0) * 100
        p_east = probs.get('p_arabic_persian', 0) * 100
        p_med = probs.get('p_greek_latin', 0) * 100
        p_west = probs.get('p_western', 0) * 100
        print(f"  • Olasılık Dağılımı         : Öz Türkçe: %{p_native:.1f} | Doğu (Arap/Fars): %{p_east:.1f} | Akdeniz (Grek/Erm): %{p_med:.1f} | Batı: %{p_west:.1f}")
        print(f"  • Lehçe Yayılımı Skorlama   : %{cog_eval.get('spreading_ratio', 0)*100:.0f} ({cog_eval.get('assessment')})")
        # Alıntı durumunda `reconstruction_notes` ÇOK SATIRLI gelir (alıntı
        # detektörünün `explain()` dökümü) ve biçimsiz bir blok hâlinde
        # akıyordu. Ayrıntı artık aşağıdaki "RAKİP KÖKEN HİPOTEZLERİ" bloğunda
        # yapılandırılmış olarak basılıyor; burada ilk satır yeter. JSON yükü
        # değişmez — API ve web paneli tam metni almaya devam eder.
        recon_note = str(recon_eval.get("reconstruction_notes") or "").strip()
        print(f"  • Rekonstrüksiyon Değerlend: {recon_note.splitlines()[0] if recon_note else '—'}")
        # Tanık tanıklığı ENGELLEYİCİ değil, görünür: sütun uyumu tanıkların
        # birbiriyle uyuşmasını ölçer, gerçek olup olmadıklarını değil.
        # Uydurma bir kökün uydurulmuş tanıkları da kusursuz uyumludur.
        _attested = recon_eval.get("attested_witness_count")
        if _attested == 0:
            print(
                "  ⚠️  Tanık Tanıklığı        : hiçbir tanık biçimi sözlükte "
                "bulunamadı — kökün tanıkların kendi uyumundan başka dayanağı yok"
            )
        elif isinstance(_attested, int) and _attested > 0:
            print(f"  • Tanık Tanıklığı          : {_attested} tanık biçimi sözlükte doğrulandı")

        # Kullanıcıya giden sayı HAM skor değil kalibre skordur (ham skorun
        # ECE'si 0,43 ölçüldü). Motor bunu zaten hesaplıyordu ama CLI basmıyor,
        # yalnızca kalibre edilmemiş ara skorlar görünüyordu.
        calibrated = recon_eval.get("calibrated_confidence")
        if calibrated is not None:
            print(f"  • Kalibre Güven              : {calibrated:.2f}  {recon_eval.get('confidence_badge', '')}")
            if recon_eval.get("calibration_note"):
                print(f"       ↳ {recon_eval['calibration_note']}")

        # Morfolojik soyma sessiz kalmamalı: rekonstrüksiyon sorulan kelime
        # üzerinden değil, soyulmuş kök üzerinden yapıldıysa rapor iki farklı
        # girdi biçimi gösteriyor (`bardak` başlıkta, `barda` alt satırda).
        if recon_eval.get("stripped_from"):
            suffixes = ", ".join(s for s in (recon_eval.get("stripped_suffixes") or []) if s)
            print(
                f"  ⚠️  Soyulmuş kök kullanıldı  : '{recon_eval.get('word')}' "
                f"(<- '{recon_eval['stripped_from']}', soyulan ek: {suffixes or '—'})"
            )

    # 6b. RAKİP KÖKEN HİPOTEZLERİ VE RED GEREKÇELERİ
    ranked = nlp_analysis.get("ranked_hypotheses") or {}
    hypotheses = ranked.get("hypotheses") or []
    if hypotheses:
        selected_claim = (ranked.get("selected") or {}).get("claim", "")
        print("\n" + "─" * 80)
        print(" ⚖️  RAKİP KÖKEN HİPOTEZLERİ (sıralı — reddedilenler gerekçesiyle kalır)")
        print("─" * 80)
        for i, hypo in enumerate(hypotheses, 1):
            claim = hypo.get("claim") or hypo.get("label", "")
            mark = "✓" if claim == selected_claim else " "
            state = " ❌ REDDEDİLDİ" if hypo.get("rejected") else ""
            print(f"  {i}. {mark} {claim}  [skor {hypo.get('score', 0):.3f}]{state}")
            for support in (hypo.get("supporting") or [])[:3]:
                print(f"       + {support}")
            for against in (hypo.get("against") or [])[:2]:
                print(f"       − {against}")
            # Veri olmadığı için çalışamayan sinyaller karşı kanıt değildir.
            if hypo.get("not_evaluated"):
                print(f"       · ölçülemedi: {'; '.join(hypo['not_evaluated'][:3])}")
            if hypo.get("rejected_because"):
                print(f"       ❌ gerekçe: {hypo['rejected_because']}")
        margin = ranked.get("margin")
        if margin is not None:
            contested = "  ⚠️ ÇEKİŞMELİ" if ranked.get("is_contested") else ""
            print(f"\n  • Birinci ile ikinci hipotez arası fark: {margin:.3f}{contested}")

    # 6c. KAYNAK VERİMİ — hangi kaynak gerçekten kanıt üretti?
    # Motor her kaynağın sonucunu `diagnostics.sources` altında zaten
    # tutuyordu ama hiçbir yerde göstermiyordu. Sessiz bozulma bu yüzden
    # görünmezdi: ölü bir uç nokta ya da kırık bir ayrıştırıcı, "kaynak
    # portföyü" satırında adı geçmediği için fark edilmiyordu.
    source_diag = (finding.get("diagnostics") or {}).get("sources") or {}
    if source_diag:
        produced = [n for n, d in source_diag.items() if d.get("status") == "ok"]
        silent = [n for n, d in source_diag.items() if d.get("status") == "empty"]
        failed = [n for n, d in source_diag.items() if d.get("status") == "error"]
        print("\n" + "─" * 80)
        print(" 📡 KAYNAK VERİMİ")
        print("─" * 80)
        print(
            f"  • Kanıt üreten: {len(produced)}/{len(source_diag)}"
            f"  |  sessiz (veri yok): {len(silent)}  |  hata: {len(failed)}"
        )
        for name in silent:
            print(f"     ➖ {name}")
        for name in failed:
            errs = (source_diag[name].get("errors") or [""])[0]
            print(f"     ❌ {name} -> {errs[:90]}")

    # 7. EN ALTA FİNAL SENTEZİ OLARAK: Qwen2.5 Otonom Yapay Zeka Ajanı Analizi
    if ai_enrichment:
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│ 🤖 QWEN2.5 OTONOM BİLİMSEL AJAN AKIL YÜRÜTME & FİNAL SENTEZ PARAGRAFI  │")
        print("├" + "─" * 78 + "┤")
        lines = ai_enrichment.split("\n")
        for line in lines:
            if line.strip():
                wrapped_words = line.strip().split()
                current_line = "│ "
                for w in wrapped_words:
                    if len(current_line) + len(w) + 1 > 77:
                        print(f"{current_line:<79}│")
                        current_line = "│ " + w + " "
                    else:
                        current_line += w + " "
                if current_line.strip() != "│":
                    print(f"{current_line:<79}│")
            else:
                print("│" + " " * 78 + "│")
        print("└" + "─" * 78 + "┘")

    print("═" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Türki Diller Etimoloji Araştırma Motoru CLI")
    subparsers = parser.add_subparsers(dest="command", help="Komutlar")

    search_parser = subparsers.add_parser("search", help="Bir kelimenin etimolojisini ve Türki dillerdeki anlamlarını arar")
    search_parser.add_argument("word", type=str, help="Aranacak kelime (örn: su, deniz, göz, us, tetik, güzellik)")
    search_parser.add_argument("--json", action="store_true", help="Çıktıyı ham JSON formatında basar")
    search_parser.add_argument("--ai", action="store_true", help="Qwen2.5 otonom web araştırma ajanı ile derinleştirilmiş arama yap")
    search_parser.add_argument("--no-save", action="store_false", dest="save", help="Sonucu veritabanına kaydetme")
    search_parser.add_argument("--no-cache", action="store_false", dest="use_cache",
                               help="Önbellekteki eski bulguyu kullanma, yeniden araştır")

    validate_parser = subparsers.add_parser("validate", help="Bir etimoloji hipotezini A-HVP protokolü ile bilimsel olarak doğrular")
    validate_parser.add_argument("word", type=str, help="Hedef kelime")
    validate_parser.add_argument("--origin", type=str, default=None, help="Önerilen ata biçim / kök")
    validate_parser.add_argument("--donor", type=str, default="Öz Türkçe", help="Önerilen donör/kaynak dil")
    validate_parser.add_argument("--attestation", type=str, default="11. yüzyıl Divanü Lugati't-Türk", help="Yazılı ilk kayıt dönemi")

    bulk_parser = subparsers.add_parser("bulk", help="Bir metin dosyasındaki tüm kelimelerin etimolojisini topluca sorgular")
    bulk_parser.add_argument("--file", type=str, required=True, help="Kelimelerin bulunduğu metin dosyası")

    subparsers.add_parser("list", help="Veritabanına kaydedilmiş tüm kelimeleri listeler")

    show_parser = subparsers.add_parser("show", help="Veritabanındaki bir kelimenin bulgusunu detaylı gösterir")
    show_parser.add_argument("word", type=str, help="Gösterilecek kelime")
    show_parser.add_argument("--json", action="store_true", help="JSON formatında göster")

    export_parser = subparsers.add_parser(
        "export", help="Kayıtlı bulguyu CLDF (Cross-Linguistic Data Formats) olarak dışa aktarır"
    )
    export_parser.add_argument("word", type=str, help="Dışa aktarılacak kelime")
    export_parser.add_argument("--out", type=str, default=None, help="Çıktı dizini (varsayılan: cldf_export/)")

    parser.add_argument("--verbose", "-v", action="store_true", help="Ayrıntılı log çıktısı")

    args = parser.parse_args()
    set_verbose(getattr(args, "verbose", False))

    if not args.command:
        parser.print_help()
        sys.exit(1)

    db_manager = DatabaseManager()
    engine = SearchEngine(db_manager=db_manager)

    if args.command == "search":
        try:
            if args.ai:
                print("🤖 Qwen2.5 Otonom Web Keşif Ajanı Devrede... (Derin Web & Makale Taraması Yapılıyor)")
            finding = engine.search(args.word, save_to_db=args.save, use_qwen_agent=args.ai, use_cache=args.use_cache)
            if args.json:
                print(json.dumps(finding, ensure_ascii=False, indent=2))
            else:
                print_finding_formatted(finding)
        except Exception as e:
            logger.error("Arama başarısız: %s", args.word, exc_info=True)
            print(f"❌ Arama başarısız: {type(e).__name__}. Ayrıntı için --verbose kullanın.", file=sys.stderr)
            sys.exit(1)

    elif args.command == "validate":
        protocol = HypothesisValidationProtocol()
        origin_form = args.origin or args.word
        hypothesis = {
            "hypothesis_type": f"Özel Etimoloji Hipotezi ({args.donor})",
            "origin_form": origin_form,
            "donor_language": args.donor,
            "proof_summary": f"Kullanıcı Hipotezi: {args.donor} kökenli {origin_form}",
            "historical_meaning": args.word
        }
        attestation = {"first_attestation_record": args.attestation}

        report = protocol.validate_hypothesis(args.word, hypothesis, attestation)
        print("\n" + "═" * 80)
        print(f" ⚖️  A-HVP AKADEMİK DOĞRULAMA VE HAKEM HİPOTEZİ RAPORU: {args.word.upper()}")
        print("═" * 80)
        print(f"  • Hakem Kararı & Rozet      : {report['badge']}")
        print(f"  • Hakem Skoru (% Yüzde)      : {report['score_percentage']}")
        print(f"  • Önerilen Ata Kök          : {origin_form}")
        print(f"  • Önerilen Donör Dil         : {args.donor}")
        print(f"  • İlk Yazılı Tanıklama      : {args.attestation}")

        stages = report.get("stage_breakdown", {})
        s1 = stages.get("stage1_phonetic_chain", {})
        s2 = stages.get("stage2_time_lock", {})
        s3 = stages.get("stage3_semantic_drift", {})
        s4 = stages.get("stage4_cognate_triangulation", {})

        print("\n  📌 5 KADEMELİ HAKEM AŞAMA DETAYLARI:")
        v_s2_reason = s2.get("reason") or s2.get("violation") or ""
        print(f"   1. Fonetik Halka (IPA Evrimi): {_stage_mark(s1, '❌ İHLAL')} -> {', '.join(s1.get('matched_rules', []))}")
        print(f"   2. Kronolojik Zaman Kilidi  : {_stage_mark(s2, '❌ ANAKRONİZM')}" + (f" -> {v_s2_reason}" if v_s2_reason else ""))
        print(f"   3. Diyakronik Semantik Sınır: {_stage_mark(s3, '❌ İHLAL')} -> {s3.get('reason')}")
        print(f"   4. Akraba Dil Triangulation : {_stage_mark(s4, '❌ İHLAL')} -> Numune: {', '.join(s4.get('sample_cognates', [])[:4])}")
        v_missing = report.get("missing_evidence") or []
        if v_missing:
            print(f"   Ölçülemeyen aşamalar       : {', '.join(v_missing)} (kapsam %{report.get('evidence_coverage', 0) * 100:.0f})")

        rejections = report.get("rejection_reasons", [])
        if rejections:
            print("\n  ⚠️  AKADEMİK HAKEM RED GEREKÇELERİ:")
            for rej in rejections:
                print(f"     ❌ {rej}")
        print("═" * 80 + "\n")

    elif args.command == "bulk":
        if not os.path.exists(args.file):
            print(f"❌ Dosya bulunamadı: {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file, encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
        print(f"\n📦 TOPLU ETIMOLOJI TARAMASI BAŞLATILDI ({len(words)} Kelime)\n")
        for i, w in enumerate(words, 1):
            print(f"[{i}/{len(words)}] Aratılıyor: {w} ...")
            finding = engine.search(w, save_to_db=True)
            print(f"  ✓ Tamamlandı: {w} (Kök: {finding['root']['proto_turkic']})")
        print("\n✅ Tüm toplu arama sonuçları veritabanına kaydedildi.\n")

    elif args.command == "list":
        findings = db_manager.list_findings()
        print(f"\n📂 VERİTABANINDA KAYITLI ETIMOLOJİ BULGULARI ({len(findings)} Kayıt)")
        print("-" * 75)
        for f in findings:
            print(f"  • {f['query_word']:<15} | Kök: {f['proto_turkic_root']:<12} | Anlam: {f['root_meaning']:<25} | Tarih: {f['created_at']}")
        print("-" * 75 + "\n")

    elif args.command == "show":
        finding = db_manager.get_finding(args.word)
        if not finding:
            print(f"❌ '{args.word}' kelimesi veritabanında bulunamadı.", file=sys.stderr)
            sys.exit(1)
        if args.json:
            print(json.dumps(finding, ensure_ascii=False, indent=2))
        else:
            print_finding_formatted(finding)

    elif args.command == "export":
        finding = db_manager.get_finding(args.word)
        if not finding:
            print(f"❌ '{args.word}' kelimesi veritabanında bulunamadı. Önce `search` çalıştırın.", file=sys.stderr)
            sys.exit(1)
        out_dir = Path(args.out) if args.out else Path("cldf_export")
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle = CldfExporter().export_to_cldf(finding)
        written = []
        file_map = {
            "forms.csv": bundle["cldf_forms_csv"],
            "cognates.csv": bundle["cldf_cognates_csv"],
            "cldf-metadata.json": json.dumps(bundle["metadata"], ensure_ascii=False, indent=2),
        }
        for name, content in file_map.items():
            path = out_dir / name
            path.write_text(content, encoding="utf-8")
            written.append(str(path))
        print(f"\n📦 CLDF dışa aktarımı tamamlandı ({len(written)} dosya):")
        for w in written:
            print(f"  • {w}")
        print()

if __name__ == "__main__":
    main()
