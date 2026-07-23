import sys
import os
import json
import argparse
from typing import Dict, Any

from engine.search_engine import SearchEngine
from engine.db.database import DatabaseManager

def print_finding_formatted(finding: Dict[str, Any]) -> None:
    query_word = finding.get("query_word", "")
    morphology = finding.get("morphology", "Yalın Kök")
    root = finding.get("root", {})
    timeline = finding.get("timeline", [])
    related_cognates = finding.get("related_cognates", [])
    turkic_languages = finding.get("turkic_languages", [])
    sources = finding.get("sources", [])
    from_cache = finding.get("from_cache", False)

    print("\n" + "=" * 75)
    print(f" 🔍 ETIMOLOJI BULGUSU: {query_word.upper()} {'(Veritabanı Önbelleği)' if from_cache else ''}")
    print("=" * 75)
    print(f" 🧩 Morfoloji & Yapı         : {morphology}")
    print(f" 📌 Ana Kök / Rekonstrüksiyon: {root.get('proto_turkic', 'Bilinmiyor')}")
    print(f" 📖 Anlam                     : {root.get('meaning', 'Bilinmiyor')}")
    print(f" 📚 Kaynak Portföyü           : {', '.join(sources)}")

    if timeline:
        print("-" * 75)
        print(" ⏳ TARİHSEL ZAMAN ÇİZELGESİ (EVRİM)")
        print("-" * 75)
        for step in timeline:
            print(f"  • {step}")

    if related_cognates:
        print("-" * 75)
        print(" 🔗 KÖK AKRABA SÖZCÜK AĞI")
        print("-" * 75)
        print(f"  • Aynı kökten türeyen akraba kelimeler: {', '.join(related_cognates)}")

    print("-" * 75)
    print(f" 🌍 TÜRKİ DİLLERDEKİ ANLAMLARI VE KARŞILIKLARI ({len(turkic_languages)} Dil/Katman)")
    print("-" * 75)

    if not turkic_languages:
        print("  ⚠️  Herhangi bir Türki dilde karşılık bulunamadı.")
    else:
        for entry in turkic_languages:
            lang_name = entry.get("lang_name", "")
            word = entry.get("word", "")
            meaning = entry.get("meaning", "")
            shift = entry.get("phonetic_shift", "")
            
            shift_info = f" [Ses Değişimi: {shift}]" if shift and shift != "Standart Lehçe Ses Uyumu" else ""
            print(f"  • {lang_name:<28} : {word:<22} [{'Anlam: ' + meaning if meaning else 'N/A'}]{shift_info}")

    print("=" * 75 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Türki Diller Etimoloji Araştırma Motoru CLI")
    subparsers = parser.add_subparsers(dest="command", help="Komutlar")

    # Search komutu
    search_parser = subparsers.add_parser("search", help="Bir kelimenin etimolojisini ve Türki dillerdeki anlamlarını arar")
    search_parser.add_argument("word", type=str, help="Aranacak kelime (örn: su, deniz, göz, tetik, güzellik)")
    search_parser.add_argument("--json", action="store_true", help="Çıktıyı ham JSON formatında basar")
    search_parser.add_argument("--no-save", action="store_false", dest="save", help="Sonucu veritabanına kaydetme")

    # Bulk komutu
    bulk_parser = subparsers.add_parser("bulk", help="Bir metin dosyasındaki tüm kelimelerin etimolojisini topluca sorgular")
    bulk_parser.add_argument("--file", type=str, required=True, help="Kelimelerin bulunduğu metin dosyası")

    # List komutu
    list_parser = subparsers.add_parser("list", help="Veritabanına kaydedilmiş tüm kelimeleri listeler")

    # Show komutu
    show_parser = subparsers.add_parser("show", help="Veritabanındaki bir kelimenin bulgusunu detaylı gösterir")
    show_parser.add_argument("word", type=str, help="Gösterilecek kelime")
    show_parser.add_argument("--json", action="store_true", help="JSON formatında göster")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    db_manager = DatabaseManager()
    engine = SearchEngine(db_manager=db_manager)

    if args.command == "search":
        try:
            finding = engine.search(args.word, save_to_db=args.save)
            if args.json:
                print(json.dumps(finding, ensure_ascii=False, indent=2))
            else:
                print_finding_formatted(finding)
        except Exception as e:
            print(f"❌ Hata oluştu: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "bulk":
        if not os.path.exists(args.file):
            print(f"❌ Dosya bulunamadı: {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
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

if __name__ == "__main__":
    main()
