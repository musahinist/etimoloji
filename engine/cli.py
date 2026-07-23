import sys
import json
import argparse
from typing import Dict, Any

from engine.search_engine import SearchEngine
from engine.db.database import DatabaseManager

def print_finding_formatted(finding: Dict[str, Any]) -> None:
    query_word = finding.get("query_word", "")
    root = finding.get("root", {})
    turkic_languages = finding.get("turkic_languages", [])
    sources = finding.get("sources", [])
    from_cache = finding.get("from_cache", False)

    print("\n" + "=" * 65)
    print(f" 🔍 ETIMOLOJI BULGUSU: {query_word.upper()} {'(Veritabanı Önbelleği)' if from_cache else ''}")
    print("=" * 65)
    print(f" 📌 Ana Kök / Rekonstrüksiyon: {root.get('proto_turkic', 'Bilinmiyor')}")
    print(f" 📖 Anlam: {root.get('meaning', 'Bilinmiyor')}")
    print(f" 📚 Kaynaklar: {', '.join(sources)}")
    print("-" * 65)
    print(f" 🌍 TÜRKİ DİLLERDEKİ ANLAMLARI VE KARŞILIKLARI ({len(turkic_languages)} Dil)")
    print("-" * 65)

    if not turkic_languages:
        print("  ⚠️  Herhangi bir Türki dilde karşılık bulunamadı.")
    else:
        for entry in turkic_languages:
            lang_name = entry.get("lang_name", "")
            word = entry.get("word", "")
            meaning = entry.get("meaning", "")
            script = entry.get("script", "")
            print(f"  • {lang_name:<24} : {word:<20} [{'Anlam: ' + meaning if meaning else 'N/A'}]")

    print("=" * 65 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Türki Diller Etimoloji Araştırma Motoru CLI")
    subparsers = parser.add_subparsers(dest="command", help="Komutlar")

    # Search komutu
    search_parser = subparsers.add_parser("search", help="Bir kelimenin etimolojisini ve Türki dillerdeki anlamlarını arar")
    search_parser.add_argument("word", type=str, help="Aranacak kelime (örn: su, deniz, göz)")
    search_parser.add_argument("--json", action="store_true", help="Çıktıyı ham JSON formatında basar")
    search_parser.add_argument("--no-save", action="store_false", dest="save", help="Sonucu veritabanına kaydetme")

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

    elif args.command == "list":
        findings = db_manager.list_findings()
        print(f"\n📂 VERİTABANINDA KAYITLI ETIMOLOJİ BULGULARI ({len(findings)} Kayıt)")
        print("-" * 65)
        for f in findings:
            print(f"  • {f['query_word']:<15} | Kök: {f['proto_turkic_root']:<12} | Anlam: {f['root_meaning']:<25} | Tarih: {f['created_at']}")
        print("-" * 65 + "\n")

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
