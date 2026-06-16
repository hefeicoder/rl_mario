"""Structural validator for ppo-from-zero-to-hero.html.

Checks (no rendering): the file parses, every internal #anchor resolves to an
element id, KaTeX is linked, and the expected number of chapters is present.
Usage: python scripts/check_doc.py [expected_chapter_count]
"""
import sys
from html.parser import HTMLParser


class DocParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.anchors = []
        self.chapter_count = 0
        self.has_katex = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"):
            self.ids.add(a["id"])
        if tag == "a" and a.get("href", "").startswith("#"):
            self.anchors.append(a["href"][1:])
        if tag == "section" and "chapter" in a.get("class", "").split():
            self.chapter_count += 1
        if tag == "link" and "katex" in a.get("href", ""):
            self.has_katex = True


def main():
    expected_chapters = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    html = open("ppo-from-zero-to-hero.html", encoding="utf-8").read()
    p = DocParser()
    p.feed(html)

    errors = []
    missing = [a for a in p.anchors if a and a not in p.ids]
    if missing:
        errors.append(f"broken anchors (no matching id): {sorted(set(missing))}")
    if not p.has_katex:
        errors.append("KaTeX stylesheet not linked")
    if expected_chapters and p.chapter_count != expected_chapters:
        errors.append(f"expected {expected_chapters} chapters, found {p.chapter_count}")

    if errors:
        print("FAIL:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"OK: {p.chapter_count} chapters, {len(p.ids)} ids, "
          f"{len(set(p.anchors))} anchors all resolve, KaTeX linked")


if __name__ == "__main__":
    main()
