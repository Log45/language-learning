import argparse
import json
import time
import sys

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse


def extract_entry_from_container(container) -> dict:
    # kanji is usually in a span with class containing 'xlarge'
    kanji_tag = container.find("span", class_=lambda c: c and "xlarge" in c)

    # hiragana often sits in a span right after the kanji span, or is a span containing hiragana characters
    kana_tag = None
    if kanji_tag:
        kana_tag = kanji_tag.find_next_sibling(lambda t: t.name == "span")
    if not kana_tag:
        # fallback: find any span containing hiragana chars
        kana_tag = container.find("span", string=lambda s: s and any("\u3040" <= ch <= "\u309F" for ch in s))

    # romaji often in an <i> with class containing 'xsmall'
    romaji_tag = container.find("i", class_=lambda c: c and "xsmall" in c)

    # badges
    badges = []
    jlpt = None
    for b in container.find_all("span", class_=lambda c: c and "badge" in c):
        text = b.get_text(strip=True)
        title = b.get("title") or None
        # detect JLPT badge like 'JLPT N5' and extract level 'N5'
        if text and text.upper().startswith("JLPT"):
            # take remainder after 'JLPT'
            level = text[4:].strip()
            if level:
                jlpt = level
            continue
        badges.append({
            "text": text,
            "title": title,
        })

    # meanings — look for list items under any list-unstyled inside this container
    meanings = []
    for ul in container.find_all("ul", class_=lambda c: c and "list-unstyled" in c):
        for li in ul.find_all("li"):
            text = li.get_text(separator=" ", strip=True)
            if text:
                meanings.append(text)
    kanji = kanji_tag.get_text(strip=True) if kanji_tag else None
    hiragana = kana_tag.get_text(strip=True) if kana_tag else None
    romaji = romaji_tag.get_text(strip=True) if romaji_tag else None

    # Only return entries that have at least one of kanji, hiragana, or meanings
    if not (kanji or hiragana or meanings):
        return {}

    result = {
        "kanji": kanji,
        "hiragana": hiragana,
        "romaji": romaji,
        "badges": badges,
        "meanings": meanings,
    }
    if jlpt:
        result["jlpt"] = jlpt
    return result


def _url_with_page(base_url: str, page: int) -> str:
    parsed = urlparse(base_url)
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    qs["page"] = str(page)
    new_q = urlencode(qs, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_q, parsed.fragment))


def scrape(url: str, delay: float = 0.25, max_pages: int = 0) -> list[dict]:
    """Scrape all paginated pages starting from `url` until no new entries are found.

    Args:
        url: base URL of the list (may already contain query params)
        delay: seconds to sleep between requests
        max_pages: optional safety limit (0 = unlimited)
    """

    results: list[dict] = []
    seen = set()
    page = 1

    while True:
        page_url = _url_with_page(url, page)
        try:
            resp = requests.get(page_url, headers={"User-Agent": "language-learning-scraper/1.0"})
            resp.raise_for_status()
        except requests.RequestException:
            break

        soup = BeautifulSoup(resp.text, "lxml")

        # Find entry containers used by the site. Entries are wrapped in
        # divs containing both 'd-inline-block' and 'w-100' classes.
        containers = soup.find_all("div", class_=lambda c: c and "d-inline-block" in c and "w-100" in c)

        added_this_page = 0
        for container in containers:
            # skip containers that are inside typical footer elements
            if container.find_parent("footer"):
                continue
            entry = extract_entry_from_container(container)
            if not entry:
                continue

            # dedupe by key (kanji, hiragana, romaji, meanings)
            key = (
                entry.get("kanji"),
                entry.get("hiragana"),
                entry.get("romaji"),
                "|".join(entry.get("meanings") or []),
            )
            if key in seen:
                continue
            seen.add(key)
            results.append(entry)
            added_this_page += 1

        # stop conditions
        if added_this_page == 0:
            break
        page += 1
        if max_pages and page > max_pages:
            break
        time.sleep(delay)
        print(f"Scraped page {page - 1}, total entries: {len(results)}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Scrape Japanese words from japandict list pages")
    parser.add_argument("--url", default="https://www.japandict.com/lists/jlpt/jlpt5", help="Page URL to scrape")
    parser.add_argument("--output", default="jlpt5_words.json", help="Output JSON filename")
    args = parser.parse_args()

    print(f"Fetching {args.url}...")
    entries = scrape(args.url)
    print(f"Found {len(entries)} entries. Writing to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
        
def scrape_dictionaries():
    base_url = "https://www.japandict.com/lists/jlpt/jlpt"
    for level in range(5, 0, -1):
        url = f"{base_url}{level}"
        output_file = f"n{level}.json"
        print(f"Scraping JLPT N{level} from {url}...")
        entries = scrape(url)
        print(f"Found {len(entries)} entries. Writing to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        scrape_dictionaries()
