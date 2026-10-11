"""
gist_set.py — the test for "does Mimir get the gist?"

Splits Grimms' Fairy Tales (data/reader/books/shelf/grimms_fairy_tales.txt)
into its tales, and for each tale that has a Wikipedia article with a plot
section, saves the tale and that plot summary (the human-written gist).

    python3 tools/gist_set.py      → data/gist/<tale>.story.txt
                                     data/gist/<tale>.summary.txt
                                     data/gist/index.tsv  (tale, article, url)

Wikipedia text is CC BY-SA 4.0; index.tsv records the article each summary
came from.
"""
import json, os, re, sys, time, urllib.error, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
BOOK = os.path.join(ROOT, "data", "reader", "books", "shelf", "grimms_fairy_tales.txt")
OUT = os.path.join(ROOT, "data", "gist")

# Grimm title (as in the book) → English Wikipedia article
ARTICLES = {
    "THE GOLDEN BIRD": "The Golden Bird",
    "HANS IN LUCK": "Hans in Luck",
    "JORINDA AND JORINDEL": "Jorinde and Joringel",
    "THE TRAVELLING MUSICIANS": "Town Musicians of Bremen",
    "THE TWELVE DANCING PRINCESSES": "The Twelve Dancing Princesses",
    "THE FISHERMAN AND HIS WIFE": "The Fisherman and His Wife",
    "THE FROG-PRINCE": "The Frog Prince",
    "THE GOOSE-GIRL": "The Goose Girl",
    "RAPUNZEL": "Rapunzel",
    "FUNDEVOGEL": "Foundling-Bird",
    "THE VALIANT LITTLE TAILOR": "The Valiant Little Tailor",
    "HANSEL AND GRETEL": "Hansel and Gretel",
    "MOTHER HOLLE": "Frau Holle",
    "THE ROBBER BRIDEGROOM": "The Robber Bridegroom",
    "RUMPELSTILTSKIN": "Rumpelstiltskin",
    "CLEVER ELSIE": "Clever Elsie",
    "SWEETHEART ROLAND": "Sweetheart Roland",
    "SNOWDROP": "Snow White",
    "THE WHITE SNAKE": "The White Snake",
    "THE WOLF AND THE SEVEN LITTLE KIDS": "The Wolf and the Seven Young Goats",
    "THE QUEEN BEE": "The Queen Bee",
    "THE ELVES AND THE SHOEMAKER": "The Elves and the Shoemaker",
    "THE JUNIPER-TREE": "The Juniper Tree (fairy tale)",
    "THE FOUR CLEVER BROTHERS": "The Four Skillful Brothers",
    "THE BLUE LIGHT": "The Blue Light",
    "THE GOLDEN GOOSE": "The Golden Goose",
    "THE WATER OF LIFE": "The Water of Life (German fairy tale)",
    "THE KING OF THE GOLDEN MOUNTAIN": "The King of the Golden Mountain",
    "THE SALAD": "Donkey Cabbages",
    "THE STORY OF THE YOUTH WHO WENT FORTH TO LEARN WHAT FEAR WAS": "The Story of the Youth Who Went Forth to Learn What Fear Was",
    "KING GRISLY-BEARD": "King Thrushbeard",
    "IRON HANS": "Iron John",
    "THE SIX SWANS": "The Six Swans",
    "THE THREE LITTLE MEN IN THE WOOD": "The Three Little Men in the Wood",
    "THE GOLDEN KEY": "The Golden Key (fairy tale)",
    "THE NIX OF THE MILL-POND": "The Nixie of the Mill-Pond",
    "THE SEVEN RAVENS": "The Seven Ravens",
    "SNOW-WHITE AND ROSE-RED": "Snow-White and Rose-Red",
}
SECTIONS = ("Plot", "Synopsis", "Story", "Summary", "Plot summary", "Tale")


def split_tales(text):
    """Every title in the book's CONTENTS list starts a tale (the contents
    list is the first time each title appears; the heading is the second)."""
    lines = text.split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "CONTENTS:")
    titles, i = [], start + 1
    while i < len(lines) and (not lines[i].strip() or lines[i].startswith(" ")):
        t = lines[i].strip()
        if t and t.upper() == t and not t[0].isdigit():
            titles.append(t)
        i += 1
    titles = set(titles)
    tales, cur, body = {}, None, []
    for ln in lines[i:]:
        t = ln.strip()
        if t in titles and not ln.startswith(" "):
            if cur:
                tales[cur] = "\n".join(body).strip()
            cur, body = t, []
            continue
        if cur:
            body.append(ln)
    if cur:
        tales[cur] = "\n".join(body).strip()
    return tales


def plot_of(article):
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                                "titles": article, "format": "json"})
    req = urllib.request.Request("https://en.wikipedia.org/w/api.php?" + q,
                                 headers={"User-Agent": "mimir-gist-set/1.0 (research; callmedraxx/mimir)"})
    for wait in (0, 10, 30, 60, 120):           # be polite: back off when told to
        time.sleep(wait)
        try:
            data = json.load(urllib.request.urlopen(req, timeout=30))
            break
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
    else:
        raise RuntimeError("rate limited")
    page = next(iter(data["query"]["pages"].values()))
    text = page.get("extract", "")
    for sec in SECTIONS:
        m = re.search(r"\n==+ *" + re.escape(sec) + r" *==+\n(.*?)(?=\n==[^=])", text + "\n== End ==\n", re.S)
        if m:
            body = re.sub(r"\n===+.*?===+\n", "\n", m.group(1)).strip()
            if len(body.split()) >= 60:
                return page.get("title", article), body
    return page.get("title", article), None


def main():
    text = open(BOOK, encoding="utf-8").read()
    tales = split_tales(text)
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for title, article in ARTICLES.items():
        story = tales.get(title)
        if not story or len(story.split()) < 300:
            print(f"  skip {title}: tale not found in the book")
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        if os.path.exists(os.path.join(OUT, slug + ".summary.txt")):  # fetched before
            open(os.path.join(OUT, slug + ".story.txt"), "w").write(story + "\n")
            plot = open(os.path.join(OUT, slug + ".summary.txt")).read().strip()
            rows.append((slug, article, "https://en.wikipedia.org/wiki/" + urllib.parse.quote(article.replace(" ", "_")),
                         len(story.split()), len(plot.split())))
            continue
        try:
            got, plot = plot_of(article)
        except Exception as e:
            print(f"  skip {title}: {e}")
            continue
        if not plot:
            print(f"  skip {title}: no plot section in '{got}'")
            continue
        open(os.path.join(OUT, slug + ".story.txt"), "w").write(story + "\n")
        open(os.path.join(OUT, slug + ".summary.txt"), "w").write(plot + "\n")
        url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(got.replace(" ", "_"))
        rows.append((slug, got, url, len(story.split()), len(plot.split())))
        print(f"  {slug:40s} story {len(story.split()):5d} words, summary {len(plot.split()):4d} words  ← {got}")
        time.sleep(3)
    with open(os.path.join(OUT, "index.tsv"), "w") as f:
        f.write("tale\tarticle\turl\tstory_words\tsummary_words\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"{len(rows)} tales with a human summary")


if __name__ == "__main__":
    main()
