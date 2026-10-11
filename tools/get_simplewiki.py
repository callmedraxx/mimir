"""
get_simplewiki.py — Simple English Wikipedia as plain reading text.

Streams the dump (simplewiki-latest-pages-articles.xml.bz2, from
dumps.wikimedia.org) and writes the articles' running text, one paragraph
per line, without templates, tables, references, file links or headings.
Text is CC BY-SA 4.0 (Wikipedia contributors). The output is large (about
200 MB), so it is not kept in git: run this to make it.

    python3 tools/get_simplewiki.py DUMP.xml.bz2 OUT.txt
    python3 tools/get_simplewiki.py thin OUT.txt OUT30M.txt   (drop bot templates, keep 30M words)
"""
import bz2, html, re, sys

SKIP_TITLES = re.compile(r"^(Wikipedia|Template|Category|File|Help|Portal|MediaWiki|Module|Draft|List of)\b")


def strip_nested(s, open_, close):
    out, depth, i = [], 0, 0
    while i < len(s):
        if s.startswith(open_, i):
            depth += 1
            i += len(open_)
        elif depth and s.startswith(close, i):
            depth -= 1
            i += len(close)
        else:
            if not depth:
                out.append(s[i])
            i += 1
    return "".join(out)


def clean(wt):
    wt = html.unescape(wt)
    wt = re.sub(r"<!--.*?-->", "", wt, flags=re.S)
    wt = re.sub(r"<ref[^>/]*/>", "", wt)
    wt = re.sub(r"<ref.*?</ref>", "", wt, flags=re.S)
    wt = strip_nested(wt, "{{", "}}")
    wt = strip_nested(wt, "{|", "|}")
    wt = re.sub(r"\[\[(?:File|Image|Category|[a-z]{2,3}):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", "", wt)
    wt = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"\2", wt)
    wt = re.sub(r"\[\[([^\]]*)\]\]", r"\1", wt)
    wt = re.sub(r"\[https?://[^ \]]+ ([^\]]*)\]", r"\1", wt)
    wt = re.sub(r"\[https?://[^\]]*\]", "", wt)
    wt = re.sub(r"<[^>]+>", "", wt)
    wt = re.sub(r"'{2,}", "", wt)
    paras = []
    for line in wt.split("\n"):
        line = line.strip()
        if not line or line.startswith(("=", "*", "#", ":", ";", "|", "!", "__")):
            continue
        if len(line.split()) < 6:
            continue
        paras.append(re.sub(r"\s+", " ", line))
    return paras


def main(dump, out):
    title, in_text, buf, pages, words = None, False, [], 0, 0
    with bz2.open(dump, "rt", encoding="utf-8") as f, open(out, "w") as o:
        for line in f:
            if "<title>" in line:
                title = re.sub(r".*<title>(.*)</title>.*", r"\1", line.strip())
            elif "<text" in line:
                in_text = True
                buf = [re.sub(r".*<text[^>]*>", "", line)]
                if "</text>" in line:
                    in_text = False
                    buf[0] = buf[0].split("</text>")[0]
                else:
                    continue
            elif in_text:
                if "</text>" in line:
                    buf.append(line.split("</text>")[0])
                    in_text = False
                else:
                    buf.append(line)
                    continue
            else:
                continue
            text = "".join(buf)
            if not title or SKIP_TITLES.match(title) or text.lstrip().upper().startswith("#REDIRECT"):
                continue
            paras = clean(text)
            if not paras:
                continue
            for p in paras:
                o.write(p + "\n")
                words += len(p.split())
            pages += 1
            if pages % 20000 == 0:
                print(f"  {pages} articles, {words / 1e6:.1f}M words", flush=True)
    print(f"{pages} articles, {words / 1e6:.1f}M words → {out}")



def thin(src, out, max_words=30_000_000, max_copies=3):
    """Drop bot-made template lines ("X is a commune in the Y department in
    France" thousands of times): a line whose shape (names and numbers taken
    out) has been seen max_copies times is skipped. Stops at max_words."""
    seen, words, kept, dropped = {}, 0, 0, 0
    with open(src) as f, open(out, "w") as o:
        for line in f:
            shape = hash(re.sub(r"\b[A-Z][\w'\-]*|\d[\d,.]*", "X", line).lower())  # a hash keeps memory small
            n = seen.get(shape, 0)
            if n >= max_copies or sum(c.isdigit() for c in line) > len(line) / 8:
                dropped += 1
                continue
            seen[shape] = n + 1
            o.write(line)
            kept += 1
            words += len(line.split())
            if words >= max_words:
                break
    print(f"kept {kept} lines, {words / 1e6:.1f}M words; dropped {dropped} template lines → {out}")


if __name__ == "__main__":
    if sys.argv[1] == "thin":
        thin(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2])
