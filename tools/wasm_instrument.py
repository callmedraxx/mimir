"""
wasm_instrument.py — make a traced copy of src/ for the browser build.

For every function definition in src/*.c, inserts MIMIR_TRACE("name"); as
the first statement.  In the browser build that macro reports entry and,
through C's cleanup attribute, exit on every return path.  The source files
in src/ are never modified; the copy goes to the output directory.

Usage: python3 tools/wasm_instrument.py src build/wasm-src
"""
import os, re, sys

SKIP = {"if", "for", "while", "switch", "return", "sizeof"}
NO_TRACE = {"__cyg_profile_func_enter", "__cyg_profile_func_exit",
            "mimir_probe", "mimir_trace_enter", "mimir_trace_leave"}
FUNC = re.compile(
    r"^(?:static\s+)?(?:inline\s+)?[A-Za-z_][\w\s\*]*?\b([A-Za-z_]\w*)\s*\(([^;{}]*)\)\s*\{",
    re.M)


def blank_comments(s):
    """Same length as s, with comments and string/char literals blanked
    (newlines kept), so offsets found here are valid in the original."""
    out = list(s)
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if s.startswith("/*", i):
            j = s.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif s.startswith("//", i):
            j = s.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and s[j] != c:
                j += 2 if s[j] == "\\" else 1
            for k in range(i + 1, min(j, n)):
                if out[k] != "\n":
                    out[k] = " "
            i = j + 1
        else:
            i += 1
    return "".join(out)


def instrument(src):
    code = blank_comments(src).replace("__attribute__((unused))", " " * 23)
    inserts = []
    for m in FUNC.finditer(code):
        name = m.group(1)
        if name in SKIP or name in NO_TRACE:
            continue
        # skip things that are not at file scope (e.g. a cast followed by a block)
        if code[:m.start()].count("{") != code[:m.start()].count("}"):
            continue
        inserts.append((m.end(), name))
    out, last = [], 0
    for pos, name in inserts:
        out.append(src[last:pos])
        out.append(f' MIMIR_TRACE("{name}");')
        last = pos
    out.append(src[last:])
    return "".join(out), [n for _, n in inserts]


def main():
    src_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for f in sorted(os.listdir(src_dir)):
        p = os.path.join(src_dir, f)
        text = open(p, encoding="utf-8").read()
        if f.endswith(".c") and f != "wasm_trace.c":
            text, names = instrument(text)
            total += len(names)
        with open(os.path.join(out_dir, f), "w", encoding="utf-8") as o:
            o.write(text)
    print(f"wasm_instrument: traced {total} functions -> {out_dir}")


if __name__ == "__main__":
    main()
