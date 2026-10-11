"""
wiring.py — read src/*.c and write Mimir's wiring: every function, what it
calls, and whether main() can reach it.

Workbench Lab's Mimir bench draws this as the board: one chip per function,
one copper trace per call.  Run it after adding or removing a function and
the board follows.

Usage: python3 tools/wiring.py [out.json]      (default build/wiring.json)

Output:
  {"functions": [{"name", "file", "line", "lines", "doc", "calls": [...],
                  "reachable": bool, "benchOnly": bool, "unusedAttr": bool}],
   "declaredOnly": [names declared in mimir.h with no body]}
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, HERE)
from wasm_instrument import FUNC, SKIP, NO_TRACE, blank_comments  # same parser as the trace build

BENCH_ROOT = "run_xor_benchmark"   # calls under it only feed the start-up table


def doc_before(src, start):
    """First sentence of the comment block right above a definition."""
    pre = src[:start].rstrip()
    text = ""
    if pre.endswith("*/"):
        text = pre[pre.rfind("/*") + 2:-2]
    else:
        lines = []
        for l in reversed(pre.split("\n")):
            if l.strip().startswith("//"):
                lines.append(l.strip()[2:])
            else:
                break
        text = " ".join(reversed(lines))
    text = re.sub(r"^\s*\*+", " ", text, flags=re.M)
    text = re.sub(r"[─═]+", " ", text)
    text = " ".join(text.split())
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    return (m.group(1) if m else text)[:240]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "wiring.json")
    funcs, bodies = {}, {}
    for f in sorted(os.listdir(SRC)):
        if not f.endswith(".c") or f == "wasm_trace.c":
            continue
        src = open(os.path.join(SRC, f), encoding="utf-8").read()
        code = blank_comments(src)
        unused_spans = [m.start() for m in re.finditer(r"__attribute__\(\(unused\)\)", code)]
        code = code.replace("__attribute__((unused))", " " * 23)
        for m in FUNC.finditer(code):
            name = m.group(1)
            if name in SKIP or name in NO_TRACE:
                continue
            if code[:m.start()].count("{") != code[:m.start()].count("}"):
                continue
            i, depth = m.end(), 1
            while depth:
                depth += {"{": 1, "}": -1}.get(code[i], 0)
                i += 1
            bodies[name] = code[m.end():i]
            funcs[name] = {
                "name": name, "file": f,
                "line": src[:m.start()].count("\n") + 1,
                "lines": src[m.start():i].count("\n") + 1,
                "doc": doc_before(src, m.start()),
                "unusedAttr": any(m.start() <= u < m.end() for u in unused_spans),
            }
    names = list(funcs)
    word = {n: re.compile(r"\b" + re.escape(n) + r"\b") for n in names}
    for n in names:
        funcs[n]["calls"] = sorted(c for c in names if c != n and word[c].search(bodies[n]))

    def reach(skip=()):
        seen, stack = set(), ["main"]
        while stack:
            n = stack.pop()
            if n in seen or n not in funcs:
                continue
            seen.add(n)
            stack += [c for c in funcs[n]["calls"] if c not in skip]
        return seen

    live, live_nb = reach(), reach({BENCH_ROOT})
    for n in names:
        funcs[n]["reachable"] = n in live
        funcs[n]["benchOnly"] = n in live and n not in live_nb

    header = blank_comments(open(os.path.join(SRC, "mimir.h"), encoding="utf-8").read())
    declared = set(re.findall(r"^[A-Za-z_][\w\s\*]*?\b([a-z_]\w*)\s*\([^;{}]*\)\s*;", header, re.M))
    CTYPES = {"int", "void", "float", "double", "char", "bool", "long", "short", "unsigned", "signed"}
    declared_only = sorted(d for d in declared if d not in funcs and d not in NO_TRACE and d not in CTYPES)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as o:
        json.dump({"functions": [funcs[n] for n in names], "declaredOnly": declared_only}, o, indent=1)
    print(f"wiring: {len(names)} functions, {sum(f['reachable'] for f in funcs.values())} reachable, "
          f"{sum(len(f['calls']) for f in funcs.values())} calls, declared only: {declared_only} -> {out_path}")


if __name__ == "__main__":
    main()
