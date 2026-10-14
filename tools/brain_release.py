# The school brain with its cortex, kept as a GitHub release (brain-school-cx) so a Colab session need not rebuild it.
#   get(dst)   download it if the release exists (checked by SHA-256); True if it is there
#   put(src)   upload it (needs the owner's token in Colab Secrets as GH_TOKEN: a fine-grained token for this repo,
#              Contents read and write); creates the release if missing
import hashlib, json, os, subprocess, urllib.request
REPO, TAG = "callmedraxx/mimir", "brain-school-cx"
URL = "https://github.com/%s/releases/download/%s/" % (REPO, TAG)
FILES = ("school_cx.brain.gz", "school_cx.brain.cx.gz")
def sh(cmd): print("$", cmd, flush=True); subprocess.run(cmd, shell=True, check=True)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
def get(dst):
    if os.path.exists(dst) and os.path.exists(dst + ".cx"): return True
    d = os.path.dirname(dst) or "."
    if subprocess.run("curl -sfL -o %s/SHA256SUMS.cx %sSHA256SUMS" % (d, URL), shell=True).returncode: print("no release", TAG, flush=True); return False
    for f in FILES: sh("curl -sfL -o %s/%s %s%s" % (d, f, URL, f))
    sh("cd %s && sha256sum -c SHA256SUMS.cx" % d)
    sh("gunzip -c %s/%s > %s && gunzip -c %s/%s > %s.cx && rm %s/%s %s/%s" % (d, FILES[0], dst, d, FILES[1], dst, d, FILES[0], d, FILES[1]))
    return True
def token():
    try:
        from google.colab import userdata
        return userdata.get("GH_TOKEN")
    except Exception as e:
        print("no GH_TOKEN in Colab Secrets (%s): the brain is not saved to GitHub" % type(e).__name__, flush=True); return None
def api(method, url, tok, data=None, ctype="application/json"):
    req = urllib.request.Request(url, data=data, method=method, headers={"Authorization": "Bearer " + tok, "Accept": "application/vnd.github+json", "Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=3600) as r: return json.loads(r.read() or b"{}")
def put(src):
    tok = token()
    if not tok: return False
    d = os.path.dirname(src) or "."
    sh("gzip -1 -c %s > %s/%s && gzip -1 -c %s.cx > %s/%s" % (src, d, FILES[0], src, d, FILES[1]))
    open("%s/SHA256SUMS.cx" % d, "w").write("".join("%s  %s\n" % (sha("%s/%s" % (d, f)), f) for f in FILES))
    try: rel = api("GET", "https://api.github.com/repos/%s/releases/tags/%s" % (REPO, TAG), tok)
    except urllib.error.HTTPError:
        rel = api("POST", "https://api.github.com/repos/%s/releases" % REPO, tok, json.dumps({"tag_name": TAG, "target_commitish": "reader", "name": "School brain with its cortex",
              "body": "The school brain read with the cortex learning from its facts (docs/CORTEX_LINK.md): BRAIN and BRAIN.cx, gzipped; SHA256SUMS."}).encode())
    for a in rel.get("assets", []): api("DELETE", "https://api.github.com/repos/%s/releases/assets/%d" % (REPO, a["id"]), tok)
    up = rel["upload_url"].split("{")[0]
    for f in FILES + ("SHA256SUMS.cx",):
        name = "SHA256SUMS" if f == "SHA256SUMS.cx" else f
        print("uploading", name, os.path.getsize("%s/%s" % (d, f)) // 1000000, "MB", flush=True)
        with open("%s/%s" % (d, f), "rb") as fh: api("POST", "%s?name=%s" % (up, name), tok, fh.read(), "application/octet-stream")
    sh("rm %s/%s %s/%s" % (d, FILES[0], d, FILES[1])); print("saved to", URL, flush=True); return True
