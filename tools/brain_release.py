# The school brain with its cortex, kept as the GitHub release brain-school-cx2 (the shelf read by the reader of
# docs/COMPREHEND.md) so a Colab session need not rebuild it.
# Built and published by .github/workflows/brain.yml (GitHub's own token for that job; no key anywhere else).
#   get(dst)   download it if the release exists (checked by SHA-256); True if it is there
import os, subprocess
REPO, TAG = "callmedraxx/mimir", "brain-school-cx2"
URL = "https://github.com/%s/releases/download/%s/" % (REPO, TAG)
FILES = ("school_cx.brain.gz", "school_cx.brain.cx.gz")
def sh(cmd): print("$", cmd, flush=True); subprocess.run(cmd, shell=True, check=True)
def get(dst):
    if os.path.exists(dst) and os.path.exists(dst + ".cx"): return True
    d = os.path.dirname(dst) or "."
    if subprocess.run("curl -sfL -o %s/SHA256SUMS %sSHA256SUMS" % (d, URL), shell=True).returncode: print("no release", TAG, flush=True); return False
    for f in FILES: sh("curl -sfL -o %s/%s %s%s" % (d, f, URL, f))
    sh("cd %s && sha256sum -c SHA256SUMS" % d)
    sh("gunzip -c %s/%s > %s && gunzip -c %s/%s > %s.cx && rm %s/%s %s/%s" % (d, FILES[0], dst, d, FILES[1], dst, d, FILES[0], d, FILES[1]))
    return True
