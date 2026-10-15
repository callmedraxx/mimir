# The school brain with its cortex, kept as the GitHub release brain-school-cx3 (the shelf read by the reader of
# docs/COMPREHEND.md and docs/READ2.md) so a Colab session need not rebuild it.
# Built and published by .github/workflows/brain.yml (GitHub's own token for that job; no key anywhere else).
#   get(dst)   download it and the shelf it read (data/school/, which open book re-reads: without it that part is 0)
#              if the release exists (checked by SHA-256); True if both are there
import os, subprocess
REPO, TAG = "callmedraxx/mimir", "brain-school-cx3"
URL = "https://github.com/%s/releases/download/%s/" % (REPO, TAG)
FILES = ("school_cx.brain.gz", "school_cx.brain.cx.gz", "school_shelf.tar.gz")
def sh(cmd): print("$", cmd, flush=True); subprocess.run(cmd, shell=True, check=True)
def get(dst, tag=TAG):
    """the release `tag` (brain-school-cx3 unless another is named) to dst, dst.cx and data/school/"""
    URL = "https://github.com/%s/releases/download/%s/" % (REPO, tag)
    if os.path.exists(dst) and os.path.exists(dst + ".cx") and os.path.exists("data/school/07_vital.txt"): return True
    d = os.path.dirname(dst) or "."
    if subprocess.run("curl -sfL -o %s/SHA256SUMS %sSHA256SUMS" % (d, URL), shell=True).returncode: print("no release", tag, flush=True); return False
    for f in FILES: sh("curl -sfL -o %s/%s %s%s" % (d, f, URL, f))
    sh("cd %s && sha256sum -c SHA256SUMS" % d)
    sh("gunzip -c %s/%s > %s && gunzip -c %s/%s > %s.cx && tar -xzf %s/%s && rm %s/%s %s/%s %s/%s" % (d, FILES[0], dst, d, FILES[1], dst, d, FILES[2], d, FILES[0], d, FILES[1], d, FILES[2]))
    return True
