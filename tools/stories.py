# The stories in data/story/ in the passage layout bg passages reads (S id / T the story on one line / Q questions).
#   python3 tools/stories.py OUTDIR   writes OUTDIR/<set>.txt for each question file
import os, sys
SETS = {"clock": ("clock_questions.tsv", ["clock"]),
        "dev_stories": ("passage_tests.tsv", ["bicycle", "stoneowl", "swim"]),          # PASSAGE's test; dev since COMPREHEND
        "comprehend_tests": ("comprehend_tests.tsv", ["lighthouse", "dragon", "nightbus"]),
        "episode_tests": ("episode_tests.tsv", ["salt", "train", "kite"]),
        "story_tests": ("story_tests.tsv", ["harbor", "garden", "snow"])}
def write(out, name):
    qfile, names = SETS[name]; D = "data/story/"
    qs = [l.rstrip("\n").split("\t") for l in open(D + qfile) if l.strip()]
    with open(os.path.join(out, name + ".txt"), "w") as f:
        for s in names:
            f.write("S\tstory.%s\nT\t%s\n" % (s, " ".join(open(D + s + ".txt").read().split())))
            for x in qs:
                if x[0].rsplit("_", 1)[0] == s: f.write("Q\t%s\t%s\t%s\t%s\n" % (x[0], x[1], "\t".join(x[3:7]), "ABCD"[int(x[7])]))
if __name__ == "__main__":
    os.makedirs(sys.argv[1], exist_ok=True)
    for n in SETS: write(sys.argv[1], n)
