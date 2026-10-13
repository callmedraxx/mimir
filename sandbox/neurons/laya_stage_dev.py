import re, sys
W=re.compile(r"\b(not|least|except|false|unlikely|cannot|never|incorrect|untrue|isn't|doesn't|can't|won't|wouldn't|aren't)\b", re.I)
def acc(want,s):
    b=max(s); t=[i for i,v in enumerate(s) if v>b-1e-9]; return (1/len(t)) if want in t else 0
Q={f:[l.split("\t")[1] for l in open("/root/repos/mimir/data/shelf/arc/%s.tsv"%f[:-4])] for f in ("easy_dev.tsv","challenge_dev.tsv","obqa_dev.tsv")}
B=(1,0.5,2,4,0)
for path in sys.argv[1:]:
    rows=[l.rstrip("\n").split("\t") for l in open(path)]; by={}
    for r in rows: by.setdefault(r[0],[]).append(r)
    out=[]; tot=[0,0]; n=0
    for f in ("easy_dev.tsv","challenge_dev.tsv","obqa_dev.tsv"):
        a=[0,0]
        for i,r in enumerate(by.get(f,[])):
            opts=[tuple(map(float,x.split())) for x in r[3:3+int(r[2])]]; s=[sum(x*y for x,y in zip(B,o)) for o in opts]
            a[0]+=acc(int(r[1]),s); neg=bool(W.search(Q[f][i])); a[1]+=acc(int(r[1]),[-v for v in s] if neg else s)
        m=len(by.get(f,[])); n+=m; tot[0]+=a[0]; tot[1]+=a[1]
        out.append("%s %.2f/%.2f"%(f.split("_")[0],100*a[0]/max(m,1),100*a[1]/max(m,1)))
    print("%-22s all %.2f / with polarity %.2f  |  %s"%(path.split("/")[-1],100*tot[0]/max(n,1),100*tot[1]/max(n,1),"  ".join(out)))
