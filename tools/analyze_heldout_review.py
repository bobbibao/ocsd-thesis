import re,json,sys,random
from math import comb
from collections import defaultdict,Counter
S=sys.argv[1]
key=json.load(open(f'{S}/res/review_key.json'))
txt=open(f'{S}/rev/REVIEW_heldout/review/REVIEW_TEMPLATE.txt',encoding='utf-8').read()
rows=[];picks={}
for blk in re.split(r'^=== ',txt,flags=re.M)[1:]:
    head,body=blk.split(' ===',1)
    m=re.match(r'\d+_(ho_qd_\S+)_seed(\d)',head.strip()); sid,seed=m.group(1),int(m.group(2))
    k=key[f'{sid}_s{seed}']
    for L in 'ABCD':
        mm=re.search(rf'^{L}: count=(\S+)\s+colour=(\S+)\s+shape=(\S+)\s+position/relation=(\S+)\s+natural\(0/1/2\)=(\S+)',body,re.M)
        assert mm,(head,L)
        c,co,sh,pr,nat=mm.groups()
        rows.append(dict(sid=sid,seed=seed,arm=k[L],count=c,colour=co,shape=sh,pos=pr,nat=nat,
            ok=int(c=='pass' and co=='pass' and sh=='pass'), unclear=any(x=='unclear' for x in (c,co,sh))))
    p=re.search(r'right structure \(A/B/C/D/tie/none\):\s*(\S*)',body).group(1)
    picks[(sid,seed)]=k.get(p,p)
print('rows',len(rows),'unclear',sum(r['unclear'] for r in rows), 'blank', Counter(r[f] for r in rows for f in('count','colour','shape','pos','nat')))
arms=['direct_box','direct_scribble','controlnet','instance_repair_count_matte_soft']
sids=sorted({r['sid'] for r in rows})
rate={a:{s:sum(r['ok'] for r in rows if r['arm']==a and r['sid']==s)/2 for s in sids} for a in arms}
rng=random.Random(20261104)
def boot(f,B=10000):
    v=sorted(f([rng.choice(sids) for _ in sids]) for _ in range(B)); return v[int(.025*B)],v[int(.975*B)-1]
out={}
for a in arms:
    m=sum(rate[a].values())/18; lo,hi=boot(lambda ss:sum(rate[a][s] for s in ss)/len(ss))
    fails={f:sum(r[f]=='fail' for r in rows if r['arm']==a) for f in('count','colour','shape','pos')}
    nat=[int(r['nat']) for r in rows if r['arm']==a and r['nat'] in '012']
    print(f"{a:34s} success {sum(r['ok'] for r in rows if r['arm']==a)}/36 rate {m:.3f} [{lo:.3f},{hi:.3f}] fails {fails} nat_mean {sum(nat)/len(nat):.2f} nat2 {nat.count(2)} picks {sum(v==a for v in picks.values())}")
    out[a]=dict(success=sum(r['ok'] for r in rows if r['arm']==a),rate=m,ci=[lo,hi],fails=fails,nat_mean=sum(nat)/len(nat),nat_dist=[nat.count(i) for i in range(3)],picks=sum(v==a for v in picks.values()))
print('picks',Counter(picks.values()))
def sign(a,b):
    w=sum(rate[a][s]>rate[b][s] for s in sids); l=sum(rate[a][s]<rate[b][s] for s in sids); n=w+l
    p=min(1,2*sum(comb(n,i) for i in range(0,min(w,l)+1))/2**n) if n else 1
    d=sum(rate[a][s]-rate[b][s] for s in sids)/18; lo,hi=boot(lambda ss:sum(rate[a][s]-rate[b][s] for s in ss)/len(ss))
    return dict(a=a,b=b,wins=w,losses=l,ties=18-n,p=p,diff=d,ci=[lo,hi])
prim=[sign('direct_box','controlnet'),sign('direct_box','instance_repair_count_matte_soft')]
ps=sorted(range(2),key=lambda i:prim[i]['p'])
prim[ps[0]]['p_holm']=min(1,2*prim[ps[0]]['p']); prim[ps[1]]['p_holm']=min(1,max(prim[ps[0]]['p_holm'],prim[ps[1]]['p']))
expl=[sign('direct_scribble','direct_box'),sign('direct_scribble','controlnet'),sign('direct_scribble','instance_repair_count_matte_soft'),sign('instance_repair_count_matte_soft','controlnet')]
for t in prim+expl: print(t)
nb=json.load(open(f'{S}/res/assets_nobj.json')) if False else None
bins=defaultdict(list)
for s in sids: bins[s.split('_')[2]].append(s)
for b,ss in sorted(bins.items()):
    print(b, {a:f"{sum(rate[a][s] for s in ss)*2:.0f}/{2*len(ss)}" for a in arms})
json.dump(dict(per_arm=out,primary=prim,exploratory=expl,picks={f'{k[0]}_s{k[1]}':v for k,v in picks.items()},
  by_bin={b:{a:sum(rate[a][s] for s in ss)*2 for a in arms} for b,ss in bins.items()},rows=rows),open(f'{S}/human_analysis.json','w'),indent=1)
