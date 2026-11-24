import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd
plt.rcParams['font.family']='DejaVu Sans'
G,PE,O,O2,INK='#9AA1AD','#F2B48C','#D9692B','#E8945A','#1F2430'
names=['Mô hình nền\n(ControlNet)','GLIGEN','Ghép ảnh\n(Collage)','OCSD\ngốc','OCSD\ntự nhiên','OCSD\nbám nét']
cols=[G,G,G,PE,O,O]
data={'opr':([65.5,80.1,82.4,71.1,81.9,82.5],'Giữ đủ đối tượng (OPR)','{:.0f}%'),
      'miou':([49.2,65.3,60.7,52.9,51.9,62.3],'Đúng vị trí, hình dạng (mIoU)','{:.0f}%'),
      'nat':([0.75,1.44,0.28,0.22,1.08,0.75],'Độ tự nhiên của ảnh (0–2)','{:.2f}')}
for k,(v,t,f) in data.items():
    fig,ax=plt.subplots(figsize=(4.05,3.0),dpi=250)
    b=ax.bar(range(6),v,color=cols,width=0.66)
    for i,x in enumerate(v): ax.text(i,x+max(v)*0.02,f.format(x).replace('.',','),ha='center',va='bottom',fontsize=8,fontweight='bold',color=INK)
    ax.set_xticks(range(6)); ax.set_xticklabels(names,fontsize=6.3,color=INK)
    ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ['top','right','left']]
    ax.set_ylim(0,max(v)*1.18); ax.set_title(t,fontsize=9,color=INK,pad=6)
    fig.tight_layout(); fig.savefig(f'c_{k}.png'); plt.close()
d=pd.read_csv('/mnt/project-files/results/xl_paper_v1/tables/by_count_bin.csv')
fig,ax=plt.subplots(figsize=(7.6,4.93),dpi=220)
st={'controlnet':('Mô hình nền (ControlNet)',G,'--',2),'gligen':('GLIGEN','#6F8CC0',':',2),'collage':('Ghép ảnh (Collage)','#3E9474','-.',2),
    'ocsd':('OCSD gốc','#E9A27A','-',2.5),'ocsd_xl_natural':('OCSD tự nhiên',O2,'--',3),'ocsd_xl_faithful':('OCSD bám nét',O,'-',3.5)}
xs=['1','3','5','8+']
for m,(lab,c,ls,lw) in st.items():
    y=[d[(d.method==m)&(d.count_bin==b)].opr.iloc[0]*100 for b in xs]
    ax.plot(range(4),y,ls,color=c,lw=lw,marker='o',ms=6,label=lab)
ax.set_xticks(range(4)); ax.set_xticklabels(xs,fontsize=12); ax.tick_params(axis='y',labelsize=12)
ax.set_xlabel('Số đối tượng trong phác thảo',fontsize=13); ax.set_ylabel('Giữ đủ đối tượng (%)',fontsize=13)
ax.set_ylim(30,103); ax.grid(axis='y',color='#E5E7EB'); [ax.spines[s].set_visible(False) for s in ['top','right']]
ax.legend(fontsize=10.5,frameon=False,loc='lower left')
fig.tight_layout(); fig.savefig('c_count.png'); plt.close()
