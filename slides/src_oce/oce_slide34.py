"""v52 (Bobby 2026-11-26 'hãy đưa OCE vào'): slide 34 gets a 5th chart + card for R2 = OCE theo lớp (oce_c, the
thesis' main count metric, ch4 eq ocec), mean of 72 scenes x 2 seeds from results/m4_study_v1/tables/per_image.csv;
lower is better. Count accuracy tests (summary.md): OCSD vs GLIGEN/Collage/ControlNet not significant after Holm.
All five charts redrawn narrower (same numbers and colours as v50, mIoU keeps the old-M4 bar).
Order R1 OPR, R2 OCE, R3 mIoU, R4 RA, PickScore. Usage: oce_slide34.py IN(v51) OUT"""
import sys, io, copy
from decimal import Decimal, ROUND_HALF_UP
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
from pptx import Presentation
from pptx.util import Inches as I, Pt
G, O, LO, INK = '#9AA1AD', '#D9692B', '#F2B48C', '#1F2430'
def fmt(f, x):
    q = Decimal(str(x)).quantize(Decimal('1') if f == '%' else Decimal('0.01'), ROUND_HALF_UP)
    return (f'{float(q):.0f}%' if f == '%' else f'{float(q):.2f}').replace('.', ',')
N6 = ['Mô hình nền\n(ControlNet)', 'T2I-\nAdapter', 'Zhang và\ncs. (2025)', 'GLIGEN', 'Collage', 'OCSD']
charts = [
    ('opr', 'Giữ đủ vật thể (OPR)', N6, [65.5, 62.4, 63.3, 80.1, 82.4, 78.5], [G] * 5 + [O], '%', 0),
    ('oce', 'Đúng số lượng (OCE, thấp = tốt)', N6, [2.514, 2.674, 2.410, 2.062, 1.500, 2.174], [G] * 5 + [O], '2', 0),
    ('miou', 'Đúng vị trí, hình dạng (mIoU)', N6[:5] + ['OCSD\n(M4 cũ)', 'OCSD'],
     [0.492, 0.437, 0.435, 0.653, 0.607, 0.529, 0.484], [G] * 5 + [LO, O], '2', 0),
    ('ra', 'Đúng tương quan (RA)', N6, [32.1, 28.2, 33.7, 56.3, 61.6, 50.8], [G] * 5 + [O], '%', 0),
    ('pick', 'Độ tự nhiên (PickScore)', N6, [21.67, 21.53, 20.73, 21.41, 21.52, 21.89], [G] * 5 + [O], '2', 20.3)]
CW, CH, X0, GAP = 2.38, 2.6, 0.45, 0.06
def chart(t, names, v, cols, f, lo):
    fig, ax = plt.subplots(figsize=(CW, CH), dpi=300)
    n = len(v); ax.bar(range(n), [x - lo for x in v], bottom=lo, color=cols, width=0.68)
    top = max(v)
    for i, x in enumerate(v):
        ax.text(i, x + (top - lo) * 0.02, fmt(f, x), ha='center', va='bottom', fontsize=5.6 if n > 6 else 6.2,
                fontweight='bold', color=O if cols[i] in (O, LO) else INK)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, fontsize=4.3 if n > 6 else 4.7, color=INK)
    ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ['top', 'right', 'left']]
    ax.set_ylim(lo, lo + (top - lo) * 1.2); ax.set_title(t, fontsize=7.6, color=INK, pad=4)
    fig.tight_layout(pad=0.3); b = io.BytesIO(); fig.savefig(b, format='png'); plt.close(); b.seek(0); return b
p = Presentation(sys.argv[1]); s = p.slides[33]
sh = {x.name: x for x in s.shapes}
assert sh['TextBox 12'].text_frame.text.startswith('Vị trí, hình dạng')
# new picture + card for OCE, cloned from the RA ones
new = []
for n in ('Picture 20', 'Rounded Rectangle 13', 'TextBox 14'):
    el = copy.deepcopy(sh[n]._element); s.shapes._spTree.append(el); new.append(s.shapes[-1])
mid = max(x.shape_id for x in s.shapes)
for k, (x, nm) in enumerate(zip(new, ('Picture OCE', 'Rounded Rectangle OCE', 'TextBox OCE'))):
    x._element.xpath('./*[1]/p:cNvPr')[0].set('id', str(mid + 1 + k)); x.name = nm
sh = {x.name: x for x in s.shapes}
tf = sh['TextBox OCE'].text_frame
tf.paragraphs[0].runs[0].text = 'Đúng số lượng'
body = tf.paragraphs[1]
body.runs[0].text = ('2,17: ít lỗi đếm hơn mô hình nền (2,51), T2I-Adapter (2,67), Zhang và cs. (2,41); '
                     'nhiều hơn GLIGEN (2,06) và Collage (1,50). Tỉ lệ đếm đúng: khác biệt chưa có ý nghĩa.')
for r in body.runs[1:]: r._r.getparent().remove(r._r)
pics = ['Picture 18', 'Picture OCE', 'Picture 19', 'Picture 20', 'Picture 21']
cards = [('Rounded Rectangle 9', 'TextBox 10'), ('Rounded Rectangle OCE', 'TextBox OCE'),
         ('Rounded Rectangle 11', 'TextBox 12'), ('Rounded Rectangle 13', 'TextBox 14'), ('Rounded Rectangle 15', 'TextBox 16')]
for i, (pn, (cn, tn), c) in enumerate(zip(pics, cards, charts)):
    x = X0 + i * (CW + GAP)
    pic = sh[pn]; _, rid = pic.part.get_or_add_image_part(chart(*c[1:])); pic._element.blipFill.blip.rEmbed = rid
    pic.left, pic.top, pic.width, pic.height = I(x), I(1.45), I(CW), I(CH)
    card, txt = sh[cn], sh[tn]
    card.left, card.top, card.width, card.height = I(x + 0.04), I(4.2), I(CW - 0.08), I(2.2)
    txt.left, txt.top, txt.width, txt.height = I(x + 0.12), I(4.28), I(CW - 0.24), I(2.05)
    txt.text_frame.word_wrap = True
    for k, para in enumerate(txt.text_frame.paragraphs):
        for r in para.runs: r.font.size = Pt(12.5 if k == 0 else 9.5)
nt = s.notes_slide.notes_text_frame; t = nt.text; a = 'Đúng tương quan: 51 phần trăm'
assert a in t
nt.text = t.replace(a, 'Đúng số lượng, OCE theo lớp, càng thấp càng tốt: OCSD 2,17, ít lỗi đếm hơn mô hình nền 2,51, T2I-Adapter 2,67 và Zhang 2,41, nhưng nhiều hơn GLIGEN 2,06 và Collage 1,50; tỉ lệ ảnh đếm đúng của OCSD là 33 phần trăm, khác GLIGEN và Collage chưa có ý nghĩa thống kê. ' + a)
p.save(sys.argv[2]); print('ok')
