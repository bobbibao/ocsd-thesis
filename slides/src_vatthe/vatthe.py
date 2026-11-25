"""Chain step 14 (last): Bobby 2026-11-25, write 'vật thể' (object) instead of 'đối tượng' for a thing in the image,
everywhere in the deck (text, tables, groups, notes, images), keep 'Đối tượng (và phạm vi) nghiên cứu'; add the
first-use definition from thesis v15 (/mnt/project-files/thesis/RQ_FINAL_v15.md) on the agenda slide.
Needs img/new_*.png from figs_vatthe.py. Usage (run here): vatthe.py IN OUT"""
import hashlib, re, sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR
p = Presentation(sys.argv[1])
KEEP = ['Đối tượng và phạm vi nghiên cứu', 'Đối tượng nghiên cứu', 'đối tượng nghiên cứu']
PAIRS = [('ĐỐI TƯỢNG', 'VẬT THỂ'), ('Đối tượng', 'Vật thể'), ('đối tượng', 'vật thể'), ('Đối Tượng', 'Vật Thể')]
DEF = ('Vật thể (object) là một thứ riêng lẻ xuất hiện trong ảnh và được người dùng vẽ trong bản phác thảo, như một con chó, '
       'một chiếc ghế hay một cái cây; nhất quán vật thể nghĩa là ảnh sinh ra có đúng các vật thể đó, đúng số lượng, '
       'đúng vị trí và đúng quan hệ như trong phác thảo.')

def fix(t):
    for k, w in enumerate(KEEP): t = t.replace(w, f'\x00{k}\x00')
    for a, b in PAIRS: t = t.replace(a, b)
    for k, w in enumerate(KEEP): t = t.replace(f'\x00{k}\x00', w)
    return t
def fix_para(para):
    rs = list(para.runs)
    if not rs: return
    full = ''.join(r.text for r in rs); new = fix(full)
    if new == full: return
    if all(fix(r.text) == r.text for r in rs) or len(new) != len(full) and True:
        pass
    # rewrite run by run: map each match onto the run where it starts, remove the rest of its chars from later runs
    spans = []                      # (start, end) of every run in the paragraph
    pos = 0
    for r in rs: spans.append((pos, pos + len(r.text))); pos += len(r.text)
    out = [list(r.text) for r in rs]
    owner = [k for k, (a, b) in enumerate(spans) for _ in range(b - a)]
    keep_mask = [True] * len(full)
    for kk, w in enumerate(KEEP):
        for m in re.finditer(re.escape(w), full):
            for j in range(m.start(), m.end()): keep_mask[j] = False
    edits = []
    for a, b in PAIRS:
        for m in re.finditer(re.escape(a), full):
            if all(keep_mask[m.start():m.end()]): edits.append((m.start(), m.end(), b))
    edits.sort()
    texts = [r.text for r in rs]
    for s0, e0, b in reversed(edits):
        k0 = owner[s0]
        for k in range(len(rs)):
            a_, b_ = spans[k]
            lo, hi = max(s0, a_), min(e0, b_)
            if lo >= hi: continue
            t = texts[k]
            texts[k] = t[:lo - a_] + (b if k == k0 else '') + t[hi - a_:]
    for r, t in zip(rs, texts): r.text = t
def walk(shapes):
    for x in shapes:
        if x.shape_type == MSO_SHAPE_TYPE.GROUP: walk(x.shapes); continue
        if x.has_text_frame:
            for para in x.text_frame.paragraphs: fix_para(para)
        if getattr(x, 'has_table', False) and x.has_table:
            for row in x.table.rows:
                for cell in row.cells:
                    for para in cell.text_frame.paragraphs: fix_para(para)
for s in p.slides:
    walk(s.shapes)
    if s.has_notes_slide:
        tf = s.notes_slide.notes_text_frame; t = fix(tf.text)
        if t != tf.text: tf.text = t
# ---------------------------------------------------------------- images with the old word
md5 = lambda b: hashlib.md5(b).hexdigest()
SWAP = {}                      # md5 of the picture blobs as they sit in the deck (v10)
import json
for k, v in json.load(open('img/hashes.json')).items(): SWAP[v] = 'img/new_' + k + '.png'
n_sw = 0
for s in p.slides:
    for x in list(s.shapes):
        if x.shape_type == MSO_SHAPE_TYPE.PICTURE and md5(x.image.blob) in SWAP:
            new = s.shapes.add_picture(SWAP[md5(x.image.blob)], x.left, x.top, x.width, x.height)
            if x.line.fill.type is not None:
                try: new.line.color.rgb = x.line.color.rgb; new.line.width = x.line.width
                except Exception: pass
            x._element.addprevious(new._element); x._element.getparent().remove(x._element); n_sw += 1
# ---------------------------------------------------------------- definition on the agenda slide (first use)
s = next(s for s in p.slides if any(x.has_text_frame and x.text_frame.text.strip() == 'Nội dung trình bày' for x in s.shapes))
for x in list(s.shapes):
    if x.name == 'VatThe def': x._element.getparent().remove(x._element)
b = s.shapes.add_shape(1, I(8.2), I(5.15), I(4.55), I(1.6)); b.name = 'VatThe def'
b.fill.solid(); b.fill.fore_color.rgb = C(0xFB, 0xE9, 0xDD); b.line.fill.background(); b.shadow.inherit = False
t = s.shapes.add_textbox(I(8.35), I(5.22), I(4.25), I(1.46)); t.name = 'VatThe def'; tf = t.text_frame; tf.word_wrap = True
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
for m in 'margin_left', 'margin_right', 'margin_top', 'margin_bottom': setattr(tf, m, 0)
pa = tf.paragraphs[0]
for txt, b_, col in (('Vật thể (object) ', True, C(0xD9, 0x69, 0x2B)), (DEF[len('Vật thể (object) '):], False, C(0x1F, 0x24, 0x30))):
    r = pa.add_run(); r.text = txt; r.font.size = Pt(11); r.font.bold = b_; r.font.color.rgb = col; r.font.name = 'Arial'
tfn = s.notes_slide.notes_text_frame
if 'Vật thể (object)' not in tfn.text: tfn.text = tfn.text + ' Trong bài, ' + DEF[0].lower() + DEF[1:]
p.save(sys.argv[2]); print('images swapped', n_sw)
