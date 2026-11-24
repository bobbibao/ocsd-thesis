"""Dataset slide: replace the 'Kiểm tra' sentences (and the 3-number rows) with real example images from the two
benchmarks: QuickDraw-Scenes sketches with 1/3/5/8+ objects, COCO-Sketch sketches made from real photos.
Usage: python patch_dataset_imgs.py IN.pptx OUT.pptx"""
import sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
SRC, OUT = sys.argv[1:3]
p = Presentation(SRC)
INK, MUT, ORG, BLU, LINE, WHT = C(0x1F,0x24,0x30), C(0x5B,0x62,0x70), C(0xD9,0x69,0x2B), C(0x2E,0x5A,0x9C), C(0xD5,0xD8,0xDE), C(255,255,255)
s = [s for s in p.slides if any(x.name == 'Text 1' and x.text_frame.text.startswith('Hai bộ dữ liệu kiểm thử do nhóm') for x in s.shapes)][0]
for x in list(s.shapes):
    if x.name in ('TextBox 73','TextBox 74','TextBox 75','TextBox 76','TextBox 78','TextBox 79','TextBox 80','TextBox 81'):
        x._element.getparent().remove(x._element)
def tb(x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    t = s.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in 'margin_left', 'margin_right', 'margin_top', 'margin_bottom': setattr(tf, m, 0)
    para = tf.paragraphs[0]; para.alignment = align
    for txt, sz, col, b in runs:
        r = para.add_run(); r.text = txt; r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = b; r.font.name = 'Arial'
    return t
def thumbs(x0, files, labels, col, size=1.02, gap=0.13, y=4.36):
    for k, (f, lab) in enumerate(zip(files, labels)):
        x = x0 + k * (size + gap)
        q = s.shapes.add_picture(f, I(x), I(y), I(size), I(size)); q.line.color.rgb = LINE; q.line.width = Pt(0.75)
        t = tb(x + 0.02, y + size - 0.22, size - 0.04, 0.2, [(lab, 9, col, True)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        t.fill.solid(); t.fill.fore_color.rgb = WHT
# QuickDraw panel (Rectangle 72: x 0.6, y 3.95, w 5.9, h 1.5)
tb(0.75, 4.0, 5.6, 0.32, [('72', 17, BLU, True), (' cảnh đánh giá   ', 10.5, MUT, False), ('3', 17, BLU, True),
                           (' mức nét vẽ   ', 10.5, MUT, False), ('đáp án: ', 10.5, MUT, False), ('vùng từng vật', 10.5, INK, True)], anchor=MSO_ANCHOR.MIDDLE)
thumbs(0.75, [f'grid_1_{c}.png' for c in range(4)], ['1 vật', '3 vật', '5 vật', '8+ vật'], BLU)
tb(5.4, 4.36, 1.0, 1.02, [('Càng nhiều vật, càng khó →', 9.5, ORG, True)], anchor=MSO_ANCHOR.MIDDLE)
# COCO panel (Rectangle 77: x 6.83)
tb(6.98, 4.0, 5.6, 0.32, [('32', 17, ORG, True), (' cảnh đánh giá   ', 10.5, MUT, False), ('1 – 8', 17, ORG, True),
                           (' vật mỗi cảnh, phác thảo từ ', 10.5, MUT, False), ('ảnh chụp thật', 10.5, INK, True)], anchor=MSO_ANCHOR.MIDDLE)
thumbs(6.98, ['coco_0.png', 'coco_1.png', 'coco_2.png'], ['người, ô', 'người, bàn ăn', 'xe máy, rào'], ORG)
tb(10.5, 4.36, 2.1, 1.02, [('Nét nhiễu, vật che nhau như ngoài đời: OCSD có còn giữ đúng vật?', 9.5, ORG, True)], anchor=MSO_ANCHOR.MIDDLE)
p.save(OUT)
print('ok')
