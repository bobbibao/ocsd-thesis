"""Bobby 2026-11-26: remove everything about the old (hard-blend) M4 from his deck, editing only those spots.
Usage (run here): drop_old_m4.py IN OUT"""
import hashlib, sys
from pptx import Presentation
from pptx.util import Inches as I
p = Presentation(sys.argv[1])
md5 = lambda b: hashlib.md5(b).hexdigest()
F = {md5(open(f'../src_final/charts/f_{k}.png', 'rb').read()): f'charts/g_{k}.png' for k in ('opr', 'miou', 'ra', 'pick', 'count', 'dev')}
def title(s): return next((x.text_frame.text.strip() for x in s.shapes if x.name == 'Text 1' and x.has_text_frame), '')
def find(prefix): return next(s for s in p.slides if title(s).startswith(prefix))
def sh(s, n): return next(x for x in s.shapes if x.name == n)
def rep(shape, a, b):
    hit = False
    for para in shape.text_frame.paragraphs:
        for r in para.runs:
            if a in r.text: r.text = r.text.replace(a, b); hit = True
    assert hit, (shape.name, a)
def note(s, a, b):
    tf = s.notes_slide.notes_text_frame; assert a in tf.text, a; tf.text = tf.text.replace(a, b)
# charts (slides 30, 31, 32): PowerPoint re-encoded the images, so pick them by slide and position
n = 0
for pre, ks in (('Tinh chỉnh M4', ['dev']), ('Khi số vật thể tăng', ['count']), ('OCSD so với các phương pháp khác', ['opr', 'miou', 'ra', 'pick'])):
    s = find(pre)
    pics = sorted([x for x in s.shapes if x.shape_type == 13], key=lambda x: x.left)
    assert len(pics) == len(ks), (pre, len(pics))
    for x, k in zip(pics, ks):
        new = s.shapes.add_picture(f'charts/g_{k}.png', x.left, x.top, x.width, x.height)
        x._element.addprevious(new._element); x._element.getparent().remove(x._element); n += 1
assert n == 6, n
# M4 slide: drop the "Khác M4 cũ" card, widen the method card
s = find('M4-Tổng hợp cảnh')
box = sh(s, 'TextBox 89')
card = next(x for x in s.shapes if x.shape_type == 1 and abs(x.left - I(7.1)) < I(0.05) and abs(x.top - I(5.0)) < I(0.05))
for x in (box, card): x._element.getparent().remove(x._element)
left = next(x for x in s.shapes if x.shape_type == 1 and abs(x.left - I(0.6)) < I(0.05) and abs(x.top - I(5.0)) < I(0.05))
left.width = I(12.1)
txt = next(x for x in s.shapes if x.has_text_frame and x.text_frame.text.startswith('Cách trộn ở mỗi bước'))
txt.width = I(11.75)
note(s, 'So với M4 cũ: M4 cũ dán cứng vật suốt 27 bước rồi chỉ thả 3 bước cuối, nên vật trông như dán vào và lộ viền trắng. M4 mới nhả dần, vật và nền khử nhiễu chung, mặt nạ được co và làm mềm. '
        'Trên 72 cảnh, độ có mặt của vật tăng từ 71 lên 79 phần trăm và ảnh tự nhiên hơn rõ rệt. ',
     'Vật và nền khử nhiễu chung một quỹ đạo, mặt nạ được co và làm mềm nên không lộ viền. ')
# tuning slide
s = find('Tinh chỉnh M4')
rep(sh(s, 'TextBox 8'), 'M4 cũ, M4 mới, 7 biến thể mỗi cái đổi một chỗ', 'M4 mới, các biến thể mỗi cái đổi một chỗ')
rep(sh(s, 'TextBox 10'), ' so với M4 cũ', '')
rep(sh(s, 'TextBox 8'), 'Thử 10 cấu hình M4', 'Thử 9 cấu hình M4')
rep(sh(s, 'TextBox 8'), '7 cấu hình tiêu biểu', '6 cấu hình tiêu biểu')
tf = s.notes_slide.notes_text_frame; tf.text = tf.text.replace('Nhóm thử mười cấu hình: M4 cũ, M4 mới, bảy biến thể', 'Nhóm thử M4 mới, các biến thể')
# by-count slide + comparison slide
s = find('Khi số vật thể tăng')
tf = s.notes_slide.notes_text_frame; t = tf.text
a = t.find('đường cam nhạt là OCSD với M4 cũ'); b = t.find('. ', a)
tf.text = t[:a].rstrip('; ') + t[b:]
s = find('OCSD so với các phương pháp khác')
rep(sh(s, 'TextBox 16'), '; hơn M4 cũ (21,16)', '')
note(s, ', thanh cam nhạt là OCSD với M4 cũ', '')
note(s, 'so với sáu phương pháp', 'so với năm phương pháp')
# delete the slide "M4 mới so với M4 cũ"
lst = p.slides._sldIdLst
i = next(k for k, s in enumerate(p.slides) if title(s).startswith('M4 mới so với M4 cũ'))
sid = lst[i]; p.part.drop_rel(sid.rId); lst.remove(sid)
# renumber
import re
N = len(p.slides)
for k, s in enumerate(p.slides, 1):
    for x in s.shapes:
        if x.has_text_frame and re.fullmatch(r'\s*\d+ / \d+\s*', x.text_frame.text):
            r = x.text_frame.paragraphs[0].runs; r[0].text = f'{k} / {N}'
            for q in r[1:]: q.text = ''
p.save(sys.argv[2]); print('slides', N)
