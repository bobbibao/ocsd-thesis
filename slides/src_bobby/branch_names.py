"""Bobby 2026-11-26: the two branches are 'nhánh xử lý vật thể' (M2, M3) and 'nhánh xử lý cảnh' (M4, M5);
rename everywhere after slide 21 too (tracker bars, agenda/overview texts, notes). Usage: branch_names.py IN OUT"""
import sys
from pptx import Presentation
R = [('NHÁNH VẬT THỂ', 'NHÁNH XỬ LÝ VẬT THỂ'), ('NHÁNH CẢNH', 'NHÁNH XỬ LÝ CẢNH'),
     ('nhánh vật thể', 'nhánh xử lý vật thể'), ('nhánh cảnh', 'nhánh xử lý cảnh'),
     ('hai nhánh vật thể và cảnh', 'hai nhánh: xử lý vật thể và xử lý cảnh')]
p = Presentation(sys.argv[1]); n = 0
def fix(t):
    for a, b in R:
        if b not in t: t = t.replace(a, b)
    return t
for s in p.slides:
    for sh in s.shapes:
        if sh.has_text_frame:
            for para in sh.text_frame.paragraphs:
                for r in para.runs:
                    t = fix(r.text)
                    if t != r.text: r.text = t; n += 1
    if s.has_notes_slide:
        tf = s.notes_slide.notes_text_frame; t = fix(tf.text)
        if t != tf.text: tf.text = t; n += 1
p.save(sys.argv[2]); print('changed', n)
