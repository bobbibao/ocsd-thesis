"""v46 (Bobby 2026-11-26): always 'bản phác thảo' (never bare 'phác thảo') on every slide and in notes;
slide 5 'người dùng' (no 'phổ thông'); slide 6 title 'Năm kiểu lỗi khi tạo ảnh từ bản phác thảo', 'bò 1' -> 'bò'.
Usage: ban_phac_thao.py IN OUT"""
import sys, re
from pptx import Presentation
p = Presentation(sys.argv[1])
def fix(t):
    t = re.sub(r'(?<![Bb]ản )phác thảo', 'bản phác thảo', t)
    t = re.sub(r'(?<![Bb]ản )Phác thảo', 'Bản phác thảo', t)
    t = re.sub(r'(?<!BẢN )PHÁC THẢO', 'BẢN PHÁC THẢO', t)
    return t
n = 0
for k, s in enumerate(p.slides, 1):
    for x in s.shapes:
        frames = [x.text_frame] if x.has_text_frame else []
        if x.has_table: frames = [c.text_frame for row in x.table.rows for c in row.cells]
        for tf in frames:
            for pa in tf.paragraphs:
                for r in pa.runs:
                    t = fix(r.text)
                    if k == 5: t = t.replace('cho người dùng phổ thông', 'cho người dùng')
                    if k == 6:
                        t = t.replace('Năm kiểu lỗi khi cảnh có nhiều vật thể', 'Năm kiểu lỗi khi tạo ảnh từ bản phác thảo')
                        t = t.replace('bò 1 + xe', 'bò + xe').replace('Bò 1 và xe', 'Bò và xe')
                    if t != r.text: r.text = t; n += 1
    if s.has_notes_slide:
        nt = s.notes_slide.notes_text_frame
        for pa in nt.paragraphs:
            for r in pa.runs:
                t = fix(r.text)
                if t != r.text: r.text = t; n += 1
p.save(sys.argv[2]); print('runs changed', n)
