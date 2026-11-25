"""Bobby 2026-11-26: use 'tạo ảnh' (create), never 'sinh ảnh' (generate), everywhere: slide text and notes.
'Sinh viên' is kept. Usage: tao_anh.py IN OUT"""
import re, sys
from pptx import Presentation
R = [('Mô hình sinh ảnh từ văn bản tạo ảnh đẹp', 'Mô hình tạo ảnh từ văn bản cho ảnh đẹp'), ('M2-Sinh ảnh cấp vật thể', 'M2-Tạo ảnh cấp vật thể'), ('M2-Sinh vật thể', 'M2-Tạo vật thể'), ('Sinh ảnh', 'Tạo ảnh'), ('sinh ảnh', 'tạo ảnh'),
     ('ảnh sinh ra', 'ảnh tạo ra'), ('ảnh sinh thật', 'ảnh tạo ra thật'), ('kết quả sinh của', 'kết quả tạo ảnh của'),
     ('Sinh video', 'Tạo video'), ('sinh video', 'tạo video'), ('sinh được ảnh', 'tạo được ảnh'), ('sau khi sinh', 'sau khi tạo ảnh'),
     ('sau sinh', 'sau khi tạo ảnh'), ('sinh hai ứng viên', 'tạo hai ứng viên'), ('sinh theo hạt giống', 'tạo theo hạt giống'),
     ('sinh vật thể và', 'tạo ảnh vật thể và'), ('tự sinh vùng', 'tự tạo vùng'), ('sinh vùng', 'tạo vùng'), ('sinh câu mô tả', 'tạo câu mô tả')]
def fix(t):
    for a, b in R: t = t.replace(a, b)
    return t
p = Presentation(sys.argv[1]); n = 0
def runs(tf):
    for para in tf.paragraphs:
        yield from para.runs
for s in p.slides:
    tfs = [sh.text_frame for sh in s.shapes if sh.has_text_frame]
    for sh in s.shapes:
        if sh.has_table:
            tfs += [c.text_frame for row in sh.table.rows for c in row.cells]
    for tf in tfs:
        for r in runs(tf):
            t = fix(r.text)
            if t != r.text: r.text = t; n += 1
    if s.has_notes_slide:
        tf = s.notes_slide.notes_text_frame; t = fix(tf.text)
        if t != tf.text: tf.text = t; n += 1
left = []
for i, s in enumerate(p.slides, 1):
    txt = ' '.join(sh.text_frame.text for sh in s.shapes if sh.has_text_frame) + ' ' + (s.notes_slide.notes_text_frame.text if s.has_notes_slide else '')
    left += [(i, m.group(0)) for m in re.finditer(r'.{0,20}\b[Ss]inh\b(?! viên).{0,20}', txt)]
p.save(sys.argv[2]); print('changed', n, 'left', left)
