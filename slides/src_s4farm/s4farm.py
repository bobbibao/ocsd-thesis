"""Bobby 2026-11-26: slide 4 clicks 1-2 use the deck's farm caption too. Click 1 = SD 1.5 text only (notebook 11,
seed 1); click 2 = clean QuickDraw sketch, layout B, + caption through SD 1.5 + ControlNet (seed 7, all 5 objects).
Click 3 unchanged (abstract farm sketch, baselines fail). --state N writes the static copy for the PDF (slide 4 only).
Usage: s4farm.py IN OUT IMG_DIR [--state N]"""
import sys
from pptx import Presentation
from pptx.util import Pt
IN, OUT, IMG = sys.argv[1:4]
state = int(sys.argv[sys.argv.index('--state') + 1]) if '--state' in sys.argv else 0
p = Presentation(IN); s = p.slides[3]
S = {x.name: x for x in s.shapes}
CAP = '“three small cows, a tall tree and a red truck on a farm”'
def swap(name, f):
    o = S[name]; q = s.shapes.add_picture(f, o.left, o.top, o.width, o.height); q.name = name
    q._element.nvPicPr.cNvPr.id = o.shape_id  # keep the id: the click animation targets it
    if o.line.fill.type is not None:
        q.line.color.rgb = o.line.color.rgb if o.line.color and o.line.color.type else q.line.color.rgb
    o._element.addprevious(q._element); o._element.getparent().remove(o._element); S[name] = q
swap('ANIM4 Picture 22', f'{IMG}/text_only_s1.png')
swap('ANIM4 Picture 25', f'{IMG}/sketchB.png')
swap('ANIM4 Picture 28', f'{IMG}/sketchB_text_s7.png')
S['ANIM4 Picture 25'].line.color.rgb = __import__('pptx.dml.color', fromlist=['RGBColor']).RGBColor(0xD0, 0xD4, 0xDA)
S['ANIM4 Picture 25'].line.width = Pt(0.75)
def settext(name, txt, size=None):
    para = S[name].text_frame.paragraphs[0]; para.runs[0].text = txt
    for r in para.runs[1:]: r.text = ''
    if size:
        for r in para.runs: r.font.size = Pt(size)
settext('ANIM4 TextBox 20', CAP, 13)
settext('ANIM4 TextBox 26', CAP, 12)
settext('ANIM4 TextBox 23', 'Chỉ 1 con bò thay vì 3, xe tải chiếm gần hết ảnh: bố cục do mô hình tự chọn, không theo ý người dùng')
settext('ANIM4 TextBox 29', 'Đủ 3 bò, cây, xe đỏ; đúng vị trí, đúng cỡ như nét vẽ')
s.notes_slide.notes_text_frame.text = (
    'Mô hình tạo ảnh từ văn bản rất giỏi diễn tả cái gì có trong ảnh, nhưng kém ở chỗ vật thể nằm ở đâu và bao nhiêu cái. '
    'Click 1: chỉ đưa câu mô tả “three small cows, a tall tree and a red truck on a farm” cho SD 1.5: ảnh đẹp nhưng chỉ có một con bò, '
    'chiếc xe chiếm gần hết khung hình; bố cục do mô hình tự chọn. '
    'Click 2: cùng câu mô tả, thêm một phác thảo rõ nét, các vật cách xa nhau, qua SD 1.5 và ControlNet: đủ ba con bò, cái cây và xe tải đỏ, '
    'nằm đúng chỗ và đúng cỡ như nét vẽ. Phác thảo giải quyết được vấn đề vị trí và số lượng. '
    'Click 3: nhưng khi phác thảo trừu tượng và các vật chen sát nhau, như phác thảo tay thật của cùng cảnh này, các phương pháp hiện có lại lỗi: '
    'ControlNet biến cây thành đống đất và con bò dưới thành xe tải đỏ, T2I-Adapter ra ảnh như tranh cắt giấy, GLIGEN đủ vật nhưng thành bóng đen, sai bối cảnh. '
    'Nguyên nhân chung là mô hình không biết nét nào thuộc vật thể nào. Đây chính là cảnh nhóm dùng làm ví dụ xuyên suốt phần sau.')
if state:
    st1 = [S[f'ANIM4 {n}'] for n in ('TextBox 18', 'Rounded Rectangle 19', 'TextBox 20', 'Right Arrow 21', 'Picture 22', 'TextBox 23')]
    st2 = [S[f'ANIM4 {n}'] for n in ('TextBox 24', 'Picture 25', 'TextBox 26', 'Right Arrow 27', 'Picture 28', 'TextBox 29')]
    st3 = [S[f'TextBox {k}'] for k in range(6, 10)] + [S[f'Picture {k}'] for k in range(30, 34)] + [S[f'TextBox {k}'] for k in range(14, 18)]
    hide = (st2 + st3) if state == 1 else st3 if state == 2 else (st1 + st2)
    for x in hide: x._element.getparent().remove(x._element)
    tf = S['Text 4'].text_frame._txBody
    for para in tf.findall('{http://schemas.openxmlformats.org/drawingml/2006/main}p')[state:]: tf.remove(para)
    s._element.remove(s._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}timing'))
    ids = p.slides._sldIdLst
    for i, sid in reversed(list(enumerate(list(ids)))):
        if i != 3: p.part.drop_rel(sid.rId); ids.remove(sid)
p.save(OUT); print('ok', state)
