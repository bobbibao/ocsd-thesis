"""Bobby 2026-11-26: slide 4 as a 3-click build.
Click 1: bullet 1 + caption -> SD 1.5 text-only image (layout not controllable).
Click 2: bullet 2 + same caption + sketch -> SD 1.5 + ControlNet image (layout follows the sketch).
Click 3: steps 1-2 fade out, bullet 3 + the existing failure row (more objects / abstract sketch).
Usage: slide4_anim.py IN OUT TEXT_ONLY.png SKETCH.png SKETCH_TEXT.png [--state N]
--state N (1, 2, 3) writes a static copy showing only what is visible after click N (for the PDF)."""
import sys, copy
from lxml import etree
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

INK, MUT, ORG, RED, GRN, WHT = C(0x1F,0x24,0x30), C(0x5B,0x62,0x70), C(0xD9,0x69,0x2B), C(0xB4,0x23,0x18), C(0x1E,0x7B,0x45), C(255,255,255)
CARD = C(0xF3,0xF4,0xF6)
args = [a for a in sys.argv[1:] if not a.startswith('--')]
state = int(sys.argv[sys.argv.index('--state') + 1]) if '--state' in sys.argv else 0
if state: args = [a for a in args if a != str(state)]
IN, OUT, IMG_T, IMG_SK, IMG_ST = args

p = Presentation(IN); s = p.slides[3]; S = {x.name: x for x in s.shapes}
assert s._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}timing') is None, 'slide 4 already animated'

# bullet 3: also abstract sketches (Bobby's wording)
for r in S['Text 4'].text_frame.paragraphs[2].runs:
    r.text = r.text.replace('khi phác thảo có nhiều vật thể,', 'khi phác thảo có nhiều vật thể hoặc nét vẽ trừu tượng,')

def tb(x, y, w, h, runs, size=12, col=INK, bold=False, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, italic=False):
    t = s.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    para = tf.paragraphs[0]; para.alignment = align
    for txt in ([runs] if isinstance(runs, str) else runs):
        r = para.add_run(); r.text = txt; f = r.font; f.size = Pt(size); f.color.rgb = col; f.bold = bold; f.italic = italic; f.name = 'Arial'
    return t
def box(x, y, w, h, fill):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(x), I(y), I(w), I(h)); b.adjustments[0] = 0.1
    b.fill.solid(); b.fill.fore_color.rgb = fill; b.line.fill.background(); b.shadow.inherit = False; return b
def arrow(x, y, w=0.5):
    a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, I(x), I(y), I(w), I(0.36))
    a.fill.solid(); a.fill.fore_color.rgb = ORG; a.line.fill.background(); a.shadow.inherit = False; return a
def pic(path, x, y, z):
    q = s.shapes.add_picture(path, I(x), I(y), I(z), I(z)); q.line.color.rgb = C(0xD0,0xD4,0xDA); q.line.width = Pt(0.75); return q

CAPTION = S['TextBox 14'].text_frame.text.strip()          # the deck's caption for this sketch, used in all 3 steps
def cap_style(box_, pt=13):    # caption a bit more prominent than the sketch labels: bigger, bold italic, dark
    for para in box_.text_frame.paragraphs:
        for r in para.runs: f = r.font; f.size = Pt(pt); f.bold = True; f.italic = True; f.color.rgb = INK
cap = []
Z, Y, HY = 2.6, 3.5, 3.16      # image size, image top, header top (same band as the failure row)
# step 1: caption only -> SD 1.5 text-only image (left half)
st1 = [tb(0.6, HY, 5.6, 0.3, 'Chỉ dùng câu mô tả (SD 1.5)', 12.5, INK, True)]
st1.append(box(0.6, Y + 0.6, 2.25, 1.4, C(0xFB,0xE9,0xDD)))
st1.append(tb(0.7, Y + 0.6, 2.05, 1.4, CAPTION, 14, INK, True, italic=True))
st1.append(arrow(2.93, Y + Z / 2 - 0.18, 0.46))
st1.append(pic(IMG_T, 3.47, Y, Z))
st1.append(tb(0.6, Y + Z + 0.08, 5.6, 0.42, 'Thiếu bò, nhà nằm xa trong góc: bố cục do mô hình tự chọn, không theo ý người dùng', 11, RED, True))
# step 2: sketch + caption -> SD 1.5 + ControlNet image (right half)
X2 = 6.85
st2 = [tb(6.8, HY, 5.9, 0.3, 'Phác thảo + câu mô tả (SD 1.5 + ControlNet)', 12.5, INK, True)]
st2.append(pic(IMG_SK, X2, Y, Z))
st2.append(tb(X2 - 0.15, Y + Z + 0.06, Z + 0.3, 0.5, CAPTION, 13, INK, True, italic=True))
st2.append(arrow(X2 + Z + 0.07, Y + Z / 2 - 0.18, 0.46))
st2.append(pic(IMG_ST, X2 + Z + 0.6, Y, Z))
st2.append(tb(X2 + Z + 0.5, Y + Z + 0.08, Z + 0.2, 0.42, 'Vật thể nằm đúng vị trí, đúng cỡ như nét vẽ', 11, GRN, True))
# step 3: the existing failure row (same sketch and caption); caption under the sketch made more prominent
# Bobby 2026-11-26: click 3 = a sketch with MORE objects (farm scene qd_5_medium_005, 5 objects, seed 0, real m4-study outputs);
# the same scene is the deck's example for background, M1-M5 and the metric slide
FARM = ('img4/farm_sketch.png', 'img4/farm_controlnet.jpg', 'img4/farm_t2i_adapter.png', 'img4/farm_gligen.jpg')
FARM_TXT = ('“three small cows, a tall tree and a red truck on a farm”', 'Cây thành đống đất; bò dưới thành xe tải đỏ',
            'Như tranh cắt giấy; bò dưới thành xe tải', 'Đủ vật nhưng thành bóng đen, sai bối cảnh')
for k in range(4):
    old_ = S[f'Picture {10 + k}']
    q = s.shapes.add_picture(FARM[k], old_.left, old_.top, old_.width, old_.height)
    if k == 0: q.line.color.rgb = C(0xD0,0xD4,0xDA); q.line.width = Pt(0.75)
    old_._element.addprevious(q._element); old_._element.getparent().remove(old_._element)
    S[f'Picture {10 + k}'] = q
    para = S[f'TextBox {14 + k}'].text_frame.paragraphs[0]
    para.runs[0].text = FARM_TXT[k]
    for r in para.runs[1:]: r.text = ''
cap_style(S['TextBox 14'])
st3 = [S[f'TextBox {k}'] for k in range(6, 10)] + [S[f'Picture {k}'] for k in range(10, 14)] + [S[f'TextBox {k}'] for k in range(14, 18)]
for x in cap + st1 + st2: x.name = 'ANIM4 ' + x.name

if state:   # static copy for the PDF: drop what is not visible after click `state`
    keep_paras = state
    hide = (st2 + st3) if state == 1 else st3 if state == 2 else (st1 + st2)
    for x in hide: x._element.getparent().remove(x._element)
    tf = S['Text 4'].text_frame._txBody
    for para in tf.findall('{http://schemas.openxmlformats.org/drawingml/2006/main}p')[keep_paras:]: tf.remove(para)
    p.save(OUT); print('static state', state); sys.exit()

# ---- click animations (fade in / fade out), written as p:timing XML
P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
nid = [2]
def cid(): nid[0] += 1; return nid[0]
def tgt(spid, para=None):
    return (f'<p:tgtEl><p:spTgt spid="{spid}"><p:txEl><p:pRg st="{para}" end="{para}"/></p:txEl></p:spTgt></p:tgtEl>'
            if para is not None else f'<p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>')
def fx(spid, kind, node, para=None, grp=0):
    t = tgt(spid, para)
    if kind == 'in':
        body = (f'<p:set><p:cBhvr><p:cTn id="{cid()}" dur="1" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>{t}'
                f'<p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr><p:to><p:strVal val="visible"/></p:to></p:set>'
                f'<p:animEffect transition="in" filter="fade"><p:cBhvr><p:cTn id="{cid()}" dur="500"/>{t}</p:cBhvr></p:animEffect>')
        cls = 'entr'
    else:
        body = (f'<p:animEffect transition="out" filter="fade"><p:cBhvr><p:cTn id="{cid()}" dur="400"/>{t}</p:cBhvr></p:animEffect>'
                f'<p:set><p:cBhvr><p:cTn id="{cid()}" dur="1" fill="hold"><p:stCondLst><p:cond delay="399"/></p:stCondLst></p:cTn>{t}'
                f'<p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr><p:to><p:strVal val="hidden"/></p:to></p:set>')
        cls = 'exit'
    me = cid()
    return (f'<p:par><p:cTn id="{me}" presetID="10" presetClass="{cls}" presetSubtype="0" fill="hold" grpId="{grp}" nodeType="{node}">'
            f'<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>{body}</p:childTnLst></p:cTn></p:par>')
def click(effects):
    a, b = cid(), cid()
    return (f'<p:par><p:cTn id="{a}" fill="hold"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst><p:childTnLst>'
            f'<p:par><p:cTn id="{b}" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
            + ''.join(effects) + '</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>')
sid = lambda x: x.shape_id
T4 = sid(S['Text 4'])
clicks = []
clicks.append(click([fx(T4, 'in', 'clickEffect', 0)] + [fx(sid(x), 'in', 'withEffect') for x in cap + st1]))
clicks.append(click([fx(T4, 'in', 'clickEffect', 1)] + [fx(sid(x), 'in', 'withEffect') for x in st2]))
clicks.append(click([fx(sid(x), 'out', 'clickEffect' if i == 0 else 'withEffect', grp=1) for i, x in enumerate(st1 + st2)]
                    + [fx(T4, 'in', 'withEffect', 2)] + [fx(sid(x), 'in', 'withEffect') for x in st3]))
bld = f'<p:bldP spid="{T4}" grpId="0" build="p"/>' + ''.join(
    f'<p:bldP spid="{sid(x)}" grpId="{g}" animBg="1"/>' for g in (0, 1) for x in st1 + st2 if x.has_text_frame) + ''.join(
    f'<p:bldP spid="{sid(x)}" grpId="0" animBg="1"/>' for x in cap + st3 if x.has_text_frame)
xml = (f'<p:timing xmlns:p="{P}"><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>'
       f'<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>' + ''.join(clicks) +
       '</p:childTnLst></p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
       '<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>'
       f'</p:childTnLst></p:cTn></p:par></p:tnLst><p:bldLst>{bld}</p:bldLst></p:timing>')
timing = etree.fromstring(xml)
root = s._element
ext = root.find(f'{{{P}}}extLst')
(ext.addprevious(timing) if ext is not None else root.append(timing))
p.save(OUT); print('ok', len(clicks), 'clicks', len(st1), len(st2), len(st3))
