"""Small helpers to write PowerPoint click animations and slide transitions as raw XML (python-pptx has no API).
add_clicks(slide, clicks): clicks = list of steps; each step = list of (shape, effect) where effect is
'fade' | 'wipe' (wipe from left) | 'zoom'; a step may also hold ('para', shape, idx) to fade in one paragraph.
set_transition(slide, kind): 'fade' | 'push' | 'morph'."""
from lxml import etree

P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
NS = f'xmlns:p="{P}"'


def add_clicks(slide, clicks, after_prev=False):
    root = slide._element
    old = root.find(f'{{{P}}}timing')
    if old is not None:
        root.remove(old)
    nid = [2]

    def cid():
        nid[0] += 1
        return nid[0]

    def tgt(spid, para=None):
        if para is None:
            return f'<p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>'
        return f'<p:tgtEl><p:spTgt spid="{spid}"><p:txEl><p:pRg st="{para}" end="{para}"/></p:txEl></p:spTgt></p:tgtEl>'

    def fx(spid, eff, node, para=None):
        t = tgt(spid, para)
        vis = (f'<p:set><p:cBhvr><p:cTn id="{cid()}" dur="1" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>{t}'
               f'<p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr><p:to><p:strVal val="visible"/></p:to></p:set>')
        if eff == 'wipe':
            preset, sub = 22, 8
            body = vis + f'<p:animEffect transition="in" filter="wipe(left)"><p:cBhvr><p:cTn id="{cid()}" dur="600"/>{t}</p:cBhvr></p:animEffect>'
        elif eff == 'zoom':
            preset, sub = 53, 16
            body = vis + (
                f'<p:anim calcmode="lin" valueType="num"><p:cBhvr><p:cTn id="{cid()}" dur="500" fill="hold"/>{t}'
                f'<p:attrNameLst><p:attrName>ppt_w</p:attrName></p:attrNameLst></p:cBhvr><p:tavLst>'
                f'<p:tav tm="0"><p:val><p:fltVal val="0"/></p:val></p:tav><p:tav tm="100000"><p:val><p:strVal val="#ppt_w"/></p:val></p:tav></p:tavLst></p:anim>'
                f'<p:anim calcmode="lin" valueType="num"><p:cBhvr><p:cTn id="{cid()}" dur="500" fill="hold"/>{t}'
                f'<p:attrNameLst><p:attrName>ppt_h</p:attrName></p:attrNameLst></p:cBhvr><p:tavLst>'
                f'<p:tav tm="0"><p:val><p:fltVal val="0"/></p:val></p:tav><p:tav tm="100000"><p:val><p:strVal val="#ppt_h"/></p:val></p:tav></p:tavLst></p:anim>'
                f'<p:animEffect transition="in" filter="fade"><p:cBhvr><p:cTn id="{cid()}" dur="500"/>{t}</p:cBhvr></p:animEffect>')
        else:
            preset, sub = 10, 0
            body = vis + f'<p:animEffect transition="in" filter="fade"><p:cBhvr><p:cTn id="{cid()}" dur="500"/>{t}</p:cBhvr></p:animEffect>'
        me = cid()
        return (f'<p:par><p:cTn id="{me}" presetID="{preset}" presetClass="entr" presetSubtype="{sub}" fill="hold" grpId="0" nodeType="{node}">'
                f'<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>{body}</p:childTnLst></p:cTn></p:par>')

    pars, bld = [], {}
    for k, step in enumerate(clicks):
        effs = []
        for i, item in enumerate(step):
            first = i == 0
            node = ('afterEffect' if (after_prev and k == 0) else 'clickEffect') if first else 'withEffect'
            if item[0] == 'para':
                _, sh, idx = item
                effs.append(fx(sh.shape_id, 'fade', node, idx)); bld[sh.shape_id] = 'p'
            else:
                sh, eff = item
                effs.append(fx(sh.shape_id, eff, node))
                if sh.has_text_frame and sh.shape_id not in bld:
                    bld[sh.shape_id] = 'whole'
        a, b = cid(), cid()
        delay = '0' if (after_prev and k == 0) else 'indefinite'
        pars.append(f'<p:par><p:cTn id="{a}" fill="hold"><p:stCondLst><p:cond delay="{delay}"/></p:stCondLst><p:childTnLst>'
                    f'<p:par><p:cTn id="{b}" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
                    + ''.join(effs) + '</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>')
    bl = ''.join(f'<p:bldP spid="{i}" grpId="0" build="p"/>' if v == 'p' else f'<p:bldP spid="{i}" grpId="0" animBg="1"/>'
                 for i, v in bld.items())
    xml = (f'<p:timing {NS}><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>'
           f'<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>' + ''.join(pars) +
           '</p:childTnLst></p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
           '<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>'
           f'</p:childTnLst></p:cTn></p:par></p:tnLst>' + (f'<p:bldLst>{bl}</p:bldLst>' if bl else '') + '</p:timing>')
    _insert(root, etree.fromstring(xml), before=('extLst',))


def set_transition(slide, kind):
    root = slide._element
    for el in list(root):
        tag = etree.QName(el).localname
        if tag in ('transition', 'AlternateContent'):
            root.remove(el)
    if kind == 'morph':
        xml = ('<mc:AlternateContent xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">'
               '<mc:Choice xmlns:p159="http://schemas.microsoft.com/office/powerpoint/2015/09/main" Requires="p159">'
               f'<p:transition {NS} xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" spd="slow" p14:dur="1400">'
               '<p159:morph option="byObject"/></p:transition></mc:Choice>'
               f'<mc:Fallback><p:transition {NS} spd="slow"><p:fade/></p:transition></mc:Fallback></mc:AlternateContent>')
    elif kind == 'push':
        xml = f'<p:transition {NS} spd="slow"><p:push dir="l"/></p:transition>'
    else:
        xml = f'<p:transition {NS} spd="med"><p:fade/></p:transition>'
    _insert(root, etree.fromstring(xml), before=('timing', 'extLst'))


def _insert(root, el, before):
    for name in before:
        ref = root.find(f'{{{P}}}{name}')
        if ref is not None:
            ref.addprevious(el)
            return
    root.append(el)
