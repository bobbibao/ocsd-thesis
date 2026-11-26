"""Re-encode large opaque PNG photos inside a pptx as JPEG (q 90) and drop unreferenced media.
Usage: shrink_media.py IN OUT"""
import sys, zipfile, re, io
from PIL import Image
zin = zipfile.ZipFile(sys.argv[1]); names = zin.namelist()
data = {n: zin.read(n) for n in names}
rels = [n for n in names if n.endswith('.rels')]
used = set()
for r in rels:
    for t in re.findall(rb'Target="([^"]+)"', data[r]):
        used.add(t.decode().split('/')[-1])
ren = {}
for n in names:
    if not n.startswith('ppt/media/'): continue
    b = n.split('/')[-1]
    if b not in used: del data[n]; continue
    if b.endswith('.png') and len(data[n]) > 250_000:
        im = Image.open(io.BytesIO(data[n]))
        if im.mode in ('RGBA', 'LA', 'P'):
            a = im.convert('RGBA').getchannel('A')
            if a.getextrema()[0] < 255: continue
        out = io.BytesIO(); im.convert('RGB').save(out, 'JPEG', quality=90, optimize=True)
        if out.tell() < len(data[n]) * 0.7:
            nb = b[:-4] + '_j.jpeg'; del data[n]; data['ppt/media/' + nb] = out.getvalue(); ren[b] = nb
for r in rels:
    s = data[r]
    for o, nw in ren.items():
        s = s.replace(f'/{o}"'.encode(), f'/{nw}"'.encode())
    data[r] = s
ct = data['[Content_Types].xml']
if b'Extension="jpeg"' not in ct:
    ct = ct.replace(b'<Default ', b'<Default Extension="jpeg" ContentType="image/jpeg"/><Default ', 1)
data['[Content_Types].xml'] = ct
with zipfile.ZipFile(sys.argv[2], 'w', zipfile.ZIP_DEFLATED) as z:
    for n in names:
        if n in data: z.writestr(n, data[n])
    for n in data:
        if n not in names: z.writestr(n, data[n])
print('re-encoded', len(ren))
