"""Build a standalone Vietnamese review page from an exported instance screen (CPU only)."""
import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path
import re
import textwrap


def static_review(scenes, images):
    """Render the evidence before JavaScript; a restricted viewer still shows images."""
    def picture(digest, label):
        return (f'<img data-image-hash="{digest}" src="{images[digest]}" '
                f'alt="{html.escape(label, quote=True)}">')
    parts = []
    for i, scene in enumerate(scenes, 1):
        parts.append(f'<section class="panel"><h2>Cảnh {i}</h2><div class="input">')
        parts.append(picture(scene['sketch'], 'Phác thảo đầu vào'))
        parts.append('<div><p>' + html.escape(scene['caption_vi'] or scene['caption']) + '</p>')
        parts.append('<small>Mô tả gốc: ' + html.escape(scene['caption']) + '</small></div></div><div class="grid">')
        for group in scene['groups']:
            label = ' = '.join(group['labels'])
            parts.append('<article><h3>Ảnh ' + html.escape(label) + '</h3>')
            parts.append(picture(group['image'], 'Kết quả ' + label) + '</article>')
        parts.append('</div></section>')
    return ''.join(parts)


def comparison_images(scenes, images, out):
    """Plain PNG evidence, with the same anonymous labels and no browser dependency."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 22)
    except OSError:
        font = ImageFont.load_default()
    def decode(digest):
        return Image.open(io.BytesIO(base64.b64decode(images[digest].split(',', 1)[1]))).convert('RGB')
    names = []
    for i, scene in enumerate(scenes, 1):
        sheet = Image.new('RGB', (1024, 288 + 544 * ((len(scene['groups']) + 1) // 2)), 'white')
        draw = ImageDraw.Draw(sheet)
        sketch = decode(scene['sketch']); sketch.thumbnail((256, 256))
        sheet.paste(sketch, (8, 24))
        draw.text((280, 24), f'Scene {i} - sketch and caption', fill='black', font=font)
        for line, text in enumerate(textwrap.wrap(scene['caption'], width=48)):
            draw.text((280, 70 + line*28), text, fill='black', font=font)
        for j, group in enumerate(scene['groups']):
            x, y = j % 2 * 512, 288 + j // 2 * 544
            draw.text((x+8, y+2), ' = '.join(group['labels']), fill='black', font=font)
            img = decode(group['image']); img.thumbnail((512, 512))
            sheet.paste(img, (x, y+32))
        name = f'SCENE_{i:02}.png'; sheet.save(out / name); names.append(name)
    return names

SCHEMA = 'ocsd-human-review-v1'
CRITERIA = [
    ('presence', 'Đúng loại', 'Có đủ các loại được yêu cầu (chó, mèo, ghế…)? Chưa xét số lượng.'),
    ('count', 'Đúng số lượng', 'Đếm riêng từng loại. Ví dụ 4 ngựa + 1 chó khác 5 ngựa. Thừa cũng là lỗi.'),
    ('layout', 'Đúng vị trí', 'Mỗi đối tượng có ở vùng và kích thước tương ứng trong phác thảo?'),
    ('shape', 'Đúng hình dáng', 'So đường bao, tư thế, hướng nhìn với nét phác thảo; không chỉ so khung bao.'),
    ('relations', 'Đúng quan hệ', 'So trái/phải, trên/dưới và chồng lấp giữa các đối tượng với phác thảo. Không đoán quan hệ trước/sau nếu không rõ.'),
    ('attributes', 'Đúng đặc điểm', 'Đúng màu, kích thước hoặc chất liệu mà mô tả yêu cầu? Không tự thêm yêu cầu.'),
    ('background', 'Đúng bối cảnh', 'Ví dụ mô tả phòng khách thì ảnh ngoài sân không đạt.'),
    ('realism', 'Ảnh tự nhiên', 'Có bộ phận méo, vật thể dính nhau, viền ghép hoặc ánh sáng bất thường không?'),
]
CAPTIONS = {
 'a black airplane at an airport':'Một máy bay đen ở sân bay.',
 'a horse on a farm':'Một con ngựa ở trang trại.',
 'a tree in a green meadow':'Một cây trên đồng cỏ xanh.',
 'two green chairs and a black chair on a sandy beach':'Hai ghế xanh lá và một ghế đen trên bãi biển cát (tổng 3 ghế).',
 'two black elephants and a yellow truck in a savanna at sunset':'Hai con voi đen và một xe tải vàng trên đồng cỏ savanna lúc hoàng hôn.',
 'a yellow bicycle and two blue trains near a train station':'Một xe đạp vàng và hai đoàn tàu xanh dương gần ga tàu.',
 'a wooden house, two red sailboats and two cats on a sandy beach':'Một nhà gỗ, hai thuyền buồm đỏ và hai con mèo trên bãi biển cát.',
 'four small horses and a brown dog in a forest':'Bốn con ngựa nhỏ và một con chó nâu trong rừng.',
 'two chairs, a white umbrella, a red chair and a gray cat in a living room':'Hai ghế không quy định màu, một ô trắng, thêm một ghế đỏ và một mèo xám trong phòng khách (tổng 3 ghế).',
 'four black dogs, two green chairs, a black umbrella, an orange umbrella and a golden cat in a living room':'Bốn chó đen, hai ghế xanh lá, một ô đen, một ô cam và một mèo màu vàng trong phòng khách.',
 'two green buses, a green train, two black dogs, two small traffic lights and a black bicycle near a train station':'Hai xe buýt xanh lá, một đoàn tàu xanh lá, hai chó đen, hai đèn giao thông nhỏ và một xe đạp đen gần ga tàu.',
 'three birds, two green chairs, a small rabbit, a small tree and a duck in a park':'Ba con chim, hai ghế xanh lá, một thỏ nhỏ, một cây nhỏ và một con vịt trong công viên. Vịt được yêu cầu riêng; không đếm lại vịt vào ba chim.',
}


def build(root, out, criteria=None):
    selected = CRITERIA
    if criteria is not None:
        available = {c[0]: c for c in CRITERIA}
        if not criteria or len(set(criteria)) != len(criteria) or set(criteria) - set(available):
            raise ValueError('Expected unique known review criteria')
        selected = [available[c] for c in criteria]
    root = Path(root).resolve()
    manifest = json.loads((root / 'run_manifest.json').read_text())
    rid = manifest['run_id']
    def read(rel):
        p = (root / rel).resolve()
        if not p.is_relative_to(root):
            raise ValueError('Artifact path escapes bundle')
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest['artifacts'].get(rel):
            raise ValueError(f'Missing or changed manifest artifact: {rel}')
        return data
    rp = f'results/runs/{rid}/'
    op = f'outputs/runs/{rid}/quickdraw/'
    key = json.loads(read(rp + 'review_key.json'))
    images, scenes = {}, []
    def embed(data):
        digest = hashlib.sha256(data).hexdigest()
        images[digest] = 'data:image/png;base64,' + base64.b64encode(data).decode()
        return digest
    for item, labels in sorted(key.items()):
        if set(labels) != set('ABCD'):
            raise ValueError('Expected exactly four labels per scene')
        specs, groups = [], {}
        for label, method in sorted(labels.items()):
            stem = op + method + '/' + item
            meta = json.loads(read(stem + '.json'))
            if meta.get('scene_spec'):
                specs.append(meta['scene_spec'])
            digest = embed(read(stem + '.png'))
            groups.setdefault(digest, []).append(label)
        if not specs or any(s != specs[0] for s in specs):
            raise ValueError(f'Missing or inconsistent scene specification: {item}')
        spec = specs[0]
        sid = item.rsplit('_s', 1)[0]
        sketch = embed(read(rp + 'review_blind/inputs/' + sid + '.png'))
        scenes.append(dict(item=item, caption=spec['caption'], caption_vi=CAPTIONS.get(spec['caption'], ''),
                           objects=spec['objects'], sketch=sketch,
                           groups=[dict(image=h, labels=ls) for h, ls in groups.items()]))
    if not scenes:
        raise ValueError('Empty review packet')
    # No method key, metrics, acceptance decisions or recommended ratings in the reviewer payload.
    payload = dict(schema=SCHEMA, run_id=rid, criteria=selected,
                   all_criteria_keys=[c[0] for c in CRITERIA], scenes=scenes, images=list(images))
    serialized = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    template = Path(__file__).with_name('templates').joinpath('human_review.html').read_text()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    replacements = {'__PAYLOAD__': serialized, '__STATIC_REVIEW__': static_review(scenes, images)}
    (out / 'REVIEW.html').write_text(re.sub(r'__(?:PAYLOAD|STATIC_REVIEW)__',
                                           lambda match: replacements[match.group()], template))
    static_images = comparison_images(scenes, images, out)
    summary = dict(run_id=rid, scenes=len(scenes), original_rows=sum(len(x['labels']) for s in scenes for x in s['groups']),
                   unique_rating_cards=sum(len(s['groups']) for s in scenes), schema=SCHEMA,
                   criteria=[c[0] for c in selected], static_images=static_images)
    (out / 'packet_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-root', required=True); p.add_argument('--out', required=True)
    p.add_argument('--criteria', nargs='+', choices=[c[0] for c in CRITERIA],
                   help='Optional focused review; omitted criteria remain unanswered, never inferred')
    args = p.parse_args()
    print(json.dumps(build(args.bundle_root, args.out, args.criteria), indent=2))
