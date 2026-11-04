"""Build the fixed stage-1 development scenes from public QuickDraw records.

Run once before any output; the result (assets/direct_dev_v1/scenes.json) is
committed and the Colab run never re-downloads. Selection rule: the first
recognized records at zero-based line index >= 4000 (the canary used >= 3000,
the usual downloader the first 3000). For objects marked `wide`, the first such
record whose stroke bounding box has width/height >= 2.2.
"""
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson'
START = 4000
MIN_WIDE = 2.2

# Layout, captions and phrases are fixed here, before any generated image.
SCENES = [
    dict(sid='dev1_cars_house', group='same_class_colours',
         caption='a red car on the left and a white car on the right, parked in front of a house',
         caption_vi='Một ô tô đỏ bên trái, một ô tô trắng bên phải, đỗ trước một ngôi nhà.',
         objects=[('car', 'a red car', [20, 280, 240, 450]), ('car', 'a white car', [272, 280, 492, 450]),
                  ('house', 'a house', [156, 30, 356, 240])]),
    dict(sid='dev2_cats_tree', group='same_class_colours',
         caption='a black cat on the left and a white cat in the middle, sitting on a lawn next to a tree',
         caption_vi='Một con mèo đen bên trái, một con mèo trắng ở giữa, ngồi trên bãi cỏ cạnh một cái cây.',
         objects=[('cat', 'a black cat', [30, 260, 210, 470]), ('cat', 'a white cat', [220, 280, 380, 470]),
                  ('tree', 'a tree', [330, 30, 500, 270])]),
    dict(sid='dev3_umbrellas_palm', group='same_class_colours',
         caption='a red umbrella and a blue umbrella on a sandy beach, with a palm tree on the right',
         caption_vi='Một cây dù đỏ và một cây dù xanh dương trên bãi cát, có một cây dừa bên phải.',
         objects=[('umbrella', 'a red umbrella', [20, 140, 190, 330]), ('umbrella', 'a blue umbrella', [205, 160, 350, 330]),
                  ('palm tree', 'a palm tree', [370, 40, 500, 420])]),
    dict(sid='dev4_apples_cup', group='same_class_colours',
         caption='a green apple and a red apple next to a white cup on a wooden table',
         caption_vi='Một quả táo xanh lá và một quả táo đỏ cạnh một chiếc cốc trắng trên bàn gỗ.',
         objects=[('apple', 'a green apple', [40, 290, 180, 430]), ('apple', 'a red apple', [195, 300, 325, 430]),
                  ('cup', 'a white cup', [350, 230, 480, 430])]),
    dict(sid='dev5_dog_couch', group='overlap',
         caption='a brown dog sitting on a gray couch next to a floor lamp in a living room',
         caption_vi='Một con chó nâu ngồi trên ghế sofa xám, cạnh một cây đèn đứng trong phòng khách.',
         objects=[('couch', 'a gray couch', [20, 220, 400, 440]), ('dog', 'a brown dog', [150, 130, 300, 330]),
                  ('floor lamp', 'a floor lamp', [420, 40, 500, 440])]),
    dict(sid='dev6_bird_fence', group='overlap',
         caption='a small bird perched on a wooden fence in a garden, with a flower in front of the fence',
         caption_vi='Một chú chim nhỏ đậu trên hàng rào gỗ trong vườn, có một bông hoa trước hàng rào.',
         objects=[('fence', 'a wooden fence', [20, 250, 492, 410]), ('bird', 'a small bird', [190, 200, 320, 290]),
                  ('flower', 'a flower', [370, 320, 480, 490])]),
    dict(sid='dev7_train_tree_sun', group='wide_box',
         caption='a long train passing by a tree under the sun',
         caption_vi='Một đoàn tàu dài chạy qua một cái cây dưới mặt trời.',
         objects=[('train', 'a long train', [20, 260, 400, 400], True), ('tree', 'a tree', [410, 120, 500, 420]),
                  ('sun', 'the sun', [40, 30, 150, 140])]),
    dict(sid='dev8_canoe_mountain', group='wide_box',
         caption='a red canoe on a calm lake, with a mountain behind it and a cloud in the sky',
         caption_vi='Một chiếc xuồng đỏ trên mặt hồ phẳng lặng, phía sau là ngọn núi và một đám mây trên trời.',
         objects=[('canoe', 'a red canoe', [90, 340, 430, 420], True), ('mountain', 'a mountain', [150, 70, 492, 290]),
                  ('cloud', 'a cloud', [20, 20, 140, 100])]),
]


def records(cls):
    url = URL.format(urllib.parse.quote(cls))
    with urllib.request.urlopen(url) as stream:
        for index, line in enumerate(stream):
            if index >= START:
                yield url, index, json.loads(line)


def pick(cls, wide, used):
    for url, index, rec in records(cls):
        if not rec['recognized'] or (cls, index) in used:
            continue
        xs = [x for s in rec['drawing'] for x in s[0]]; ys = [y for s in rec['drawing'] for y in s[1]]
        aspect = (max(xs)-min(xs))/max(1, max(ys)-min(ys))
        if wide and aspect < MIN_WIDE:
            continue
        used.add((cls, index))
        return dict(source_url=url, line_index=index, key_id=rec['key_id'], recognized=True,
                    countrycode=rec.get('countrycode'), drawing=rec['drawing'])
    raise ValueError('No record for ' + cls)


def main(out):
    used, scenes = set(), []
    for spec in SCENES:
        objects = []
        for item in spec['objects']:
            cls, phrase, box = item[:3]; wide = len(item) > 3 and item[3]
            objects.append(dict(cls=cls, phrase=phrase, placement_box=box, wide_rule=wide,
                                source=pick(cls, wide, used)))
            print(spec['sid'], cls, objects[-1]['source']['line_index'], flush=True)
        scenes.append(dict(sid=spec['sid'], group=spec['group'], size=512, caption=spec['caption'],
                           caption_vi=spec['caption_vi'], objects=objects,
                           selection=f'First recognized QuickDraw records at zero-based line index >= {START} '
                                     f'(wide objects: first with width/height >= {MIN_WIDE}); fixed before outputs; '
                                     'development scenes, not a held-out benchmark.'))
    Path(out).write_text(json.dumps(dict(schema='ocsd-direct-dev-scenes-v1', scenes=scenes), indent=1) + '\n')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT/'assets/direct_dev_v1/scenes.json')
