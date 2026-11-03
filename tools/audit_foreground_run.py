"""Verify a completed four-arm foreground export and reproduce its seeds on CPU.

No learned inference, weight download or image generation. Figures are labeled
engineering diagnostics, not the anonymous human-review packet.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ocsd.clean_repair import clean_canvas
from ocsd.foreground import METHODS, REFINED_METHODS, MODEL_SHA256, place_foreground
from ocsd.sketch import square_crop


def audit(bundle, out):
    root = Path(bundle).resolve()
    manifest_bytes = (root / 'run_manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    rid = manifest['run_id']
    assert manifest['status'] == 'complete', 'Run is incomplete'
    assert manifest['contract']['experiment']['tier'] == 'foreground_screen'
    assert all(j['todo'] == 0 for j in manifest['jobs'].values())
    artifacts = manifest['artifacts']
    omitted = []
    for rel, digest in artifacts.items():
        path = (root / rel).resolve()
        assert path.is_relative_to(root), 'Artifact path escapes bundle'
        if not path.exists():
            parts = Path(rel).parts
            assert parts[:5] == ('outputs', 'runs', rid, 'quickdraw', '_objects'), rel
            omitted.append(rel)
        else:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, rel

    def read(rel):
        assert rel in artifacts and rel not in omitted, 'Required artifact missing: ' + rel
        return (root / rel).read_bytes()

    rp = f'results/runs/{rid}/'
    plan = json.loads(read(rp + 'quickdraw/plan.json'))
    assert set(plan) == set(METHODS)
    pairs = plan[METHODS[0]]
    assert len(pairs) == 1 and all(plan[m] == pairs for m in METHODS)
    sid, seed = pairs[0]
    item = f'{sid}_s{seed}'
    means = list(csv.DictReader(io.StringIO(read(rp + 'tables/screen_means.csv').decode())))
    assert len(means) == 4 and {r['method'] for r in means} == set(METHODS)
    metrics, logs, images, checks = [], {}, {}, {}
    for method in METHODS:
        prefix = f'outputs/runs/{rid}/quickdraw/{method}/'
        log = json.loads(read(prefix + item + '.json'))
        logs[method] = log
        assert (log['method'], log['sid'], log['seed']) == (method, sid, seed)
        rows = list(csv.DictReader(io.StringIO(read(rp + 'quickdraw/per_image_' + method + '.csv').decode())))
        assert len(rows) == 1 and (rows[0]['method'], rows[0]['sid'], int(rows[0]['seed'])) == (method, sid, seed)
        assert log['sha256'] == hashlib.sha256(read(prefix + item + '.png')).hexdigest()
        for rel, digest in log['diagnostics'].items():
            assert hashlib.sha256(read(f'outputs/runs/{rid}/' + rel)).hexdigest() == digest
        d = {}
        for name in ('initial', 'proposal', 'proposal_mask', 'placed_proposal', 'placed_proposal_mask',
                     'local_edit_mask', 'local_excluded_mask', 'local_conditioning_control',
                     'local_clean_background', 'local_clean_seed', 'local_decoded_candidate', 'local_candidate'):
            d[name] = np.array(Image.open(io.BytesIO(read(prefix + '_diagnostics/' + item + '/' + name + '.png'))).convert('RGB'))
        final = np.array(Image.open(io.BytesIO(read(prefix + item + '.png'))).convert('RGB'))
        d['final'] = final
        mask = d['local_edit_mask'][..., 0] > 0
        assert np.array_equal(final[~mask], d['initial'][~mask]), 'Changed protected pixels'
        assert np.array_equal(final, d['local_candidate']), 'Final differs from accepted candidate'
        assert log['route'] == 'count_replacement' and log['verify'][-1]['accepted']
        expected_canvas = dict(clean_init=True, clean_blend=True, extra_vae_encodes=int(method == METHODS[0]),
                               context_anchor=method != METHODS[0], background_fill='telea_radius_3')
        assert log['repair_canvas'] == expected_canvas
        target = log['target']
        for obj in log['scene_spec']['objects']:
            if obj['index'] == target:
                continue
            x0, y0, x1, y1 = map(int, obj['box'])
            assert np.array_equal(final[y0:y1, x0:x1], d['initial'][y0:y1, x0:x1])
        original_mask = d['placed_proposal_mask'][..., 0] > 0
        seed_image, background = clean_canvas(d['initial'], mask, d['placed_proposal'], original_mask)
        if method in REFINED_METHODS:
            for name in ('refined_probability', 'local_foreground_probability'):
                d[name] = np.array(Image.open(io.BytesIO(read(prefix + '_diagnostics/' + item + '/' + name + '.png'))).convert('RGB'))
            probability = d['refined_probability'][..., 0].astype(np.float32) / 255
            box = log['scene_spec']['objects'][target]['box']
            _, info = square_crop(np.zeros(mask.shape, np.uint8), box, final.shape[0])
            rgb, alpha = place_foreground(d['proposal'], probability, d['proposal_mask'][..., 0] > 0,
                                         info, log['foreground_placement'], final.shape[0], original_mask,
                                         hard=method.endswith('_hard'))
            alpha *= mask
            assert np.array_equal((alpha * 255).round().astype(np.uint8), d['local_foreground_probability'][..., 0])
            seed_image = np.rint(background * (1 - alpha[..., None]) + rgb * alpha[..., None]).clip(0, 255).astype(np.uint8)
            assert log['foreground_model']['sha256'] == MODEL_SHA256
        assert np.array_equal(background, d['local_clean_background']), 'Background reconstruction failed'
        assert np.array_equal(seed_image, d['local_clean_seed']), 'Seed reconstruction failed'
        fields = ('unet_sample_calls', 'controlnet_sample_calls', 'gligen_unet_sample_calls')
        work = sum(log['compute'].get(k, 0) + log.get('gligen_compute_cached', {}).get(k, 0)
                   + log['m2_compute'].get(k, 0) - log['m2_compute_in_this_call'].get(k, 0) for k in fields)
        assert work == log['standalone_sampling_work'] == 280
        row = rows[0]
        row['standalone_time_s'] = next(r['standalone_time_s'] for r in means if r['method'] == method)
        row['sampling_work'] = work
        metrics.append(row)
        checks[method] = dict(outside_pixels_unchanged=True, train_rectangles_unchanged=True,
                              accepted_candidate_is_final=True, exact_cpu_seed_reconstruction=True,
                              sampling_work=work, final_sha256=log['sha256'],
                              repair_canvas=log['repair_canvas'], foreground_model=log.get('foreground_model'))
        images[method] = d
    for method in METHODS[1:]:
        for name in ('initial', 'proposal', 'proposal_mask', 'placed_proposal', 'placed_proposal_mask',
                     'local_edit_mask', 'local_excluded_mask', 'local_conditioning_control', 'local_clean_background'):
            assert np.array_equal(images[method][name], images[METHODS[0]][name]), name
        for name in ('target', 'count_extra_ids', 'cfg', 'm2_key', 'm2_object_ids', 'm2_compute', 'scene_spec', 'focus_conditioning'):
            assert logs[method][name] == logs[METHODS[0]][name], name
    assert np.array_equal(images[METHODS[0]]['local_clean_seed'], images[METHODS[1]]['local_clean_seed'])
    hard, soft = REFINED_METHODS
    assert np.array_equal(images[hard]['refined_probability'], images[soft]['refined_probability'])
    assert logs[hard]['foreground_placement'] == logs[soft]['foreground_placement']
    actual_work = sum(sum(log['compute'].get(k, 0) for k in fields) for log in logs.values())
    assert actual_work == 460
    result = dict(run_id=rid, source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  source_git=manifest['git'], verified_artifacts=len(artifacts)-len(omitted),
                  omitted_cache_artifacts=omitted, checks=checks, actual_shared_sampling_work=actual_work,
                  common_inputs_masks_configuration=True, hard_soft_common_probability_placement=True,
                  cpu_opencv=cv2.__version__, quality_scope='engineering inspection only; no new human ratings')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    with (out / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    canvas = Image.new('RGB', (1024, 1088), 'white'); draw = ImageDraw.Draw(canvas)
    for i, method in enumerate(METHODS):
        x, y = i % 2 * 512, i // 2 * 544
        draw.text((x+8, y+8), method.removeprefix('instance_repair_count_'), fill='black')
        canvas.paste(Image.fromarray(images[method]['final']), (x, y+28))
    canvas.save(out / 'finals.jpg', quality=94)
    # Same crop and scale for every stage; includes residuals below the target.
    box = logs[soft]['scene_spec']['objects'][logs[soft]['target']]['box']
    crop = (int(box[0])-20, int(box[1])-45, int(box[2])+25, int(box[3])+50)
    canvas = Image.new('RGB', (1200, 1092), 'white'); draw = ImageDraw.Draw(canvas)
    for i, method in enumerate(METHODS[1:]):
        for j, name in enumerate(('local_clean_seed', 'local_decoded_candidate', 'final')):
            img = Image.fromarray(images[method][name]).crop(crop).resize((400, 320))
            x, y = j*400, i*364
            draw.text((x+6, y+8), method.removeprefix('instance_repair_count_') + ' / ' + name, fill='black')
            canvas.paste(img, (x, y+30))
    canvas.save(out / 'stages.jpg', quality=94)
    d = images[soft]; mask = d['local_edit_mask'][..., 0] > 0
    overlay = d['initial'].copy()
    overlay[mask] = (overlay[mask] * .55 + np.array([255, 0, 0]) * .45).astype(np.uint8)
    canvas = Image.new('RGB', (1200, 285), 'white'); draw = ImageDraw.Draw(canvas)
    for i, (title, pixels) in enumerate((('Initial', d['initial']), ('Editable pixels (red)', overlay),
                                        ('Soft clean seed', d['local_clean_seed']), ('Soft final', d['final']))):
        draw.text((i*300+5, 8), title, fill='black')
        canvas.paste(Image.fromarray(pixels).crop(crop).resize((300, 255)), (i*300, 28))
    canvas.save(out / 'regions.jpg', quality=94)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-root', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.bundle_root, args.out), indent=2))
