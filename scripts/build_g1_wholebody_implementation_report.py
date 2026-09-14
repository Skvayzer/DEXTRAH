#!/usr/bin/env python3
"""Build the September 14 implementation report from exported experiment data.

No training, simulation, invented evaluation, or checkpoint mutation. Uses
TensorBoard scalars, bootstrap logs and saved task contracts. Plot data, file
hashes, vector figures and page previews are written beside the report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT = Path(__file__).resolve().parents[1]
W, H = 841.89, 595.28
INK, MUTED, TEAL, BLUE, ORANGE = '#183347', '#596D7B', '#007D76', '#316CA4', '#AC501A'
LIGHT, LINE, GREEN_BG, BLUE_BG, AMBER_BG = '#F4F7F9', '#DCE5EA', '#EAF5F1', '#EEF3FA', '#FFF3E8'
CODE = 'https://github.com/Skvayzer/DEXTRAH/blob/8a5c1b5/'
WB = 'https://wandb.ai/skvayzer/adept/runs/unique_id_0_sonic_sapg_touch_547'
DATE = '14 SEPTEMBER 2026'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def save_plot(fig, target):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(target.with_suffix('.'+ext), dpi=200, facecolor='white')
    plt.close(fig)


def moving_mean(y, window=100):
    y = np.asarray(y, float)
    prefix = np.r_[0., np.cumsum(y)]
    starts = np.maximum(0, np.arange(len(y))+1-window)
    return (prefix[1:]-prefix[starts])/(np.arange(len(y))+1-starts)


def measurements(inputs, out):
    train, bc = inputs/'training_547', inputs/'bootstrap_503'
    result, progress = read(train/'training_result.json'), read(train/'progress.json')
    contract, summary = read(train/'task_contract.json'), read(train/'wandb-summary.json')
    bootstrap, dataset = read(bc/'result.json'), read(bc/'dataset_manifest.json')
    assert result['num_envs'] == 9216 and not result['completed']
    assert 'OutOfMemoryError' in result['error']
    assert contract['physics_hz'] == 120 and contract['policy_hz'] == 60
    assert contract['tactile_hz'] == 70 and not contract['self_collision']
    assert dataset['task_dim'] == 249 and dataset['tactile_training']
    assert bootstrap['best_update'] == 500
    # Old bootstrap result.json has a stale tactile_training=False flag.
    # Source identity, config.task_dim and dataset manifest resolve the variant.
    config = read(bc/'config.json')
    assert config['source_teacher_sha256'] == contract['source_sha256']

    val = [dict(update=0, **bootstrap['initial_validation'])]
    for line in (bc/'slurm-sonic-distill-503.out').read_text().splitlines():
        if line.startswith('DISTILLATION_VALIDATION '):
            val.append(json.loads(line[len('DISTILLATION_VALIDATION '):]))
    assert len(val) == 13
    selected = next(x for x in val if x['update'] == bootstrap['best_update'])
    events = list(train.glob('events.out.tfevents.*'))
    assert len(events) == 1
    assert hashlib.sha256(events[0].read_bytes()).hexdigest() == 'e1a91962dcce9f87879080fc2e3feae50c11568bdf08921dbef718d9b4c1eb5d', 'Event-file export differs from workstation checksum'
    acc = EventAccumulator(str(events[0]), size_guidance={'scalars': 0})
    acc.Reload()
    wanted = ['rewards/step', 'episode_final/ever_lifted_rate',
              'episode_final/any_goal_success_rate', 'losses/a_loss', 'losses/c_loss',
              'episode_final/robot_fall']
    series = {}
    for tag in wanted:
        rows = acc.Scalars(tag)
        values = np.array([[x.wall_time, x.step, x.value] for x in rows])
        if not np.isfinite(values).all():
            raise ValueError(f'Non-finite plotted data: {tag}')
        assert len(values) > 1000, (tag, len(values))
        series[tag] = values
        with (out/(tag.replace('/', '_')+'.csv')).open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['wall_time_unix_s', 'logged_cumulative_transitions', 'value'])
            writer.writerows(values)
    assert max(series['rewards/step'][:, 1]) > 1.23e9, 'Incomplete event-file copy'
    memory = [json.loads(line) for line in (train/'memory_trace.jsonl').read_text().splitlines()]
    mem = [r for r in memory if r['stage'] == 'train_epoch:after']
    assert mem[-1]['epoch'] >= 8300
    write(out/'memory_epoch_samples.json', mem)
    write(out/'bootstrap_validation.json', val)
    audit = dict(as_of='2026-09-14', training_run=547, training_snapshot=contract['source_commit'],
        final_frame=progress['frame'], final_epoch=progress['epoch'],
        resumed_frame=result['resume']['frame'], new_transitions=progress['frame']-result['resume']['frame'],
        final_training_return=float(summary['rewards/step']),
        final_training_lift_rate=float(summary['episode_final/ever_lifted_rate']),
        final_training_goal_rate=float(summary['episode_final/any_goal_success_rate']),
        final_training_robot_fall_rate=float(summary['episode_final/robot_fall']),
        maximum_logged_goal_rate=float(series['episode_final/any_goal_success_rate'][:, 2].max()),
        selected_bootstrap=selected, bootstrap_dataset=dataset,
        bootstrap_stale_flag='result.json tactile_training=False conflicts with config and dataset; identity checked from SHA/task_dim',
        plot_smoothing='Raw logged values plus trailing 100 logged updates; not an evaluation or ablation',
        checkpoint_archive_metadata=dict(latest=dict(epoch=8256, frame=1223688192),
            best=dict(epoch=8207, frame=1216462848), crc_checks_passed=True,
            checked_on='2026-09-14 via read-only zipfile/pickletools inspection'),
        input_sha256={str(p.relative_to(inputs)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(inputs.rglob('*')) if p.is_file()})
    write(out/'report_evidence.json', audit)
    return dict(audit=audit, result=result, progress=progress, contract=contract,
                summary=summary, bootstrap=bootstrap, dataset=dataset, val=val,
                selected=selected, series=series, memory=mem)


def figures(data, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'text.color': INK, 'axes.labelcolor': MUTED, 'axes.edgecolor': LINE,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False})
    val, selected = data['val'], data['selected']
    fig, ax = plt.subplots(figsize=(5.15, 2.9), layout='constrained')
    ax.plot([v['update'] for v in val], [v['arm_rmse_rad'] for v in val],
            color=TEAL, marker='o', ms=3, lw=1.8)
    ax.scatter([500], [selected['arm_rmse_rad']], color=ORANGE, s=38, zorder=4)
    ax.annotate(f"Selected at 500\n{selected['arm_rmse_rad']:.4f} rad",
                (500, selected['arm_rmse_rad']), xytext=(630, .23),
                fontsize=9, color=ORANGE, arrowprops=dict(arrowstyle='->', color=ORANGE))
    ax.set(xlabel='Supervised update', ylabel='Temporal-holdout arm RMSE [rad]',
           xlim=(-20, 1240), ylim=(0, .55))
    ax.grid(axis='y', color=LINE, lw=.6)
    save_plot(fig, out/'bootstrap_503')

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 2.65), layout='constrained')
    panels = [('rewards/step', 'Leader training return', 1., TEAL),
              ('episode_final/ever_lifted_rate', 'Training lift / robot-fall rates [%]', 100., BLUE),
              ('episode_final/any_goal_success_rate', 'Training episodes with a goal [%]', 100., ORANGE)]
    for ax, (tag, label, scale, color) in zip(axes, panels):
        rows = data['series'][tag]
        x, y = rows[:, 1]/1e9, rows[:, 2]*scale
        ax.plot(x, y, color=color, alpha=.23, lw=.45)
        ax.plot(x, moving_mean(y), color=color, lw=1.65)
        ax.set(xlabel='Cumulative whole-body transitions [B]', ylabel=label, xlim=(.078, 1.24))
        ax.set_xticks([.1, .4, .8, 1.2]); ax.grid(axis='y', color=LINE, lw=.6)
        if scale == 100:
            ax.set_ylim(-2, 102)
        if tag == 'episode_final/ever_lifted_rate':
            rows = data['series']['episode_final/robot_fall']
            ax.plot(rows[:, 1]/1e9, moving_mean(rows[:, 2]*100), color=ORANGE, lw=1.3,
                    label='Robot fall')
            ax.plot([], [], color=BLUE, label='Ever lifted')
            ax.legend(fontsize=8, loc='lower right', frameon=False)
        if tag.endswith('any_goal_success_rate') and y.max() == 0:
            ax.text(.52, .55, 'No recorded goal successes', transform=ax.transAxes,
                    ha='center', color=ORANGE, fontsize=9)
    save_plot(fig, out/'training_547')

    rows = data['memory']
    x = [r['elapsed_s']/3600 for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 3.25), layout='constrained')
    for key, label, color in [('device_used_gib', 'Total device usage', ORANGE),
                              ('reserved_gib', 'PyTorch reserved', BLUE),
                              ('allocated_gib', 'Live tensors after update', TEAL)]:
        ax.plot(x, [r[key] for r in rows], lw=1.3, color=color, label=label)
    capacity = rows[-1]['device_used_gib']+rows[-1]['device_free_gib']
    ax.axhline(capacity, color=ORANGE, ls='--', lw=.8, label='CUDA-reported capacity')
    ax.set(xlabel='Hours since training-loop start', ylabel='GPU memory [GiB]',
           xlim=(0, 10), ylim=(0, 52))
    ax.grid(axis='y', color=LINE, lw=.6)
    ax.legend(loc='lower left', fontsize=8, ncol=2, frameon=False)
    ax.annotate('OOM at next update', xy=(x[-1], rows[-1]['device_used_gib']),
                xytext=(6.7, 50), fontsize=9, color=ORANGE,
                arrowprops=dict(arrowstyle='->', color=ORANGE))
    save_plot(fig, out/'memory_547')

    # Equations remain vector glyphs when embedded into the PDF.
    equations = {
        'policy_equation': [r'$h_t=f_{\mathrm{SAPG}}(o_t^{\mathrm{task}},e_g,h_{t-1})$',
            r'$v_t=\mathrm{SiLU}(W_0[z_t^{\mathrm{stand}},s_t]+b_0+W_{\mathrm{task}}h_t)$',
            r'$q_t^{*\mathrm{body}}=q_0+\alpha\,D_{\mathrm{rest}}(v_t),\quad a_t^R=\mu_{\mathrm{finger}}(h_t)$'],
        'bootstrap_equation': [r'$\mathcal{L}_{\mathrm{BC}}=\mathcal{L}_{\mathrm{arm}}+\mathcal{L}_{\mathrm{finger}}'
            r'+\mathcal{L}_{\mathrm{other\ body}}+10\mathcal{L}_{\mathrm{standing}}$'],
        'bps_equation': [r'$d_i=\min_{p\in S}\|b_i-p\|_2,\quad i=1,\ldots,128$'],
    }
    for name, lines in equations.items():
        fig = plt.figure(figsize=(7.7 if name == 'policy_equation' else 5.5, .37*len(lines)+.10))
        for i, line in enumerate(lines):
            fig.text(.01, 1-(i+.78)/(len(lines)+.15), line, fontsize=12, color=INK)
        save_plot(fig, out/name)


class Report:
    def __init__(self, out):
        fonts = Path(matplotlib.get_data_path())/'fonts/ttf'
        for name, file in [('Body', 'DejaVuSans.ttf'), ('Bold', 'DejaVuSans-Bold.ttf')]:
            pdfmetrics.registerFont(TTFont(name, str(fonts/file)))
        pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold')
        self.out, self.page, self.charts = out, 0, []
        self.c = canvas.Canvas(str(out/'layout.pdf'), pagesize=(W, H))
        self.c.setTitle('G1 + Revo2: SONIC / SAPG implementation report')
        self.c.setAuthor('Konstantin Smirnov')

    def text(self, x, top, text, size=10, color=INK, bold=False):
        self.c.setFont('Bold' if bold else 'Body', size)
        self.c.setFillColor(colors.HexColor(color))
        self.c.drawString(x, H-top-size, text)

    def para(self, x, top, width, text, size=10, color=INK, max_height=None, leading=None):
        p = Paragraph(text, ParagraphStyle('body', fontName='Body', fontSize=size,
            leading=leading or size*1.42, textColor=colors.HexColor(color)))
        _, height = p.wrap(width, H)
        if max_height is not None and height > max_height+.1:
            raise ValueError(f'Page {self.page}: text exceeds {max_height} by {height-max_height}: {text}')
        p.drawOn(self.c, x, H-top-height)
        return height

    def rect(self, x, top, width, height, fill=LIGHT, stroke=LINE, radius=6):
        self.c.setFillColor(colors.HexColor(fill))
        self.c.setStrokeColor(colors.HexColor(stroke)); self.c.setLineWidth(.6)
        self.c.roundRect(x, H-top-height, width, height, radius, fill=1, stroke=1)

    def box(self, x, top, width, height, title, body, fill=LIGHT, color=INK, size=10):
        self.rect(x, top, width, height, fill)
        self.text(x+12, top+10, title, 11, color, True)
        self.para(x+12, top+29, width-24, body, size, color, max_height=height-38)

    def arrow(self, points, color=MUTED):
        self.c.setStrokeColor(colors.HexColor(color)); self.c.setFillColor(colors.HexColor(color))
        self.c.setLineWidth(1.25)
        p = self.c.beginPath(); p.moveTo(points[0][0], H-points[0][1])
        for x, t in points[1:]: p.lineTo(x, H-t)
        self.c.drawPath(p)
        end, prev = np.array(points[-1]), np.array(points[-2])
        direction = (end-prev)/np.linalg.norm(end-prev)
        side = np.array([-direction[1], direction[0]])
        p = self.c.beginPath(); p.moveTo(end[0], H-end[1])
        for a in (end-direction*6+side*2.5, end-direction*6-side*2.5): p.lineTo(a[0], H-a[1])
        p.close(); self.c.drawPath(p, fill=1, stroke=0)

    def chart(self, name, x, top, width, height):
        self.charts.append((self.page-1, name, pymupdf.Rect(x, top, x+width, top+height)))

    def begin(self, number, title, subtitle):
        if self.page: self.c.showPage()
        self.page += 1
        self.text(32, 18, f'IMPLEMENTATION REPORT  /  {DATE}', 8.4, TEAL, True)
        self.text(32, 39, title, 24, INK, True)
        self.para(33, 76, 775, subtitle, 10.2, MUTED, max_height=30)
        self.c.setStrokeColor(colors.HexColor(LINE)); self.c.line(32, 32, W-32, 32)
        self.text(32, H-24, 'G1 + Revo2  |  Measured implementation; not a deployment or success claim', 7.4, MUTED)
        self.text(755, H-24, f'{number} / 6', 8.2, MUTED)

    def table(self, x, top, widths, rows, row_height=28, size=9.5):
        for i, row in enumerate(rows):
            self.rect(x, top+i*row_height, sum(widths), row_height,
                      BLUE_BG if i == 0 else ('#FFFFFF' if i % 2 else LIGHT), radius=0)
            px = x
            for value, width in zip(row, widths):
                self.para(px+8, top+i*row_height+6, width-16,
                    f'<b>{value}</b>' if i == 0 else value,
                    size, max_height=row_height-7, leading=size*1.22)
                px += width

    def link(self, x, top, label, url, size=8.5):
        self.text(x, top, label, size, BLUE)
        width = pdfmetrics.stringWidth(label, 'Body', size)
        self.c.linkURL(url, (x, H-top-size-2, x+width, H-top+2), relative=0)

    def finish(self, pdf):
        self.c.showPage(); self.c.save()
        document = pymupdf.open(self.out/'layout.pdf')
        for index, name, rect in self.charts:
            with pymupdf.open(self.out/(name+'.pdf')) as chart:
                document[index].show_pdf_page(rect, chart, 0)
        document.set_metadata(dict(title='SONIC + SAPG: proposal implementation and measured results',
            author='Konstantin Smirnov', subject='BPS-128 + touch transfer into full-body G1; status 2026-09-14'))
        document.save(pdf, garbage=4, deflate=True)
        document.close()
        checks = []
        with pymupdf.open(pdf) as doc:
            assert len(doc) == 6
            content = '\n'.join(p.get_text() for p in doc)
            for required in ['78,544,896', '1,232,535,552', 'self-collision', '70 Hz',
                             'No online imitation loss', 'Bimanual', 'out of memory']:
                assert required in content, f'Missing claim: {required}'
            for page in doc:
                for block in page.get_text('dict')['blocks']:
                    if block['type'] != 0: continue
                    for line in block['lines']:
                        for span in line['spans']:
                            assert page.rect.contains(pymupdf.Rect(span['bbox'])), span['text']
                page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(
                    self.out/f'page_{page.number+1:02d}.png')
                checks.append(dict(page=page.number+1, text_characters=len(page.get_text()),
                                   links=len(page.get_links()), text_within_page=True))
        write(self.out/'document_checks.json', dict(pages=checks, required_claims_present=True,
                                                   pdf=str(pdf), vector_charts=len(self.charts)))


def compose(data, out, pdf):
    r = Report(out); a, dataset = data['audit'], data['dataset']
    r.begin(1, 'One policy for manipulation and the whole body',
        'Implemented route 2: reuse the trained SAPG skill and fine-tune SONIC itself. '
        'This supersedes the frozen-decoder / latent-adapter proposal.')
    r.rect(32, 111, 778, 31, AMBER_BG, AMBER_BG)
    r.text(44, 119, 'STATUS: RL implemented and run to 1.233B transitions; stopped by CUDA out of memory.', 10, ORANGE, True)
    r.text(34, 154, 'ACTUAL CONTROL GRAPH', 9, MUTED, True)
    r.box(34, 178, 187, 57, 'Body history · 930 values', 'Posture, motion, prior actions', BLUE_BG, BLUE, 9.5)
    r.box(34, 253, 187, 72, 'Standing reference · 64-D', 'Frozen SONIC encoder / FSQ.<br/>No future demonstration input.', BLUE_BG, BLUE, 9.5)
    r.box(294, 178, 245, 112, 'Trainable SONIC body decoder',
          'Task features enter its first hidden layer.<br/><br/>29 joint targets: both arms, waist and legs.', GREEN_BG, TEAL, 10)
    r.box(611, 190, 197, 84, '29 body commands', 'Joint-limit-normalized<br/>absolute position targets', GREEN_BG, TEAL, 10)
    r.box(34, 355, 187, 78, 'Manipulation input · 249-D', 'Object / goal / hand state<br/>+ BPS-128 + right touch', BLUE_BG, BLUE, 9.5)
    r.box(294, 355, 245, 78, 'Copied SAPG LSTM + MLP', 'Produces 512-D task features.<br/>Fine-tuned during whole-body RL.', GREEN_BG, TEAL, 10)
    r.box(611, 355, 197, 78, '6 right-finger commands', 'Copied finger head; source<br/>absolute-target pipeline', GREEN_BG, TEAL, 10)
    r.arrow([(221, 206), (294, 206)], BLUE)
    r.arrow([(221, 288), (253, 288), (253, 262), (294, 262)], BLUE)
    r.arrow([(539, 232), (611, 232)], TEAL)
    r.arrow([(221, 394), (294, 394)], BLUE)
    r.arrow([(539, 394), (611, 394)], TEAL)
    r.arrow([(416, 355), (416, 290)], TEAL)
    r.text(427, 314, 'Learned 512 → 2,048 projection', 9.4, TEAL)
    r.para(34, 450, 778,
        '<b>One motor owner:</b> a single actor emits 35 actions. The old seven-arm SAPG output is not '
        'another controller; it is used only as an offline teaching target. The copied critic gains body inputs. '
        'During RL, decoder, task projection, copied task features and finger head can all learn.', 10.5, max_height=52)
    r.para(34, 505, 778,
        '<b>Manipulation is connected to leg control, not to a footstep planner.</b> The robot starts beside the table '
        'with a standing reference. It can shift weight or step, but purposeful walking-to-grasp and disturbance '
        'recovery have not been validated.', 10.2, MUTED, max_height=44)

    r.begin(2, 'Reuse first, then learn with full-body physics',
        'The completed BPS + touch teacher is reused. Offline imitation initializes the changed body-action interface; '
        'online SAPG then trains the combined controller.')
    r.box(32, 117, 236, 77, '1 · Manipulation teacher 383',
        '17.364B-step BPS + 70 Hz touch policy.<br/>13 actions: 7 arm + 6 hand.', BLUE_BG, BLUE, 9.5)
    r.box(302, 117, 236, 77, '2 · Bootstrap 503',
        'Fit SONIC body outputs to saved labels.<br/>Select supervised update 500.', GREEN_BG, TEAL, 9.5)
    r.box(572, 117, 236, 77, '3 · Whole-body SAPG',
        'Migrate to 35 actions, then fine-tune.<br/>Later resumes retain both optimizers.', GREEN_BG, TEAL, 9.5)
    r.arrow([(268, 157), (302, 157)]); r.arrow([(538, 157), (572, 157)])
    r.text(34, 212, 'WHAT THE BOOTSTRAP ACTUALLY SAW', 10, INK, True)
    r.para(34, 235, 359,
        '<b>Four 30-second teacher clips at 60 Hz:</b> brush, eraser, hammer and spatula; 7,200 captured samples. '
        'The right-arm motion is lifted into a nominal standing body <b>kinematically</b>, not replayed through '
        'full-body contact physics.<br/><br/>'
        f'<b>{dataset["training_samples"]:,} training / {dataset["validation_samples"]:,} validation rows</b>, '
        'split within episodes with a 50-step gap. After recurrent burn-in, 1,668 holdout rows are scored. '
        'These are <b>not unseen objects</b>.<br/><br/>'
        'Separate real standing-physics rehearsal: 693 training / 247 validation rows at 50 Hz. '
        'All body histories retain SONIC’s 20 ms spacing.', 10, max_height=175)
    r.chart('bootstrap_503', 417, 208, 391, 224)
    r.para(429, 433, 369,
        f'<b>Selected arm-target RMSE: {data["selected"]["arm_rmse_rad"]:.4f} rad</b> '
        '(initial: 0.5094 rad). Selection combines arm, finger, other-body and standing errors. '
        'This measures target imitation—not grasping success.', 9.7, max_height=59)
    r.table(34, 424, [172, 86, 109], [
        ['Module', 'Bootstrap', 'Online RL'],
        ['SONIC body / task projection', 'Train', 'Train'],
        ['SAPG task / finger features', 'Frozen', 'Train'],
        ['Reference encoder / FSQ', 'Frozen', 'Frozen'],
    ], row_height=28, size=9)
    r.para(430, 504, 366,
        '<b>No online imitation loss is currently added.</b> Teacher/standing supervision is offline only; '
        'whole-body RL uses the original manipulation reward. This differs from the old proposal’s “SAPG + teacher loss” wording.',
        9.6, ORANGE, max_height=48)

    r.begin(3, 'Preserve the task; change the body interface',
        'The environment subclasses the actual Play2Perfect → BPS → touch task. Startup checks reject unintended '
        'reward, goal, observation, timing or object-bank changes.')
    r.table(32, 118, [147, 239], [
        ['Simulation component', 'Implemented setup'],
        ['Floating G1 + two Revo2', '51 joints / 68 rigid bodies; gravity on'],
        ['Policy / physics / touch', '60 Hz / 120 Hz / 70 Hz'],
        ['Controlled outputs', '29 body + 6 right hand = 35'],
        ['Left hand', 'Physical joints present, held neutral'],
        ['Object bank', '1,200 procedural tools; same bank SHA'],
        ['Right-hand drives', 'Source PD + five target followers'],
        ['Fabrics / PCA / arm torques', 'All disabled in this run'],
    ], row_height=31, size=9.1)
    r.text(439, 119, 'SHAPE AND TOUCH REACH THE POLICY', 10, INK, True)
    r.box(439, 143, 368, 86, 'BPS describes shape; keypoints describe the goal',
        '128 distances to normalized surface samples + centroid (3) + radius (1) = 132 features. '
        'The original four goal-alignment keypoints are retained separately.', BLUE_BG, BLUE, 10)
    r.chart('bps_equation', 449, 239, 344, 32)
    r.box(439, 289, 108, 76, 'PhysX · 120 Hz', 'Normal + friction<br/>contact reports', LIGHT, INK, 8.8)
    r.box(569, 289, 108, 76, 'Pad · 70 Hz', 'Crop / rotate /<br/>sample and hold', BLUE_BG, BLUE, 8.8)
    r.box(699, 289, 108, 76, 'Actor · 60 Hz', '5 pads ×<br/>5 channels', GREEN_BG, TEAL, 8.8)
    r.arrow([(547, 328), (569, 328)]); r.arrow([(677, 328), (699, 328)])
    r.para(439, 380, 368,
        '<b>Per pad:</b> normal force, two shear components, validity and packet age. '
        'Contacts are cropped to fingertip pads and transformed into sensor coordinates. '
        'Object, table and floor partners are included, without exposing partner identity.<br/><br/>'
        'The 25-value signal is a physics-based tactile proxy—not RGB, TacMap or a validated model of capacitive electronics. '
        'This run does not change contact materials.', 9.7, max_height=124)
    r.text(34, 386, 'OBSERVATION ACCOUNTING', 10, INK, True)
    r.para(34, 408, 376,
        '<b>Actor:</b> 92 task + 132 BPS + 25 touch = 249.<br/>'
        '<b>Body extension:</b> 930 history + 64 standing code = 994.<br/>'
        '<b>Total:</b> actor 1,243; critic 1,265, before SAPG coefficient conditioning.<br/><br/>'
        'Body targets are absolute; source finger smoothing, target coupling and action delay are preserved. '
        'Original arm-delta calculations do not write the body joints.', 9.7, max_height=119)
    r.rect(32, 529, 776, 28, AMBER_BG, AMBER_BG)
    r.text(42, 537, 'Known fidelity gap: self-collision is OFF; removing the fixed torso proxy does not preserve arm–torso collision pairs.', 8.7, ORANGE)

    r.begin(4, 'Online SAPG learns lifting, not yet reposing',
        'Run 547, September 13–14. These are logged training statistics from exploratory environments; '
        'no held-out evaluation or baseline-matched ablation was run.')
    final_lift = a['final_training_lift_rate']*100
    cards = [('1.233B', 'cumulative transitions'), (f'{a["final_training_return"]:.1f}', 'last leader training return'),
             (f'{final_lift:.1f}%', 'last training lift-rate window'),
             (f'{a["final_training_goal_rate"]*100:.1f}%', 'last training goal-rate window')]
    for i, (value, label) in enumerate(cards):
        x = 32+i*198
        r.rect(x, 116, 182, 62, GREEN_BG if i < 3 else AMBER_BG)
        r.text(x+12, 122, value, 23, TEAL if i < 3 else ORANGE, True)
        r.text(x+12, 153, label, 8.1, MUTED)
    r.chart('training_547', 30, 194, 783, 204)
    r.text(36, 400, 'Raw reward/lift/goal values + trailing 100-update means; robot fall shows the trailing mean. Rates are recent training-episode windows.', 7.9, MUTED)
    r.text(36, 413, 'Last window: 67% robot falls despite 91% lifting. These events can occur in the same episode; this is not stable reposing.', 8.3, ORANGE, True)
    r.text(34, 428, 'SAME MANIPULATION OBJECTIVE', 10, INK, True)
    r.para(34, 451, 372,
        'Approach progress, lifting, orientation/keypoint progress and goal bonuses; original arm/finger motion penalties. '
        '<b>No new balance reward or online teacher loss.</b><br/><br/>'
        '<b>Goal success:</b> all four keypoint errors &lt; 0.015 m; 10 accumulated near-goal control steps. '
        'Lifting once is not a reposing success. A body fall terminates the episode and loses future task reward.',
        9.7, max_height=102)
    r.text(439, 428, 'UNCHANGED SAPG TRAINER; NEW ACTOR INTERFACE', 10, INK, True)
    r.para(439, 451, 368,
        '6 exploration groups × 1,536 envs; horizon / recurrent sequence 16; minibatch 36,864; two mini-epochs. '
        'Actor LR 10⁻⁵; critic LR 10⁻⁴; γ = 0.99; λ = 0.95; PPO clip 0.1; mixed precision.<br/><br/>'
        'Existing leader/follower experience sharing and learned exploration embeddings are retained. '
        '<b>Best means highest leader training return</b>, not the best evaluated success rate.',
        9.7, max_height=102)

    r.begin(5, 'One-GPU training works; long-run memory does not',
        'Job 547 resumed at 78,544,896 transitions and completed 7,826 new updates on one RTX 6000 Ada. '
        'It failed at 05:55 Dubai time on September 14, after 9 h 57 min 57 s allocated wall time.')
    r.chart('memory_547', 27, 120, 474, 251)
    r.para(38, 379, 453,
        'Stage-level samples: every update initially, then every 16 updates. '
        '“Live” is measured after train_epoch, before outer-loop logging/cleanup. '
        'Reserved memory includes live allocations and cache; the device gap includes simulator/runtime allocations.',
        9, MUTED, max_height=48)
    r.box(521, 119, 287, 103, 'Confirmed failure: observation-batch copy',
        'SAPG torch.cat requested 1.37 GiB.<br/>Only 303.75 MiB free; 31.07 GiB live PyTorch allocations. '
        'OOM occurred while repeating observations across exploration groups.', AMBER_BG, ORANGE, 9.5)
    r.para(523, 238, 281,
        '<b>Why the early fit was insufficient:</b> device usage grew from ~26 to ~47 GiB. '
        'The failure site is known, but the cause of growing tensor lifetimes / workload peaks is not yet established. '
        'This is not proof of a specific leak or fragmentation bug.<br/><br/>'
        '<b>Earlier failures:</b> 517 OOM at 12,288 envs; 528 native segfault at 9,216. '
        'Periodic stack dumping was disabled as a suspected trigger before 547. No repeat segfault was observed in this run.',
        9.7, max_height=177)
    r.table(32, 438, [143, 307, 326], [
        ['Evidence', 'What is verified', 'What is not verified'],
        ['49 CPU tests / 4 preflight', 'Contracts, action/history handling, resume checks', 'Native stability or learned manipulation success'],
        ['Ground-contact check 515', 'Normal-force reconstruction error ≤ 1.91e-6 N', 'Real fingertip force calibration / all contacts'],
        ['Finite-speed safeguards', 'Affected env resets above 1,000 rad/s', 'Numerical cause; 2,385 outliers in run 547'],
    ], row_height=28, size=8.6)

    r.begin(6, 'What was delivered—and what remains',
        'Implementation follows the requested direct SONIC fine-tuning route. It is not a reproduction of the full '
        'GRAIL video-data pipeline or a completed bimanual locomotion policy.')
    r.table(32, 118, [197, 113, 466], [
        ['Proposal item', 'Status', 'Implemented mechanism / remaining limit'],
        ['Reuse the trained SAPG skill', 'Implemented', 'Teacher 383 → bootstrap 503 → one 35-action SONIC/SAPG actor'],
        ['Use the original manipulation task', 'Implemented', 'Source reward/goal/DR + BPS-128 + 70 Hz touch; contract checks'],
        ['Learn whole-body manipulation', 'Partial', 'Last training window: 91% lift / 67% robot fall / 0% goal'],
        ['Walk and stabilize while grasping', 'Not validated', 'Legs are controlled; standing reference, no footstep planner/curriculum'],
        ['Bimanual manipulation', 'Not implemented', 'Left fingers are neutral; 41-action extension still needs training/data'],
        ['Stable one-GPU long training', 'Unresolved', '9,216 envs ran nearly 10 hours, then CUDA OOM; no active job'],
    ], row_height=31, size=9.1)
    r.text(34, 351, 'NEXT WORK, IN ORDER', 10.5, INK, True)
    r.para(34, 375, 376,
        '<b>1.</b> Trace growing SAPG allocations; reduce unnecessary batch copies without changing learning math; '
        'test long-run memory with the full optimizer workload.<br/>'
        '<b>2.</b> Resume, then validate precise reposing and balance across objects, payloads and perturbations.<br/>'
        '<b>3.</b> Add varying approach distances and stepping requirements; then left-hand observations, commands '
        'and demonstrations for the bimanual stage.', 9.8, max_height=112)
    r.text(438, 351, 'RECOVERABLE STATE', 10.5, INK, True)
    r.para(438, 375, 368,
        '<b>Last completed:</b> 1,232,535,552 transitions / epoch 8,316.<br/>'
        '<b>Latest saved:</b> 1,223,688,192 / epoch 8,256.<br/>'
        '<b>Best-return saved:</b> 1,216,462,848 / epoch 8,207.<br/>'
        'Both checkpoint archives passed CRC checks. Complete saves include actor and critic optimizers; '
        'atomic latest saves occur every 64 updates. Source checkpoints remain unchanged.', 9.8, max_height=107)
    r.text(34, 500, 'AUDIT LINKS  ·  figures use exported raw logs; provenance and plotted CSV/JSON are supplied alongside the PDF', 8.4, MUTED)
    links = [(34, 522, 'Actor / critic implementation', CODE+'dextrah_lab/wholebody/sapg_network.py'),
             (241, 522, 'Environment and body actions', CODE+'dextrah_lab/tasks/g1_revo2_adept/g1_sonic_touch_env.py'),
             (461, 522, 'Trainer / source-task checks', CODE+'scripts/train_g1_sonic_sapg.py'),
             (666, 522, 'W&B run 547', WB),
             (34, 540, 'SONIC source', 'https://github.com/NVlabs/GR00T-WholeBodyControl'),
             (241, 540, 'GRAIL reference project', 'https://github.com/NVlabs/GRAIL'),
             (461, 540, 'Related diagnostic-crash report', 'https://github.com/python/cpython/issues/116008')]
    for item in links: r.link(*item, size=8.3)
    r.finish(pdf)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, default=PROJECT/'outputs/implementation_report_20260914/data')
    p.add_argument('--assets-output', type=Path, required=True)
    p.add_argument('--pdf', type=Path, required=True)
    p.add_argument('--research-directory', type=Path)
    args = p.parse_args()
    args.assets_output.mkdir(parents=True, exist_ok=True)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    data = measurements(args.inputs, args.assets_output)
    print('Raw measurements verified and exported', flush=True)
    figures(data, args.assets_output)
    compose(data, args.assets_output, args.pdf)
    if args.research_directory:
        research = args.research_directory
        research.mkdir(parents=True, exist_ok=True)
        for directory in (PROJECT/'docs', research):
            old = directory/'G1_SONIC_SAPG_Proposal.pdf'
            backup = directory/'G1_SONIC_SAPG_Proposal_20260912.pdf'
            if old.exists() and not backup.exists(): shutil.copy2(old, backup)
            if old.resolve() != args.pdf.resolve(): shutil.copy2(args.pdf, old)
        target = research/'G1_SONIC_SAPG_Implementation_Report.pdf'
        if target.resolve() != args.pdf.resolve(): shutil.copy2(args.pdf, target)
    print(f'Created and checked six-page report: {args.pdf}', flush=True)


if __name__ == '__main__':
    main()
