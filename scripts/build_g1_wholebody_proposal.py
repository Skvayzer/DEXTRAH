#!/usr/bin/env python3
"""One-page, evidence-backed route-2 proposal. No invented rollout results.

Uses exported run-485 logs, run-482 standing states, run-491 probe states and
original G1/Revo2 CAD. The render depicts the initial scene, not a grasp.
Install reportlab, matplotlib, pymupdf, scipy and trimesh in an isolated env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
import numpy as np
import pymupdf
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from scipy.spatial.transform import Rotation
import trimesh

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from dextrah_lab.wholebody.kinematics import UrdfKinematics, pose_matrix

NAVY = '#183347'
INK = '#243D4B'
TEAL = '#008D83'
BLUE = '#3972AC'
ORANGE = '#BD601E'
GRAY = '#61717D'
LINE = '#DDE5E9'
LIGHT = '#F5F8FA'
W, H = 841.89, 595.28


def read_json(path):
    return json.loads(Path(path).read_text())


def save_plot(fig, stem):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(stem.with_suffix('.'+ext), dpi=240, facecolor='white')
    plt.close(fig)


def plots(data, out):
    result = read_json(data/'bootstrap_485/result.json')
    records = [json.loads(line) for line in (data/'bootstrap_485/metrics.jsonl').read_text().splitlines()]
    validation = [dict(update=0, **result['initial_validation'])]
    prefix = 'DISTILLATION_VALIDATION '
    for line in (data/'bootstrap_485/slurm-sonic-distill-485.out').read_text().splitlines():
        if line.startswith(prefix):
            validation.append(json.loads(line[len(prefix):]))
    assert [v['update'] for v in validation] == list(range(0, 1201, 100))
    best = next(v for v in validation if v['update'] == result['best_update'])
    assert result['fullbody_rl_updates'] == 0 and result['source_checkpoints_unchanged']

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
        'axes.labelcolor':GRAY,'text.color':INK,'axes.edgecolor':LINE,
        'xtick.color':GRAY,'ytick.color':GRAY,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(3.77, 1.86))
    fig.subplots_adjust(left=.16, right=.975, top=.96, bottom=.27)
    ax.semilogy([r['update'] for r in records],[r['arm_rmse_rad'] for r in records],
                color=BLUE,alpha=.7,lw=1,label='Training minibatches')
    ax.semilogy([r['update'] for r in validation],[r['arm_rmse_rad'] for r in validation],
                color=TEAL,lw=1.7,marker='o',ms=2.4,label='Temporal holdout')
    ax.axvline(result['best_update'],color=ORANGE,lw=.8,ls='--')
    ax.scatter([best['update']],[best['arm_rmse_rad']],s=24,color=ORANGE,zorder=5)
    ax.annotate(f"Selected: {best['arm_rmse_rad']:.3f} rad",(best['update'],best['arm_rmse_rad']),
        xytext=(540,.42),fontsize=7.5,color=ORANGE,
        arrowprops=dict(arrowstyle='-',color=ORANGE,lw=.7))
    ax.set(xlabel='Supervised updates (not RL steps)',ylabel='Arm target RMSE [rad]',
        xlim=(-25,1225),ylim=(.006,2.))
    ax.set_xticks([0,400,800,1200]); ax.grid(axis='y',which='major',color=LINE,lw=.6)
    ax.legend(loc='lower left',fontsize=6.5,frameon=False,handlelength=1.5,borderpad=.1)
    save_plot(fig,out/'distillation_learning')

    standing = np.load(data/'standing_482/trace.npz',allow_pickle=False)
    validation_stand = read_json(data/'standing_482/validation.json')
    z = standing['root_state'][...,2]
    assert validation_stand['standing_probe_passed']
    t = (np.arange(len(z))+1)/50.
    progress=[]
    for line in (data/'contact_491/slurm-fullbody-probe-491.out').read_text().splitlines():
        if line.startswith('FULLBODY_PROBE {'):
            progress.append(json.loads(line[len('FULLBODY_PROBE '):]))
    rejection = np.load(data/'contact_491/rejected_action.npz',allow_pickle=False)
    stop = float(rejection['step'])/50.
    contact_t = [r['simulated_s'] for r in progress]+[stop]
    contact_z = [r['min_height_m'] for r in progress]+[float(rejection['root_state'][:,2].min())]
    fig, ax = plt.subplots(figsize=(3.25,1.86))
    fig.subplots_adjust(left=.18,right=.975,top=.96,bottom=.27)
    ax.fill_between(t,z.min(1),z.max(1),color=TEAL,alpha=.2,lw=0)
    ax.plot(t,z.min(1),color=TEAL,lw=1.4,label='Standing only (482 / student 479)')
    ax.plot(contact_t,contact_z,color=ORANGE,lw=1.4,marker='o',ms=2.5,
        label='Object task (491 / student 485)')
    ax.plot(stop,contact_z[-1],marker='x',color=ORANGE,ms=5)
    ax.annotate(f'Command guard\nat {stop:.2f} s',(stop,contact_z[-1]),
        xytext=(7,.62),fontsize=7,color=ORANGE,
        arrowprops=dict(arrowstyle='-',color=ORANGE,lw=.7))
    # Include the rejected final state: clipping it out would hide the drift.
    lower=min(.45,min(contact_z)-.025)
    ax.set(xlabel='Simulated time [s]',ylabel='Pelvis height [m]',
        xlim=(0,20),ylim=(lower,.79))
    ax.set_xticks([0,5,10,15,20]);ax.set_yticks([.5,.6,.7])
    ax.grid(axis='y',color=LINE,lw=.6)
    ax.legend(loc='lower right',frameon=False,fontsize=5.9,handlelength=1.3,borderpad=.1)
    save_plot(fig,out/'closed_loop_validation')
    (out/'plotted_measurements.json').write_text(json.dumps(dict(
        validation=validation,selected=best,contact_time_s=contact_t,
        contact_min_pelvis_m=contact_z,standing_source='482, student 479, 200 Hz physics',
        contact_source='491, student 485, 400 Hz physics, no replication',
        comparisons_are_diagnostics_not_controlled_ablation=True),indent=2)+'\n')
    return best,stop


def render_initial_scene(inputs,out):
    """CPU orthographic CAD rendering, no generated or successful-grasp imagery."""
    path=PROJECT/'outputs/teacher_transfer_inputs/g1_29dof_mode_15_brainco_hand.urdf'
    tree=UrdfKinematics(path)
    xml=ET.parse(path).getroot()
    names=read_json(inputs/'data/contact_491/runtime_motors.json')['joint_names']
    initial=np.load(PROJECT/'outputs/live_task_probe_487/initial_task_state.npz',allow_pickle=False)
    positions=dict(zip(names,initial['joint_pos'][0]))
    root=np.eye(4);root[2,3]=.76
    triangles=[];base_colors=[]

    def add(mesh,transform,color):
        vertices=trimesh.transform_points(np.asarray(mesh.vertices),transform)
        triangles.append(vertices[np.asarray(mesh.faces)])
        base_colors.append(np.broadcast_to(to_rgb(color),(len(mesh.faces),3)).copy())

    for link in xml.findall('link'):
        name=link.get('name')
        transform=root@tree.transform(name,positions,1)[0]
        color=TEAL if name.startswith('right_') and any(p in name for p in
            ('shoulder','elbow','wrist','base','thumb','index','middle','ring','pinky')) else '#607481'
        if 'head' in name or name=='torso_link':color='#344D5E'
        for visual in link.findall('visual'):
            element=visual.find('geometry/mesh')
            if element is None:continue
            mesh=trimesh.load(inputs/'assets/robot'/element.get('filename'),force='mesh',process=False)
            mesh.apply_scale(np.fromstring(element.get('scale','1 1 1'),sep=' '))
            local=np.eye(4);origin=visual.find('origin')
            if origin is not None:
                local[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                local[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            add(mesh,transform@local,color)
    progress=read_json(inputs/'data/contact_491/progress.json')
    alignment=np.asarray(progress['live_task']['source_scene_alignment'])
    source=np.load(PROJECT/'outputs/teacher_transfer_inputs/capture_459_trace.npz',allow_pickle=False)
    for name,color in [('table','#B9C5CC'),('object','#EE9B31')]:
        mesh=trimesh.load(inputs/f'assets/{name}.glb',force='mesh',process=False)
        add(mesh,alignment@pose_matrix(source[name][0]),color)
    triangles=np.concatenate(triangles)
    bases=np.concatenate(base_colors)
    # Global triangle painter ordering handles robot/table occlusions between
    # links, unlike separate per-link 3-D axes artists. Original faces retained.
    eye=np.array([2.8,-3.3,1.95]);target=np.array([.16,-.03,.70])
    view=eye-target;view/=np.linalg.norm(view)
    right=np.cross([0.,0.,1.],view);right/=np.linalg.norm(right)
    up=np.cross(view,right)
    basis=np.column_stack((right,up,view))
    projected=(triangles-target)@basis
    normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    light=np.array([.35,-.5,1.]);light/=np.linalg.norm(light)
    shade=.47+.53*np.clip(normal@light,0,1)
    shades=np.clip(bases*shade[:,None],0,1)
    order=np.argsort(projected[:,:,2].mean(1))
    fig,ax=plt.subplots(figsize=(4.8,4.7),dpi=240)
    fig.subplots_adjust(0,0,1,1)
    ax.add_collection(PolyCollection(projected[order,:,:2],facecolors=shades[order],
        edgecolors='none',antialiased=False,rasterized=True))
    xy=projected[:,:,:2].reshape(-1,2)
    lo,hi=xy.min(0),xy.max(0);center=(lo+hi)/2
    span=max((hi-lo)[0]/(4.8/4.7), (hi-lo)[1])*1.08
    ax.set_xlim(center[0]-span*(4.8/4.7)/2,center[0]+span*(4.8/4.7)/2)
    ax.set_ylim(center[1]-span/2,center[1]+span/2)
    ax.set_aspect('equal');ax.axis('off')
    fig.savefig(out/'g1_revo2_initial_scene.png',dpi=240,facecolor='white')
    plt.close(fig)
    (out/'render_provenance.json').write_text(json.dumps(dict(
        original_urdf=str(path),urdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        initial_measured_joint_positions='fullbody_probe_487/initial_task_state.npz',
        joint_names=names,root_pose_xyz_wxyz=[0,0,.76,1,0,0,0],
        source_scene_alignment=alignment.tolist(),triangles_rendered=len(triangles),
        rendering='CPU orthographic CAD re-render of initial setup',
        successful_grasp_depicted=False,physics_rollout_image=False),indent=2)+'\n')


def make_page(out,pdf,best,stop):
    fonts=Path(matplotlib.get_data_path())/'fonts/ttf'
    for family,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(family,str(fonts/file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold')
    c=canvas.Canvas(str(out/'layout.pdf'),pagesize=(W,H))
    c.setTitle('SONIC + SAPG: standing manipulation — implementation and proposal')
    c.setAuthor('Konstantin Smirnov')

    def text(x,y,value,size=9,color=INK,bold=False):
        c.setFillColor(colors.HexColor(color));c.setFont('Bold' if bold else 'Body',size)
        c.drawString(x,y,value)

    def paragraph(x,top,width,value,size=8.4,color=INK,max_height=None,leading=None):
        style=ParagraphStyle('p',fontName='Body',fontSize=size,leading=leading or size*1.42,
            textColor=colors.HexColor(color))
        p=Paragraph(value,style);_,height=p.wrap(width,H)
        if max_height is not None and height>max_height:
            raise ValueError(f'Text overflow ({height}>{max_height}): {value}')
        p.drawOn(c,x,top-height)
        return height

    def rect(x,y,w,h,fill=LIGHT,edge=LINE,radius=6):
        c.setFillColor(colors.HexColor(fill));c.setStrokeColor(colors.HexColor(edge));c.setLineWidth(.65)
        c.roundRect(x,y,w,h,radius,stroke=1,fill=1)

    def box(x,y,w,h,title,body,fill=LIGHT,color=INK):
        rect(x,y,w,h,fill)
        text(x+9,y+h-14,title,8.7,color,True)
        paragraph(x+9,y+h-20,w-18,body,7.6,color,max_height=h-22,leading=10)

    def arrow(points,color=GRAY,dashed=False):
        c.setStrokeColor(colors.HexColor(color));c.setFillColor(colors.HexColor(color));c.setLineWidth(1.1)
        c.setDash(3,2) if dashed else c.setDash()
        p=c.beginPath();p.moveTo(*points[0])
        for point in points[1:]:p.lineTo(*point)
        c.drawPath(p)
        c.setDash()
        end=np.array(points[-1]);delta=end-np.array(points[-2]);delta=delta/np.linalg.norm(delta)
        side=np.array([-delta[1],delta[0]])
        a=end-delta*5+side*2.2;b=end-delta*5-side*2.2
        p=c.beginPath();p.moveTo(*end);p.lineTo(*a);p.lineTo(*b);p.close();c.drawPath(p,fill=1,stroke=0)

    text(28,574,'RESEARCH PROPOSAL  /  12 SEPTEMBER 2026',8.2,TEAL,True)
    text(28,548,'Teaching SONIC to manipulate while standing',23,NAVY,True)
    text(28,531,'Reuse a trained SAPG skill in one full-body G1 + Revo2 controller.  One RTX 6000 Ada, 48 GB.',9.6,GRAY)
    rect(28,504,786,18,'#EDF7F4','#EDF7F4',4)
    text(37,510,'IMPLEMENTED: INITIAL DISTILLATION     |     NOT YET STARTED: FULL-BODY SAPG RL',8.1,TEAL,True)

    # Architecture: every motor has one owner; no separate teacher arm writer.
    rect(28,291,544,203,'#FFFFFF')
    text(41,475,'01  One student, two pretrained sources',11,NAVY,True)
    box(41,418,143,39,'Body history','Joints, velocity, gravity')
    box(41,367,143,40,'SONIC reference encoder','Standing ref. → 64 tokens',fill='#EEF3F8',color=BLUE)
    box(217,411,167,48,'SONIC body decoder','All 10.18M weights trainable',fill='#E9F5F1',color=TEAL)
    box(427,411,128,48,'29 body targets','Arms, waist and legs',fill='#E9F5F1',color=TEAL)
    box(41,311,143,44,'Current task input','Object / goal / hand + BPS-128')
    box(217,311,167,44,'Copied SAPG LSTM / MLP','Saved 8B BPS teacher features',fill='#EEF3F8',color=BLUE)
    box(427,311,128,44,'Copied finger head','6 right-hand targets',fill='#EEF3F8',color=BLUE)
    arrow([(184,438),(217,438)])
    arrow([(184,387),(200,387),(200,424),(217,424)],BLUE)
    arrow([(384,435),(427,435)],TEAL)
    arrow([(184,333),(217,333)])
    arrow([(384,333),(427,333)],BLUE)
    arrow([(300,355),(300,410)],TEAL)
    text(310,382,'Learned task',7.7,TEAL)
    text(310,372,'projection',7.7,TEAL)
    text(41,297,'35 actions • no separate SAPG arm override • copied task/finger weights frozen only during bootstrap',7.3,GRAY)

    rect(586,291,228,203,'#FFFFFF')
    text(599,475,'02  Full-physics setup',11,NAVY,True)
    c.drawImage(str(out/'g1_revo2_initial_scene.png'),596,311,width=208,height=158,
        preserveAspectRatio=True,anchor='c',mask='auto')
    text(599,302,'Initial CAD scene—not a successful grasp.',7.4,GRAY)

    # Actual vector charts are inserted below after layout generation.
    rect(28,99,276,180,'#FFFFFF')
    text(41,261,'03  Measured skill transfer',10.8,NAVY,True)
    text(41,247,'4 × 30 s clips; brush, eraser, hammer, spatula',7.7,GRAY)
    paragraph(41,120,248,'Temporal split + gap; synthetic standing-body bootstrap.<br/>'
        '1,359 scored holdout samples. Error ≠ grasp success.',7.5,GRAY,max_height=22,leading=10)

    rect(315,99,240,180,'#FFFFFF')
    text(328,261,'04  Closed-loop evidence',10.8,NAVY,True)
    text(328,247,'Standing passes; manipulation does not yet.',7.7,GRAY)
    paragraph(328,120,214,'Min. over 4 envs; different settings—not an ablation.<br/>'
        '20 s standing passed; object task stopped at 3.94 s.',7.4,GRAY,max_height=22,leading=10)

    rect(567,99,247,180,'#F6F8FA')
    text(580,261,'05  Proposed next stages',10.8,NAVY,True)
    stages=[('1','Validate loaded physics',
        'Resolve contact instability; add goal/reset logic.'),
        ('2','Fine-tune the same student with SAPG',
        '35-action actor/critic + teacher loss; learn balance.'),
        ('3','Integrate and size the complete workload',
        '70 Hz tactile; benchmark rollouts + optimizer on 48 GB.'),
        ('4','Expand after right-hand validation',
        'Disturbances and payloads, then two hands: 41 actions.')]
    for i,(number,title,body) in enumerate(stages):
        y=235-i*34
        c.setFillColor(colors.HexColor(TEAL));c.circle(587,y+3,7.5,fill=1,stroke=0)
        text(584.5,y+.4,number,7.5,'#FFFFFF',True)
        text(602,y+4,title,8.1,INK,True)
        paragraph(602,y-1,199,body,7.1,GRAY,max_height=23,leading=9.5)

    paragraph(28,86,786,
        '<b>Why this route:</b> SAPG provides manipulation skill; SONIC provides the body-control initialization. '
        'Imitation transfers arm targets and finger commands, but online full-body RL must learn balance and load transfer. '
        '<b>Both hands, feet, gravity and self-contact stay in physics; no fabrics/PCA.</b>',8.6,INK,max_height=27,leading=12)
    text(28,47,'Evidence: 1,200 supervised updates; 20 targeted tests passed. No validated standing-manipulation success rate yet.',8,ORANGE)
    c.setStrokeColor(colors.HexColor(LINE));c.line(28,37,814,37)
    text(28,24,'Sources: run 485 (distillation), 482 (standing), 491 (contact); code 112d92b. Original checkpoints unchanged.',7.1,GRAY)
    text(28,12,'Current bootstrap uses the 8B BPS-only SAPG teacher. Whole-body tactile and optimizer-inclusive GPU capacity remain unvalidated.',6.9,GRAY)
    text(684,24,'CODE',7.2,BLUE,True)
    text(730,24,'W&B RUN',7.2,BLUE,True)
    c.linkURL('https://github.com/Skvayzer/DEXTRAH/tree/feature/g1-wholebody-sapg',(682,20,718,34),relative=0)
    c.linkURL('https://wandb.ai/skvayzer/adept/runs/3kzird8t',(728,20,789,34),relative=0)
    c.showPage();c.save()

    doc=pymupdf.open(out/'layout.pdf')
    page=doc[0]
    # ReportLab bottom-left -> MuPDF top-left rectangles. These retain vector
    # text and axes rather than rasterizing the complete document.
    for name,x,y,w,h in [('distillation_learning',30,124,272,121),
                         ('closed_loop_validation',317,124,236,121)]:
        chart=pymupdf.open(out/f'{name}.pdf')
        page.show_pdf_page(pymupdf.Rect(x,H-y-h,x+w,H-y),chart,0)
        chart.close()
    doc.set_metadata({'title':'SONIC + SAPG: standing manipulation — proposal and measured progress',
        'author':'Konstantin Smirnov','subject':'Initial distillation complete; full-body SAPG RL not started'})
    doc.save(pdf,garbage=4,deflate=True)
    doc.close()
    check=pymupdf.open(pdf)
    assert len(check)==1
    text_content=check[0].get_text()
    for required in ['35 actions','NOT YET STARTED','No validated standing-manipulation','8B BPS-only']:
        if required not in text_content:raise AssertionError(f'Missing document claim: {required}')
    for block in check[0].get_text('dict')['blocks']:
        if block['type']==0:
            for line in block['lines']:
                for span in line['spans']:
                    r=pymupdf.Rect(span['bbox'])
                    if not check[0].rect.contains(r):raise AssertionError(f'Text outside page: {span["text"]}')
    check[0].get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False).save(out/'proposal_preview.png')
    check.close()
    (out/'document_checks.json').write_text(json.dumps(dict(pages=1,required_claims_present=True,
        text_within_page=True,pdf=str(pdf),selected_arm_rmse_rad=best['arm_rmse_rad'],
        object_probe_duration_s=stop,render_is_initial_scene=True),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,default=PROJECT/'outputs/proposal_report')
    p.add_argument('--assets-output',type=Path,required=True)
    p.add_argument('--pdf',type=Path,required=True)
    args=p.parse_args()
    args.assets_output.mkdir(parents=True,exist_ok=True)
    args.pdf.parent.mkdir(parents=True,exist_ok=True)
    best,stop=plots(args.inputs/'data',args.assets_output)
    print('Measured-result figures rendered',flush=True)
    render_initial_scene(args.inputs,args.assets_output)
    print('Initial G1/Revo2 CAD scene rendered',flush=True)
    make_page(args.assets_output,args.pdf,best,stop)
    print(f'Created and checked: {args.pdf}',flush=True)


if __name__=='__main__':
    main()
