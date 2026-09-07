#!/usr/bin/env python3
"""Render actual offline Revo2 data and explicitly labeled PCA illustrations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_pca_presentation import load_npz

W, H, FPS = 1920, 1080, 30
BG, INK, GRAY = (247, 249, 252), (25, 51, 72), (91, 106, 123)
TEAL, PURPLE, ORANGE = (0, 143, 149), (119, 85, 190), (218, 130, 40)


class HandRenderer:
    def __init__(self, urdf_path):
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Use the user's Slurm GPU allocation")
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        import pyrender
        from pyrender.platforms import egl
        import trimesh
        import yourdfpy
        import torch
        from dextrah_lab.retargeting import Revo2Kinematics, REVO2_RIGHT_ACTUATED_JOINTS
        torch.set_num_threads(2)
        candidates = [i for i, device in enumerate(egl.query_devices()) if b"EGL_NV_device_cuda" in
            (egl._eglQueryDeviceStringEXT(device._display, 0x3055) or b"")]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one accessible NVIDIA device, got {candidates}")
        os.environ["EGL_DEVICE_ID"] = str(candidates[0])
        self.torch, self.hand, self.names = torch, Revo2Kinematics(urdf_path), REVO2_RIGHT_ACTUATED_JOINTS
        self.urdf = yourdfpy.URDF.load(urdf_path)
        self.scene = pyrender.Scene(bg_color=np.array([*BG, 255])/255, ambient_light=[.40]*3)
        self.handles = []
        for node in self.urdf.scene.graph.nodes_geometry:
            transform, geometry_name = self.urdf.scene.graph.get(node)
            link = self.urdf.scene.graph.transforms.parents[node]
            while link not in self.urdf.link_map:
                link = self.urdf.scene.graph.transforms.parents[link]
            mesh = self.urdf.scene.geometry[geometry_name].copy()
            mesh.apply_transform(np.linalg.inv(self.urdf.get_transform(link)) @ transform)
            color = [.20, .27, .34, 1] if "base" in link else [.72, .77, .82, 1]
            material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color, metallicFactor=.1,
                roughnessFactor=.7, doubleSided=True)
            handle = self.scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))
            self.handles.append((link, handle))
        eye, target = np.array([.30, .22, .27]), np.array([.03, .0, .083])
        z = eye - target; z /= np.linalg.norm(z)
        x = np.cross([0, 0, 1], z); x /= np.linalg.norm(x)
        pose = np.eye(4); pose[:3,:3] = np.column_stack((x, np.cross(z,x), z)); pose[:3,3] = eye
        self.scene.add(pyrender.OrthographicCamera(xmag=.125, ymag=.133, znear=.01, zfar=3), pose=pose)
        self.scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.4), pose=pose)
        fill = pose.copy(); fill[:3,3] = [-.2, -.25, .3]
        self.scene.add(pyrender.PointLight(color=np.ones(3), intensity=.08), pose=fill)
        self.renderer = pyrender.OffscreenRenderer(640, 680)
        self.max_fk_error = 0.
        self.frames_checked = 0
        print("PCA_RENDER_DEVICE", candidates[0], os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

    def render(self, q):
        if np.any(q < self.hand.lower.numpy()-1.e-8) or np.any(q > self.hand.upper.numpy()+1.e-8):
            raise ValueError("Presentation pose outside hand joint limits")
        self.urdf.update_cfg(dict(zip(self.names, q)))
        # Independent differentiable FK checks both mimic and palm-frame placement.
        expected = self.hand.fingertip_positions(self.torch.tensor(q, dtype=self.torch.float64)).numpy()
        actual = np.stack([self.urdf.get_transform(link)[:3,3] for link in self.hand.fingertip_link_names])
        err = float(np.abs(expected-actual).max())
        if err > 1.e-7:
            raise RuntimeError(f"Rendered URDF differs from retargeting FK: {err}")
        self.max_fk_error = max(self.max_fk_error, err)
        self.frames_checked += 1
        for link, handle in self.handles:
            self.scene.set_pose(handle, self.urdf.get_transform(link))
        rgb, _ = self.renderer.render(self.scene)
        return Image.fromarray(rgb)


class Presentation:
    def __init__(self, root):
        self.root = root
        self.info = json.loads((root / "data" / "measurements.json").read_text())
        self.a = load_npz(root / "data" / "analysis.npz")
        self.hr = HandRenderer(self.info["hand_urdf"])
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        self.fonts = {size: ImageFont.truetype(font, size) for size in (20, 23, 26, 30, 34)}
        self.title_font = ImageFont.truetype(bold, 42)
        self.outputs = []

    def canvas(self, title, subtitle, headings, footer, notes=None):
        image = Image.new("RGB", (W,H), BG)
        draw = ImageDraw.Draw(image)
        draw.text((50, 30), title, font=self.title_font, fill=INK)
        draw.text((50, 91), subtitle, font=self.fonts[26], fill=GRAY)
        for i,(heading,color) in enumerate(headings):
            draw.line((i*640+35,155,i*640+605,155), fill=color,width=5)
            draw.text((i*640+35,177),heading,font=self.fonts[30],fill=color)
        draw.line((40, 970, 1880, 970), fill=(211, 219, 228),width=2)
        draw.text((50, 995), footer, font=self.fonts[23], fill=GRAY)
        if notes:
            for i,note in enumerate(notes):
                draw.text((i*640+35, 930),note,font=self.fonts[23],fill=GRAY)
        return image

    def panels(self, canvas, configurations):
        for i,q in enumerate(configurations):
            if q is not None:
                canvas.paste(self.hr.render(q), (640*i, 235))
        return canvas

    def video(self, name, frames, description):
        path = self.root / "videos" / name
        if path.exists():
            raise FileExistsError(path)
        encoder = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-n",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{W}x{H}",
            "-framerate", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-crf", "18",
            "-preset", "fast", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)
        n = 0
        try:
            for frame in frames:
                encoder.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
                n += 1
                if n % 150 == 0:
                    print("PCA_VIDEO_FRAME", name, n, flush=True)
        finally:
            encoder.stdin.close()
            code = encoder.wait(timeout=60)
        if code:
            raise RuntimeError(f"Video encoding failed: {path}")
        self.outputs.append(dict(file=str(path.relative_to(self.root)), frames=n, fps=FPS,
                                 seconds=n/FPS, description=description))
        print("PCA_VIDEO_PASS", name, n, flush=True)

    def retargeting(self):
        for clip, entry in enumerate(self.info["motions"]):
            a = load_npz(self.root / "data" / entry["file"])
            for f in range(len(a["gamma"])):
                canvas = self.canvas("From human grasping to a robot-motion prior",
                    f"{entry['label']}  |  source: {entry['subject']} / {entry['capture']}  |  frame {int(a['frame_indices'][f])}",
                    [("Recorded human motion", ORANGE), ("Offline retargeting (6 motors)", TEAL), ("Frozen PCA reconstruction", PURPLE)],
                    "Matched DexYCB frames (CC BY-NC 4.0). Robot panels are kinematic replay, not RL or physical grasp tests.",
                    ["Valid source frames shown at 0.5× speed", f"Human weight γ = {a['gamma'][f]:.2f}", "5 components; joint-limit clipping shown"])
                rgb = Image.open(str(a["rgb_paths"][f])).convert("RGB")
                rgb.thumbnail((620, 620))
                canvas.paste(rgb, ((640-rgb.width)//2, 235+(680-rgb.height)//2))
                self.panels(canvas, [None, a["joint_positions"][f], a["reconstructed"][f]])
                if f in (0,len(a["gamma"])//2,len(a["gamma"])-1):
                    canvas.save(self.root / "stills" / f"retarget_{clip+1:02d}_{f:03d}.png")
                repeats = 2 + (FPS if f in (0, len(a["gamma"])-1) else 0)
                for _ in range(repeats):
                    yield canvas

    def component_sweeps(self):
        mean, components = self.a["mean"], self.a["components"]
        for pc, (low,high) in enumerate(self.a["sweeps"]):
            for f in range(180):
                scalar = low + (high-low) * .5*(1-np.cos(2*np.pi*f/179))
                q = mean + scalar*components[pc]
                canvas = self.canvas("What does one PCA motion direction do?",
                    f"PC {pc+1}  |  {100*self.a['explained_variance_ratio'][pc]:.3f}% of robot-configuration variance  |  scalar = {scalar:+.3f} rad",
                    [("Mean hand posture", GRAY), (f"Vary PC {pc+1} only", PURPLE), ("Coordinated motor change", TEAL)],
                    "Illustrative single-component sweep, not a recorded human trial. Range stays within joint limits and empirical 5–95% coordinates.")
                self.panels(canvas, [mean, q, None])
                draw = ImageDraw.Draw(canvas)
                labels = ["Thumb rotation", "Thumb flexion", "Index", "Middle", "Ring", "Little"]
                for j,label in enumerate(labels):
                    y = 285 + j*87
                    draw.text((1320,y),label,font=self.fonts[23],fill=GRAY)
                    center, length = 1580, int(200*scalar*components[pc,j])
                    draw.line((center,y+40,center+length,y+40),fill=PURPLE,width=12)
                    draw.line((center,y+25,center,y+55),fill=GRAY,width=2)
                    draw.text((1740,y+12),f"{scalar*components[pc,j]:+.3f}",font=self.fonts[23],fill=INK)
                if f == 90:
                    canvas.save(self.root / "stills" / f"pca_component_{pc+1:02d}.png")
                yield canvas

    def presets(self):
        modes = ["power", "precision_pinch", "precision_tripod"]
        arrays = [load_npz(self.root / "data" / f"preset_{mode}.npz") for mode in modes]
        assert len({len(a["gamma"]) for a in arrays}) == 1
        n = len(arrays[0]["gamma"])
        for f in range(n):
            canvas = self.canvas("Different closure presets through the same frozen PCA",
                "Same human input; different offline closure masks. The full 501-capture PCA artifact is unchanged.",
                [("Power / full-hand closure", TEAL), ("Two-finger pinch extension", PURPLE), ("Tripod extension", ORANGE)],
                "Pinch and tripod are inspection extensions, not extra training families or measured grasp success. Kinematic replay only.",
                [f"γ = {arrays[0]['gamma'][f]:.2f}", "Thumb + index closure attractors", "Thumb + index + middle attractors"])
            self.panels(canvas, [a["reconstructed"][f] for a in arrays])
            if f in (0,n//2,n-1):
                canvas.save(self.root / "stills" / f"grasp_presets_{f:03d}.png")
            repeats = 3 + (FPS if f in (0,n-1) else 0)
            for _ in range(repeats):
                yield canvas

    def soft_prior(self):
        mean, residual = self.a["mean"], self.a["residual"]
        low,high = self.a["residual_bounds"]
        for f in range(300):
            scalar = low + (high-low)*.5*(1-np.cos(4*np.pi*f/299))
            q = mean + scalar*residual
            full = np.clip(mean + (q-mean) @ self.a["components"].T @ self.a["components"], self.a["lower"], self.a["upper"])
            soft = q+.05*(full-q)
            canvas = self.canvas("The current prior nudges; it does not remove a motor action",
                "Synthetic sweep along the omitted direction. SAPG still predicts six independent hand targets.",
                [("Nominal motor targets", GRAY), ("Actual setting: 5% correction", TEAL), ("100% projection: comparison", PURPLE)],
                "Not an RL rollout. Away from clipping: the five retained directions have gain 1; the residual direction has gain 0.95.",
                ["Residual motion retained: 100%", "Residual motion retained: 95%", "Residual motion retained: 0%"])
            self.panels(canvas, [q,soft,full])
            if f == 75:
                canvas.save(self.root / "stills" / "soft_prior_comparison.png")
            yield canvas

    def run(self, only):
        choices = {
            "retarget": ("01_human_to_revo2_pca.mp4", self.retargeting, "Four matched DexYCB human/IK/PCA comparisons; endpoint holds and 0.5x replay"),
            "components": ("02_five_pca_motion_directions.mp4", self.component_sweeps, "Five independent, joint-feasible PCA basis sweeps; illustrative, not human-recorded"),
            "presets": ("03_power_pinch_tripod_priors.mp4", self.presets, "Power and optional pinch/tripod closure presets projected by the same frozen artifact"),
            "soft": ("04_actual_five_percent_prior.mp4", self.soft_prior, "Nominal vs actual 5% blend vs 100% projection; synthetic residual-direction sweep")}
        try:
            for key,(name,generator,description) in choices.items():
                if only in ("all",key):
                    self.video(name,generator(),description)
        finally:
            self.hr.renderer.delete()
        result = dict(videos=self.outputs, checked_hand_poses=self.hr.frames_checked,
                      max_fk_error_m=self.hr.max_fk_error, joint_limits_checked=True,
                      physics_simulated=False, artifact_sha256=self.info["artifact_sha256"])
        (self.root / "data" / f"render_validation_{only}.json").write_text(json.dumps(result,indent=2))
        print("PCA_RENDER_VALIDATED",json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle",type=Path)
    parser.add_argument("--only",choices=("all","retarget","components","presets","soft"),default="all")
    args = parser.parse_args()
    Presentation(args.bundle).run(args.only)


if __name__ == "__main__":
    main()
