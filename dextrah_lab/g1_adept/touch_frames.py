"""A single anatomical convention for PhysX, packets and the viewer."""
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from .contact import FINGERS


def sensor_frames(urdf, pads):
    tree = ET.parse(urdf).getroot()
    frames = []
    for finger, pad in zip(FINGERS, pads):
        # The fixed tip marker defines the anatomical longitudinal direction,
        # but is never used as the contact surface or material target.
        joint = next(j for j in tree.findall('joint')
                     if j.find('parent').get('link') == f'right_{finger}_distal_link'
                     and 'tip' in j.find('child').get('link'))
        origin = joint.find('origin')
        tip = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        n = pad.outward / np.linalg.norm(pad.outward)
        p = pad.mesh.centroid.copy()
        p += n * np.dot(pad.surface_point-p, n)
        x = tip-p
        x -= np.dot(x, n)*n
        if np.linalg.norm(x) < 1e-6:
            raise ValueError(f'Undefined toward-tip axis for {finger}')
        x /= np.linalg.norm(x)
        y = np.cross(x, n)  # clockwise viewed from outside pad
        # [compression, toward-tip, clockwise] is right-handed.
        frame = np.stack((-n, x, y), -1)
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-6)
        frames.append(frame)
    return np.asarray(frames, dtype=np.float32)
