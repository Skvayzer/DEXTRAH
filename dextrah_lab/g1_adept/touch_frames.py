"""A single anatomical convention for PhysX, packets and the viewer."""
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from .contact import FINGERS


def pose_matrix(position, quaternion):
    result = trimesh.transformations.quaternion_matrix(quaternion)
    result[:3, 3] = position
    return result


def display_pose(world_to_display, position, quaternion):
    """One rigid transform for all meshes, pads, probes and force vectors."""
    result = world_to_display @ pose_matrix(position, quaternion)
    return result[:3, 3], trimesh.transformations.quaternion_from_matrix(result)


def ray_surface_point(mesh, origin, direction):
    """Nearest ray/triangle hit and face normal; no rtree dependency."""
    origin, direction = np.asarray(origin), np.asarray(direction)
    direction = direction / np.linalg.norm(direction)
    triangles = mesh.triangles
    a, b = triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]
    h = np.cross(direction,b)
    determinant = np.sum(a*h,axis=-1)
    nonparallel = np.abs(determinant)>1e-12
    inv = np.divide(1.,determinant,out=np.zeros_like(determinant),where=nonparallel)
    s = origin-triangles[:,0]
    u = np.sum(s*h,axis=-1)*inv
    q = np.cross(s,a)
    v = np.sum(direction*q,axis=-1)*inv
    distance = np.sum(b*q,axis=-1)*inv
    valid = nonparallel & (u>=-1e-7) & (v>=-1e-7) & (u+v<=1+1e-7) & (distance>=0)
    distances = np.where(valid,distance,np.inf)
    i = distances.argmin()
    if not np.isfinite(distances[i]):
        raise ValueError('Ray missed the finger collision mesh')
    n = np.cross(a[i],b[i])
    n /= np.linalg.norm(n)
    if np.dot(n,direction)>0:
        n=-n
    return origin+distances[i]*direction,n


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
