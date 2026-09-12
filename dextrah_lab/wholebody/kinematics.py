"""Small, mesh-independent URDF FK for offline teacher frame conversion.

Not a collision or dynamics solver. All transforms are metres, radians and
active local-to-parent rotations; pose arrays use xyz + wxyz quaternions.
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape[-1:] != (7,) or not np.isfinite(pose).all():
        raise ValueError('Expected finite xyz+wxyz poses')
    if not np.allclose(np.linalg.norm(pose[..., 3:], axis=-1), 1., atol=1e-3):
        raise ValueError('Non-unit pose quaternion')
    result = np.broadcast_to(np.eye(4), (*pose.shape[:-1], 4, 4)).copy()
    result[..., :3, :3] = Rotation.from_quat(pose[..., [4, 5, 6, 3]].reshape(-1, 4)).as_matrix().reshape(*pose.shape[:-1], 3, 3)
    result[..., :3, 3] = pose[..., :3]
    return result


def matrix_pose(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[-2:] != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Expected finite homogeneous transforms')
    quat = Rotation.from_matrix(matrix[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(*matrix.shape[:-2], 4)
    return np.concatenate((matrix[..., :3, 3], quat[..., [3, 0, 1, 2]]), axis=-1)


class UrdfKinematics:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.joints = {j.get('name'): j for j in root.findall('joint')}
        self.parents = {j.find('child').get('link'): j for j in self.joints.values()}
        links = {link.get('name') for link in root.findall('link')}
        roots = links - set(self.parents)
        if len(roots) != 1 or len(self.parents) != len(self.joints):
            raise ValueError('Expected one connected URDF tree')
        self.root = roots.pop()
        self.links = links

    def _angle(self, joint, positions, visiting=()):
        name = joint.get('name')
        if name in visiting:
            raise ValueError('Cyclic joint mimic relation')
        # Explicit measured follower positions take precedence over ideal mimic.
        if name in positions:
            return positions[name]
        mimic = joint.find('mimic')
        if mimic is not None:
            return (self._angle(self.joints[mimic.get('joint')], positions, (*visiting, name))
                    * float(mimic.get('multiplier', '1')) + float(mimic.get('offset', '0')))
        limit = joint.find('limit')
        if limit is not None and limit.get('lower') == limit.get('upper'):
            return float(limit.get('lower'))
        if joint.get('type') == 'fixed':
            return 0.
        raise ValueError(f'Missing independent joint position: {name}')

    def transform(self, link, positions, count):
        if link not in self.links or count < 1:
            raise ValueError('Invalid link or batch count')
        chain, seen = [], set()
        child = link
        while child != self.root:
            if child in seen or child not in self.parents:
                raise ValueError('Disconnected or cyclic URDF tree')
            seen.add(child)
            joint = self.parents[child]
            chain.append(joint)
            child = joint.find('parent').get('link')
        result = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
        for joint in reversed(chain):
            origin = joint.find('origin')
            matrix = np.eye(4)
            if origin is not None:
                matrix[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                matrix[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            result = result @ matrix
            kind = joint.get('type')
            if kind == 'fixed':
                continue
            if kind not in ('revolute', 'continuous', 'prismatic'):
                raise ValueError(f'Unsupported URDF joint type: {kind}')
            value = np.broadcast_to(self._angle(joint, positions), (count,))
            axis_element = joint.find('axis')
            axis = np.fromstring(axis_element.get('xyz') if axis_element is not None else '1 0 0', sep=' ')
            if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0 or not np.isfinite(value).all():
                raise ValueError('Invalid joint axis/position')
            axis /= np.linalg.norm(axis)
            motion = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
            if kind == 'prismatic':
                motion[:, :3, 3] = value[:, None] * axis
            else:
                motion[:, :3, :3] = Rotation.from_rotvec(value[:, None] * axis).as_matrix()
            result = result @ motion
        return result
