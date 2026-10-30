"""Phase 1 episodes on tools (plan rev. 2.1 §4 and §B): scripted episodes in the C MuJoCo testbed that start from the
harvested held states of the manipulation expert (tool_holds.py; usable = held or shifted) and cover carrying,
setting down, release and incidental contact, with finger commands that vary independently of the state.

Episode: restore the post-settle state of a hold (its tool compiled into the testbed); the table is shifted so the
object is within 8 cm of its centre (never into contact). Then
  carry   1-2 s: min-jerk moves through 1-3 random palm waypoints (up to 10 cm, 20 deg; 30% jerky: 2-5 short moves);
          25% of episodes regrip once (commands open by 0.1-0.4 rad, then close back);
  lower   palm straight down at 0.08-0.3 m/s until the object or the hand touches the table; after an object-first
          touchdown the palm presses 0-8 mm on;
  release every command ramps to open (random delay and duration per command), palm still;
  retreat palm backs off (away from the palm side and up) 5-12 cm;
  sweep   30% of episodes: the palm passes horizontally through or beside the object (offset up to 3 cm): pushes,
          brushes and near-misses;
  settle  0.3 s.
If the object leaves the hand during the carry (no hand contact for 0.25 s), the episode jumps to the sweep / settle;
it ends when the object falls 0.15 m below the table top.
Noise throughout: an Ornstein-Uhlenbeck perturbation per finger command (stationary sigma 0.02-0.15 rad, correlation
time 0.1-0.3 s, drawn per episode) and an OU offset of the palm pose (0.5-3 mm, 0.2-1 deg).

Logged per control step t: the integration state before the step (mjSTATE_INTEGRATION, with the solver warm start:
exact restore with set_state, verified bit-identical on replay), the step's input (float64) (palm pose at the
end of the step, palm twist, the 6 commands with and without noise, the 11 targets), the joint angles and velocities,
object pose and twist and per-finger forces after the step, the phase; every hand-object, object-table and hand-table
contact after each physics substep (CSR rows: step, substep, pair, hand body, distance, position, frame, 6-D force).
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from ..testbed import HAND_JOINTS, PALM_NORMAL, PHYSICS_DT, PREFIX, SUBSTEPS, TABLE_SIZE, Testbed, TestbedSpec
from .held_states import COMMAND_JOINTS, axis_angle, mat, pose, random_unit, targets_from_commands

DT = SUBSTEPS*PHYSICS_DT
PHASES = ('carry', 'regrip', 'lower', 'press', 'release', 'retreat', 'sweep', 'settle')
PAIRS = ('hand_object', 'object_table', 'hand_table')
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky', 'palm')
CMD_INDEX = [HAND_JOINTS.index(j) for j in COMMAND_JOINTS]
STATE = mujoco.mjtState.mjSTATE_INTEGRATION          # physics + controls + solver warm start: exact restores
INPUTS = ('palm_next', 'cmd', 'targets')             # stored in float64 so that a restored step replays exactly
CONTACT_COLUMNS = (('step', 1), ('substep', 1), ('pair', 1), ('hand_body', 1), ('dist', 1), ('pos', 3), ('frame', 9),
                   ('force', 6))                       # force in the contact frame: normal, 2 tangential, 3 torques


@dataclass
class EpisodeSpec:
    carry_s: tuple = (1., 2.)
    waypoints: tuple = (1, 3)
    jerky_p: float = .3
    jerky_waypoints: tuple = (2, 5)
    max_shift_m: float = .10
    max_rot_deg: float = 20.
    regrip_p: float = .25
    regrip_open: tuple = (.1, .4)
    regrip_s: tuple = (.1, .3)
    lower_speed: tuple = (.08, .3)
    max_lower_m: float = .6
    press_m: tuple = (0., .008)
    press_s: tuple = (.1, .3)
    release_delay_s: tuple = (0., .2)
    release_s: tuple = (.15, .6)
    retreat_m: tuple = (.05, .12)
    retreat_s: tuple = (.3, .6)
    sweep_p: float = .3
    sweep_speed: tuple = (.1, .3)
    sweep_offset_m: float = .03
    settle_s: float = .3
    finger_sigma: tuple = (.02, .15)
    finger_tau: tuple = (.1, .3)
    palm_sigma_m: tuple = (.0005, .003)
    palm_sigma_deg: tuple = (.2, 1.)
    palm_tau: tuple = (.1, .3)
    table_offset_m: float = .08
    lost_s: float = .25
    fall_m: float = .15


def set_state(tb, state):
    """Exact restore of a logged state (mjSTATE_INTEGRATION)."""
    mujoco.mj_setState(tb.m, tb.d, state, STATE)
    mujoco.mj_forward(tb.m, tb.d)


def min_jerk(s):
    return s*s*s*(10-15*s+6*s*s)


def interp(a, b, s):
    """Pose between a and b (7: pos, wxyz) at fraction s (position linear, rotation along the shortest arc)."""
    Ra, Rb = mat(a[3:]), mat(b[3:])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (Ra.T @ Rb).reshape(-1))
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.)
    ang = np.linalg.norm(v)
    R = Ra @ (axis_angle(v, s*ang) if ang > 1e-12 else np.eye(3))
    return pose(a[:3]+s*(b[:3]-a[:3]), R)


class OU:
    """Discrete Ornstein-Uhlenbeck process with stationary std sigma and correlation time tau (per dimension)."""

    def __init__(self, rng, n, sigma, tau):
        self.sigma, self.tau = sigma, tau
        self.rng, self.a = rng, np.exp(-DT/np.asarray(tau, float))
        self.s = np.asarray(sigma, float)*np.sqrt(1-self.a**2)
        self.x = np.zeros(n)                                 # starts at zero: no jump at the first step

    def __call__(self):
        self.x = self.a*self.x+self.s*self.rng.normal(size=self.x.shape)
        return self.x


class Recorder:
    """Per-step arrays and per-substep contact rows of one episode."""

    STEP = ('state', 'palm_next', 'palm_twist', 'cmd', 'cmd_nominal', 'targets', 'q', 'qd', 'object', 'object_vel',
            'forces', 'object_table_n', 'phase')

    def __init__(self, tb):
        self.tb = tb
        self.state = np.empty(mujoco.mj_stateSize(tb.m, STATE))
        self.steps = {k: [] for k in self.STEP}
        self.contacts = []
        self.t = 0
        self.body_of_geom = tb.m.geom_bodyid

    def substep(self, k):
        for c in self.tb.contacts():
            hb = int(self.body_of_geom[c['hand_geom']]) if c['hand_geom'] >= 0 else -1
            self.contacts.append(np.concatenate(([self.t, k, PAIRS.index(c['pair']), hb, c['dist']], c['pos'],
                                                 c['frame'], c['force_local'])))

    def get_state(self):
        mujoco.mj_getState(self.tb.m, self.tb.d, self.state, STATE)
        return self.state.copy()

    def arrays(self):
        out = {k: np.array(v) for k, v in self.steps.items()}
        out['contacts'] = np.array(self.contacts, dtype=np.float32).reshape(-1, sum(n for _, n in CONTACT_COLUMNS))
        return out


class Episode:
    def __init__(self, tb, hold, rng, spec=EpisodeSpec()):
        self.tb, self.hold, self.rng, self.spec = tb, hold, rng, spec
        m = tb.m
        jid = [m.joint(PREFIX+j).id for j in COMMAND_JOINTS]
        self.cmd_lo, self.cmd_hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
        self.cmd0 = np.asarray(hold['hand_target'])[CMD_INDEX]
        self.coupling_error = float(np.abs(targets_from_commands(self.cmd0)-hold['hand_target']).max())
        u = lambda r: rng.uniform(*r)
        self.f_ou = OU(rng, 6, u(spec.finger_sigma), u(spec.finger_tau))
        self.p_ou = OU(rng, 3, u(spec.palm_sigma_m), u(spec.palm_tau))
        self.r_ou = OU(rng, 3, np.radians(u(spec.palm_sigma_deg)), u(spec.palm_tau))
        self.rec = Recorder(tb)
        self.labels = dict(regrip=False, sweep=False, jerky=False, touchdown=None, lost_at=None, fell_at=None,
                           finger_sigma=float(self.f_ou.sigma), finger_tau=float(self.f_ou.tau),
                           palm_sigma_m=float(self.p_ou.sigma), palm_sigma_deg=float(np.degrees(self.r_ou.sigma)))

    # --- state helpers
    def palm(self):
        d, a = self.tb.d, self.tb.palm_qadr
        return d.qpos[a:a+7].copy()

    def obj(self):
        d, a = self.tb.d, self.tb.object_qadr
        return d.qpos[a:a+7].copy()

    def in_hand(self):
        return sum(self.forces[:6]) > .5

    def table_top(self):
        return float(self.tb.d.mocap_pos[self.tb.table_mocap][2]+TABLE_SIZE[2]/2)

    def place_table(self):
        """Shift the table so the object is within table_offset_m of its centre (in x and y); keep the old place if
        the new one would touch the hand or the object."""
        tb, d, s = self.tb, self.tb.d, self.spec
        old = d.mocap_pos[tb.table_mocap].copy()
        new = old.copy()
        new[:2] = self.obj()[:2]-self.rng.uniform(-s.table_offset_m, s.table_offset_m, 2)
        d.mocap_pos[tb.table_mocap] = new
        mujoco.mj_forward(tb.m, d)
        if any(c['pair'] != 'hand_object' for c in tb.contacts()):
            d.mocap_pos[tb.table_mocap] = old
            mujoco.mj_forward(tb.m, d)
            return False
        return True

    # --- one control step
    def step(self, nominal_palm, cmd_nominal, phase):
        tb, s = self.tb, self.spec
        prev = self.palm()
        off_p, off_r = self.p_ou(), self.r_ou()
        nxt = pose(nominal_palm[:3]+off_p, axis_angle(off_r, np.linalg.norm(off_r)) @ mat(nominal_palm[3:])
                   if np.linalg.norm(off_r) > 0 else mat(nominal_palm[3:]))
        cmd = np.clip(cmd_nominal+self.f_ou(), self.cmd_lo, self.cmd_hi)
        targets = targets_from_commands(cmd)
        r = self.rec
        r.steps['state'].append(r.get_state())
        tb.step(targets, prev, nxt, on_substep=r.substep)
        d = tb.d
        r.steps['palm_next'].append(nxt)
        r.steps['palm_twist'].append(d.qvel[tb.palm_dadr:tb.palm_dadr+6].copy())
        r.steps['cmd'].append(cmd)
        r.steps['cmd_nominal'].append(np.asarray(cmd_nominal, float).copy())
        r.steps['targets'].append(targets)
        r.steps['q'].append(d.qpos[tb.qadr].copy())
        r.steps['qd'].append(d.qvel[tb.dadr].copy())
        r.steps['object'].append(self.obj())
        r.steps['object_vel'].append(d.qvel[tb.object_dadr:tb.object_dadr+6].copy())
        f = tb.finger_forces()
        self.forces = np.array([f[k] for k in FINGERS])
        r.steps['forces'].append(self.forces)
        ot = sum(c['force_local'][0] for c in tb.contacts() if c['pair'] == 'object_table')
        r.steps['object_table_n'].append(ot)
        r.steps['phase'].append(PHASES.index(phase))
        r.t += 1
        self.last_cmd = np.asarray(cmd_nominal, float).copy()
        self.nominal = nominal_palm
        self.hand_table = any(c['pair'] == 'hand_table' for c in tb.contacts())
        self.object_table = ot > 0
        if self.labels['fell_at'] is None and self.obj()[2] < self.table_top()-s.fall_m:
            self.labels['fell_at'] = r.t
        return self.labels['fell_at'] is None

    # --- phases
    def move(self, target, seconds, cmd, phase, check_lost=False):
        start, n = self.nominal, max(1, int(round(seconds/DT)))
        lost = 0
        for k in range(n):
            if not self.step(interp(start, target, min_jerk((k+1)/n)), cmd(k, n) if callable(cmd) else cmd, phase):
                return 'fell'
            lost = 0 if self.in_hand() else lost+1
            if check_lost and lost*DT >= self.spec.lost_s:
                self.labels['lost_at'] = self.rec.t
                return 'lost'
        return 'ok'

    def carry(self):
        s, rng, home = self.spec, self.rng, self.nominal
        jerky = rng.random() < s.jerky_p
        self.labels['jerky'] = jerky
        n = rng.integers(*(s.jerky_waypoints if jerky else s.waypoints), endpoint=True)
        total = rng.uniform(*s.carry_s)
        w = rng.dirichlet(np.ones(n))*total
        regrip_at = int(rng.integers(n)) if rng.random() < s.regrip_p else -1
        for i in range(n):
            R = axis_angle(random_unit(rng), np.radians(rng.uniform(0, s.max_rot_deg))) @ mat(home[3:])
            target = pose(home[:3]+rng.uniform(-s.max_shift_m, s.max_shift_m, 3), R)
            if i == regrip_at:
                out = self.regrip(target, w[i])
            else:
                out = self.move(target, max(w[i], .15), self.cmd0, 'carry', check_lost=True)
            if out != 'ok':
                return out
        return 'ok'

    def regrip(self, target, seconds):
        s, rng = self.spec, self.rng
        self.labels['regrip'] = True
        depth = rng.uniform(*s.regrip_open, size=6)*(rng.random(6) < .7)
        t_open, t_hold, t_close = rng.uniform(*s.regrip_s), rng.uniform(0, .2), rng.uniform(*s.regrip_s)
        T = max(seconds, t_open+t_hold+t_close)

        def cmd(k, n):
            t = (k+1)*DT
            a = min(t/t_open, 1.) if t < t_open+t_hold else max(1-(t-t_open-t_hold)/t_close, 0.)
            return self.cmd0-depth*min_jerk(a)
        return self.move(target, T, cmd, 'regrip', check_lost=True)

    def lower(self):
        s, rng = self.spec, self.rng
        v = rng.uniform(*s.lower_speed)
        start = self.nominal.copy()
        n_max = int(s.max_lower_m/(v*DT))
        for k in range(n_max):
            p = start.copy()
            p[2] -= v*DT*(k+1)
            if not self.step(p, self.cmd0, 'lower'):
                return 'fell'
            if self.object_table or self.hand_table:
                self.labels['touchdown'] = dict(step=self.rec.t, by='object' if self.object_table else 'hand', speed=v)
                break
        else:
            return 'no_touchdown'
        if not self.object_table:                         # the hand touched first: no pressing into the table
            return 'ok'
        target = self.nominal.copy()
        target[2] -= rng.uniform(*s.press_m)
        return self.move(target, rng.uniform(*s.press_s), self.cmd0, 'press')

    def release(self):
        s, rng = self.spec, self.rng
        delay, dur = rng.uniform(*s.release_delay_s, 6), rng.uniform(*s.release_s, 6)
        start = self.cmd0.copy()
        T = float((delay+dur).max())

        def cmd(k, n):
            a = np.clip(((k+1)*DT-delay)/dur, 0., 1.)
            return start+(self.cmd_lo-start)*min_jerk(a)
        return self.move(self.nominal, T, cmd, 'release')

    def retreat(self):
        s, rng = self.spec, self.rng
        n_world = mat(self.nominal[3:]) @ PALM_NORMAL
        away = -n_world+np.array([0., 0., .7])
        target = self.nominal.copy()
        target[:3] += rng.uniform(*s.retreat_m)*away/np.linalg.norm(away)
        return self.move(target, rng.uniform(*s.retreat_s), self.cmd_lo, 'retreat')

    def sweep(self):
        s, rng = self.spec, self.rng
        self.labels['sweep'] = True
        o, p = self.obj()[:3], self.nominal[:3]
        h = o-p
        h[2] = 0.
        if np.linalg.norm(h) < 1e-3:
            h = random_unit(rng)*np.array([1., 1., 0.])
        h /= np.linalg.norm(h)
        side = np.array([-h[1], h[0], 0.])
        through = o+side*rng.uniform(-s.sweep_offset_m, s.sweep_offset_m)
        through[2] = max(o[2], self.table_top()+.02)+rng.uniform(0., .03)
        start = self.nominal.copy()
        end = start.copy()
        end[:3] = through+h*.08
        dist = np.linalg.norm(end[:3]-start[:3])
        return self.move(end, dist/rng.uniform(*s.sweep_speed), self.cmd_lo, 'sweep')

    def run(self):
        """The whole episode from the hold's post-settle state; returns (arrays, labels)."""
        s, rng = self.spec, self.rng
        self.tb.set_state(self.hold['state'])
        self.labels['table_moved'] = self.place_table()
        self.nominal = self.palm()
        self.forces = np.zeros(6)
        out = self.carry()
        self.labels['carry'] = out
        if out == 'ok':
            out = self.lower()
            self.labels['lower'] = out
            if out == 'ok':
                out = self.release()
            if out == 'ok':
                out = self.retreat()
        if out != 'fell' and rng.random() < s.sweep_p:
            out = self.sweep()
        if out != 'fell':
            self.move(self.nominal, s.settle_s, self.last_cmd, 'settle')
        self.labels['steps'] = self.rec.t
        self.labels['coupling_error'] = self.coupling_error
        self.rec.steps['state'].append(self.rec.get_state())         # final state (one more than steps)
        return self.rec.arrays(), self.labels
