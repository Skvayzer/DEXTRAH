"""Revo2 finger servo (dextrah_lab.mj.robot.hand_servo, Testbed.servo_ctrl) with an optional soft backward pass.

Forward (exact): v_cmd = clip(g (q* - q), -vmax, vmax), g = kp / kv; the actuator force kv (v_cmd - qd) is applied by
the dynamics (implicit in qd). The linear band |q* - q| <= vmax / g = effort / kp is 0.4-1.7 mrad, so the command is
saturated almost always and d v_cmd / d q* = 0 there.

Backward (band > 0): straight-through v_exact + (v_soft - stopgrad(v_soft)) with v_soft = vmax tanh((q* - q) / band):
the forward value is unchanged; the gradient is that of a servo whose linear band is `band` wide. Plan rev. 2.1 A1:
band ~ sigma of the executed-target noise of the reference gradients.
"""
import torch


def straight_through(exact, soft):
    return exact+(soft-soft.detach())


def servo_velocity(targets, q, gain, vmax, band=None):
    exact = torch.maximum(torch.minimum(gain*(targets-q), vmax), -vmax)
    if band is None or band <= 0:
        return exact
    return straight_through(exact, vmax*torch.tanh((targets-q)/band))
