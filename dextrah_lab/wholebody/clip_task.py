"""Small LIVE object/table task for route-2 integration, not a SAPG trainer.

One captured training object, one held goal, no trajectory replay after reset.
All subsequent actor observations come from current full-body simulation. The
constant inverse scene alignment preserves the original teacher task frame.
Observation noise/delays and tactile are not yet enabled in this diagnostic.
"""
import json
from pathlib import Path
import numpy as np
import torch

from .contract import BODY_JOINTS,hand_joints,joint_indices
from .kinematics import UrdfKinematics,matrix_pose
from .reference import Segment
from .teacher_bridge import RIGHT_ARM,align_teacher_segment

TIP_NAMES=tuple('right_'+f+'_tip' for f in ('index','middle','ring','thumb','pinky'))
CORNERS=((1,1,1),(1,1,-1),(-1,-1,1),(-1,-1,-1))


def qmul(a,b):
    aw,av=a[...,:1],a[...,1:];bw,bv=b[...,:1],b[...,1:]
    return torch.cat((aw*bw-(av*bv).sum(-1,keepdim=True),aw*bv+bw*av+torch.cross(av,bv,dim=-1)),-1)


def qapply(q,v):
    uv=torch.cross(q[...,1:],v,dim=-1)
    return v+2*(q[...,:1]*uv+torch.cross(q[...,1:],uv,dim=-1))


def compose(a,b):
    a,b=torch.broadcast_tensors(a,b)
    return torch.cat((a[...,:3]+qapply(a[...,3:],b[...,:3]),qmul(a[...,3:],b[...,3:])),-1)


class LiveClipTask:
    def __init__(self,clip,urdf,output,device):
        self.clip=Path(clip);self.device=device;self.urdf=Path(urdf)
        self.metadata=json.loads((self.clip/'metadata.json').read_text())
        if self.metadata['actor_dim']!=224 or self.metadata['tactile']:
            raise ValueError('First live integration probe requires the audited BPS source clip')
        with np.load(self.clip/'reference_trace.npz',allow_pickle=False) as f:
            source=dict(f)
        boundaries=np.flatnonzero(np.diff(source['resets'])!=0)+1
        stop=int(boundaries[0]) if len(boundaries) else len(source['step'])
        aligned,audit=align_teacher_segment(source,self.metadata,Segment(0,stop,0),urdf)
        self.audit=audit
        self.initial_body=aligned['body_q'][0]
        self.initial_hand=aligned['right_hand_q'][0]
        self.initial_poses={k:aligned[k+'_pose'][0] for k in ('object','table','goal')}
        self.source_from_world=torch.tensor(matrix_pose(np.linalg.inv(np.asarray(audit['scene_alignment']))),device=device,dtype=torch.float32)
        self.object_scale=torch.tensor(source['teacher_observation'][0,89:92],device=device)
        with np.load(self.clip/'bps_geometry.npz',allow_pickle=False) as geom:
            features=np.r_[geom['distances'],geom['centroid'],geom['radius']].astype(np.float32)
        # Match exactly the shape descriptor in the captured actor input.
        np.testing.assert_allclose(features,source['teacher_observation'][0,92:224],rtol=1e-6,atol=1e-7)
        self.bps=torch.as_tensor(features,device=device)
        self.output=Path(output)
        self.tree=UrdfKinematics(urdf)

    def add_assets(self,scene_cfg):
        from isaaclab.assets import RigidObjectCfg
        import isaaclab.sim as sim
        from isaaclab.sensors import ContactSensorCfg
        from isaacsimenvs.tasks.play.utils.scene_utils import _convert_urdf_to_usd,_bake_usd
        for name in ('object','table'):
            raw=_convert_urdf_to_usd(str(self.clip/f'{name}.urdf'),self.output/'task_usd',fix_base=False)
            baked=_bake_usd(raw,self.output/'task_baked',name,props=dict(
                kinematic_enabled=name=='table',disable_gravity=name=='table',articulation_enabled=False))
            pose=self.initial_poses[name]
            cfg=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/'+name.capitalize(),
                spawn=sim.UsdFileCfg(usd_path=baked,activate_contact_sensors=True,
                    rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=name=='table',disable_gravity=name=='table',
                        solver_position_iteration_count=32,solver_velocity_iteration_count=4),
                    physics_material=sim.RigidBodyMaterialCfg(static_friction=.5,dynamic_friction=.5)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pose[:3]),rot=tuple(pose[3:])))
            setattr(scene_cfg,name,cfg)
        scene_cfg.object_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Object/.*',update_period=0.,history_length=1)

    def _merged_frame(self,name,robot):
        # Find the surviving rigid ancestor and preserve every locked sensor
        # angle in the transform. Do not mistake a distal origin for the tip.
        ancestor=name
        while ancestor not in robot.body_names:
            joint=self.tree.parents[ancestor]
            limit=joint.find('limit')
            if joint.get('type')!='fixed' and not (limit is not None and limit.get('lower')==limit.get('upper')):
                raise ValueError(f'Missing moving body for observed frame {name}')
            ancestor=joint.find('parent').get('link')
        zeros=dict.fromkeys((*BODY_JOINTS,*hand_joints('left'),*hand_joints('right')),0.)
        transform=np.linalg.inv(self.tree.transform(ancestor,zeros,1)[0])@self.tree.transform(name,zeros,1)[0]
        return robot.body_names.index(ancestor),torch.tensor(matrix_pose(transform),device=self.device,dtype=torch.float32)

    def reset(self,scene,audit):
        self.scene=scene;self.robot=scene['robot'];self.object=scene['object']
        robot=self.robot;n=robot.num_instances
        self.ids=joint_indices(robot.joint_names,(*RIGHT_ARM,*hand_joints('right')))
        self.hand_ids=self.ids[7:]
        self.palm=self._merged_frame('right_base_link',robot)
        self.tips=[self._merged_frame(name,robot) for name in TIP_NAMES]
        q=robot.data.default_joint_pos.clone()
        q[:,joint_indices(robot.joint_names,BODY_JOINTS)]=q.new_tensor(self.initial_body)
        q[:,self.hand_ids]=q.new_tensor(self.initial_hand)
        # Initialize physical followers consistently; NEVER rewrite them during
        # rollout or replace native coupling with independent PD drives.
        for name,rel in audit['mimic_relations'].items():
            q[:,robot.joint_names.index(name)]=(q[:,robot.joint_names.index(rel['joint'])]*float(rel['multiplier'])+float(rel.get('offset',0.)))
        robot.write_joint_state_to_sim(q,torch.zeros_like(q));robot.set_joint_position_target(q)
        self.previous=q.clone()
        for name in ('object','table'):
            asset=scene[name];state=asset.data.default_root_state.clone()
            state[:,:3]+=scene.env_origins
            asset.write_root_state_to_sim(state)
        self.goal=q.new_tensor(self.initial_poses['goal'])[None].repeat(n,1)
        self.limits=robot.data.joint_pos_limits[0,self.ids].clone()
        # Actor keypoints use per-object dimensions, reward uses fixed size.
        self.kp_offsets=q.new_tensor(CORNERS)[None]*(self.object_scale*.04*1.5*.5)[None,None]
        self.reward_offsets=q.new_tensor(CORNERS)[None]*q.new_tensor([.141,.03025,.0271])[None,None]*.75
        self.initial_object_z=self.object.data.default_root_state[:,2].clone()
        self.max_lift=q.new_zeros(n);self.min_goal_error=q.new_full((n,),float('inf'))
        self.near_goal_samples=q.new_zeros(n,dtype=torch.long)
        self.peak_object_contact=0.
        self.finger_clipping_count=0
        return q

    def _body_pose(self,item):
        i,local=item
        pose=torch.cat((self.robot.data.body_pos_w[:,i]-self.scene.env_origins,self.robot.data.body_quat_w[:,i]),-1)
        return compose(pose,local)

    def observation(self):
        source=self.source_from_world
        palm=compose(source,self._body_pose(self.palm))
        tips=torch.stack([compose(source,self._body_pose(item))[:,:3] for item in self.tips],1)
        obj=compose(source,torch.cat((self.object.data.root_pos_w-self.scene.env_origins,self.object.data.root_quat_w),-1))
        goal=compose(source,self.goal)
        count=len(obj)
        offsets=self.kp_offsets.expand(count,-1,-1)
        obj_kp=obj[:,None,:3]+qapply(obj[:,None,3:].expand(-1,4,-1),offsets)
        goal_kp=goal[:,None,:3]+qapply(goal[:,None,3:].expand(-1,4,-1),offsets)
        q=self.robot.data.joint_pos[:,self.ids]
        normalized=2*(q-self.limits[:,0])/(self.limits[:,1]-self.limits[:,0])-1
        task=torch.cat((normalized,self.robot.data.joint_vel[:,self.ids],self.previous[:,self.ids],
            palm[:,:3],palm[:,[4,5,6,3]],obj[:,[4,5,6,3]],(tips-palm[:,None,:3]).flatten(1),
            (obj_kp-palm[:,None,:3]).flatten(1),(obj_kp-goal_kp).flatten(1),self.object_scale.expand(count,-1)),-1)
        if task.shape!=(count,92):
            raise ValueError('Teacher base observation layout mismatch')
        task=torch.cat((task.clamp(-10,10),self.bps.expand(count,-1)),-1)
        if not torch.isfinite(task).all():
            raise ValueError('Nonfinite live full-body task observation')
        return task

    def set_finger_targets(self,target,actions):
        clipped=actions.clamp(-1,1)
        self.finger_clipping_count+=int((actions!=clipped).sum())
        lower,upper=self.limits[7:,0],self.limits[7:,1]
        target[:,self.hand_ids]=lower+(clipped+1)*.5*(upper-lower)
        self.previous=target.clone()

    def record(self):
        pose=torch.cat((self.object.data.root_pos_w-self.scene.env_origins,self.object.data.root_quat_w),-1)
        offsets=self.reward_offsets.expand(len(pose),-1,-1)
        a=pose[:,None,:3]+qapply(pose[:,None,3:].expand(-1,4,-1),offsets)
        b=self.goal[:,None,:3]+qapply(self.goal[:,None,3:].expand(-1,4,-1),offsets)
        error=(a-b).norm(dim=-1).amax(-1)
        lift=pose[:,2]-self.initial_object_z
        self.max_lift=torch.maximum(self.max_lift,lift)
        self.min_goal_error=torch.minimum(self.min_goal_error,error)
        self.near_goal_samples+=(error<.015).long()
        force=self.scene['object_contact'].data.net_forces_w
        self.peak_object_contact=max(self.peak_object_contact,float(force.norm(dim=-1).max()))
        return dict(object_pose=pose,object_force=force,task_goal_error=error)

    def report(self):
        return dict(source_clip=str(self.clip),source_scene_alignment=self.audit['scene_alignment'],
            live_simulator_observations=True,future_demo_states_used=False,teacher_arm_action_override=False,
            goal_protocol='one held source goal, no resets; fixed-size four-keypoint diagnostic',
            max_lift_m=self.max_lift.cpu().tolist(),min_goal_error_m=self.min_goal_error.cpu().tolist(),
            samples_below_15mm=self.near_goal_samples.cpu().tolist(),peak_object_contact_n=self.peak_object_contact,
            finger_clipped_channels_count=self.finger_clipping_count,
            tactile_enabled=False,observation_delay_noise_enabled=False,
            finger_action_pipeline='absolute URDF-range targets; original teacher delays/filter not yet ported',
            training_task_ready=False,grasp_success_rate_measured=False)
