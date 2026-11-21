# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import itertools
import colorsys
import numpy as np
import os
import torch
import torchvision
import functools
import math
from isaacgym import gymtorch
import matplotlib.pyplot as plt
from isaacgym import gymapi, gymutil
import time
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.kinematics import Kinematics
import wandb
import warnings
import random
import torch.nn.functional as F
from .sdf import initialize_sdfs, compute_query_sdfs
import glob
from tqdm import tqdm
from types import SimpleNamespace
from xarm.wrapper import XArmAPI
from depth_backbone import *

if os.getenv("IS_REAL") is not None and os.getenv('IS_REAL') in ['1', 'True', 'true']:
    import rospy
    from sensor_msgs.msg import JointState
    from hardware_controller import LeapHand
    from std_msgs.msg import Float32MultiArray
    from rospy.numpy_msg import numpy_msg
    from std_msgs.msg       import Float64, Float32

import random
import omegaconf
    
class Overhand(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.lookat_id = 0
        self.cfg = cfg
        self.global_counter = 0
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        self.robot_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.arm_act_moving_average = self.cfg["env"]["armActionsMovingAverage"] if 'armActionsMovingAverage' in self.cfg["env"] else self.cfg["env"]["actionsMovingAverage"]
        self.hand_act_moving_average = self.cfg["env"]["handActionsMovingAverage"] if 'armActionsMovingAverage' in self.cfg["env"] else  self.cfg["env"]["actionsMovingAverage"]

        self.viewer = None
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.is_real = self.cfg['env'].get('real', False)#os.getenv("IS_REAL") == '1' if os.getenv("IS_REAL") is not None else False
        self.is_test = self.cfg['env'].get('test', False)#os.getenv("IS_TEST") == '1' if os.getenv("IS_TEST") is not None else False #self.cfg['env'].get('test', False)
        self.is_real_input = self.cfg['env'].get('is_real_data', False)

        self.is_phase2 = 'depth' in torch.load(self.cfg['env']['checkpoint']).keys() if self.cfg['env'].get('checkpoint', None) is not None else False #self.cfg['env']['phase2'].get('enable', False)
        
        self.debug_viz = self.cfg["env"]["enableDebugVis"] or self.is_test
        self.is_cameras_active = self.is_phase2 or self.cfg['env']['cameras'].get('enable', False)
        self.enable_latency = self.cfg['env'].get('enable_latency', False) and not self.is_real_input
        self.enable_action_latency = self.cfg['env'].get('enable_action_latency', False) and not self.is_real_input
        self.no_actions_before_latency = self.cfg['env'].get('no_actions_before_latency', False)
        self.no_actions_before = self.cfg['env'].get('no_actions_before', 0)
        if self.enable_latency and self.no_actions_before_latency:
            self.no_actions_before = self.max_latency

        self.max_latency = self.cfg['env'].get('max_latency', 1)

        self.control_type = self.cfg['env'].get('control_type', 'dof_pos')
        assert(self.control_type in ['dof_pos', 'torque'])

        self.stat_horizon = 100 if not self.is_test else 10000

        ################################################################################
        #Editable configs 
        self.arm_act_dim = self.cfg["env"].get("arm_act_dim", 6) #control pos via ik 
        self.num_arm_dofs = self.cfg["env"].get("num_arm_dofs", 6)
        self.num_hand_dofs = 16
        self.end_hand_dofs = self.num_arm_dofs + self.num_hand_dofs
        self.arm_control_mode = self.cfg["env"].get("armControlMode", "ik")

        #if self.is_phase2 and self.cfg["env"]['phase2'].get('student_arm_control_mode', None) is not None:
        #    self.arm_control_mode = self.cfg["env"]['phase2']['student_arm_control_mode']

        if self.arm_control_mode == "joint_control":
            self.cfg["env"]["ik_version"] = "NA"

        self.hand_control_mode = self.cfg["env"].get("handControlMode", "relative")

        print()
        print('IS_TEST', self.is_test)
        print('IS_REAL', self.is_real)
        print('IS_PHASE2', self.is_phase2)
        print('is_cameras_active', self.is_cameras_active)
        print('enable_latency', self.enable_latency)
        print('enable_action_latency', self.enable_action_latency)
        print('ARM', self.arm_control_mode)
        print('HAND', self.hand_control_mode)
        print()
        print()

        self.eef_upper_limits = self.cfg["env"]["eef_upper_limits"]
        self.eef_lower_limits = self.cfg["env"]["eef_lower_limits"]

        self.dt = self.cfg["sim"]["dt"]
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        self.control_dt = self.dt * control_freq_inv

        super().__init__(self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        #self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        #self.reset_position_noise = to_torch(self.reset_position_noise).to(self.device) 
        
        self.init_kinematics()

        self.hand_prev_targets = self.robot_default_dof_pos[:, self.num_arm_dofs:].clone()
        self.hand_cur_targets = self.hand_prev_targets.clone()
        self.real_hand_prev_targets = self.robot_default_dof_pos[:, self.num_arm_dofs:].clone()
        self.real_hand_cur_targets = self.real_hand_prev_targets.clone()
        self.robot_cur_targets = self.robot_default_dof_pos.clone()

        self._reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.num_resets = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.episode_success = torch.zeros(self.num_envs, device=self.device)

        self.init_isaacgym_tensors()

        self._refresh()
        self.init_depth_buffer()
        self.setup_keyboard_events()
        # self.object_reaching_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        if self.is_real:
            self.setup_real_robot()

        self.obs_input, self.obs_phase1 = compute_obs_lists(self.cfg, self.device)

    def init_isaacgym_tensors(self):
        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
    

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        robot_dof_state = self.dof_state.view(self.num_envs, -1, 2)#[:, :self.num_robot_dofs]
        print(robot_dof_state.shape[1], self.num_robot_dofs)
        assert(robot_dof_state.shape[1] == self.num_robot_dofs)
        self.robot_dof_pos = robot_dof_state[..., 0]
        self.robot_dof_vel = robot_dof_state[..., 1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.robot_torques = gymtorch.wrap_tensor(dof_force_tensor).view(-1, self.num_dofs)

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(-1, 13)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(-1, 3)

        self.robot_rb_states = self.rigid_body_states[self.robot_rb_indices].view(self.num_envs, -1, 13)
        self.robot_contact_forces = self.contact_forces[self.robot_rb_indices].view(self.num_envs, -1, 3)

        self.object_root_indices = []
        self.object_rb_forces = torch.zeros((self.num_envs, 3), device=self.device)
        self.rb_forces = torch.zeros_like(self.contact_forces)

        cur = 0
        for _, object_rb_count in enumerate(self.object_rb_count):
            self.object_root_indices.append(cur + self.robot_rb_count)
            cur += self.robot_rb_count + object_rb_count
        self.object_root_indices = torch.tensor(self.object_root_indices, device=self.device).long()

    def _refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.robot_rb_states = self.rigid_body_states[self.robot_rb_indices].view(self.num_envs, -1, 13)
        self.robot_contact_forces = self.contact_forces[self.robot_rb_indices].view(self.num_envs, -1, 3)

    def load_assets(self):
        assets_type = self.cfg['env'].get('asset_dataset_type', 'manual')

        asset_root = self.cfg['env']['asset_root']
        asset_directory = self.cfg['env']['asset_directory']
        if assets_type == 'manual':
            asset_files = self.load_assets_manual(os.path.abspath(asset_root))
        elif assets_type == 'meshdata':
            asset_files = self.load_assets_from_meshdata(os.path.abspath(f'{asset_root}/{asset_directory}'))
        else:
            raise NotImplemented

        self.object_assets = {}
        self.object_files = {}
        asset_options = self.get_object_asset_options()
        for asset_name, asset_file in tqdm(asset_files.items(), desc='Loading assets'):
            asset = self.gym.load_asset(self.sim, '/', asset_file, asset_options)
            self.object_assets[asset_name] = asset
            self.object_files[asset_name] = asset_file
        print('done')

        self.object_names = list(self.object_assets.keys())
        self.num_objects = len(self.object_names)
        
        assert(self.num_objects > 0)
        
        self.probabilities = np.ones(self.num_objects, dtype=float) / self.num_objects
        print(self.num_objects)

    def close(self):
        pass

    def setup_real_robot(self):
        if not self.is_real:
            raise RuntimeError("Real robot is not enabled in the config file")

        if self.num_envs != 1:
            raise RuntimeError("Real robot only supports one environment")

        rospy.init_node('isaacgym')
        self.rate = rospy.Rate(1 / self.control_dt) # 10hz

        if self.is_real:
            self.cam_sub1 = rospy.Subscriber("/cam/obs_dxdydz", Float32MultiArray, self._ros_cam_get_obs)
            self.cam_sub2 = rospy.Subscriber("/cam/ep_time", Float32MultiArray, self._ros_cam_get_time)
        if not self.cfg["env"].get("disable_arm", False):
            print('Starting arm')
            self.arm = XArmAPI("192.168.1.227", is_radian=True)

            self.arm_joint_pub = rospy.Publisher("/arm/joint_angles", JointState, queue_size=10)
            #self.arm_eef_pose_pub = rospy.Publisher("/arm/cmd_eef_pose", JointState, queue_size=10)
            self.arm_joint_sub = rospy.Subscriber("/arm/obs_joint_angles", JointState, self._ros_set_arm_joints)
            #self.eef_pose_sub = rospy.Subscriber("/arm/obs_eef_pose", JointState, self._ros_set_eef_pose)
        
        if not self.cfg["env"].get("disable_hand", False):
            print('Starting hand')
            self.leap_hand = LeapHand()
            self.leap_hand.leap_dof_lower = self.robot_dof_lower_limits[self.num_arm_dofs:].cpu().numpy()
            self.leap_hand.leap_dof_upper = self.robot_dof_upper_limits[self.num_arm_dofs:].cpu().numpy()
            self.leap_hand.sim_to_real_indices = np.array(self.cfg["env"].get("sim_to_real_indices", [1, 0, 2, 3, 9, 8, 10, 11, 13, 12, 14, 15, 4, 5, 6, 7]))
            self.leap_hand.real_to_sim_indices = np.array(self.cfg["env"].get("real_to_sim_indices", [1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11]))
            print('Waiting for service')
            rospy.wait_for_service('/leap_position')
            print('Done')

        self.real_robot_eef_target_pose = self.default_ee_pose.clone()
        self.real_robot_cur_targets = self.robot_default_dof_pos.clone()
    
        #self.move_robot_to_start(None)

    def _ros_set_arm_joints(self, data):
        pose = list(data.position)
        
        self.real_arm_dof_pos = torch.tensor(pose, device=self.device).float()[None, :]
        eef_pos, eef_quat = self.kinematics.forward_kinematics(self.real_arm_dof_pos)
        # This pose is calculated via FK on observed joints
        # self.real_eef_pose is directly obtained from the hardware
        self.real_eef_pose_fk = torch.cat((eef_pos, eef_quat), dim=-1) 

    def _ros_cam_get_obs(self, data):
        image = data.data
        self.cam_obs_tensor = torch.tensor(image, device=self.device).float().reshape(self.num_envs, -1, *self.camera_res)

    def _ros_cam_get_time(self, data):
        time = data.data
        
        time = torch.tensor(time, device=self.device).float().reshape(self.num_envs,) 
        self.cam_ep_ended = time < 0
        self.cam_time = torch.maximum(time, torch.zeros_like(time))
        self.cam_progress_buf = self.cam_time / self.dt
        #self.progress_buf = self.cam_progress_buf
        self.progress_buf[self.cam_ep_ended] = 0

    def _ros_set_eef_pose(self, data):
        pose = list(data.position)
        eef_xyz, eef_euler = pose[:3], pose[3:]
        
        eef_xyz = torch.tensor(eef_xyz, device=self.device)
        eef_euler = torch.tensor(eef_euler, device=self.device)

        roll = torch.tensor([eef_euler[0]], device=self.device)
        pitch = torch.tensor([eef_euler[1]], device=self.device)
        yaw = torch.tensor([eef_euler[2]], device=self.device)

        eef_quat = quat_from_euler_xyz(roll, pitch, yaw).squeeze()
        self.real_eef_pose = torch.cat((eef_xyz, eef_quat))[None, :]

    def init_depth_buffer(self):
        if not self.is_cameras_active:
            return

        self.phase2_buffer_type = self.cfg["env"]['phase2'].get("buffer_type", 'xyz')
        self.num_camera_frames = self.cfg["env"]['phase2'].get("num_camera_frames", 1)
        self.camera_res = self.cfg["env"]["cameras"]["camera_resolution"]
        self.num_cameras = len(self.cfg["env"]["cameras"]["camera_names"])
        self.diff_rgb_threshold = self.cfg["env"]["cameras"].get("diff_rgb_threshold", 0.05)
        self.depth_buffer = torch.zeros((self.num_envs, self.num_cameras, *self.camera_res, ), device=self.device)
        self.diff_buffer = torch.zeros((self.num_envs, self.num_cameras, *self.camera_res, ), device=self.device)
        self.rgb_buffer = torch.zeros((self.num_envs, self.num_cameras, 3, *self.camera_res, ), device=self.device)
        self.xyz_buffer = torch.zeros((self.num_envs, self.num_cameras, 3, *self.camera_res, ), device=self.device)
        self.segm_buffer = torch.zeros((self.num_envs, self.num_cameras, 2, *self.camera_res, ), device=self.device)
        self.phase2_buffers = []
        self.new_depth_image = False
        self.refresh_interval = round(1/(self.cfg["env"]["cameras"]["refresh_frequency"]*self.control_dt))
        self.depth_image_displayed = torch.zeros((self.num_cameras, *self.camera_res), device=self.device)

    def process_image(self, image):
        if not hasattr(self, 'crop_left'):
            self.crop_left = self.cfg["env"]["cameras"].get("crop_left", 0)
            self.crop_right = self.cfg["env"]["cameras"].get("crop_right", None)
            self.crop_right = -self.crop_right if self.crop_right is not None else None
            self.crop_top = self.cfg["env"]["cameras"].get("crop_top", 0)
            self.crop_bottom = self.cfg["env"]["cameras"].get("crop_bottom", None)
            self.crop_bottom = -self.crop_bottom if self.crop_bottom is not None else None

            self.depth_near_clip = self.cfg["env"]["cameras"].get('depth_near_clip', 0) 
            self.depth_far_clip = self.cfg["env"]["cameras"].get('depth_far_clip', 1) 
            self.xyz_bg = self.cfg["env"]["cameras"].get('xyz_bg', [5, 0, 0])
            self.xyz_bg_noise_scale = self.cfg["env"]["cameras"].get('xyz_bg_noise_scale', [0, 0, 0])
            self.xyz_fg_noise_scale = self.cfg["env"]["cameras"].get('xyz_fbg_noise_scale', [0, 0, 0])
            self.xyz_min_clip = self.cfg["env"].get('xyz_min_clip', [0, -0.5, 0])
            self.xyz_max_clip = self.cfg["env"].get('xyz_max_clip', [5, +0.5, 2])
            self.resize_transform = torchvision.transforms.Resize(self.camera_res, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            #self.blur_transform = torchvision.transforms.GaussianBlur(kernel_size=(5, 5), sigma=self.cfg["env"]["cameras"]['depth_blur_scale'])

            self.depth_noise_scale = self.cfg['env']['cameras'].get('depth_noise_scale', 0)
            self.far_depth_noise_scale = self.cfg['env']['cameras'].get('far_depth_noise_scale', 0)
            self.far_threshold = self.cfg['env']['cameras'].get('far_threshold', 10)


        image = image[self.crop_top:self.crop_bottom, self.crop_left:self.crop_right].float() # crop
        image = self.resize_transform(image[None, :]).squeeze() # resiwe
        return image


    def init_kinematics(self):
        joint_indices = {
                "joint1": 0,
                "joint2": 1,
                "joint3": 2,
                "joint4": 3,
                "joint5": 4,
                "joint6": 5,
                "xarm6_leap": 0
            }
        
        urdf_path = os.path.join(self.asset_root, self.robot_asset_file)
        self.kinematics = Kinematics(urdf_path, "link_base", "link6", joint_indices, self.device)

        default_dof_pos = self.robot_default_dof_pos[:, :self.num_arm_dofs]
        eef_pos, eef_quat = self.kinematics.forward_kinematics(default_dof_pos)
        self.default_ee_pose = torch.cat((eef_pos, eef_quat), dim=-1)
        self.robot_eef_target_pose = self.default_ee_pose

    def setup_keyboard_events(self):
        if self.viewer is None:
            return

        self.lookat_id = 0
        self.lookat_vec = torch.tensor([0.9, 0.0, 0.4], requires_grad=False, device=self.device)
        self.offset_pos = torch.tensor([0.0, 0.0, 0.3], requires_grad=False, device=self.device)
        self.free_cam = False
        self.keyboard_delta_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "DECREMENT_X")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "INCREMENT_X")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "DECREMENT_Y")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "INCREMENT_Y")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_PAGE_DOWN, "DECREMENT_Z")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_PAGE_UP, "INCREMENT_Z")

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "free_cam")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT_BRACKET, "prev_id")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT_BRACKET, "next_id")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "enable_vr_glove")

        self.lookat(self.lookat_id)

    def render(self, **kwargs):
        res = super().render(**kwargs)

        if self.viewer is not None:
            # check for keyboard events
            self.keyboard_delta_pos[:] = 0.0

            for evt in self.gym.query_viewer_action_events(self.viewer):
                print("event detected: ", evt.action)

                if evt.action == "DECREMENT_X" and evt.value > 0:
                    self.keyboard_delta_pos[self.lookat_id, 0] = -0.1
                if evt.action == "INCREMENT_X" and evt.value > 0:
                    self.keyboard_delta_pos[self.lookat_id, 0] = 0.1
                if evt.action == "DECREMENT_Y" and evt.value > 0:
                    self.keyboard_delta_pos[self.lookat_id, 1] = -0.1
                if evt.action == "INCREMENT_Y" and evt.value > 0:
                    self.keyboard_delta_pos[self.lookat_id, 1] = 0.1
                if evt.action == "DECREMENT_Z" and evt.value > 0:
                    self.keyboard_delta_pos[self.lookat_id, 2] = -0.1
                if evt.action == "INCREMENT_Z" and evt.value > 0:
                    self.keyboard_delta_pos[self.lookat_id, 2] = 0.1

                if not self.free_cam:
                    if evt.action == "prev_id" and evt.value > 0:
                        self.lookat_id  = (self.lookat_id-1) % self.num_envs
                        self.lookat(self.lookat_id)
                    if evt.action == "next_id" and evt.value > 0:
                        self.lookat_id  = (self.lookat_id+1) % self.num_envs
                        self.lookat(self.lookat_id)
                if evt.action == "free_cam" and evt.value > 0:
                    self.free_cam = not self.free_cam
                    print("free cam: ", self.free_cam)
                    
                    if self.free_cam:
                        self.gym.viewer_camera_look_at(self.viewer, None, self.default_cam_pos, self.default_cam_target)
                    else:
                        self.lookat(self.lookat_id)

            self.gym.sync_frame_time(self.sim)
            
        return res

    def set_camera(self, position, lookat):
        """ 
        Set camera position and direction
        """

        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, self.envs[self.lookat_id], cam_pos, cam_target)

    def lookat(self, i):
        look_at_pos = self.get_arm_base_pos()[i, :].clone() + self.offset_pos
        cam_pos = look_at_pos + self.lookat_vec
        self.set_camera(cam_pos, look_at_pos)

    # def set_defaults(self):
    #     if "ik_version" not in self.cfg["env"]:
    #         self.cfg["env"]["ik_version"] = "v1"
        
    #     if "ik_max_iters" not in self.cfg["env"]:
    #         self.cfg["env"]["ik_max_iters"] = 100

    #     if "ik_tolerance" not in self.cfg["env"]:
    #         self.cfg["env"]["ik_tolerance"] = 1e-3

    #     if "override_object_start_pos" not in self.cfg["env"]:
    #         self.cfg["env"]["override_object_start_pos"] = None
            
    #     if "arm_actions_ema" not in self.cfg["env"]:
    #         self.cfg["env"]["arm_actions_ema"] = 1.0

    #     if "max_joint_speed" not in self.cfg["env"]:
    #         self.cfg["env"]["max_joint_speed"] = None

    #     if "publish_joint_angles" not in self.cfg["env"]:
    #         self.cfg["env"]["publish_joint_angles"] = False

    #     if "disable_object_collision" not in self.cfg["env"]:
    #         self.cfg["env"]["disable_object_collision"] = False
    
    #     if "randomize_rotation" not in self.cfg["env"]:
    #         self.cfg["env"]["randomize_rotation"] = True

    #     if "randomize_goal_rotation" not in self.cfg["env"]:
    #         self.cfg["env"]["randomize_goal_rotation"] = True

    #     if "goal_type" not in self.cfg["env"]:
    #         self.cfg["env"]["goal_type"] = "fixed"

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.epoch_count = 1
        # https://forums.developer.nvidia.com/t/camera-example-and-headless-mode/178901/11
        self.graphics_device_id = self.device_id
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        #self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        if self.randomize:
            self.apply_randomizations(self.randomization_params)
            self.update_randomization_params(torch.arange(self.num_envs), torch.zeros(self.num_envs, device=self.device))
            self.log_randomizations()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def get_object_indices(self, names):
        link_indices = [self.link_dict[name] for name in names]
        link_indices = torch.tensor(link_indices, device=self.device).long()
        return link_indices

    def get_object_rb_indices(self, link_indices):
        indices = []
        cur = 0
        for env_idx in range(self.num_envs):
            indices.append(link_indices + cur)        
            cur += self.robot_rb_count + self.object_rb_count[env_idx]

        return torch.cat(indices, dim=0)

    def load_assets_from_meshdata(self, object_directory):
        object_directory = os.path.abspath(object_directory)
        urdfs = sorted(glob.glob(f'{object_directory}/**/*.urdf', recursive=True))
        asset_files_dict = {}
        random.seed(self.cfg['env']['asset_sample_seed'])
        n_assets = self.cfg['env']['max_num_assets']
        n_assets = len(urdfs) if n_assets < 0 else n_assets
        random.shuffle(urdfs)
        for value in urdfs[:n_assets]:
            key = value#os.path.basename(os.path.dirname(os.path.dirname(value)))
            asset_files_dict[key] = value   
        return asset_files_dict

    def load_assets_manual(self, object_directory):
        asset_files_dict = {
            #"block": "urdf/objects/cube_multicolor_allegro.urdf",
            #"egg": "pregrasp_assets/assets/objects/egg.xml",
            #"egg": "mjcf/open_ai_assets/hand/egg.xml",
            #"pen": "mjcf/open_ai_assets/hand/pen.xml",
            # "hammer": "pregrasp_assets/assets/objects/hammer.xml",
            #"drill_dexycb": "drill_dexycb/coacd/coacd.urdf",
            # "mug": "pregrasp_assets/assets/objects/mug.xml",
            # "waterbottle": "pregrasp_assets/assets/objects/waterbottle.xml",
            # "train": "pregrasp_assets/assets/objects/train.xml",
            # "fryingpan": "pregrasp_assets/assets/objects/fryingpan.xml",
            # "phone": "pregrasp_assets/assets/objects/phone.xml",
            "plate": "dinnerware/plate.urdf",
            # "stanfordbunny": "pregrasp_assets/assets/objects/stanfordbunny.xml",
            #"thin_plate": "bowl_1/object.urdf",
            #"dinner_plate": "dinner_plate/object.urdf",
            #"drill": "drill/object.urdf",
            # "headphones": "headphones/headphones_1/object.urdf",
            #"cylindersmall": "pregrasp_assets/assets/objects/cylindersmall.xml",
            #"table": "pregrasp_assets/assets/objects/table.xml",
            "book": "cuboid/0000.urdf",
            # "ycb_cup": "ycb/065-c_cups/065-c_cups.urdf",
            # "ycb_airplane": "ycb_urdfs/072-f_toy_airplane/072-f_toy_airplane.urdf",
            # "ycb_clamp_medium": "ycb_urdfs/050_medium_clamp/050_medium_clamp.urdf",
            # "ycb_scissors": "ycb_urdfs/037_scissors/037_scissors.urdf",
            #"drill": "drill/object.urdf",
            #"marker": "040_large_marker/model_normalized.urdf",
            #"pen_cylinder": "cylinder/default/0010.urdf",
            "ball": "ball.urdf",
            "smallBall" : "throwObjects/balls/ball20.urdf",
            "marble" : "throwObjects/balls/ball10.urdf",
            "cuboid" : "throwObjects/cuboid/default/0035.urdf",
            # "pyramidmedium" : "pregrasp_assets/assets/objects/pyramidmedium.xml",
            # "banana" : "pregrasp_assets/assets/objects/banana.xml",
            # "binoculars" : "pregrasp_assets/assets/objects/binoculars.xml",
            # "tennisBall" : "urdf/objects/tennis_ball.urdf",
            # "cuboid22" : "/throwObjects/cuboid/default/0022.urdf",
            "apple" : "pregrasp_assets/assets/urdf/apple.urdf",
            "rubberduck" : "pregrasp_assets/assets/urdf/rubberduck.urdf",
            "toothpaste" : "pregrasp_assets/assets/urdf/toothpaste.urdf",
            "lightbulb" : "pregrasp_assets/assets/urdf/lightbulb.urdf",
            "flashlight" : "pregrasp_assets/assets/urdf/flashlight.urdf",
            "wristwatch" : "pregrasp_assets/assets/urdf/wristwatch.urdf",
            "flute" : "pregrasp_assets/assets/urdf/flute.urdf"
        }

        #keys = self.cfg["env"]["object_mixture"].keys() if 'object_mixture' in self.cfg['env'] else asset_files_dict.keys()
        #asset_files_dict = {key:value for key, value in asset_files_dict.items() if key in keys}
        asset_files_dict = {key: f'{object_directory}/{value}' for key, value in asset_files_dict.items()}
        print(asset_files_dict)
        return asset_files_dict

    def log_randomizations(self):
        # if wandb.run is None:
        #     return

        # for param in self.dr_randomizations:
        #     for value in self.dr_randomizations[param]:
        #         if self.dr_randomizations[param]["wandb_log"]:
        #             wandb.log({"{}_{}".format(param, value): self.dr_randomizations[param][value]})
        pass

    def randomize_initial_dof_pos(self, env_ids):
        if not hasattr(self, "clean_robot_default_dof_pos"):
            train_hand_position = self.cfg['env'].get('train_hand_position', 'underhand')
            assert(train_hand_position in ['underhand', 'overhand'])
            if train_hand_position == 'underhand':
                self.robot_default_dof_pos = self.robot_lower_pos.clone()
            else:
                self.robot_default_dof_pos = self.robot_upper_pos.clone()
            
            self.robot_default_dof_vel = torch.zeros_like(self.robot_default_dof_pos)

            # this is required since this function may be called multiple times during a curriculum
            self.clean_robot_default_dof_pos = self.robot_default_dof_pos.clone() 

        if not hasattr(self, "kinematics"):
            self.init_kinematics()

        self.robot_default_dof_pos[env_ids] = self.clean_robot_default_dof_pos[env_ids]

        if "init_ee_pos_rand_scale" in self.cfg["env"] or "init_ee_quat_rand_scale" in self.cfg["env"]:
            if not hasattr(self, 'robot_rand_ee_joints'):
                self.num_ee_rand = 50000
                pos_a = torch.tensor(self.cfg["env"].get("init_ee_pos_rand_start", [0, 0, 0]), device=self.device)[None, :]
                pos_b = torch.tensor(self.cfg["env"].get("init_ee_pos_rand_end", [0, 0, 0]), device=self.device)[None, :]
                delta_ee_pos = pos_a + torch.rand((self.num_ee_rand, 3), device=self.device) * (pos_b - pos_a)
                delta_ee_quat = torch.rand((self.num_ee_rand, 3), device=self.device) * torch.tensor(self.cfg["env"].get("init_ee_quat_rand_scale", [0, 0, 0]), device=self.device)[None, :]
                delta_pose = torch.cat([delta_ee_pos, delta_ee_quat], dim=-1)

                rand_ee_init = self.clean_robot_default_dof_pos[:1].repeat(self.num_ee_rand, 1)
                joint_angles, _ = self.kinematics.inverse_kinematics(delta_pose, rand_ee_init[:, :self.num_arm_dofs])

                lower, upper = self.robot_dof_lower_limits[:self.num_arm_dofs].unsqueeze(0), self.robot_dof_upper_limits[:self.num_arm_dofs].unsqueeze(0)
                mask = ((joint_angles >= lower).sum(dim=-1) == self.num_arm_dofs) & ((joint_angles <= upper).sum(dim=-1) == self.num_arm_dofs)

                self.num_ee_rand = mask.sum()
                self.robot_rand_ee_delta = delta_pose[mask, :]
                self.robot_rand_ee_joints = joint_angles[mask, :]
                self.robot_default_delta_pose = torch.zeros((self.num_envs, 6), device=self.device)
                print(f'num_ee_rand: {self.num_ee_rand}')
                assert(self.num_ee_rand == 50000)

            idxs = torch.randperm(self.num_ee_rand, device=self.device)[:len(env_ids)]
            self.robot_default_delta_pose[env_ids] = self.robot_rand_ee_delta[idxs]
            #self.robot_default_dof_pos[env_ids, :self.num_arm_dofs] = self.robot_rand_ee_joints[idxs]

        if not hasattr(self, 'init_dof_pos_rand_scale'):
            self.init_dof_pos_rand_scale = self.cfg["env"].get("init_dof_pos_rand_scale", 0)
            self.init_hand_dof_pos_rand_scale = self.cfg["env"].get("init_hand_dof_pos_rand_scale", 0)

        #self.robot_default_dof_pos[env_ids] += torch.randn_like(self.robot_default_dof_pos[env_ids]) * self.init_dof_pos_rand_scale
        #self.robot_default_dof_pos[env_ids, self.num_arm_dofs:] += torch.randn_like(self.robot_default_dof_pos[env_ids, self.num_arm_dofs:]) * self.init_hand_dof_pos_rand_scale

        if not hasattr(self, 'randomize_arm_flag'):
            self.randomize_arm_flag = self.cfg["env"].get("randomize_arm", 'none')
        if self.randomize_arm_flag == 'hand_rot':
            self.robot_default_dof_pos[env_ids, self.num_arm_dofs-1] = torch.rand(self.num_envs, device=self.device)[env_ids] * (2*np.pi)

        #if hasattr(self, 'robot_cur_targets'):
        #    self.robot_cur_targets[env_ids] = self.robot_default_dof_pos[env_ids]
        # if "pregrasp_cache" in self.cfg["env"]:
        #     pregrasp_cache = np.load(self.cfg["env"]["pregrasp_cache"])
        #     choices = np.random.randint(0, high=pregrasp_cache.shape[0], size=self.num_envs)
        #     pregrasp_joints = pregrasp_cache[choices]
        #     pregrasp_joints = torch.from_numpy(pregrasp_joints).to(self.device).float()
        #     self.robot_default_dof_pos[:, self.num_arm_dofs:] = pregrasp_joints

        # if self.cfg["env"].get("init_from_random_egrasp", False):
        #     num_egrasps = self.init_egrasp_matrix()
        #     u_hand = torch.randn((self.num_envs, num_egrasps), device=self.device)
        #     self.robot_default_dof_pos[:, self.num_arm_dofs:] = self.fetch_hand_joints_from_egrasp(u_hand)

        # if "init_dof_range" in self.cfg["env"]:
        #     lower = torch.tensor(self.cfg["env"]["init_dof_range"]["lower"], device=self.device)
        #     higher = torch.tensor(self.cfg["env"]["init_dof_range"]["higher"], device=self.device)

        #     self.robot_default_dof_pos[:, self.num_arm_dofs:] = (higher - lower) * torch.rand_like(self.robot_default_dof_pos[:, self.num_arm_dofs:]) + lower

    # def init_projectile_static(self):
    #     rng_pos = 0.5
    #     rng_vel = 10
    #     rng_angvel = 10

    #     start_pos = [np.random.uniform(-rng_pos, rng_pos), np.random.uniform(-rng_pos, rng_pos), np.random.uniform(-rng_pos, rng_pos)]
    #     goal_pos = [np.random.uniform(-rng_pos, rng_pos), np.random.uniform(-rng_pos, rng_pos), np.random.uniform(-rng_pos, rng_pos)]
    #     start_vel = [np.random.uniform(-rng_vel, rng_vel), np.random.uniform(-rng_vel, rng_vel), np.random.uniform(-rng_vel, rng_vel)]
    #     start_angvel = [np.random.uniform(-rng_angvel, rng_angvel), np.random.uniform(-rng_angvel, rng_angvel), np.random.uniform(-rng_angvel, rng_angvel)]

    #     u, v, w = np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)
    #     start_rot = [np.sqrt(1-u)*np.sin(2*np.pi*v), np.sqrt(1-u)*np.cos(2*np.pi*v), np.sqrt(u)*np.sin(2*np.pi*w), np.sqrt(u)*np.cos(2*np.pi*w)]

    #     u, v, w = np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)
    #     goal_rot = [np.sqrt(1-u)*np.sin(2*np.pi*v), np.sqrt(1-u)*np.cos(2*np.pi*v), np.sqrt(u)*np.sin(2*np.pi*w), np.sqrt(u)*np.cos(2*np.pi*w)]

    #     obj_cfg = {
    #         'start_pos': start_pos,
    #         'goal_pose': [*goal_pos, *goal_rot],
    #         'start_rot': start_rot,
    #         'start_vel': start_vel,
    #         'start_angvel': start_angvel,
    #     }
    #     obj_cfg['init_state'] = [*obj_cfg['start_pos'], *obj_cfg['start_rot'], *obj_cfg['start_vel'], *obj_cfg['start_angvel']]
    #     return obj_cfg

    # def reward_static_obj_goal_quat_diff(self):
    #     if not self.cfg['env'].get('static', False):
    #         return 0
    #     goal_object_quat = self.object_goal_pose[:,3:]
    #     curr_object_quat = self.get_obj_pose()[:, 3:]
    #     quat_diff = quat_mul(curr_object_quat, quat_conjugate(goal_object_quat))
    #     reward = torch.abs(quat_diff[:, :3]).sum(dim=-1)
    #     return reward

    # def reward_static_obj_goal_pos_diff(self):
    #     if not self.cfg['env'].get('static', False):
    #         return 0
    #     goal_object_pos = self.object_goal_pose[:,:3]
    #     curr_object_pos = self.get_obj_pose()[:, :3]
    #     reward = torch.abs(curr_object_pos - goal_object_pos).sum(dim=-1)
    #     return reward

    #generate a random toss 
    def init_projectile(self):
        #if self.cfg['env'].get('static', False):
        #    return self.init_projectile_static()

        if not hasattr(self, 'end_pos_rng'):
            self.end_pos_rng = self.cfg["env"]["end_pos_range"]
            self.start_vel_rng = self.cfg["env"]["start_vel_range"]
            self.start_angvel_rng = self.cfg["env"]["start_ang_vel_range"]
            self.time_throw_range = self.cfg["env"]["time_throw_range"]
            self.fast_speed_flag = self.cfg["env"].get("fast_speed", False)

            self.time_curriculum_flag = self.cfg['env'].get('time_curriculum', False) and not self.is_test and not self.is_phase2
            if self.time_curriculum_flag:
                self.time_curriculum_flag = self.cfg['env'].get('time_curriculum_version', 'v1')
            else:
                self.time_curriculum_flag = 'none'
            self.time_curriculum_end = self.cfg['env'].get('time_curriculum_end', 800)


            #self.throw_curriculum = self.cfg['env'].get('throw_curriculum', False)
            #self.throw_curriculum_end = self.cfg['env'].get('throw_curriculum_end', 300)

        u, v, w = np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)
        rotation = [
            np.sqrt(1-u)*np.sin(2*np.pi*v), 
            np.sqrt(1-u)*np.cos(2*np.pi*v), 
            np.sqrt(u)*np.sin(2*np.pi*w), 
            np.sqrt(u)*np.cos(2*np.pi*w)
        ]

        wx_o = random.uniform(self.start_angvel_rng[0][0], self.start_angvel_rng[0][1]) #range of .7 on spec sheet
        wy_o = random.uniform(self.start_angvel_rng[1][0], self.start_angvel_rng[1][1]) #range .7 on spec
        wz_o = random.uniform(self.start_angvel_rng[2][0], self.start_angvel_rng[2][1]) #max rang betweeen 400 -950 mm

        #alpha_throw = min(self.epoch_count / self.throw_curriculum_end, 1)
        x_f = random.uniform(self.end_pos_rng[0][0], self.end_pos_rng[0][1]) #range of .7 on spec sheet
        y_f = random.uniform(self.end_pos_rng[1][0], self.end_pos_rng[1][1]) #range .7 on spec

        vx_o = random.uniform(self.start_vel_rng[0][0], self.start_vel_rng[0][1]) #range of .7 on spec sheet
        vy_o = np.abs(vx_o)*np.sin(np.pi/180*random.uniform(self.start_vel_rng[1][0], self.start_vel_rng[1][1])) #range .7 on spec

        a, b = self.time_throw_range
        alpha = 1

        if self.time_curriculum_flag == 'v1':
                b = a + (b - a) * min(self.epoch_count / self.time_curriculum_end, 1)
        elif self.time_curriculum_flag == 'v2':
                alpha += 0.5*(1 - min(self.epoch_count / self.time_curriculum_end, 1))
        if self.fast_speed_flag:
            alpha *= 2

        time_impact_start, time_impact_end = a / alpha, b / alpha

        t_impact = np.random.uniform(time_impact_start, time_impact_end) 

        x_o = x_f - vx_o * t_impact
        y_o = y_f - vy_o * t_impact

        z_f = random.uniform(self.end_pos_rng[2][0], self.end_pos_rng[2][1]) #max rang betweeen 400 -950 mm
        vz_o = min(np.abs(vx_o)*np.sin(np.pi/180*random.uniform(self.start_vel_rng[2][0], self.start_vel_rng[2][1])), z_f/t_impact + 0.5*9.81*t_impact) #max rang betweeen 400 -950 mm
        z_o = z_f - vz_o * t_impact + 0.5*9.81*t_impact**2

        if not self.headless:
            print(f'New throw:\t{t_impact:.2}s\tstart=({x_o:.2}, {y_o:.2}, {z_o:.2})\tend=({x_f:.2}, {y_f:.2}, {z_f:.2})')

        obj_cfg = {
            'start_pos': [x_o, y_o, z_o],
            'end_pos': [x_f, y_f, z_f],
            't_impact': t_impact,
            'start_rot': rotation,
            'start_vel': [vx_o, vy_o, vz_o],
            'start_angvel': [wx_o, wy_o, wz_o],
            'start_speed': math.sqrt(vx_o **2 + vy_o **2 + vz_o ** 2),
            'start_angspeed': math.sqrt(wx_o **2 + wy_o **2 + wz_o ** 2),
        }
        obj_cfg['init_state'] = [*obj_cfg['start_pos'], *obj_cfg['start_rot'], *obj_cfg['start_vel'], *obj_cfg['start_angvel']]
        return obj_cfg


    def get_robot_asset_options(self):
        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        #asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        #asset_options.angular_damping = 0.01
        #asset_options.collapse_fixed_joints = False

        # https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/docs/rl_examples.md#factory-fast-contact-for-robotic-assembly
        #asset_options.density = 1000.0  # default = 1000.0
        #asset_options.armature = 0.01  # default = 0.0
        #asset_options.enable_gyroscopic_forces = True
        # stability
        #asset_options.linear_damping = 100.0  # default = 0.0; increased to improve stability
        #asset_options.max_linear_velocity = 10.0  # default = 1000.0; reduced to prevent CUDA errors
        #asset_options.angular_damping = 100.0  # default = 0.5; increased to improve stability
        #asset_options.max_angular_velocity = 64  # default = 64.0; reduced to prevent CUDA errors


        if not hasattr(self, 'robot_vhacd_enabled'):
            self.robot_vhacd_enabled = self.cfg["env"].get("robot_vhacd_enabled", True)
        if self.robot_vhacd_enabled:
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.resolution = 10000
            asset_options.vhacd_params.max_convex_hulls = 10
            asset_options.vhacd_params.max_num_vertices_per_ch = 32

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True


        return asset_options

    def get_object_asset_options(self):
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = False
        #object_asset_options.override_com = True
        #object_asset_options.override_inertia = True
        #object_asset_options.enable_gyroscopic_forces = True
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        #object_asset_options.linear_damping = 100.0  # default = 0.0; increased to improve stability
        #object_asset_options.max_linear_velocity = 10.0  # default = 1000.0; reduced to prevent CUDA errors
        #object_asset_options.angular_damping = 100.0  # default = 0.5; increased to improve stability
        #object_asset_options.max_angular_velocity = 64  # default = 64.0; reduced to prevent CUDA errors

        if not hasattr(self, 'object_vhacd_enabled'):
            self.object_vhacd_enabled = self.cfg["env"].get("object_vhacd_enabled", True)
        if self.object_vhacd_enabled:
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params.resolution = 10000
            object_asset_options.vhacd_params.max_convex_hulls = 10
            object_asset_options.vhacd_params.max_num_vertices_per_ch = 32

            if "object_density" in self.cfg["env"]:
                object_asset_options.density = self.cfg["env"]['object_density']
        #     object_asset_options.disable_gravity = True

        return object_asset_options

    def compute_robot_indices(self):
        urdf_version = self.cfg['env']['asset'].get('version', '2.0')
        if urdf_version == '2.0': 
            finger_names = \
                ["end_link", "end_link_2", "end_link_3", "end_link_4"] + \
                ["middle_link", "middle_link_2", "middle_link_3", 'thumb_midddle_link_6mm'] + \
                ["rotating_link", "rotating_link_2", "rotating_link_3", 'thumb_rotating_link'] + \
                ['link6']
            fingertip_names = ["end_link", "end_link_2", "end_link_3", "end_link_4"]
            arm_names = ["link5", "link4", "link3", "link2", "link1", "link_base"] #"link_base", "link1", "link2", "link3", "link4", 
            palm_names = ['link6']
        elif urdf_version == '4.0':
            finger_names = \
                ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"] + \
                ["pip", "pip_2", "pip_3", "thumb_pip"] + \
                ["mcp_joint", "mcp_joint_2", "mcp_joint_3", "pip_4"] + \
                ["dip", "dip_2", "dip_3", "thumb_dip"] + \
                ['link6']
            fingertip_names = ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]
            arm_names = ["link5", "link4", "link3", "link2", "link1", "link_base"] #"link_base", "link1", "link2", "link3", "link4", 
            palm_names = ['link6']
        elif urdf_version == '5.0':
            finger_names = \
                ["nail", "nail2", "nail3", "nail4"] + \
                ["pip", "pip_2", "pip_3", "thumb_pip"] + \
                ["mcp_joint", "mcp_joint_2", "mcp_joint_3", "pip_4"] + \
                ["dip", "dip_2", "dip_3", "thumb_dip"] + \
                ['link6']
            fingertip_names = ["nail", "nail2", "nail3", "nail4"]
            arm_names = ["link5", "link4", "link3", "link2", "link1", "link_base"] #"link_base", "link1", "link2", "link3", "link4", 
            palm_names = ['link6']
        else:
            raise "Unrecognized urdf version"
        fingertip_palm_names = fingertip_names + palm_names

        #self.gym.find_actor_rigid_body_handle(self.envs[0], 0, self.eef_name)
        self.robot_rb_indices = self.get_object_rb_indices(torch.arange(self.num_robot_bodies, device=self.device))

        self.link_base_index = self.get_object_indices(["link_base"])[0]
        self.eef_index = self.get_object_indices(['link6'])[0]

        
        self.palm_indices = self.get_object_indices(palm_names)
        self.finger_indices = self.get_object_indices(finger_names)
        self.fingertip_indices = self.get_object_indices(fingertip_names)
        self.fingertip_palm_indices = self.get_object_indices(fingertip_palm_names)
        self.arm_indices = self.get_object_indices(arm_names)


    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.abspath(self.cfg['env']['asset_root'])
        print(asset_root)

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            robot_asset_file = self.cfg["env"]["asset"].get("assetFileName")

        self.asset_root = asset_root
        self.robot_asset_file = robot_asset_file
        asset_options = self.get_robot_asset_options()
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.robot_asset = robot_asset
        #print("Num dofs: ", self.num_robot_dofs)
        self.num_robot_actuators = self.num_robot_dofs #self.gym.get_asset_actuator_count(robot_asset)
        self.num_arm_dofs = self.cfg["env"].get("num_arm_dofs", 6)
        self.num_hand_dofs = 16
        assert(self.num_robot_dofs == self.num_arm_dofs + self.num_hand_dofs)

        self.hand_actuated_dof_indices = [i for i in range(self.num_hand_dofs)]

        # set robot dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        #self.robot_dof_default_pos = []
        #self.robot_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        #import ipdb ; ipdb.set_trace()
        #self.projectile_aspect_ratio = torch.tensor([[1.0, 1.0, 1.0]] * self.num_envs, device=self.device)
        arm_stiffness = self.cfg["env"].get("arm_stiffness", [1000000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0])
        arm_damping = self.cfg["env"].get("arm_stiffness", [10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0])
        for i in range(self.num_robot_dofs):
            if i < self.num_arm_dofs:
                # robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT 
                # robot_dof_props['velocity'][i] = 1.0
                # print("Max effort: ", robot_dof_props['effort'][i])
                # robot_dof_props['damping'][i] = 100.0
                # robot_dof_props['stiffness'][i] = 1000
                # robot_dof_props['damping'][i] = 100
                # robot_dof_props['effort'][i] = 20.0
                # robot_dof_props['friction'][i] = 0.1
                # robot_dof_props['armature'][i] = 0.1
                robot_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
                robot_dof_props["stiffness"][i] = arm_stiffness[i]
                robot_dof_props["damping"][i] = arm_damping[i]
                
                #robot_dof_props['effort'][i] = ??? # 300
            else:
                robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS

                robot_dof_props['effort'][i] = self.cfg["env"].get("hand_effort", 0.8)
                robot_dof_props['velocity'][i] = self.cfg["env"].get("hand_velocity", 5)
                robot_dof_props['stiffness'][i] = self.cfg["env"].get("hand_stiffness", 3)
                robot_dof_props['damping'][i] = self.cfg["env"].get("hand_damping", 0.1)
                robot_dof_props['friction'][i] = 0.01
                robot_dof_props['armature'][i] = 0.001
            
            self.robot_dof_lower_limits.append(robot_dof_props['lower'][i])
            self.robot_dof_upper_limits.append(robot_dof_props['upper'][i])
            #self.robot_dof_default_pos.append(0.0)
            #self.robot_dof_default_vel.append(0.0)
        
        self.hand_actuated_dof_indices = to_torch(self.hand_actuated_dof_indices, dtype=torch.long, device=self.device)
        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits, device=self.device)
        #self.robot_dof_default_pos = to_torch(self.robot_dof_default_pos, device=self.device)
        #self.robot_dof_default_vel = to_torch(self.robot_dof_default_vel, device=self.device)

        #self.robot_default_dof_pos = torch.zeros(self.num_robot_dofs, dtype=torch.float, device=self.device)
        #self.robot_upper_pos = torch.zeros(self.num_robot_dofs, dtype=torch.float, device=self.device)
        #self.robot_default_dof_pos[6:] = ((self.robot_dof_lower_limits + self.robot_dof_upper_limits)/2)[6:]
        #self.robot_default_dof_pos[6:] = 0.0

        if "robot_lower_pos" in self.cfg["env"]:
            self.robot_lower_pos = to_torch(self.cfg["env"]["robot_lower_pos"], device=self.device).repeat((self.num_envs, 1))

        if "robot_upper_pos" in self.cfg["env"]:
            self.robot_upper_pos = to_torch(self.cfg["env"]["robot_upper_pos"], device=self.device).repeat((self.num_envs, 1))
                
        # load manipulated object and goal assets
        asset_options = self.get_object_asset_options()
        robot_start_pose = gymapi.Transform()

        # compute aggregate size
        max_agg_bodies = self.num_robot_bodies + 2
        max_agg_shapes = self.num_robot_shapes + 2

        self.robots = []
        self.envs = []
        self.hand_start_states = []
        self.hand_indices = []

        self.link_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.robot_rb_count = self.gym.get_asset_rigid_body_count(robot_asset)


        self.object_indices = []

        #print('num robot bodies', robot_rb_count)

        self.object_handles = []
        self.robot_handles = []
        self.env_ptrs = []
        self.object_rb_count = []



        self.time_impact = torch.zeros(self.num_envs, device=self.device)
        self.env_object_names = ['' for i in range(self.num_envs)]

        self.load_assets()

        self.proc_object_ids = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)

        for indice in range(self.robot_rb_count):
            sensor_pose = gymapi.Transform()
            self.gym.create_asset_force_sensor(robot_asset, indice, sensor_pose)

        self.obj_sem_id = 2
        self.robot_sem_id = 1
        print('generating throws')
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            robot_actor = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", i, -1, self.robot_sem_id) # TODO: -1
            robot_idx = self.gym.get_actor_index(env_ptr, robot_actor, gymapi.DOMAIN_SIM)
            #self.robot_indices.append(robot_idx)
            self.hand_start_states.append([robot_start_pose.p.x, robot_start_pose.p.y, robot_start_pose.p.z,
                                           robot_start_pose.r.x, robot_start_pose.r.y, robot_start_pose.r.z, robot_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.enable_actor_dof_force_sensors(env_ptr, robot_actor)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, robot_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # create fingertip force-torque sensors
            # if self.obs_type == "full_state" or self.asymmetric_obs:
            #self.gym.enable_actor_dof_force_sensors(env_ptr, robot_actor)
            #env_sensors.append(self.gym.create_force_sensor(env_ptr, ft_handle, sensor_pose))
            #self.sensors.append(env_sensors)
            #self.gym.enable_actor_dof_force_sensors(env_ptr, robot_actor)

            # add object
            # disable collision
            # collision_group = -(i+2)
            collision_group = i

            # if "procedural" in self.cfg["env"]["objectType"]:
            #self.object_scale_log = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device) 
            self.object_init_state = torch.zeros((self.num_envs,13), dtype=torch.float, device=self.device) 
            self.object_goal_pose = torch.zeros((self.num_envs,7), dtype=torch.float, device=self.device)
            
            #object_rb_count = 0
            #while object_rb_count != 1:
            
                
            object_id = np.random.choice(len(self.probabilities), p=self.probabilities)
            object_name = self.object_names[object_id]
            self.env_object_names[i] = object_name
            self.proc_object_ids[i] = object_id
            object_asset = self.object_assets[object_name]
            object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
            #print(object_asset, object_rb_count)
            assert(object_rb_count > 0)

            self.object_rb_count.append(object_rb_count)



            assert(self.cfg["env"].get("randomizeThrow", False))
            obj_cfg = self.init_projectile()
            self.object_init_state[i] = torch.tensor(obj_cfg['init_state'])
            self.time_impact[i] = obj_cfg['t_impact']
            if 'goal_pose' in obj_cfg:
                self.object_goal_pose[i] = torch.tensor(obj_cfg['goal_pose'])
            object_start_pose = gymapi.Transform()
            object_start_pose.p = gymapi.Vec3(*obj_cfg['start_pos'])
            object_start_pose.r = gymapi.Quat(*obj_cfg['start_rot'])
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "root", collision_group, 0, self.obj_sem_id)

            self.env_ptrs.append(env_ptr)
            self.robot_handles.append(robot_actor)
            self.object_handles.append(object_handle)


            #for k in range(self.robot_rb_count):
            #    self.gym.set_rigid_body_segmentation_id(env_ptr, robot_actor, k, self.robot_sem_id)
            #for k in range(object_rb_count):
            #    self.gym.set_rigid_body_segmentation_id(env_ptr, object_handle, k, self.obj_sem_id)
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.robots.append(robot_actor)
            self.attach_cameras(i, env_ptr, robot_actor)

        print('done')
        self.compute_robot_indices()


        self.randomize_initial_dof_pos(torch.arange(self.num_envs, device=self.device).long())

        #object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        #self.object_rb_masses = [prop.mass for prop in object_rb_props] 

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        #self.object_goal_state = to_torch(self.object_goal_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        #self.ep_object_init_state = self.object_init_state.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        # self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)

        
        #self.object_rb_handles = list(range(self.robot_rb_count, self.robot_rb_count + object_rb_count))

        #self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        #self.robot_indices = to_torch(self.robot_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.init_sdf()
        

        self.phase_success_time = self.cfg['env'].get('phase_success_time', 0.4)
        self.episode_success = (self.time_impact + self.phase_success_time)/self.dt

    def attach_cameras(self, idx, env_ptr, robot_actor):
        if not self.is_cameras_active:
            return

        if not hasattr(self, "cam_handles"):
            self.cam_handles = []
        
        env_cam_handles = []

        for camera_name in self.cfg["env"]["cameras"]["camera_names"].keys():
            camera_config = self.cfg["env"]["cameras"]["camera_names"][camera_name]
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg["env"]["cameras"]["camera_resolution"][1]
            camera_props.height = self.cfg["env"]["cameras"]["camera_resolution"][0]
            camera_props.horizontal_fov = self.cfg["env"]["cameras"]["horizontal_fov"]
            camera_props.enable_tensors = True
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
            env_cam_handles.append(camera_handle)

            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(*camera_config["location"])
            local_transform.r = gymapi.Quat.from_euler_zyx(
                                    np.radians(camera_config["rotation"][2]), 
                                    np.radians(camera_config["rotation"][1]), 
                                    np.radians(camera_config["rotation"][0])
                                )

            #self.gym.set_camera_location(camera_handle, env_ptr, local_transform.p, local_transform.r)
            if camera_config.get("body_name", "root") == "root":
                body_handle = self.gym.get_actor_root_rigid_body_handle(env_ptr, robot_actor)
            else:
                body_handle = self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, camera_config["body_name"])
            
            #self.gym.attach_camera_to_body(camera_handle, env_ptr, body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
            self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(*camera_config["location"]), gymapi.Vec3(*camera_config["rotation"]))

        self.cam_handles.append(env_cam_handles)

        if self.viewer is not None:
        # all coordinates need to be different
            self.default_cam_pos = gymapi.Vec3(20.5, 8, 1.5)
            self.default_cam_target = gymapi.Vec3(0, 7, 1.25)
            self.gym.viewer_camera_look_at(self.viewer, None, self.default_cam_pos, self.default_cam_target)
        #self.gym.prepare_sim(self.sim)

    def init_data(self):
        env_ptr = self.envs[0]
        robot_handle = 0
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        self.jacobian = jacobian

        eef_name = self.cfg["env"].get("eef_name", "link6")
        self.hand_index = self.gym.get_asset_rigid_body_dict(self.robot_asset)[eef_name] 
  
        #self._j_eef = jacobian[:, hand_index - 1, :, :self.num_arm_dofs]
        #_massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "robot")
        #mm = gymtorch.wrap_tensor(_massmatrix)
        #self._mm = mm[:, :self.num_arm_dofs, :self.num_arm_dofs]


    def reset_domain_rand_bug(self, min_dist, time_thresh):
        not_moving_x = self.root_state_tensor[self.object_indices, 7].abs() <= min_dist
        arm_contact = self.reset_arm_contact()
        dof_pos_limits = self.reset_dof_pos_limits()
        eef_height_in = self.reset_eef_height_in(0.15, 0.75)
        return (not_moving_x | arm_contact | dof_pos_limits | eef_height_in) & (self.progress_buf <= time_thresh)
    
    def reset_domain_rand_bug_dof_vel(self, max_vel):
        return torch.abs(self.robot_dof_vel).sum(axis=-1) >= max_vel

    def reset_prog_freeze(self, start_episode, max_vel, decay):
        decay = np.log(decay)
        threshold = max_vel * torch.exp(decay*(self.progress_buf - start_episode))
        return (self.progress_buf >= start_episode) & (torch.abs(self.robot_dof_vel).sum(axis=-1) >= threshold)
    
    def reset_freeze(self, start_episode, max_vel):
        return (self.progress_buf >= start_episode) & (torch.abs(self.robot_dof_vel).sum(axis=-1) >= max_vel)

    def reset_sdf_dist_fingertips_small(self, start_episode, max_dist):
        return (self.progress_buf >= start_episode) & (self.reward_sdf_dist(indices='fingertip_indices') > max_dist)


    def reset_catch_lost(self):
        if not hasattr(self, 'prev_catch_time'):
            self.prev_catch_time = self.catch_time
        reset_mask = (self.prev_catch_time > 0) & (self.catch_time == 0) & (self.progress_buf > 0)
        self.prev_catch_time = self.catch_time[:]
        return reset_mask

    def reset_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits_lower = self.robot_dof_pos < self.robot_dof_lower_limits # lower limit
        out_of_limits_upper = self.robot_dof_pos > self.robot_dof_upper_limits # lower limit
        reset_mask = out_of_limits_lower.long().sum(axis=-1) + out_of_limits_upper.long().sum(axis=-1)
        return reset_mask > 0

    def reset_air_lost(self):
        if not hasattr(self, 'prev_air_time'):
            self.prev_air_time = self.air_time[:]
        reset_mask = (self.prev_air_time > 0) & (self.air_time == 0) & (self.progress_buf > 0)
        self.prev_air_time = self.air_time[:]
        return reset_mask

    def reset_obj_height_in(self, min_height, max_height):
        object_pos = self.root_state_tensor[self.object_indices, :3]
        return ((object_pos[:,2] < min_height) | (object_pos[:,2] > max_height)) 

    def reset_obj_close_range(self, min_x=-0.3):
        if not hasattr(self, "close_range_x"):
            self.close_range_x = min_x
        object_pos = self.root_state_tensor[self.object_indices, :3]
        return (object_pos[:,0] < self.close_range_x)

    def reset_obj_height_above(self, min_height):
        object_pos = self.root_state_tensor[self.object_indices, :3]
        return (object_pos[:,2] < min_height) 

    def reset_eef_height_in(self, min_height, max_height):
        eef_pos = self.get_eef_pos()
        return ((eef_pos[:,2] < min_height) | (eef_pos[:,2] > max_height)) 

    def reset_phase_target_joints(self, dist):
        mean_joint = ((self.robot_dof_pos - self.robot_upper_pos)[:, :self.num_arm_dofs-1]**2).mean(dim=-1)
        reset_mask = mean_joint >= dist
        return reset_mask & self.phase_target_pos
    
    def reset_phase_target_eef(self, dist):
        if not hasattr(self, 'obj_target_pos'):
            default_dof_pos = self.robot_upper_pos[:, :self.num_arm_dofs]
            eef_pos, eef_quat = self.kinematics.forward_kinematics(default_dof_pos)
            self.obj_target_pos = eef_pos

        mean_dist = (self.get_eef_pos() - self.obj_target_pos).norm(dim=-1)
        reset_mask = mean_dist >= dist
        return reset_mask & self.phase_target_pos
    # def reset_arm_contact(self):
    #     contact_norms = self.robot_contact_forces.norm(dim=-1)
    #     return torch.sum(contact_norms[:, self.arm_indices]) > 0

    def reset_arm_contact(self):
        contact_norms = self.robot_contact_forces.norm(dim=-1)
        return torch.sum(contact_norms[:, self.arm_indices], dim=-1) > 0.5

    #def reset_arm_contact(self, max_length):
    #    return (self.arm_time >= max_length*self.dt) & (self.progress_buf > 0)
    
    # def get_box_dims(self):
    #     if self.cfg["env"]["objectType"] != "procedural_box":
    #         raise Exception("get_box_dims only works for procedural_box objects")
        
    #     return self.box_dims
    
    def get_progress_buf_pos_enc(self):
        if not hasattr(self, "pos_embeddings"):
            dim = self.cfg["env"]["observation_args"]["progress_buf_pos_enc"]["dim"]
            if dim % 2 != 0:
                raise Exception("dim must be even")
            
            div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float, device=self.device) * -(math.log(10000.0) / dim)))
            div_term = div_term[None, :]
            length = self.cfg["env"]["episodeLength"] + 1
            position = torch.arange(0, length, dtype=torch.float, device=self.device).unsqueeze(1)
            self.pos_embeddings = torch.zeros((length, dim), device=self.device)
            self.pos_embeddings[:, 0::2] = torch.sin(position * div_term)
            self.pos_embeddings[:, 1::2] = torch.cos(position * div_term)

        return self.pos_embeddings[self.progress_buf]


    def get_additional_reset_conditions(self):
        reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        #if "additional_reset_conditions" not in self.cfg["env"]:
        #    return reset_mask
        if not hasattr(self, 'reset_names'):
            reset_dict = self.cfg["env"]["additional_reset_conditions"]
            self.reset_names = list(reset_dict.keys())
            self.reset_fns = {key: functools.partial(getattr(self, f'reset_{key}'), *args) for key, args in reset_dict.items()}
            self.reset_stats = {key:[] for key in reset_dict.keys()}
            self.reset_counts = []
            self.reset_stat_counts = []
            self.reset_stat_after_impact = []
            self.stat_ignore_resets = self.cfg["env"].get("stat_ignore_resets", [])

        masks = {}
        for reset_name in self.reset_names:
            mask = self.reset_fns[reset_name]()
            masks[reset_name] = mask
            reset_mask |= mask
            self.reset_stats[reset_name].append(mask.sum().item())
            self.reset_stats[reset_name] = self.reset_stats[reset_name][-self.stat_horizon:]

            if (self.is_test or (not self.headless)) and mask.sum() > 0:
                print(f'Reset:\t{reset_name}\tcount={mask.sum().item()}')

        self.reset_buf_stat = reset_mask.clone() | self._reset_buf
        for reset_name in self.stat_ignore_resets:
            self.reset_buf_stat &= (~masks[reset_name])

        self.reset_stat_counts.append(self.reset_buf_stat.float().sum().item())
        self.reset_stat_counts = self.reset_stat_counts[-self.stat_horizon:]

        self.after_impact = reset_mask.clone() | self._reset_buf
        self.after_impact = self.after_impact & (self.progress_buf >= (self.time_impact / self.dt - 8))
        self.reset_stat_after_impact.append(self.after_impact.float().sum().item())
        self.reset_stat_after_impact = self.reset_stat_after_impact[-self.stat_horizon:]

        self.reset_counts.append(reset_mask.float().sum().item())
        self.reset_counts = self.reset_counts[-self.stat_horizon:]

        return reset_mask


    def compute_reward(self, actions):
        self.rew_buf = torch.zeros((self.num_envs), device=self.device)
        self._reset_buf = self.progress_buf >= self.episode_success 
        extras = {}

        # concatenate two dicts - extras and self.extras
        self.extras = {**self.extras, **extras}

        self.compute_all_times()
        self._reset_buf |= self.get_additional_reset_conditions()

        if not hasattr(self, 'additional_rewards'):
            self.additional_rewards = self.cfg["env"].get("additional_rewards", {}).items()

        for reward_name, reward_config in self.additional_rewards:
            if not hasattr(self, f'reward_{reward_name}'):#
                print(reward_name, self.is_test)
                continue

            if type(reward_config) is dict:
                args_list = reward_config["args"]
                reward_scale_list = reward_config["scale"]
                suffix_list = reward_config.get("suffix", [""] * len(args_list))
            else:
                args_list = [[]]
                reward_scale_list = [reward_config]
                suffix_list = [""]

            assert(len(args_list) == len(reward_scale_list))
            assert(len(args_list) == len(suffix_list))  

            for args, reward_scale, suffix in zip(args_list, reward_scale_list, suffix_list):
                reward_value_unscaled = eval("self.reward_{}(*args)".format(reward_name)) 
                reward_value_scaled = reward_value_unscaled * reward_scale
                self.extras["reward_{}{}_scaled".format(reward_name, suffix)] = reward_value_scaled.mean()
                self.extras["reward_{}{}_unscaled".format(reward_name, suffix)] = reward_value_unscaled.mean()

                self.rew_buf += reward_value_scaled
            
            self.rew_buf *= self.reset_buf_stat.float() + (1-self._reset_buf.float())
            if self.no_actions_before > 0:
                self.rew_buf *= (self.progress_buf >= self.no_actions_before).float()

    def compute_stat(self, name, buf):
        if not hasattr(self, f'num_{name}'):
            setattr(self, f'num_{name}', [])
            setattr(self, f'num_{name}_after_impact', [])
        getattr(self, f'num_{name}').append(buf[self.reset_buf_stat].float().sum().item())
        setattr(self, f'num_{name}', getattr(self, f'num_{name}')[-self.stat_horizon:])

        getattr(self, f'num_{name}_after_impact').append(buf[self.after_impact].float().sum().item())
        setattr(self, f'num_{name}_after_impact', getattr(self, f'num_{name}_after_impact')[-self.stat_horizon:])
        self.extras[name] = buf.float().mean().item()
        self.extras[f'{name}_at_reset'] = np.sum(getattr(self, f'num_{name}')) / np.sum(self.reset_stat_counts)
        self.extras[f'{name}_at_reset_after_impact'] = np.sum(getattr(self, f'num_{name}_after_impact')) / np.sum(self.reset_stat_after_impact)

    def compute_stats(self):
        self.compute_stat('stat_reset_ratio', self._reset_buf)
        self.compute_stat('stat_catch_length', self.catch_time.float()/self.dt)
        self.compute_stat('stat_ep_length', self.progress_buf)
        self.compute_stat('stat_only_air', self.air_time > 0)
        self.compute_stat('stat_t_length', self.progress_buf - self.time_impact / self.dt)
        self.compute_stat('stat_t_force_ratio', self.phase_force_perturb)
        self.compute_stat('stat_t_pos_ratio', self.phase_target_pos)
        self.compute_stat('stat_t_success_ratio', self.progress_buf >= self.episode_success)

        accuracy = self.progress_buf >= self.episode_success#self.phase_force_perturb
        self.extras['accuracy'] = accuracy.float().mean().item()
        if not hasattr(self, 'catch_rate_per_obj'):
            self.num_catch_per_obj = {obj_name:[] for obj_name in self.object_names}
            self.num_reset_per_obj = {obj_name:[] for obj_name in self.object_names}
        for obj_idx, obj_name in enumerate(self.object_names):
            idxs = self.reset_buf_stat & (self.proc_object_ids == obj_idx)
            if idxs.sum().item() > 0:
                self.num_catch_per_obj[obj_name].append(accuracy[idxs].sum().item())
                self.num_reset_per_obj[obj_name].append(idxs.sum().item())

        if hasattr(self, 'reset_names'):
            for reset_name in self.reset_names:
                reset_rate = np.sum(self.reset_stats[reset_name][-self.stat_horizon:]) / np.sum(self.reset_counts[-self.stat_horizon:])
                self.extras[f'stat_reset_{reset_name}'] = reset_rate

        for obj_name in self.object_names:
            success_rate = np.sum(self.num_catch_per_obj[obj_name][-self.stat_horizon:]) / np.sum(self.num_reset_per_obj[obj_name][-self.stat_horizon:])
            self.extras[f'stat_fp_{obj_name}'] = success_rate

        show = self.global_counter % (self.max_episode_length*10) == 0
        if show:
            print('********************************')
            for stat in ['stat_reset_ratio', 'stat_catch_length', 'stat_ep_length', 'stat_only_air', 'stat_t_length', 'stat_t_force_ratio', 'stat_t_pos_ratio', 'stat_t_success_ratio']:
                print(f'{stat:40}:\t{self.extras[stat]:.3f}')
                print(f'{stat+"_at_reset":40}:\t{self.extras[stat+"_at_reset"]:.3f}')
            print()
            print('********************************')
            print("Success per object type breakdown: ")
            for obj_name in self.object_names:
                print(f'{obj_name:10}\t{self.extras[f"stat_fp_{obj_name}"]*100:.1f}%')
            print()
            print('********************************')
            print("Reset type breakdown: ")
            for reset_name in self.reset_names:
                print(f'{reset_name:10}\t{self.extras[f"stat_reset_{reset_name}"]*100:.1f}%')


        if not hasattr(self, 'scale_rng'):
            self.scale_rng = self.randomization_params['actor_params']['root']['scale']['range']
            self.friction_rng = self.randomization_params['actor_params']['root']['rigid_shape_properties']['friction']['range']
            self.mass_rng = self.randomization_params['actor_params']['root']['rigid_body_properties']['mass']['range']
            self.v_rng = self.cfg['env']['stat_vel_range']
            self.w_rng = self.cfg['env']['stat_ang_vel_range']
            self.t_rng = self.cfg['env']['stat_time_range']

            eef_start_pos = np.array(self.cfg['env']['init_ee_pos_rand_start'])
            eef_end_pos = np.array(self.cfg['env']['init_ee_pos_rand_start'])
            eef_scale = max(np.linalg.norm(eef_start_pos), np.linalg.norm(eef_end_pos))
            self.hand_rng = [0, 2*np.pi]
            self.eef_rng = [0, eef_scale]
            quat_scale = np.max(self.cfg['env']['init_ee_quat_rand_scale'])
            self.quat_rng = [0, quat_scale]

        def split_rng(rng, n_splits=3):
            a, b = rng
            return [a + (i/n_splits)*(b-a) for i in range(n_splits+1)]

        self.get_success_table("stat_scale", "Scale", split_rng(self.scale_rng, 3), self.get_randomization_obj_scale().squeeze(), accuracy, verbose=show)
        self.get_success_table("stat_friction", "Friction", split_rng(self.friction_rng, 3), self.get_randomization_obj_friction().squeeze(), accuracy, verbose=show)
        self.get_success_table("stat_mass", "Mass", split_rng(self.mass_rng, 3), self.get_randomization_obj_mass().squeeze(), accuracy, verbose=show)
        self.get_success_table("stat_lin_vel", "Lin. Velocity", split_rng(self.v_rng, 3), self.object_init_state[:,7:10].norm(dim=-1), accuracy, verbose=show)
        self.get_success_table("stat_ang_vel", "Ang. Velocity", split_rng(self.w_rng, 3), self.object_init_state[:,10:13].norm(dim=-1), accuracy, verbose=show)
        self.get_success_table("stat_t_throw", "Time throw", split_rng(self.t_rng, 10), self.time_impact, accuracy, verbose=show)
        self.get_success_table("stat_rand_hand", "Hand rand", split_rng(self.hand_rng, 6), self.robot_default_dof_pos[:, self.num_arm_dofs-1], accuracy, verbose=show)
        self.get_success_table("stat_rand_arm_xyz", "Arm rand pos", split_rng(self.eef_rng, 3), self.robot_default_delta_pose[:,:3].norm(dim=-1), accuracy, verbose=show)
        self.get_success_table("stat_rand_arm_rpy", "Arm rand rpy", split_rng(self.quat_rng, 6), self.robot_default_delta_pose[:,3:].norm(dim=-1), accuracy, verbose=show)

    
    def get_success_table(self, name, title, X, indices, success, verbose=False):
        #map object scale to success
        X = [float('-inf')] + X + [float('inf')]

        if not hasattr(self, name):
            setattr(self, name, [[] for x in X[:-1]])

        for i, x in enumerate(X[:-1]):
            indices_within_range = ((indices >= X[i]) & (indices < X[i+1]))
            selected_success = success[indices_within_range & self.reset_buf_stat]
            if len(selected_success) > 0 or len(getattr(self, name)) < self.stat_horizon:
                getattr(self, name)[i].append(selected_success)
                getattr(self, name)[i] = getattr(self, name)[i][-self.stat_horizon:]

        xs, ys, ns = [], [], []
        for i, x in enumerate(X[:-1]):
            mean_selected_objects_success = torch.cat(getattr(self, name)[i]).float().mean().item()
            self.extras[f'{name}_in_[{X[i]:.3f},{X[i+1]:.3f}]'] = mean_selected_objects_success
            xs.append(f'{x:.2f}')
            ys.append(f'{mean_selected_objects_success*100:.1f}%')
            # to align both x and y
            ns.append(max(len(xs[-1]), len(ys[-1])))

        if verbose:
            # spaces to align both x and y
            xs, ys = '\t'.join([f'{x:{n}}' for n,x in zip(ns,xs)] + [f'{X[-1]:.2f}']), '\t'.join([f'{y:{n}}' for n,y in zip(ns,ys)])
            print("********************************")
            print(f"Success per {title} breakdown: ")
            print(f"{title:10} \t| {xs}")
            print(f"Success rate \t| {ys}")

    def reward_action_rate_last_phase(self):
        if not hasattr(self, 'last_actions_last_phase'):
            self.last_actions_last_phase = self.actions
        reward = torch.mean(torch.square(self.last_actions_last_phase - self.actions), dim=1)
        self.last_actions_last_phase = self.actions
        return self.phase_force_perturb * reward

    def reward_arm_action_rate_last_phase(self):
        arm_actions = self.actions[:, :self.num_arm_dofs-1]
        if not hasattr(self, 'last_arm_actions_last_phase'):
            self.last_arm_actions_last_phase = arm_actions
        reward = torch.mean(torch.square(self.last_arm_actions_last_phase - arm_actions), dim=1)
        self.last_arm_actions_last_phase = arm_actions
        return self.phase_force_perturb * reward

    def reward_action_rate(self):
        if not hasattr(self, 'last_actions'):
            self.last_actions = self.actions
        reward = torch.mean(torch.square(self.last_actions - self.actions), dim=1)
        self.last_actions = self.actions
        return reward

    def reward_arm_action_rate(self):
        arm_actions = self.actions[:, :self.num_arm_dofs-1]
        if not hasattr(self, 'last_arm_actions'):
            self.last_arm_actions = arm_actions
        # Penalize changes in actions
        reward = torch.mean(torch.square(self.last_arm_actions - arm_actions), dim=1)
        self.last_arm_actions = arm_actions
        return reward

    def reward_hand_action_rate_l1(self):
        hand_actions = self.actions[:, self.num_arm_dofs:]
        if not hasattr(self, 'last_hand_actions'):
            self.last_hand_actions = hand_actions
        # Penalize changes in actions
        reward = torch.mean(torch.abs(self.last_hand_actions - hand_actions), dim=1)
        self.last_hand_actions = hand_actions
        return reward

    def reward_eef_rate_l1(self, hist=3):
        eef_pos = self.get_eef_pose()[:, :3]
        if not hasattr(self, 'last_eef_pos_l1'):
            self.last_eef_pos_l1 = [eef_pos]
        # Penalize changes in actions
        last_eef_pos = torch.stack(self.last_eef_pos_l1).mean(dim=0)
        reward = torch.mean(torch.abs(eef_pos - last_eef_pos), dim=1)
        self.last_eef_pos_l1.append(eef_pos)
        self.last_eef_pos_l1 = self.last_eef_pos_l1[-hist:]
        return reward
    def reward_eef_rate_l2(self, hist=3):
        eef_pos = self.get_eef_pose()[:, :3]
        if not hasattr(self, 'last_eef_pos_l2'):
            self.last_eef_pos_l2 = [eef_pos]
        # Penalize changes in actions
        last_eef_pos = torch.stack(self.last_eef_pos_l2).mean(dim=0)
        reward = torch.mean(torch.square(eef_pos - last_eef_pos), dim=1)
        self.last_eef_pos_l2.append(eef_pos)
        self.last_eef_pos_l2 = self.last_eef_pos_l2[-hist:]
        return reward

    def reward_weighted_hand_action_rate_l1(self, weights, hist=1):
        hand_actions = self.actions[:, self.num_arm_dofs:]
        if not hasattr(self, 'last_hand_actions'):
            self.last_hand_actions = [hand_actions]
            self.hand_action_rate_weights = torch.tensor(weights, device=self.device)
        
        # Penalize changes in actions
        last_hand_actions = torch.stack(self.last_hand_actions).mean(dim=0)
        reward = torch.mean(self.hand_action_rate_weights * torch.abs(hand_actions - last_hand_actions), dim=1)
        self.last_hand_actions.append(hand_actions)
        self.last_hand_actions = self.last_hand_actions[-hist:]
        return reward

    def reward_weighted_arm_action_rate_l1(self, weights, hist=1):
        arm_actions = self.actions[:, :self.num_arm_dofs-1]
        if not hasattr(self, 'last_arm_actions'):
            self.last_arm_actions = [arm_actions]
            self.arm_action_rate_weights = torch.tensor(weights, device=self.device)
        
        # Penalize changes in actions
        last_arm_actions = torch.stack(self.last_arm_actions).mean(dim=0)
        reward = torch.mean(self.arm_action_rate_weights * torch.abs(arm_actions - last_arm_actions), dim=1)
        self.last_arm_actions.append(arm_actions)
        self.last_arm_actions = self.last_arm_actions[-hist:]
        return reward

    def reward_arm_action_acc(self):
        arm_actions = self.actions[:, :self.num_arm_dofs-1]
        if not hasattr(self, 'prev_arm_speed'):
            self.prev_arm_speed, self.last_arm_actions = torch.zeros_like(arm_actions), arm_actions
        arm_speed = arm_actions - self.last_arm_actions

        reward = torch.mean(torch.square(arm_speed - self.prev_arm_speed), dim=1)

        self.prev_arm_speed, self.last_arm_actions = arm_speed, arm_actions

        return reward
    # def reward_arm_default_pos(self):
    #     reward = torch.mean(torch.abs((self.robot_dof_pos - self.robot_upper_pos)[:, :self.num_arm_dofs]), dim=-1)
    #     return reward

    # def reward_arm_default_pos_ep(self):
    #     dist = torch.mean(torch.abs((self.robot_dof_pos - self.robot_upper_pos)[:, :self.num_arm_dofs]), dim=-1)
    #     start = self.cfg['env'].get('arm_default_pos_episode', 30)
    #     reward = dist*torch.maximum(self.progress_buf - start, torch.zeros_like(self.progress_buf)).float() / (self.max_episode_length - start)
    #     return reward

    def reward_episode_length(self):
        return torch.maximum(self.progress_buf.float() - self.time_impact / self.dt, torch.zeros_like(self.time_impact)) / self.max_episode_length

    def compute_all_times(self):
        #contact_norms = self.robot_contact_forces.norm(dim=-1)
        #arm_contact = contact_norms[:, self.arm_indices] > 0.5
        #any_arm_contact = arm_contact.sum(dim=-1) > 0
        #reward = self.compute_filter_time('arm_time', any_arm_contact)

        if not hasattr(self, 'catch_threshold'):
            self.catch_threshold = self.cfg["env"]["catch_threshold"]
            self.air_threshold = self.cfg["env"]["air_threshold"]
            self.phase_target_pos_catch_time_threshold = self.cfg['env'].get('phase_target_pos_catch_time', 0.2)

        object_pos = self.root_state_tensor[self.object_indices, :3]
        dist_hand_obj = self.sdf_dist[:, self.fingertip_palm_indices].mean(dim=-1)
        caught = dist_hand_obj < self.catch_threshold 
        self.compute_filter_time('catch_time', caught)
        air = object_pos[:, 2] >= self.air_threshold
        self.compute_filter_time('air_time', air)


        self.phase_target_pos = (self.progress_buf * self.dt) >= (self.time_impact + self.phase_target_pos_catch_time_threshold)


    def compute_contact_filter(self):
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact_forces = self.robot_contact_forces.norm(dim=-1)
        contact = contact_forces > 1.

        if not hasattr(self, 'last_contacts'):
            self.last_contacts = torch.zeros_like(contact)
            self.last_contact_forces = torch.zeros_like(contact_forces)

        self.contact_filt = (contact | self.last_contacts) & (self.sdf_dist <= 0.05) 
        self.contact_forces_filt = (contact_forces + self.last_contact_forces)/2

        self.last_contacts = contact
        self.last_contact_forces = contact_forces
        
        # d = 0.002
        # self.contact_filt = (self.sdf_dist <= d) 
        # self.contact_forces_filt = torch.exp(-8*torch.clamp(self.sdf_dist, min=0, max=d)/d)


    def compute_filter_time(self, name, filter):
        if not hasattr(self, name):
            setattr(self, name, torch.zeros(self.num_envs, dtype=torch.float, device=self.device))
        old_time = getattr(self, name)
        new_time = (old_time*(1-self._reset_buf.float())+self.dt)*filter.float()
        setattr(self, name, new_time)
        return getattr(self, name)

    def reward_arm_time(self):
        return self.arm_time

    def reward_catch_time(self):
        return self.catch_time

    def reward_palm_time(self):
        return self.palm_time

    def reward_air_time(self):
        return self.air_time

    # https://forums.developer.nvidia.com/t/getting-contact-points-state-from-isaacgym-environment/193849/2
    # No way to distinguish which contact...
    # early on, the object will always hit the arm- we don't want to learn to avoid it
    def reward_arm_contact(self):
        contact_norms = self.robot_contact_forces.norm(dim=-1)
        arm_contact = contact_norms[:, self.arm_indices] #> 0.1
        reward = torch.mean(arm_contact.float(), dim=-1)
        return reward


    def reward_alive(self):
        return torch.ones_like(self.rew_buf)
    def reward_not_reset(self):
        return 1 - self._reset_buf.float()

    def reward_obj_eef_dist(self, names='fingertip_indices'):
        hand_pos = self.get_eef_pose()[:, :3]
        obj_pos = self.root_state_tensor[self.object_indices, :3]
        reward = torch.norm(hand_pos - obj_pos, dim=-1)**2
        return reward

    def reward_obj_dist(self, names='fingertip_indices'):
        indices = getattr(self, names)
        fingertip_pos = self.robot_rb_states[:, indices, :3]
        object_pos = self.root_state_tensor[self.object_indices, :3]
        reward = torch.norm(fingertip_pos - object_pos[:, None, :], dim=-1)
        reward = (reward**2).mean(dim=-1)

        hand_pos = self.get_eef_pose()[:, :3]
        reward = reward * (torch.norm(object_pos - hand_pos, dim=-1) <= 3*self.get_obj_bb().max(dim=-1)[0]) 

        return reward

    def reward_obj_fingertip_dist(self):
        return self.reward_obj_dist('fingertip_indices')

    def reward_obj_finger_dist(self):
        return self.reward_obj_dist('finger_indices')

    def reward_fingertip_palm_contact_norm(self):
        return self.reward_finger_contact_norm('fingertip_palm_indices')

    def reward_finger_contact_norm(self, indices='finger_indices'):
        indices = getattr(self, indices)
        finger_contacts = self.contact_filt[:, indices] * self.contact_forces_filt[:, indices]
        return torch.clamp(finger_contacts, max=2).mean(dim=-1)


    def reward_obj_perp_scaled(self):
        hand_pose = self.get_eef_pose()
        hand_pos, hand_quat = hand_pose[:,:3], hand_pose[:,3:]

        obj_pose = self.get_obj_pose()
        obj_pos, obj_quat = obj_pose[:,:3], obj_pose[:,3:]

        # Get the indices of the max and second max elements along dim=1
        obj_bb = F.normalize(self.get_obj_bb(), dim=-1)
        max_values, max_indices = torch.topk(obj_bb, k=2, dim=1)
        max_axis = F.one_hot(max_indices, num_classes=3).float()

        hand_axis = quat_apply(hand_quat, to_torch([[0, -1, 0]], device=self.device).repeat(self.num_envs, 1))
        major_axis = quat_apply(obj_quat, max_axis[:,0,:])
        minor_axis = quat_apply(obj_quat, max_axis[:,1,:])

        corr_major_minor = 1 - max_values[:,1]/max_values[:,0]

        correlation = torch.abs(torch.sum(hand_axis * major_axis, dim=-1))
        reward = correlation * corr_major_minor
        #reward = reward * (torch.norm(obj_pos - hand_pos, dim=-1) <= 0.25) 
        
        return reward


    def reward_obj_target_joints(self):
        mean_joint = ((self.robot_dof_pos - self.robot_upper_pos)[:, :self.num_arm_dofs-1]**2).mean(dim=-1)
        return mean_joint
    def reward_obj_target_joints_last_phase(self):
        mean_joint = ((self.robot_dof_pos - self.robot_upper_pos)[:, :self.num_arm_dofs-1]**2).mean(dim=-1)
        return mean_joint * self.phase_force_perturb.float()

    def reward_torques(self, max_torque=1000):
        # Penalize torques
        return torch.mean(torch.clamp(torch.abs(self.robot_torques), max=max_torque)**2, dim=1)

    def reward_hand_dof_vel(self, limit=10):
        hand_dof_vel = self.get_hand_dof_vel()
        return torch.clip(torch.abs(hand_dof_vel), max=limit).sum(dim=-1)
    def reward_arm_dof_vel(self, limit=10):
        arm_dof_vel = self.get_arm_dof_vel()
        return torch.clip(torch.abs(arm_dof_vel), max=limit).sum(dim=-1)
    # def reward_hand_pose_vae_log_prob(self):
    #     if not hasattr(self, "hand_joints_vae_encoder"):
    #         self.hand_joints_vae_encoder = torch.jit.load(self.cfg["env"]["hand_joints_vae_encoder"]).to(self.device)

    #     latent = self.hand_joints_vae_encoder(self.robot_dof_pos[:, self.num_arm_dofs:])
    #     log_prob = -0.5 * torch.sum(latent ** 2, dim=1) - 0.5 * latent.shape[1] * np.log(2 * np.pi)

    #     return log_prob

    # def reward_hand_torque_l1(self):
    #     hand_dof_torques = self.robot_torques[:, self.num_arm_dofs:]
    #     reward = torch.sum(torch.abs(hand_dof_torques), dim=-1)

    #     return reward

    # def reward_hand_obj_exp(self, scale=1.0):
    #     hand_pos = self.get_eef_pose()[:, :3]
    #     obj_pos = self.get_obj_pos()
    #     reward = torch.exp(-torch.norm(hand_pos - obj_pos, dim=-1) / scale)

    #     return reward

    # def reward_hand_obj_ref_point(self):
    #     hand_pos = self.get_eef_pose()[:, :3]
    #     ref_point_pos = self.get_object_reference_points_world()
    #     reward = torch.norm(hand_pos - ref_point_pos, dim=-1)

    #     return reward
    
    # def reward_object_pose_diff(self):
    #     default_object_quat = self.ep_object_init_state[:, 3:7]
    #     curr_object_quat = self.get_obj_pose()[:, 3:]
    #     quat_diff = quat_mul(curr_object_quat, quat_conjugate(default_object_quat))
    #     reward = torch.abs(quat_diff[:, :3]).sum(dim=-1)

    #     return reward

    def reward_ee_orn_diff(self):
        default_ee_quat = self.default_ee_pose[:, 3:7]
        curr_ee_quat = self.get_eef_pose()[:, 3:7]
        quat_diff = quat_mul(curr_ee_quat, quat_conjugate(default_ee_quat))
        reward = torch.abs(quat_diff[:, :3]).sum(dim=-1)

        return reward

    def reward_work_abs(self, max_torque=1000, max_vel=300):
        reward = torch.mean(torch.clamp(torch.abs(self.robot_torques), max=max_torque) * torch.clamp(torch.abs(self.robot_dof_vel), max=max_vel), dim=-1)
        
        return reward

    def reward_work_abs_hand(self):
        reward = torch.sum(torch.abs(self.robot_torques[:, self.num_arm_dofs:] * self.robot_dof_vel[:, self.num_arm_dofs:]), dim=-1)

        return reward
    
    def reward_hand_targets_delta(self):
        delta = self.hand_prev_targets - self.hand_cur_targets
        reward = torch.norm(delta, dim=-1)

        return reward


    def reward_dof_pos_limits(self, alpha=0.95):
        # Penalize dof positions too close to the limit
        dof_pos = unscale(self.robot_dof_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        out_of_limits = -(dof_pos + alpha).clip(max=0.) # lower limit
        out_of_limits += (dof_pos - alpha).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.robot_dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def reward_torque_limits(self):
        # penalize torques too close to the limit
        if self.control_type == 'torque':
            return torch.sum((torch.abs(self.torques) - self.torque_limits).clip(min=0.), dim=1)
        return torch.zeros((self.num_envs), device=self.device)

    def get_time_throw(self):
        return (self.progress_buf * self.dt - self.time_impact).reshape(-1,1)

    def get_robot_dof_pos(self):
        return unscale(self.robot_dof_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits) #[:, :self.num_arm_dofs]

    def get_robot_default_pos_error(self):
        return (self.robot_dof_pos - self.robot_upper_pos)[:, :self.num_arm_dofs-1]

    def get_robot_default_pos_error_unscaled(self):
        dof_pos = unscale(self.robot_dof_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        dof_def = unscale(self.robot_upper_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        return (dof_pos - dof_def)[:, :self.num_arm_dofs]

    def get_robot_dof_lower_limits(self):
        return (self.robot_dof_pos - self.robot_dof_lower_limits)#[:, :self.num_arm_dofs]
    def get_robot_dof_upper_limits(self):
        return (self.robot_dof_upper_limits - self.robot_dof_pos)#[:, :self.num_arm_dofs]

    def get_depth_latent(self, obs_input):
        if not self.is_phase2:
            raise NotImplemented
        
        if not hasattr(self, 'depth_backbone'):
            phase2_train_cfg = self.cfg['env']['phase2']
            num_input = {'xyz': 3, 'xyzs': 5, 'dxdydz': 3, 'depth': 1, 'dd': 1}
            self.depth_backbone = eval(self.cfg['env']['phase2']["backbone_class"])(
                num_cameras=self.num_cameras, 
                num_input=num_input[phase2_train_cfg["buffer_type"]]*phase2_train_cfg.get('num_camera_frames',1),
                latent_dim=len(self.obs_phase1), 
                num_obs_input=len(self.obs_input),
                num_obs_phase1=len(self.obs_phase1),
                res=self.camera_res)
            checkpoint = torch.load(self.cfg['env']['checkpoint'])

            def get_model_weights_from_ddp(items):
                return {k.replace('module.', ''): v for k, v in items.items()}


            self.depth_backbone.load_state_dict(get_model_weights_from_ddp(checkpoint['depth']), strict=True)
            
            self.depth_backbone = self.depth_backbone.to(self.device)
            self.depth_backbone.eval()
            
        assert(self.depth_buffer is not None)

        #obs_buf = self.obs_buf[:, 34:]
        depth_latent = self.depth_backbone(self.get_buffer_phase2().clone(), obs_input)
        return depth_latent
        #return torch.cat([obs_buf[:, :16], obs_buf[:, 16:17], obs_buf[:, 17:]], dim=-1)
    
    def get_hand_dof_pos(self):
        return self.robot_dof_pos[:, self.num_arm_dofs:]


    def get_fingertip_palm_contact_norm(self, indices='fingertip_palm_indices'):
        indices = getattr(self, indices)
        return self.contact_filt[:, indices] * self.contact_forces_filt[:, indices]
        
    def get_fingertip_palm_contact(self):
        return self.contact_filt[:, self.fingertip_palm_indices]

    def get_randomization_sim_gravity(self):
        return torch.zeros((self.num_envs,1), device=self.device) + self.sim_props.gravity.z # - self.original_props["sim_params"]['gravity'].z
    def get_finger_palm_sdf_dist(self):
        return self.sdf_dist[:, self.fingertip_palm_indices] #if hasattr(self, "sdf_dist") else torch.zeros((self.num_envs,len(self.fingertip_palm_indices)), device=self.device)
    
    def get_finger_palm_sdf_dist_clipped(self):
        if not hasattr(self, 'max_clip_sdf_dist'):
            self.max_clip_sdf_dist = self.cfg.get('max_clip_sdf_dist', float('inf'))
        return torch.clip(self.sdf_dist[:, self.fingertip_palm_indices], max=self.max_clip_sdf_dist) #if hasattr(self, "sdf_dist") else torch.zeros((self.num_envs,len(self.fingertip_palm_indices)), device=self.device)
    
    def get_randomization_obj_scale(self):
        #return self.obj_scale.view(-1, 1)#
        return self.rand_scale.view(-1,1)
    def get_randomization_obj_mass(self):
        return torch.tensor([[getattr(prop, 'mass')] for prop in self.obj_rb_props], device=self.device)
    def get_randomization_obj_friction(self):
        return torch.tensor([[getattr(prop, 'friction')] for prop in self.obj_rs_props], device=self.device)

    def get_randomization_robot_mass(self):
        return torch.tensor([[getattr(prop, 'mass')] for prop in self.robot_rb_props], device=self.device)
    def get_randomization_robot_friction(self):
        return torch.tensor([[getattr(prop, 'friction')] for prop in self.robot_rs_props], device=self.device)
    def get_randomization_robot_damping(self):
        return torch.tensor([prop['damping'] for prop in self.robot_dof_props], device=self.device)
    def get_randomization_robot_stiffness(self):
        return torch.tensor([prop['stiffness'] for prop in self.robot_dof_props], device=self.device)

    def get_object_name(self):
        projectile_identity = self.proc_object_ids
        projectile_identity = projectile_identity.unsqueeze(1)
        # Define the mapping dictionary
        mapping = {
            0: 0,
            2: 0,
            14: 0,
            1: 1,
            4: 1,
            13: 1,
            3: 3,
            5: 5,
            17: 5,
            18: 18,
            6: 6,
            7: 7,
            8: 7,
            9: 7,
            15: 7,
            10: 10,
            11: 11,
            16: 11,
            12: 1,
            17: 5,
            19: 5,
            20: 5,
            21: 7
        }

        projectile_identity = torch.tensor([[mapping[val.item()] for val in row] for row in projectile_identity])

        return projectile_identity

    # def get_center_of_mass(self):
    #     center_of_mass = torch.tensor(self.projectile_center_of_mass)
    #     return center_of_mass

    # def get_aspect_ratio(self):
    #     aspect_ratio = self.projectile_aspect_ratio
    #     return aspect_ratio

    def get_default_hand_dof_pos(self):
        return self.robot_default_dof_pos[:, self.num_arm_dofs:]
    
    def get_arm_dof_pos(self):
        return self.robot_dof_pos[:, :self.num_arm_dofs]

    def get_hand_dof_vel(self):
        return self.robot_dof_vel[:, self.num_arm_dofs:]
    
    def get_arm_dof_vel(self):
        return self.robot_dof_vel[:, :self.num_arm_dofs]


    def get_obj_pose(self):
        if not hasattr(self, 'object_pose'):
            return torch.zeros((self.num_envs, 7), device=self.device)
        return self.standardize_pose(self.object_pose)
    
    # def get_init_obj_pose(self):
    #     return self.standardize_pose(self.ep_object_init_state[:, :7])

    def get_obj_speed(self):
        return torch.norm(self.root_state_tensor[self.object_indices, 7:10], dim=-1).reshape(-1,1)

    def get_obj_vel(self):
        return self.root_state_tensor[self.object_indices, 7:10]
    
    def get_obj_acc(self):
        acc = torch.zeros((self.num_envs, 3), device=self.device)
        acc[:,2:3] = self.get_randomization_sim_gravity()*self.get_randomization_obj_mass()
        return acc 

    def get_obj_ang_speed(self):
        return torch.norm(self.root_state_tensor[self.object_indices, 10:13], dim=-1).reshape(-1,1)

    def get_obj_ang_vel(self):
        return self.root_state_tensor[self.object_indices, 10:13]

    def get_obj_pos(self):
        return self.object_pose[:, :3]

    def get_eef_xyzq(self):
        pose = self.robot_rb_states[:, self.eef_index, :][:,:7]
        pose = self.standardize_pose(pose).clone()
        xyz, quat = pose[:,:3], pose[:, 3:]
        if not hasattr(self, 'xyz_min_clip'):
            self.xyz_min_clip = self.cfg["env"].get('xyz_min_clip', [0, -0.5, 0])
            self.xyz_max_clip = self.cfg["env"].get('xyz_max_clip', [5, +0.5, 2])
        xyz[:,0] = 2*(torch.clip(xyz[:,0], min=self.xyz_min_clip[0], max=self.xyz_max_clip[0]) - self.xyz_min_clip[0]) / (self.xyz_max_clip[0] - self.xyz_min_clip[0]) - 1
        xyz[:,1] = 2*(torch.clip(xyz[:,1], min=self.xyz_min_clip[1], max=self.xyz_max_clip[1]) - self.xyz_min_clip[1]) / (self.xyz_max_clip[1] - self.xyz_min_clip[1]) - 1
        xyz[:,2] = 2*(torch.clip(xyz[:,2], min=self.xyz_min_clip[2], max=self.xyz_max_clip[2]) - self.xyz_min_clip[2]) / (self.xyz_max_clip[2] - self.xyz_min_clip[2]) - 1
        return torch.cat([xyz, quat], dim=1)

    def get_obj_xyz(self):
        if not hasattr(self, 'xyz_min_clip'):
            self.xyz_min_clip = self.cfg["env"].get('xyz_min_clip', [0, -0.5, 0])
            self.xyz_max_clip = self.cfg["env"].get('xyz_max_clip', [5, +0.5, 2])
        xyz = self.get_obj_pos().clone()
        xyz[:,0] = 2*(torch.clip(xyz[:,0], min=self.xyz_min_clip[0], max=self.xyz_max_clip[0]) - self.xyz_min_clip[0]) / (self.xyz_max_clip[0] - self.xyz_min_clip[0]) - 1
        xyz[:,1] = 2*(torch.clip(xyz[:,1], min=self.xyz_min_clip[1], max=self.xyz_max_clip[1]) - self.xyz_min_clip[1]) / (self.xyz_max_clip[1] - self.xyz_min_clip[1]) - 1
        xyz[:,2] = 2*(torch.clip(xyz[:,2], min=self.xyz_min_clip[2], max=self.xyz_max_clip[2]) - self.xyz_min_clip[2]) / (self.xyz_max_clip[2] - self.xyz_min_clip[2]) - 1
        return xyz

    def get_obj_rpy(self):
        quat = self.object_pose[:, 3:7]
        r, p, y = get_euler_xyz(quat)

        return torch.stack((r, p, y), dim=1)

    def get_eef_quat(self):
        return self.get_eef_pose()[:, 3:7]

    def get_eef_pose(self):
        pose = self.robot_rb_states[:, self.eef_index, :][:,:7]
        return self.standardize_pose(pose)

    def get_eef_vel(self):
        return self.robot_rb_states[:, self.eef_index, :][:,7:10]

    def get_obj_size(self):
        return self.get_obj_bb().max(axis=-1)[0].reshape(-1, 1)

    def get_obj_max_size(self):
        return self.pcs_bb.max(axis=-1)[0].reshape(-1, 1)

    def get_obj_min_size(self):
        return self.pcs_bb.min(axis=-1)[0].reshape(-1, 1)

    def get_obj_bb(self):
        return self.pcs_bb if hasattr(self, 'pcs_bb') else torch.zeros((self.num_envs, 3), device=self.device)

    def get_real_default_hand_dof_pos(self):
        return self.get_default_hand_dof_pos()

    def get_real_arm_dof_pos(self):
        #code, angles = self.arm.get_servo_angle(is_radian=True, is_real=True)
        #assert(code == 0) 
        #return torch.tensor([angles[:-1]], device=self.device)
        if hasattr(self, "real_arm_dof_pos"):
            return self.real_arm_dof_pos
        else:
            return None

    def get_real_hand_dof_pos(self):
        if not hasattr(self, "real_hand_dof_pos"):
            return self.get_hand_dof_pos()
            #return None
        else:
            return self.real_hand_dof_pos

    def get_real_robot_dof_pos(self):
        return torch.cat([
            self.get_real_arm_dof_pos(),
            self.get_real_hand_dof_pos()
        ], dim=-1)
    
    def get_real_hand_targets(self):
        return self.real_hand_cur_targets


    def get_real_hand_dof_vel(self):
        raise NotImplemented


    def get_real_arm_dof_vel(self):
        raise NotImplemented

    # def get_eef_goal_delta_pos(self):
    #     return self.get_eef_pos() - self.goal_states[:, :3]

    def get_real_eef_goal_delta_rot(self):
        raise NotImplemented
    

    def get_real_obj_pose(self):
        raise NotImplemented

    def get_real_eef_pose(self):
        real_eef_pose_version = self.cfg["env"].get("real_eef_pose_version", "hardware")

        if real_eef_pose_version == "hardware":
            if not hasattr(self, "real_eef_pose"):
                return None
            
            return self.standardize_pose(self.real_eef_pose)
        elif real_eef_pose_version == "forward_kinematics":
            if not hasattr(self, "real_eef_pose_fk"):
                return None
            
            return self.standardize_pose(self.real_eef_pose_fk)

    def get_eef_angvel(self):
        return self.robot_rb_states[:, self.eef_index, :][:,10:13]

    def get_real_eef_angvel(self):
        raise NotImplemented


    def scale_and_add_noise(self, obs, noise_scale):
        if noise_scale < 0 or noise_scale > 1:
            raise ValueError("Noise scale must be between 0 and 1")
        
        noise = torch.randn_like(obs)
        obs = noise_scale * noise + obs

        return obs

    def compute_observations(self): 
        #for i in range(self.num_envs):
        # pred_id = self.predicted_object_id[i]
        # gt_id = self.proc_object_ids[i]
        self._refresh()
        self.refresh_depth_buffer()

        #self.rigid_body_states = self.rigid_body_states_flat[self.robot_rb_mask].view(self.num_envs, -1, 13)
        #self.contact_forces = self.contact_forces_flat[self.robot_rb_mask].view(self.num_envs, -1, 3)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]


        # if self.cfg["env"].get("add_goal_object", True):
        #     self.goal_pose = self.goal_states[:, 0:7]
        #     self.goal_pos = self.goal_states[:, 0:3]
        #     self.goal_rot = self.goal_states[:, 3:7]

        # if self.obs_type == 'expl_obs':
        #     assert self.obs_buf.shape[1] == self.arm_act_dim + self.num_hand_dofs + 7
        #     #import ipdb ; ipdb.set_trace()
        #     self.obs_buf[:, :self.arm_act_dim] = self.rigid_body_states[:, self.gym.find_actor_rigid_body_handle(self.envs[0], 0, "leap_mount"), :][:,:3]
        #     self.obs_buf[:, self.arm_act_dim : self.arm_act_dim + self.num_hand_dofs] = unscale(self.robot_dof_pos[:, self.num_arm_dofs:],
        #                                                            self.robot_dof_lower_limits[self.num_arm_dofs:], 
        #                                                            self.robot_dof_upper_limits[self.num_arm_dofs:])

        #     self.obs_buf[:, -7:] =  self.object_pose

        self.compute_sdf_metrics()
        self.compute_contact_filter()

        #if self.obs_type == "custom":
        self.obs_buf[:,:] = 0
        start_idx = 0  

        if not hasattr(self, 'obs_names'):
            if not isinstance(self.cfg["env"]["observation_list"], dict):
                self.obs_names = self.cfg["env"]["observation_list"] 
            else:
                self.obs_input_names = list([key for key, val in self.cfg["env"]["observation_list"].items() if val[0] in ['input']])
                self.obs_phase1_names = list([key for key, val in self.cfg["env"]["observation_list"].items() if val[0] in ['phase1']])

            self.obs_input_fn = [getattr(self, f'get_{obs_name}') for obs_name in self.obs_input_names]
            self.obs_phase1_fn = [getattr(self, f'get_{obs_name}') for obs_name in self.obs_phase1_names]

        obs_input = torch.zeros((self.num_envs, len(self.obs_input)), device=self.device)
        obs_phase1 = torch.zeros((self.num_envs, len(self.obs_phase1)), device=self.device)

        start_idx = 0
        start_idxs = {}
        end_idxs = {}

        for i, obs_name in enumerate(self.obs_input_names):
            obs = self.obs_input_fn[i]().float()
            end_idx = start_idx + obs.shape[1]
            start_idxs[obs_name] = start_idx
            end_idxs[obs_name] = end_idx
            obs_input[:, start_idx:end_idx] = obs
            start_idx = end_idx
        assert(start_idx == obs_input.shape[1])

        start_idx = 0
        for i, obs_name in enumerate(self.obs_phase1_names):
            obs = self.obs_phase1_fn[i]().float() 
            end_idx = start_idx + obs.shape[1]
            start_idxs[obs_name] = start_idx
            end_idxs[obs_name] = end_idx
            obs_phase1[:, start_idx:end_idx] = obs
            start_idx = end_idx
        assert(start_idx == obs_phase1.shape[1])
        self.obs_buf[:, self.obs_input] = obs_input
        self.start_idxs = start_idxs
        self.end_idxs = end_idxs
        
        if self.is_phase2:
            self.obs_buf[:, self.obs_phase1] = self.get_depth_latent(obs_input)
        else:
            self.obs_buf[:, self.obs_phase1] = obs_phase1#self.env_encoder(obs_phase1)
            # self.obs_buf[:, :] = self.compute_latency(self.obs_buf, start_idxs, end_idxs)
            
        #self.resample_exec_lag(env_ids)

    def postprocess_obs(self, obs_buf, add_latency=False):
        return self.compute_latency(obs_buf, self.start_idxs, self.end_idxs, add_latency=add_latency)

    # def resample_exec_lag(self, env_ids):
    #     if not hasattr(self, "sampled_exec_lag_values"):
    #         return
    #     lo = self.cfg["env"]["exec_lag_lo"][-1]
    #     hi = self.cfg["env"]["exec_lag_hi"][-1]
    #     resampled_exec_lag = torch.randint(low=lo, high=hi + 1, size=[env_ids.shape[0]], device=self.device)

    #     self.sampled_exec_lag_values[env_ids] = resampled_exec_lag

    # def get_exec_lag(self):
    #     exec_lag = None

    #     if "exec_lag_lo" in self.cfg["env"]:
    #         if not hasattr(self, "sampled_exec_lag_values"):
    #             self.sampled_exec_lag_values = torch.ones(self.num_envs, device=self.device) 
    #             if self.cfg["env"]["test"]:
    #                 lo = self.cfg["env"]["exec_lag_lo"][-1]
    #                 hi = self.cfg["env"]["exec_lag_hi"][-1]
    #             else:
    #                 lo = self.cfg["env"]["exec_lag_lo"][0]
    #                 hi = self.cfg["env"]["exec_lag_hi"][0]

    #             self.sampled_exec_lag_values *= random.randint(lo, hi)
    #             self.sampled_exec_lag_values = self.sampled_exec_lag_values.long()
            
    #         exec_lag = self.sampled_exec_lag_values
            
    #         if wandb.run is not None:
    #             wandb.log({"sampled_exec_lag_mean": self.sampled_exec_lag_values.sum().item() / self.num_envs})

    #     #elif "exec_lag" in self.dr_randomizations: # Only use dr_randomizations if exec_lag not in main config
    #     #    exec_lag = self.dr_randomizations["exec_lag"]["noise_lambda"](torch.zeros(self.num_envs, device=self.device))

    #     return exec_lag

    def preprocess_actions(self, actions, add_latency=False):
        if not hasattr(self, 'action_history'):
            self.arm_latency_rng = self.cfg['env'].get('arm_latency_rng', [0, 0])
            self.hand_latency_rng = self.cfg['env'].get('hand_latency_rng', [0, 0])
            self.prev_action_latency = torch.zeros((self.num_envs, actions.shape[1]), device=self.device).long()
            self.action_history = []
            self.start_action_latency = self.cfg['env'].get('start_action_latency', 0)

        if not add_latency:
            if not self.enable_action_latency or self.epoch_count < self.start_action_latency:
                return actions

        self.action_history.append(actions.detach().to(self.device))
        self.action_history = self.action_history[-self.max_latency:]

        latency = torch.zeros_like(self.prev_action_latency).long()

        latency[:, :self.num_arm_dofs] = torch.randint(low=self.arm_latency_rng[0],  high=self.arm_latency_rng[1]+1,  size=(self.num_envs, 1), device=self.device)
        latency[:, self.num_arm_dofs:] = torch.randint(low=self.hand_latency_rng[0], high=self.hand_latency_rng[1]+1, size=(self.num_envs, 1), device=self.device)
        
        self.extras['arm_latency'] = latency[:, :self.num_arm_dofs].float().mean()
        self.extras['hand_latency'] = latency[:, self.num_arm_dofs:].float().mean()

        has_not_reset = (self.progress_buf > 0).unsqueeze(1).long()
        history_cnt = (self.progress_buf).unsqueeze(1)
        latency = torch.maximum(latency, (self.prev_action_latency - 1)*has_not_reset)
        latency = torch.minimum(latency, history_cnt)
        actions = torch.gather(torch.stack(self.action_history, dim=1), 1, len(self.action_history)-1-latency.unsqueeze(1)).squeeze(1)
        self.prev_action_latency = latency

        return actions


    def compute_latency(self, obs_buf, start_idxs, end_idxs, add_latency=False):
        if not hasattr(self, 'obs_history'):
            self.input_latency_rng = self.cfg['env'].get('input_latency_rng', [0, 0])
            self.phase1_latency_rng = self.cfg['env'].get('phase1_latency_rng', [0, 0])
            self.prev_obs_latency = torch.zeros((self.num_envs, obs_buf.shape[1]), device=self.device).long()
            self.obs_history = []
            self.episode_latency = torch.zeros((self.num_envs, obs_buf.shape[1]), device=self.device).long()
            #self.latency_curriculum = self.cfg['env'].get('latency_curriculum', None)
            self.start_obs_latency = self.cfg['env'].get('start_obs_latency', 0)
            self.input_latency_probs = self.cfg['env'].get('input_latency_probs', [1.0]) #15ms avg
            self.phase1_latency_probs = self.cfg['env'].get('phase1_latency_probs', [0.94, 0.03, 0.02, 0.01]) #90ms avg 

        if not add_latency:
            if not self.enable_latency or self.epoch_count < self.start_obs_latency:
                return obs_buf

        self.obs_history.append(obs_buf.detach().to(self.device))
        self.obs_history = self.obs_history[-self.max_latency:]

        # set episode latency
        assert((self.progress_buf == 0).sum().item() == 0)
        reset_idxs = (self.progress_buf == 1).nonzero() 
        if len(reset_idxs) > 0:
            dist1 = torch.distributions.Categorical(probs=torch.tensor(self.input_latency_probs))
            #self.episode_latency[reset_idxs, self.obs_input] = dist1.sample((len(reset_idxs), len(self.obs_input))).long().to(self.device)
            self.episode_latency[reset_idxs, self.obs_input] = dist1.sample((len(reset_idxs), 1)).long().to(self.device)
            dist2 = torch.distributions.Categorical(probs=torch.tensor(self.phase1_latency_probs))
            #self.episode_latency[reset_idxs, self.obs_phase1] = dist2.sample((len(reset_idxs), len(self.obs_phase1))).long().to(self.device)
            self.episode_latency[reset_idxs, self.obs_phase1] = dist2.sample((len(reset_idxs), 1)).long().to(self.device)

        # has_not_reset = (self.progress_buf > 1).unsqueeze(1).long()
        # prev_obs_latency = torch.clip((self.prev_obs_latency - 1)*has_not_reset, min=0)
        obs_history_cnt = self.progress_buf.unsqueeze(1)
        max_obs_latency = torch.clip(torch.minimum(self.episode_latency, obs_history_cnt), max=len(self.obs_history)-1)
        # latency = (torch.rand_like(obs_buf) * (max_obs_latency - prev_obs_latency) + prev_obs_latency).long()
        # for obs_name in (self.obs_input_names + self.obs_phase1_names):
        #     latency[:, start_idxs[obs_name]:end_idxs[obs_name]] = latency[:, start_idxs[obs_name]].unsqueeze(1)
        latency = max_obs_latency
        
        self.extras['phase1_latency'] = latency[:, self.obs_phase1].float().mean()
        self.extras['input_latency'] = latency[:, self.obs_input].float().mean()

        obs_buf = torch.gather(torch.stack(self.obs_history, dim=1), 1, len(self.obs_history)-1-latency.unsqueeze(1)).squeeze(1)
        self.prev_obs_latency = latency

        return obs_buf

    # def apply_obs_mask(self):
    #     if "obs_mask" not in self.cfg["env"]:
    #         return
        
    #     self.obs_buf[:, self.cfg["env"]["obs_mask"]] = 0


    def save_depth_postprocess_config(self, do_exit=False):
        cfg = {
            'camera_res': self.camera_res,
            'crop_top': self.crop_top,
            'crop_bottom': self.crop_bottom,
            'crop_left': self.crop_left,
            'crop_right': self.crop_right,
            'xyz_bg_noise_scale': self.xyz_bg_noise_scale,
            'xyz_min_clip': self.xyz_min_clip,
            'xyz_max_clip': self.xyz_max_clip,
            'view': self.camera_view_matrixs[0][0].cpu().numpy(),
            'proj': self.camera_proj_matrixs[0][0].cpu().numpy(),
            'u': self.camera_u.cpu().numpy(),
            'v': self.camera_v.cpu().numpy(),
            'w': self.camera_res[1],#self.cfg["env"]["cameras"]["raw_dimension"][1],
            'h': self.camera_res[0]#self.cfg["env"]["cameras"]["raw_dimension"][0],
        }
        print('saved cfg')
        np.savez('cfg.npz', cfg, allow_pickle=True)
        if do_exit:
            exit()

    def get_images(self):
        # xyz
        if not hasattr(self, 'camera_u'):
            #self.camera_u = torch.arange(0, self.cfg["env"]["cameras"]["raw_dimension"][1], device=self.device)
            #self.camera_v = torch.arange(0, self.cfg["env"]["cameras"]["raw_dimension"][0], device=self.device)
            self.camera_u = torch.arange(0, self.camera_res[1], device=self.device)
            self.camera_v = torch.arange(0, self.camera_res[0], device=self.device)

            #self.camera_u = torch.linspace(-1,1, steps=self.camera_res[1], device=self.device)
            #self.camera_v = torch.linspace(-1,1, steps=self.camera_res[0], device=self.device)

            self.camera_v, self.camera_u = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')

            self.camera_view_matrixs = []
            self.camera_proj_matrixs = []
            self.depth_cam_tensor = []
            self.segm_cam_tensor = []
            self.rgb_cam_tensor = []
            self.env_origin = torch.zeros((self.num_envs, 3), device=self.device)

            for i in range(self.num_envs):
                self.camera_view_matrixs.append([])
                self.camera_proj_matrixs.append([])
                self.depth_cam_tensor.append([])
                self.segm_cam_tensor.append([])
                self.rgb_cam_tensor.append([])
                for j, cam_handle in enumerate(self.cam_handles[i]):
                    cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, self.envs[i], cam_handle)).float())).to(self.device)
                    cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, self.envs[i], cam_handle), device=self.device).float()
                    self.camera_view_matrixs[i].append(cam_vinv.clone())
                    self.camera_proj_matrixs[i].append(cam_proj.clone())

                    depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], cam_handle, gymapi.IMAGE_DEPTH)
                    depth_image = gymtorch.wrap_tensor(depth_image_)
                    self.depth_cam_tensor[i].append(depth_image)

                    segm_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], cam_handle, gymapi.IMAGE_SEGMENTATION)
                    segm_image  = gymtorch.wrap_tensor(segm_image_)
                    self.segm_cam_tensor[i].append(segm_image)

                    rgb_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], cam_handle, gymapi.IMAGE_COLOR)
                    rgb_image = gymtorch.wrap_tensor(rgb_image_)
                    self.rgb_cam_tensor[i].append(rgb_image)

                origin = self.gym.get_env_origin(self.envs[i])
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z

        for i in range(self.num_envs):
            for j, cam_handle in enumerate(self.cam_handles[i]):
                segm_image = self.segm_cam_tensor[i][j].long().clone().to(self.device)
                ks = 5
                kernel = torch.ones((1,1,ks,ks)).float().to(segm_image.device)
                obj_segm_image = (segm_image == self.obj_sem_id).long().float()
                #obj_segm_image = torch.clamp(torch.nn.functional.conv2d(obj_segm_image.view(1,1,*segm_image.shape), kernel, padding=(ks//2, ks//2)), 0, 1).squeeze()
                rob_segm_image = (segm_image == self.robot_sem_id).float()
                
                prev_rgb_buffer = self.rgb_buffer[i, j].clone()
                self.rgb_buffer[i, j] = self.process_image(self.rgb_cam_tensor[i][j].clone().permute((2,0,1))[:3,:,:].to(self.device))

                obj_segm_image = self.process_image(obj_segm_image)
                rob_segm_image = self.process_image(rob_segm_image)
                depth_buffer = self.depth_cam_tensor[i][j].clone().to(self.device)
                #depth_buffer = self.blur_transform(depth_buffer[None,None,:,:]).squeeze()
                
                depth_buffer = depth_buffer + torch.randn_like(depth_buffer)*self.depth_noise_scale
                
                depth_buffer = torch.clip(depth_buffer, -1000, 0) # clip
                depth_buffer = self.process_image(depth_buffer)
                
                depth_image = torch.clip(depth_buffer, -self.depth_far_clip, self.depth_far_clip) # clip
                
                depth_image = -depth_image

                xyz_buffer = depth_image_to_point_cloud_GPU(depth_buffer.clone(), self.camera_view_matrixs[i][j], self.camera_proj_matrixs[i][j], self.camera_u, self.camera_v, self.camera_u.shape[1], self.camera_u.shape[0], self.device).reshape(*depth_buffer.shape, 3).permute(2, 0, 1)
                #print(self.camera_u.shape[1], self.camera_u.shape[0], xyz_buffer.shape)

                #self.save_depth_postprocess_config(do_exit=True)
                if self.phase2_buffer_type == 'xyzs':
                    obj_mask = (rob_segm_image == 0) & (obj_segm_image == 0)
                    xyz_buffer[0, obj_mask] = self.xyz_bg[0] + torch.randn_like(xyz_buffer[0, obj_mask])*self.xyz_bg_noise_scale[0]
                    xyz_buffer[1, obj_mask] = self.xyz_bg[1] + torch.randn_like(xyz_buffer[1, obj_mask])*self.xyz_bg_noise_scale[1]
                    xyz_buffer[2, obj_mask] = self.xyz_bg[2] + torch.randn_like(xyz_buffer[2, obj_mask])*self.xyz_bg_noise_scale[2]

                    xyz_buffer[0, ~obj_mask] = xyz_buffer[0, ~obj_mask] + torch.randn_like(xyz_buffer[0, ~obj_mask])*self.xyz_fg_noise_scale[0]
                    xyz_buffer[1, ~obj_mask] = xyz_buffer[1, ~obj_mask] + torch.randn_like(xyz_buffer[1, ~obj_mask])*self.xyz_fg_noise_scale[1]
                    xyz_buffer[2, ~obj_mask] = xyz_buffer[2, ~obj_mask] + torch.randn_like(xyz_buffer[2, ~obj_mask])*self.xyz_fg_noise_scale[2]
                else:
                    xyz_buffer[0] = xyz_buffer[0] + torch.randn_like(xyz_buffer[0])*self.xyz_bg_noise_scale[0]
                    xyz_buffer[1] = xyz_buffer[1] + torch.randn_like(xyz_buffer[1])*self.xyz_bg_noise_scale[1]
                    xyz_buffer[2] = xyz_buffer[2] + torch.randn_like(xyz_buffer[2])*self.xyz_bg_noise_scale[2]

                xyz_buffer[0] = 2*(torch.clip(xyz_buffer[0], min=self.xyz_min_clip[0], max=self.xyz_max_clip[0]) - self.xyz_min_clip[0]) / (self.xyz_max_clip[0] - self.xyz_min_clip[0]) - 1
                xyz_buffer[1] = 2*(torch.clip(xyz_buffer[1], min=self.xyz_min_clip[1], max=self.xyz_max_clip[1]) - self.xyz_min_clip[1]) / (self.xyz_max_clip[1] - self.xyz_min_clip[1]) - 1
                xyz_buffer[2] = 2*(torch.clip(xyz_buffer[2], min=self.xyz_min_clip[2], max=self.xyz_max_clip[2]) - self.xyz_min_clip[2]) / (self.xyz_max_clip[2] - self.xyz_min_clip[2]) - 1

                self.depth_buffer[i, j] = depth_image
                self.segm_buffer[i, j, 0] = (rob_segm_image != 0).float().clone()
                self.segm_buffer[i, j, 1] = (obj_segm_image != 0).float().clone() 


                self.diff_buffer[i, j] = ((self.rgb_buffer[i, j] - prev_rgb_buffer).abs().mean(dim=0)/2 >= self.diff_rgb_threshold).float()
                self.xyz_buffer[i, j] = xyz_buffer# - self.env_origin[i].reshape(3, 1, 1))

        #self.xyz_buffer = self.depth_buffer.repeat(1, 3, 1, 1)
        #for i in range(self.num_envs):
        #    for j, cam_handle in enumerate(self.cam_handles[i]):
        #        self.xyz_buffer[i, j] = depth_image_to_point_cloud_GPU(self.depth_buffer[i, j], self.camera_view_matrixs[i][j], self.camera_proj_matrixs[i][j], self.camera_u, self.camera_v, self.camera_res[1], self.camera_res[0], self.device).reshape(*self.depth_buffer[i,j].shape, 3).permute(2, 0, 1)

    
        #self.phase2_buffers.append(self.get_single_buffer_phase2())
        #self.phase2_buffers = self.phase2_buffers[-self.num_camera_frames:]

    def refresh_depth_buffer(self):
        if not hasattr(self, "depth_buffer"):
            return
        
        if self.global_counter % self.refresh_interval != 0:
            self.new_depth_image = False
            return

        # https://forums.developer.nvidia.com/t/camera-example-and-headless-mode/178901/11
        if self.headless:
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        self.get_images()

        self.gym.end_access_image_tensors(self.sim)
        self.new_depth_image = True

    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
            self.update_randomization_params(env_ids, self.obj_scale)
            self.log_randomizations()

        # generate random values
        # rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_robot_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        # self.reset_target_pose(env_ids)

        # reset rigid body forces
        # for idx in env_ids:
        #     self.rb_forces[idx][:] = 0
        self.object_rb_forces[env_ids, :] = 0.0

        #if self.cfg["env"]["randomize_projectile_init"]:
        #randomize object init state
        for env_idx in env_ids:
            obj_cfg = self.init_projectile()
            self.object_init_state[env_idx] = torch.tensor(obj_cfg['init_state'])
            self.time_impact[env_idx] = obj_cfg['t_impact']
            if 'goal_pose' in obj_cfg:
                self.object_goal_pose[env_idx] = torch.tensor(obj_cfg['goal_pose'])


        self._refresh()

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:3] = self.object_init_state[env_ids, 0:3]# + self.reset_position_noise * rand_floats[:, 0:3]

        #self.ep_object_init_state[env_ids] = self.root_state_tensor[self.object_indices[env_ids]].clone()

        # new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        # if self.object_type == "pen":
        #     rand_angle_y = torch.tensor(0.3)
        #     new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
        #                                             self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids], self.z_unit_tensor[env_ids])

        # if self.cfg["env"]["randomize_rotation"]:
        #     self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        #     object_indices = self.object_indices[env_ids].to(torch.int32)
        #     #self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        # if self.cfg["env"].get("add_goal_object", True):
        #     object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
        #                                             self.goal_object_indices[env_ids],
        #                                             self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        # else:
        object_indices = self.object_indices[env_ids].to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset random force probabilities
        if hasattr(self, 'random_force_prob'):
            self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1])) * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))
        #self.object_rb_forces *= torch.maximum(self.progress_buf.float() - force_curriculum, torch.zeros_like(self.progress_buf)).unsqueeze(1) / (self.max_episode_length - force_curriculum)

        self.randomize_initial_dof_pos(env_ids)

        if not hasattr(self, 'reset_dof_vel_noise'):
            self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.robot_dof_pos[env_ids, :] = self.robot_default_dof_pos[env_ids].clone()
        self.robot_dof_vel[env_ids, :] = self.robot_default_dof_vel[env_ids] + self.reset_dof_vel_noise * torch.rand_like(self.robot_dof_vel[env_ids])
        self.hand_prev_targets[env_ids, :] = self.robot_default_dof_pos[env_ids, self.num_arm_dofs:].clone()
        self.hand_cur_targets[env_ids, :] = self.robot_default_dof_pos[env_ids, self.num_arm_dofs:].clone()
        self.real_hand_prev_targets[env_ids, :] = self.robot_default_dof_pos[env_ids, self.num_arm_dofs:].clone()
        self.real_hand_cur_targets[env_ids, :] = self.robot_default_dof_pos[env_ids, self.num_arm_dofs:].clone()
        self.robot_cur_targets[env_ids, :] = self.robot_default_dof_pos[env_ids].clone()
        self.robot_eef_target_pose[env_ids, :] = self.default_ee_pose[env_ids].clone()

        if self.is_real:
            self.real_robot_eef_target_pose[env_ids, :] = self.default_ee_pose[env_ids].clone()

        indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.robot_cur_targets),
                                                        gymtorch.unwrap_tensor(indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(indices), len(env_ids))
        self.progress_buf[env_ids] = 0
        self._reset_buf[env_ids] = 0
        mask = self.num_resets[env_ids] == 1
        self.num_resets[env_ids[mask]] = 0
        self.reset_buf[env_ids[mask]] = 0

        self.episode_success[env_ids] = (self.time_impact[env_ids] + self.phase_success_time)/self.dt

        self.sync_extra(env_ids)

        if self.is_real:
            print("calling reset")
            self.move_robot_to_start(env_ids)

        if self.is_phase2:
            if hasattr(self, 'depth_backbone'): 
                if self.depth_backbone.is_rnn:
                    print(self.depth_backbone.hidden_states.shape)
                    self.depth_backbone.hidden_states[:, env_ids, :] = 0*self.depth_backbone.hidden_states[:, env_ids, :]

    def sync_extra(self, env_ids):
        pass

    def update_reset_buf(self):
        self.num_resets += self._reset_buf
        self.reset_buf = torch.zeros_like(self._reset_buf)
        self.reset_buf[self.num_resets == 1] = 1
        lost = (self.num_resets > 1).float().mean().item()
        if lost > 0:
            print('lost', lost)

    def move_robot_to_start(self, env_ids):
        if env_ids is not None and len(env_ids) == 0:
            return
        
        print("sleeping for 1 second...")
        time.sleep(1)
        print("done sleeping...")

        self.move_hand_to_start()
        self.move_arm_to_start()
        
        print("sleeping for 1 second...")
        time.sleep(1)
        print("done sleeping...")
        while self.is_real_input and self.cam_ep_ended.sum() > 0:
            time.sleep(0.016)

    def move_arm_to_start(self, eps=1e-2):
        if self.cfg["env"].get("disable_arm", False):
            return
        
        print("waiting for arm obs to initialize...")
        while self.get_real_arm_dof_pos() is None: # or self.get_real_eef_pose() is None:
            self.sync_with_real_time()

        print("moving arm to start position...")
        initial_time = time.time()
        pbar = tqdm()
        while True:
            cur_joints = self.get_real_arm_dof_pos()
            tgt_joints = self.robot_default_dof_pos[:,:self.num_arm_dofs]
            error = torch.abs(tgt_joints - cur_joints).max().item()

            msg = JointState()
            msg.position = np.array(tgt_joints.cpu().squeeze())
            self.arm_joint_pub.publish(msg)

            pbar.update(1)
            pbar.set_description(f'error: {error:.2f}\t cur: {cur_joints[:,:3]}\t cmd: {tgt_joints[:,:3]}')
            if error <= 0.01:
                break
            #self.sync_with_real_time()

    # def move_arm_to_start(self, eps=1e-2):
    #     if self.cfg["env"].get("disable_arm", False):
    #         return
        
    #     print("waiting for arm obs to initialize...")
    #     while self.get_real_arm_dof_pos() is None or self.get_real_eef_pose() is None:
    #         self.sync_with_real_time()

    #     print("moving arm to start position...")
    #     initial_time = time.time()
    #     pbar = tqdm()
    #     while True:
    #         # blend_ratio = min((time.time() - initial_time) / 2, 1)
    #         blend_ratio = 1
    #         error = self.default_ee_pose - self.get_real_eef_pose()
    #         targets = self.robot_default_dof_pos[:, :self.num_arm_dofs]

    #         msg = JointState()
    #         msg.position = np.array(targets.cpu().squeeze())
    #         self.arm_joint_pub.publish(msg)

    #         cur_eef_pose = self.get_real_eef_pose()
    #         eef_targets = self.default_ee_pose * blend_ratio + (1 - blend_ratio) * cur_eef_pose
    #         self.publish_arm_eef_pose(eef_targets, speed=100)

    #         #print("error: ", error)
    #         #print("real_eef_pose: ", self.real_eef_pose)
    #         #print("real_eef_target_pose: ", self.get_real_eef_target_pose())
    #         pbar.update(1)
    #         pbar.set_description(f'error: {torch.abs(error).max().item():.2f}\t cur: {cur_eef_pose[:,:3]}\t cmd: {eef_targets[:,:3]}')
    #         if torch.all(torch.abs(error) <= 0.01):
    #             break
            
    #         #self.sync_with_real_time()

    #     #print("done")

    def move_hand_to_start(self):
        if self.cfg["env"].get("disable_hand", False):
            return

        print("waiting for hand obs to initialize...")
        while self.get_real_hand_dof_pos() is None:
            self.refresh_real_hand_dof_pos()
            self.sync_with_real_time()

        print("moving hand to start position...")
        initial_time = time.time()
        pbar = tqdm()
        while True:
            #blend_ratio = min(30*(time.time() - initial_time), 1) # blend_ratio = 0
            self.refresh_real_hand_dof_pos()
            cur_joints = self.get_real_hand_dof_pos()
            tgt_joints = self.robot_default_dof_pos[:, self.num_arm_dofs:]
            #error = self.robot_default_dof_pos[0, self.num_arm_dofs:] - self.get_real_hand_dof_pos()
            error = torch.abs(tgt_joints - cur_joints).max().item()
            #targets = self.robot_default_dof_pos[0, self.num_arm_dofs:] * blend_ratio + (1 - blend_ratio) * self.get_real_hand_dof_pos()

            self.publish_hand_targets(tgt_joints)

            pbar.update(1)
            pbar.set_description(f'error: {error:.2f}\t fps: {1/(time.time() - initial_time)}')

            if error <= 0.1:
                break
            
            #print('error: ', error, " ", torch.abs(error).mean())  ]]
            #self.sync_with_real_time()

        print("done")

        return

    def control_ik_v2(self, dpose, only_pos=False, **kwargs):
        if not hasattr(self, "kinematics"):
            self.init_kinematics()
        
        # Check the FK error
        arm_dof_pos = self.get_real_arm_dof_pos()
        eef_pos, eef_quat = self.kinematics.forward_kinematics(arm_dof_pos)
        eef_error = torch.norm(eef_pos - self.get_real_eef_pos(), dim=1)

        if torch.any(eef_error > 1e-2):
            warnings.warn("The FK error is too large: {}".format(eef_error.max()))
        #     return torch.zeros_like(self.robot_dof_pos[:, :self.num_arm_dofs])

        target_joint_angles, error = self.kinematics.inverse_kinematics(dpose, arm_dof_pos, **kwargs)
        delta_joint_angles = target_joint_angles - arm_dof_pos
        self.extras["ik_error"] = error.mean()

        return delta_joint_angles

    def get_jacobian(self, robot_dof_pos):
        if not hasattr(self, "kinematics"):
            self.init_kinematics()

        if not hasattr(self, '_j_eef'):
            self.init_data()
            #assert(self._j_eef.abs().sum() == 0)

        jacobian_version = self.cfg["env"].get("jacobian_version", "native")

        if jacobian_version == "native":
            # if self.is_real:
            #     warnings.warn("cannot use IG jacobian on real robot value will be null")
            #     return None
            
            _j_eef = self.jacobian[:, self.hand_index - 1, :, :self.num_arm_dofs]
            return _j_eef
        
        if jacobian_version == "finite_diff":
            return self.kinematics._compute_jac(robot_dof_pos[:, :self.num_arm_dofs], eps=1e-3)
        
        if jacobian_version == "autograd":
            return self.kinematics._compute_jac_autograd(robot_dof_pos[:, :self.num_arm_dofs]).squeeze().unsqueeze(0)
        
        raise RuntimeError("unknown jacobian version: {}".format(jacobian_version))

    def control_ik(self, dpose, robot_dof_pos, j_eef = None, only_pos = False):
        damping = self.cfg["env"].get("ik_damping", 0.05)
        j_eef = self.get_jacobian(robot_dof_pos) if j_eef is None else j_eef

        if j_eef is None:
            return None

        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        if only_pos : 
            u = ( (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda))[:,:,:3] @ dpose[:,:3].unsqueeze(-1)).squeeze()
        else:
            u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose.unsqueeze(-1)).squeeze()
        return u

    # def fetch_vr_glove_pose(self):
    #     if hasattr(self, "ros_eef_pos"):
    #         target_pos = self.ros_eef_pos
    #         self.vr_eef_pos_offset = self.cfg["env"].get("vr_eef_pos_offset", [0.0, 0., 0.0])
    #         self.vr_eef_pos_offset = torch.tensor(self.vr_eef_pos_offset, device=self.device).float()
    #         target_pos += self.vr_eef_pos_offset
    #         target_pos = target_pos.repeat((self.num_envs, 1))
    #         target_rot = self.ros_eef_rot.repeat((self.num_envs, 1))

    #         return target_pos, target_rot
    #     else:
    #         result = self.get_eef_pose().clone()

    #         return result[:, :3], result[:, 3:]

    # def fetch_hand_joints_from_file(self):
    #     if not hasattr(self, "hand_joints_file"):
    #         self.hand_joints_file = np.load(self.cfg["env"]["hand_joints_file"])
    #         self.hand_joints_file = torch.from_numpy(self.hand_joints_file).to(self.device).float()

    #     if torch.any(self.progress_buf >= self.hand_joints_file.shape[0]):
    #         warnings.warn("hand joints file is too short, looping")

    #     actions = self.hand_joints_file[self.progress_buf % self.hand_joints_file.shape[0]]
    #     actions = self.robot_default_dof_pos[:, self.num_arm_dofs:] + actions / 10

    #     return actions

    # def fetch_hand_joints_from_vae(self, u_hand):
    #     if not hasattr(self, "hand_joints_vae"):
    #         self.hand_joints_vae = torch.jit.load(self.cfg["env"]["hand_joints_vae"]).to(self.device)

    #     with torch.no_grad():
    #         return self.hand_joints_vae(u_hand)

    def refresh_real_hand_dof_pos(self):
        obses, _ = self.leap_hand.poll_joint_position()
        obses = torch.from_numpy(obses).to(self.device)
        self.real_hand_dof_pos = obses[None, :]
        self.robot_dof_pos[:, self.num_arm_dofs:] = obses[None, :]

    def fetch_playback_actions(self):
        if not hasattr(self, "playback_actions"):
            self.playback_actions = torch.from_numpy(np.load(self.cfg["env"]["playback_actions"])).to(self.device).float()

        return self.playback_actions[self.progress_buf % self.playback_actions.shape[0]]

    # def update_real_hand_control(self, u_hand):
    #     self.refresh_real_hand_dof_pos()
    #     self.real_hand_prev_targets[:] = self.real_hand_cur_targets[:]

    #     # if "playback_actions" in self.cfg["env"]:
    #     #     u_hand = self.fetch_playback_actions()

    #     # if "disable_hand_obj_dist_threshold" in self.cfg["env"]:
    #     #     raise NotImplemented

    #     # if "object_reaching_fraction" in self.cfg["env"]:
    #     #     raise NotImplemented

    #     #hand control
    #     if self.hand_control_mode == "relative":
    #         self.real_hand_cur_targets[:] = self.real_hand_prev_targets[:] + self.robot_dof_speed_scale * self.dt * u_hand
    #     elif self.hand_control_mode == "absolute":
    #         lower, upper = self.robot_dof_lower_limits[self.num_arm_dofs:], self.robot_dof_upper_limits[self.num_arm_dofs:]
    #         self.real_hand_cur_targets[:] = scale(u_hand, lower*0.9+upper*0.1, lower*0.1+upper*0.9)
    #         self.real_hand_cur_targets[:] = (1 - self.hand_act_moving_average) * self.real_hand_cur_targets[:] + self.hand_act_moving_average * self.real_hand_prev_targets[:]
    #     # elif self.hand_control_mode == "vr" or self.hand_control_mode == "rosbag":
    #     #     if hasattr(self, "ros_hand_joints"):
    #     #         # self.hand_cur_targets[:] = self.real_to_sim(self.ros_hand_joints.repeat((self.num_envs, 1)))
    #     #         self.real_hand_cur_targets[:] = self.real_to_sim(self.ros_hand_joints.repeat((self.num_envs, 1)))
    #     # elif self.hand_control_mode == "file":
    #     #     self.real_hand_cur_targets[:] = self.fetch_hand_joints_from_file()
    #     # elif self.hand_control_mode == "vae":
    #     #     u_hand = self.fetch_hand_joints_from_vae(u_hand)
    #     #     self.real_hand_cur_targets[:] = u_hand
    #     # elif self.hand_control_mode == "vae_random":
    #     #     u_hand = torch.randn_like(u_hand)
    #     #     u_hand = self.fetch_hand_joints_from_vae(u_hand)
    #     #     self.real_hand_cur_targets[:] = u_hand
    #     elif self.hand_control_mode == "eigengrasps":
    #         self.real_hand_cur_targets[:] = self.fetch_hand_joints_from_egrasp(u_hand)
    #         self.real_hand_cur_targets[:] = (1.0 - self.hand_act_moving_average) * self.real_hand_cur_targets[:] + (self.hand_act_moving_average) * self.real_hand_prev_targets[:]
    #     elif self.hand_control_mode == "no_op":
    #         pass
    #     else:
    #         raise NotImplementedError

    #     self.real_hand_cur_targets[:] = tensor_clamp(self.real_hand_cur_targets[:],
    #                                                  self.robot_dof_lower_limits[self.num_arm_dofs:], 
    #                                                  self.robot_dof_upper_limits[self.num_arm_dofs:])

    #     if "max_delta_hand_targets" in self.cfg["env"]:
    #         clip_value = self.cfg["env"]["max_delta_hand_targets"] * self.control_dt
    #         delta_hand_targets = self.real_hand_cur_targets - self.real_hand_prev_targets
    #         delta_hand_targets = torch.clip(delta_hand_targets, -clip_value, clip_value)
    #         self.real_hand_cur_targets[:] = self.real_hand_prev_targets + delta_hand_targets
            
            
    #     self.real_robot_cur_targets[:, self.num_arm_dofs:] = self.real_hand_cur_targets

    #     self.publish_hand_targets(self.real_hand_cur_targets)
        
    #     return

    def publish_hand_targets(self, targets):
        targets_np = targets[0].cpu().numpy()
        self.leap_hand.command_joint_position(targets_np)

    def init_egrasp_matrix(self):
        self.egrasp = np.load(self.cfg["env"]["egrasp_file"])
        self.egrasp = torch.from_numpy(self.egrasp).to(self.device).float()

        if "extra_egrasp" in self.cfg["env"]:
            for _, egrasp in self.cfg["env"]["extra_egrasp"].items():
                egrasp = torch.tensor(egrasp, device=self.device)
                self.egrasp = torch.cat([self.egrasp, egrasp[None, :]], dim=0)

        self.num_egrasps = self.egrasp.shape[0]
        return self.num_egrasps

    def fetch_hand_joints_from_egrasp(self, u_hand):
        if not hasattr(self, "egrasp_file"):
            self.init_egrasp_matrix()

        hand_joints = u_hand @ self.egrasp 
        hand_joints = scale(hand_joints, self.robot_dof_lower_limits[self.num_arm_dofs:], self.robot_dof_upper_limits[self.num_arm_dofs:])
        
        return hand_joints
    
    def fetch_hand_joints_from_egrasp_residual(self, u_hand):
        if not hasattr(self, "egrasp_file"):
            self.init_egrasp_matrix()
            # find the null space of the egrasp matrix
            U, D, V = torch.linalg.svd(self.egrasp)
            self.residual_action_matrix = V[self.num_egrasps:]

        principal_actions = u_hand[:, :self.num_egrasps]
        principal_actions = principal_actions @ self.egrasp

        residual_actions = u_hand[:, self.num_egrasps:] * self.cfg["env"]["residual_action_scale"]
        residual_actions = residual_actions @ self.residual_action_matrix

        actions = principal_actions + residual_actions 
        print(actions[0])       
        return actions



    def update_hand_control(self, u_hand):   
        #     self.update_real_hand_control(u_hand)
        #     return
        
        # if "playback_actions" in self.cfg["env"]:
        #     u_hand = self.fetch_playback_actions()
        
        self.hand_prev_targets[:] = self.hand_cur_targets[:]
        mask = torch.ones(self.num_envs, device=self.device).bool()

        # if "disable_hand_obj_dist_threshold" in self.cfg["env"]:
        #     threshold = self.cfg["env"]["disable_hand_obj_dist_threshold"]
        #     object_pos = self.root_state_tensor[self.object_indices, 0:3]
        #     hand_pos = self.get_eef_pose()[:, :3]
        #     dist = torch.norm(object_pos - hand_pos, dim=-1)
        #     mask[dist > threshold] = 0

        # if "object_reaching_fraction" in self.cfg["env"]:
        #     mask[self.is_reaching_object()] = 0

        #hand control
        if self.hand_control_mode == "relative":
            self.hand_cur_targets[:][mask] = self.hand_prev_targets[:][mask] + self.robot_dof_speed_scale * self.dt * u_hand[mask]
        elif self.hand_control_mode == "absolute":
            hand_cur_targets = scale(u_hand, self.robot_dof_lower_limits[self.num_arm_dofs:], self.robot_dof_upper_limits[self.num_arm_dofs:])
            self.hand_cur_targets[:][mask] = (1 - self.hand_act_moving_average) * hand_cur_targets[mask] + (self.hand_act_moving_average) * self.hand_prev_targets[:][mask]
        #elif self.hand_control_mode == "vr" or self.hand_control_mode == "rosbag":
        #     if hasattr(self, "ros_hand_joints"):
        #         # self.hand_cur_targets[:] = self.real_to_sim(self.ros_hand_joints.repeat((self.num_envs, 1)))
        #         self.hand_cur_targets[:][mask] = self.real_to_sim(self.ros_hand_joints.repeat((self.num_envs, 1)))[mask]
        # elif self.hand_control_mode == "file":
        #     self.hand_cur_targets[:][mask] = self.fetch_hand_joints_from_file()[mask]
        # elif self.hand_control_mode == "vae":
        #     u_hand = self.fetch_hand_joints_from_vae(u_hand)
        #     self.hand_cur_targets[:][mask] = u_hand[mask]
        # elif self.hand_control_mode == "vae_random":
        #     u_hand = torch.randn_like(u_hand)
        #     u_hand = self.fetch_hand_joints_from_vae(u_hand)
        #     self.hand_cur_targets[:][mask] = u_hand[mask]
        # elif self.hand_control_mode == "eigengrasps":
        #     u_hand = self.fetch_hand_joints_from_egrasp(u_hand)
        #     self.hand_cur_targets[:][mask] = scale(u_hand[mask], self.robot_dof_lower_limits[self.num_arm_dofs:], self.robot_dof_upper_limits[self.num_arm_dofs:])
        #     self.hand_cur_targets[:][mask] = (1 - self.hand_act_moving_average) * self.hand_cur_targets[:][mask] + (self.hand_act_moving_average) * self.hand_prev_targets[:][mask]
        elif self.hand_control_mode == "eigengrasps_residual":
            u_hand = self.fetch_hand_joints_from_egrasp_residual(u_hand)
            self.hand_cur_targets[:][mask] = scale(u_hand[mask], self.robot_dof_lower_limits[self.num_arm_dofs:], self.robot_dof_upper_limits[self.num_arm_dofs:])
            self.hand_cur_targets[:][mask] = (1 - self.hand_act_moving_average) * self.hand_cur_targets[:][mask] + (self.hand_act_moving_average) * self.hand_prev_targets[:][mask]
            #self.hand_cur_targets[:][mask] = self.hand_prev_targets[:][mask] + self.robot_dof_speed_scale * self.dt * u_hand[mask]
        # elif self.hand_control_mode == "eigengrasps_random":
        #     u_hand = torch.randn_like(u_hand)
        #     self.hand_cur_targets[:][mask] = self.fetch_hand_joints_from_egrasp(u_hand)[mask]
        # elif self.hand_control_mode == "eigengrasps_axis":
        #     num_egrasps = u_hand.shape[1]
        #     u_hand = torch.arange(self.num_envs, device=self.device) % num_egrasps
        #     u_hand = torch.nn.functional.one_hot(u_hand, num_classes=num_egrasps).float() 
        #     self.hand_cur_targets[:][mask] = self.fetch_hand_joints_from_egrasp(u_hand)[mask]
        # elif self.hand_control_mode == "no_op":
        #     pass
        else:
            raise NotImplementedError

        self.hand_cur_targets[:] = tensor_clamp(self.hand_cur_targets[:],
                                                self.robot_dof_lower_limits[self.num_arm_dofs:], 
                                                self.robot_dof_upper_limits[self.num_arm_dofs:])

        if "max_delta_hand_targets" in self.cfg["env"]:
            clip_value = self.cfg["env"]["max_delta_hand_targets"] * self.control_dt
            delta_hand_targets = self.hand_cur_targets - self.hand_prev_targets
            delta_hand_targets = torch.clip(delta_hand_targets, -clip_value, clip_value)
            self.hand_cur_targets[:] = self.hand_prev_targets + delta_hand_targets

        if self.is_real:
            ema = self.cfg['env'].get('ema_hand_targets', 0.25)
            #ema = 1
            self.hand_cur_targets = self.hand_prev_targets * (1-ema) + self.hand_cur_targets * ema

        self.robot_cur_targets[:, self.num_arm_dofs:] = self.hand_cur_targets[:]

        if self.is_real:
            self.real_hand_cur_targets[:] = self.hand_cur_targets
            self.real_robot_cur_targets[:, self.num_arm_dofs:] = self.hand_cur_targets

        if self.is_real and not self.cfg["env"].get("disable_hand", False):
            self.publish_hand_targets(self.real_hand_cur_targets)

            if not hasattr(self,'prev_time'):
                self.prev_time = time.time()
            error = torch.abs(self.hand_prev_targets - self.get_real_hand_dof_pos()).max().item()
            print("fps: ", 1/(time.time() - self.prev_time), f'error: {error:.2f}')

            # if not hasattr(self, 'ik_error_plot'):
            #     self.ik_errors = [error]
            #     self.ik_error_fig, self.ik_error_ax = plt.subplots()
            #     self.ik_error_plot,  = self.ik_error_ax.plot(range(len(self.ik_errors)), self.ik_errors, label='ik error')
            # else:
            #     self.ik_errors.append(error)
            #     self.ik_error_plot.set_data(range(len(self.ik_errors)), self.ik_errors)
            #     self.ik_error_ax.relim()
            #     self.ik_error_ax.autoscale_view()

            # self.ik_error_fig.canvas.draw()
            # plt.pause(0.0001)

            #     initial_time = time.time()
            #     pbar = tqdm()
            #     cur_joints = self.get_real_hand_dof_pos()
            #     tgt_joints = self.real_hand_cur_targets
            #     total_duration = 4
            #     while True:
            #         self.refresh_real_hand_dof_pos()
            #         blend_ratio = min((time.time() - initial_time)/total_duration, 1) # blend_ratio = 0
            #         cur_joints = self.get_real_hand_dof_pos()
            #         cmd_joints = tgt_joints * blend_ratio + cur_joints * (1-blend_ratio)
            #         error = torch.abs(tgt_joints - cur_joints).max().item()

            #         self.publish_hand_targets(cmd_joints)

            #         pbar.update(1)
            #         pbar.set_description(f'error: {error:.2f}\t error={torch.abs(tgt_joints - cur_joints)}\tfps: {1/(time.time() - initial_time)}')

            #         if self.headless:
            #             self.gym.fetch_results(self.sim, True)
            #             self.gym.step_graphics(self.sim)
            #         if error <= 0.1:
            #             break

            self.prev_time = time.time()

    def get_eef_rot(self):
        return self.get_eef_pose()[:, 3:]

    # def get_object_reaching_targets(self):
    #     fraction = self.cfg["env"]["object_reaching_fraction"]
    #     offset = torch.tensor(self.cfg["env"]["object_reaching_offset"], device=self.device)
    #     final_pos = self.ep_object_init_state[:, :3] + offset[None, :]
    #     result = self.default_ee_pose[:, :3] * (1 - fraction) + fraction * final_pos

    #     return result
    
    # def is_reaching_object(self, tol=5e-2, return_error=False):
    #     targets = self.get_object_reaching_targets()
    #     error = torch.norm(targets - self.get_eef_pose()[:, :3], dim=-1)
    #     self.object_reaching_mask[error <= tol] = False

    #     if return_error:
    #         return error
    #     else:
    #         return self.object_reaching_mask

    # def publish_arm_eef_pose(self, pose, speed=10000):
    #     eef_pos, eef_quat = pose[:, :3], pose[:, 3:] 
    #     r, p, y = get_euler_xyz(eef_quat)
    #     r = move_to_range(r, -np.pi, np.pi)
    #     p = move_to_range(p, -np.pi, np.pi)
    #     y = move_to_range(y, -np.pi, np.pi)
    #     rpy = torch.stack((r, p, y), dim=-1)

    #     result = torch.zeros((1, 7), device=self.device)
    #     result[:, :6] = torch.cat((eef_pos, rpy), dim=-1)
    #     result[:, 6] = speed


    #     # if True:
    #     #     robot_dof_pos = self.get_real_arm_dof_pos()
    #     #     lower, upper = self.robot_dof_lower_limits[:self.num_arm_dofs], self.robot_dof_upper_limits[:self.num_arm_dofs]
    #     #     for i in range(self.num_envs):
    #     #         joint_angles, _ = self.kinematics.inverse_kinematics(pose[i] - robot_dof_pos[i], robot_dof_pos[i])
    #     #         joint_angles = torch.clamp(joint_angles, min=lower, max=upper)

    #     #     msg = JointState()
    #     #     msg.position = np.array(self.real_robot_cur_targets[0, :self.num_arm_dofs].cpu())
    #     #     self.arm_joint_pub.publish(msg)

    #     msg = JointState()
    #     msg.position = np.array(result[0].cpu())
    #     self.arm_eef_pose_pub.publish(msg)

    def publish_arm_eef_pose(self, pose, speed=10000):
        ik_pos_delta = pose[:, :3] - self.get_real_eef_pos()
        ik_orn_delta = orientation_error(pose[:, 3:], self.get_real_eef_rot())
        delta_joints = self.control_ik_v2(torch.cat([ik_pos_delta, ik_orn_delta], axis = -1))

        if self.cfg["env"]["max_joint_speed"] is not None:
            joint_delta_limit = self.cfg["env"]["max_joint_speed"] * self.dt
            delta_joints = torch.clamp(delta_joints, -joint_delta_limit, joint_delta_limit)

        command = self.get_real_arm_dof_pos() + delta_joints 
        command = tensor_clamp(command, self.robot_dof_lower_limits[: self.num_arm_dofs], self.robot_dof_upper_limits[: self.num_arm_dofs])
        msg = JointState()
        msg.position = np.array(command.cpu()[0])
        self.arm_joint_pub.publish(msg)

    # def update_real_arm_control(self, u_arm):
    #     #self.prev_real_robot_eef_target_pose = self.real_robot_eef_target_pose.clone()
    #     self.real_arm_prev_targets = self.real_robot_cur_targets[:, :self.num_arm_dofs]
    #     if self.arm_control_mode == "joint_control":
    #         self.real_robot_cur_targets[:, :self.num_arm_dofs] = u_arm.clone()

    #         self.real_robot_cur_targets[:, :self.num_arm_dofs] =  tensor_clamp(self.real_robot_cur_targets[:, :self.num_arm_dofs],
    #                                                                         self.robot_dof_lower_limits[: self.num_arm_dofs], 
    #                                                                         self.robot_dof_upper_limits[: self.num_arm_dofs])
    #     elif self.arm_control_mode == 'absolute':
    #         self.real_robot_cur_targets[:, :self.num_arm_dofs] = scale(u_arm, self.robot_dof_lower_limits[:self.num_arm_dofs], self.robot_dof_upper_limits[:self.num_arm_dofs])
    #         self.real_robot_cur_targets[:, :self.num_arm_dofs] = (1 - self.arm_act_moving_average) * self.real_robot_cur_targets[:, :self.num_arm_dofs] + (self.arm_act_moving_average) * self.real_arm_prev_targets
    #     elif "ik" in self.arm_control_mode:
    #         if "eef_max_speed" in self.cfg["env"]:
    #             max_speed = self.cfg["env"]["eef_max_speed"]
    #             u_arm = torch.clamp(u_arm, -max_speed, max_speed)

    #         u_arm *= self.control_dt
    #         pos_delta = u_arm[:, :3]
            
    #         if "norot" in self.arm_control_mode:
    #             orn_delta = torch.zeros((self.num_envs, 3), device=self.device)
    #         else:
    #             orn_delta = u_arm[:, 3:]

    #         orn_quat_delta = orn_delta_to_quat(orn_delta)
    #         self.robot_eef_target_pose = torch.cat((
    #             self.robot_eef_target_pose[:, :3] + pos_delta,
    #             quat_mul(orn_quat_delta, self.robot_eef_target_pose[:, 3:])
    #         ), dim=-1)

    #         self.robot_eef_target_pose[:, :3] =  tensor_clamp(
    #                                                 self.robot_eef_target_pose[:, :3], 
    #                                                 torch.tensor(self.eef_lower_limits, device=self.device), 
    #                                                 torch.tensor(self.eef_upper_limits, device=self.device)
    #                                             )
            
    #         arm_ema = self.cfg["env"].get("armActionsMovingAverage", 0.0)
    #         self.robot_eef_target_pose[:] = self.prev_robot_eef_target_pose[:] * arm_ema + (1 - arm_ema) * self.robot_eef_target_pose[:] 
            
    #         self.ik_pos_error = torch.norm(self.get_eef_pos() - self.robot_eef_target_pose[:, :3], dim=1)
    #         self.extras["ik_pos_error"] = self.ik_pos_error.mean()

    #         ik_pos_delta = self.robot_eef_target_pose[:, :3] - self.get_eef_pos()
    #         ik_orn_delta = orientation_error(self.robot_eef_target_pose[:, 3:], self.get_eef_rot())

    #         if self.cfg["env"]["ik_version"] == "v1":
    #             delta_joints = self.control_ik(torch.cat([ik_pos_delta, ik_orn_delta], axis = -1))
    #         elif self.cfg["env"]["ik_version"] == "v2":
    #             delta_joints = self.control_ik_v2(
    #                                 torch.cat([pos_delta, orn_delta], axis = -1), 
    #                                 max_iters=self.cfg["env"]["ik_max_iters"],
    #                                 tol=self.cfg["env"]["ik_tolerance"],
    #                                 damping=self.cfg["env"].get("ik_damping", 0.05)
    #                             )

    #         if self.cfg["env"].get("max_joint_speed", None) is not None:
    #             joint_delta_limit = self.cfg["env"]["max_joint_speed"] * self.dt
    #             delta_joints = torch.clamp(delta_joints, -joint_delta_limit, joint_delta_limit)

    #         self.robot_cur_targets[:, :self.num_arm_dofs] = self.robot_dof_pos[:, :self.num_arm_dofs] + delta_joints[:]

    #         self.robot_cur_targets[:, :self.num_arm_dofs] =  tensor_clamp(self.robot_cur_targets[:, :self.num_arm_dofs],
    #                                                                     self.robot_dof_lower_limits[: self.num_arm_dofs], 
    #                                                                     self.robot_dof_upper_limits[: self.num_arm_dofs])
        
    #     else:
    #         raise NotImplementedError
    #     self.arm_cur_targets = self.robot_cur_targets[:, :self.num_arm_dofs]
    #     #if self.arm_control_mode not in ['absolute']:
    #     #    self.arm_cur_targets2 = self.arm_cur_targets
    #     if self.is_real:
    #         self.real_robot_cur_targets[:, :self.num_arm_dofs] = self.robot_cur_targets[:, :self.num_arm_dofs]
        
    #     msg = JointState()
    #     msg.position = np.array(self.real_robot_cur_targets[0, :self.num_arm_dofs].cpu())
    #     self.arm_joint_pub.publish(msg)
   
    def compute_eef_target(self, u_arm, robot_eef_target_pose):
            prev_robot_eef_target_pose = robot_eef_target_pose.clone()
            u_arm = u_arm * self.control_dt
            pos_delta = u_arm[:, :3]
            

            if "norot" in self.arm_control_mode:
                orn_delta = torch.zeros((self.num_envs, 3), device=self.device)
            else:
                orn_delta = u_arm[:, 3:]

            orn_quat_delta = orn_delta_to_quat(orn_delta)
            robot_eef_target_pos = tensor_clamp(robot_eef_target_pose[:, :3] + pos_delta,
                                                    torch.tensor(self.eef_lower_limits, device=self.device), 
                                                    torch.tensor(self.eef_upper_limits, device=self.device))
            
            robot_eef_target_pose = torch.cat((
                robot_eef_target_pos,
                quat_mul(orn_quat_delta, robot_eef_target_pose[:, 3:])
            ), dim=-1)

            arm_ema = self.cfg["env"].get("armActionsMovingAverage", 0.0)
            robot_eef_target_pose = prev_robot_eef_target_pose * arm_ema + (1 - arm_ema) * robot_eef_target_pose
            return robot_eef_target_pose

    def update_hand_control(self, u_hand):   
        #     self.update_real_hand_control(u_hand)
        #     return
        
        # if "playback_actions" in self.cfg["env"]:
        #     u_hand = self.fetch_playback_actions()
        
        self.hand_prev_targets[:] = self.hand_cur_targets[:]
        mask = torch.ones(self.num_envs, device=self.device).bool()

        # if "disable_hand_obj_dist_threshold" in self.cfg["env"]:
        #     threshold = self.cfg["env"]["disable_hand_obj_dist_threshold"]
        #     object_pos = self.root_state_tensor[self.object_indices, 0:3]
        #     hand_pos = self.get_eef_pose()[:, :3]
        #     dist = torch.norm(object_pos - hand_pos, dim=-1)
        #     mask[dist > threshold] = 0

        # if "object_reaching_fraction" in self.cfg["env"]:
        #     mask[self.is_reaching_object()] = 0

        #hand control
        if self.hand_control_mode == "relative":
            self.hand_cur_targets[:][mask] = self.hand_prev_targets[:][mask] + self.robot_dof_speed_scale * self.dt * u_hand[mask]
        elif self.hand_control_mode == "absolute":
            hand_cur_targets = scale(u_hand, self.robot_dof_lower_limits[self.num_arm_dofs:], self.robot_dof_upper_limits[self.num_arm_dofs:])
            self.hand_cur_targets[:][mask] = (1 - self.hand_act_moving_average) * hand_cur_targets[mask] + (self.hand_act_moving_average) * self.hand_prev_targets[:][mask]
        #elif self.hand_control_mode == "vr" or self.hand_control_mode == "rosbag":
        #     if hasattr(self, "ros_hand_joints"):
        #         # self.hand_cur_targets[:] = self.real_to_sim(self.ros_hand_joints.repeat((self.num_envs, 1)))
        #         self.hand_cur_targets[:][mask] = self.real_to_sim(self.ros_hand_joints.repeat((self.num_envs, 1)))[mask]
        # elif self.hand_control_mode == "file":
        #     self.hand_cur_targets[:][mask] = self.fetch_hand_joints_from_file()[mask]
        # elif self.hand_control_mode == "vae":
        #     u_hand = self.fetch_hand_joints_from_vae(u_hand)
        #     self.hand_cur_targets[:][mask] = u_hand[mask]
        # elif self.hand_control_mode == "vae_random":
        #     u_hand = torch.randn_like(u_hand)
        #     u_hand = self.fetch_hand_joints_from_vae(u_hand)
        #     self.hand_cur_targets[:][mask] = u_hand[mask]
        # elif self.hand_control_mode == "eigengrasps":
        #     u_hand = self.fetch_hand_joints_from_egrasp(u_hand)
        #     self.hand_cur_targets[:][mask] = scale(u_hand[mask], self.robot_dof_lower_limits[self.num_arm_dofs:], self.robot_dof_upper_limits[self.num_arm_dofs:])
        #     self.hand_cur_targets[:][mask] = (1 - self.hand_act_moving_average) * self.hand_cur_targets[:][mask] + (self.hand_act_moving_average) * self.hand_prev_targets[:][mask]
        elif self.hand_control_mode == "eigengrasps_residual":
            u_hand = self.fetch_hand_joints_from_egrasp_residual(u_hand)
            self.hand_cur_targets[:][mask] = scale(u_hand[mask], self.robot_dof_lower_limits[self.num_arm_dofs:], self.robot_dof_upper_limits[self.num_arm_dofs:])
            self.hand_cur_targets[:][mask] = (1 - self.hand_act_moving_average) * self.hand_cur_targets[:][mask] + (self.hand_act_moving_average) * self.hand_prev_targets[:][mask]
            #self.hand_cur_targets[:][mask] = self.hand_prev_targets[:][mask] + self.robot_dof_speed_scale * self.dt * u_hand[mask]
        # elif self.hand_control_mode == "eigengrasps_random":
        #     u_hand = torch.randn_like(u_hand)
        #     self.hand_cur_targets[:][mask] = self.fetch_hand_joints_from_egrasp(u_hand)[mask]
        # elif self.hand_control_mode == "eigengrasps_axis":
        #     num_egrasps = u_hand.shape[1]
        #     u_hand = torch.arange(self.num_envs, device=self.device) % num_egrasps
        #     u_hand = torch.nn.functional.one_hot(u_hand, num_classes=num_egrasps).float() 
        #     self.hand_cur_targets[:][mask] = self.fetch_hand_joints_from_egrasp(u_hand)[mask]
        # elif self.hand_control_mode == "no_op":
        #     pass
        else:
            raise NotImplementedError

        self.hand_cur_targets[:] = tensor_clamp(self.hand_cur_targets[:],
                                                self.robot_dof_lower_limits[self.num_arm_dofs:], 
                                                self.robot_dof_upper_limits[self.num_arm_dofs:])

        if "max_delta_hand_targets" in self.cfg["env"]:
            clip_value = self.cfg["env"]["max_delta_hand_targets"] * self.control_dt
            delta_hand_targets = self.hand_cur_targets - self.hand_prev_targets
            delta_hand_targets = torch.clip(delta_hand_targets, -clip_value, clip_value)
            self.hand_cur_targets[:] = self.hand_prev_targets + delta_hand_targets

        if self.is_real:
            ema = self.cfg['env'].get('ema_hand_targets', 0.25)
            #ema = 1
            self.hand_cur_targets = self.hand_prev_targets * (1-ema) + self.hand_cur_targets * ema

        self.robot_cur_targets[:, self.num_arm_dofs:] = self.hand_cur_targets[:]

        if self.is_real:
            self.real_hand_cur_targets[:] = self.hand_cur_targets
            self.real_robot_cur_targets[:, self.num_arm_dofs:] = self.hand_cur_targets

        if self.is_real and not self.cfg["env"].get("disable_hand", False):
            self.publish_hand_targets(self.real_hand_cur_targets)

            if not hasattr(self,'prev_time'):
                self.prev_time = time.time()
            error = torch.abs(self.hand_prev_targets - self.get_real_hand_dof_pos()).max().item()
            print("fps: ", 1/(time.time() - self.prev_time), f'error: {error:.2f}')

            self.prev_time = time.time()

    def compute_arm_control(self, u_arm, arm_prev_targets, robot_dof_pos, robot_eef_target_pose, eef_pos, eef_rot, j_eef=None):
        assert("ik" in self.arm_control_mode)
        if self.arm_control_mode == "joint_control":
            arm_cur_targets = u_arm.clone()
        elif self.arm_control_mode == 'absolute':
            arm_cur_targets = scale(u_arm, self.robot_dof_lower_limits[:self.num_arm_dofs], self.robot_dof_upper_limits[:self.num_arm_dofs])
            arm_cur_targets = (1 - self.arm_act_moving_average) * arm_cur_targets + (self.arm_act_moving_average) * arm_prev_targets
        elif "ik" in self.arm_control_mode:
            robot_eef_target_pose = self.compute_eef_target(u_arm, robot_eef_target_pose)
        
            ik_pos_delta = robot_eef_target_pose[:, :3] - eef_pos
            ik_orn_delta = orientation_error(robot_eef_target_pose[:, 3:], eef_rot)

            if self.cfg["env"]["ik_version"] == "v1":
                delta_joints = self.control_ik(torch.cat([ik_pos_delta, ik_orn_delta], axis = -1), robot_dof_pos, j_eef = j_eef)
            elif self.cfg["env"]["ik_version"] == "v2":
                delta_joints = self.control_ik_v2(
                                    torch.cat([pos_delta, orn_delta], axis = -1), 
                                    max_iters=self.cfg["env"]["ik_max_iters"],
                                    tol=self.cfg["env"]["ik_tolerance"],
                                    damping=self.cfg["env"].get("ik_damping", 0.05)
                                )

            if self.cfg["env"].get("max_joint_speed", None) is not None:
                joint_delta_limit = self.cfg["env"]["max_joint_speed"] * self.dt
                delta_joints = torch.clamp(delta_joints, -joint_delta_limit, joint_delta_limit)

            arm_cur_targets = robot_dof_pos[:, :self.num_arm_dofs] + delta_joints
            arm_cur_targets = tensor_clamp(arm_cur_targets, self.robot_dof_lower_limits[:self.num_arm_dofs], self.robot_dof_upper_limits[:self.num_arm_dofs])
        else:
            raise NotImplementedError
        return arm_cur_targets, robot_eef_target_pose

    def update_arm_control(self, u_arm):
        self.prev_robot_eef_target_pose = self.robot_eef_target_pose.clone()
        
        arm_prev_targets = self.robot_cur_targets[:, :self.num_arm_dofs].clone()
        self.robot_cur_targets[:, :self.num_arm_dofs], self.robot_eef_target_pose = self.compute_arm_control(u_arm.clone(), arm_prev_targets.clone(), self.robot_dof_pos.clone(), self.robot_eef_target_pose.clone(), self.get_eef_pos().clone(), self.get_eef_rot().clone())
        self.arm_cur_targets = self.robot_cur_targets[:, :self.num_arm_dofs].clone()

        if self.is_real:
            #self.real_robot_cur_targets[:, :self.num_arm_dofs] = self.robot_cur_targets[:, :self.num_arm_dofs]
            if not self.cfg["env"].get("disable_arm", False):
                msg = JointState()
                msg.position = np.array(self.robot_cur_targets[0, :self.num_arm_dofs].cpu())
                self.arm_joint_pub.publish(msg)

    def pre_physics_step(self, actions):
        if self.is_real and not self.cfg["env"].get("disable_hand", False):
            self.refresh_real_hand_dof_pos()    

        env_ids = self._reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if hasattr(self, 'air_threshold'):
            self.compute_stats()

        self.epoch_count += 1 / self.max_episode_length

        #self.object_scale_log = to_torch([self.gym.get_actor_scale(env, handle) for env, handle in zip(self.env_ptrs, self.object_handles)], device=self.device)
        #print(self.object_scale_log)
  
        # if only goals need reset, then call set API
        #if len(goal_env_ids) > 0 and len(env_ids) == 0:
        #    self.reset_target_pose(goal_env_ids, apply_reset=True)

        # # if goals need reset in addition to other envs, call set API in reset()
        # elif len(goal_env_ids) > 0:
        #     self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        
        self.actions = actions.clone().to(self.device)

        if not hasattr(self, 'last_actions_clip'):
            self.last_actions_clip = self.actions
            self.eps = self.cfg['env'].get('action_epsilon', 0.)
        self.actions = torch.where(torch.abs(self.actions-self.last_actions_clip) >= self.eps, self.actions, self.last_actions_clip)
        
        if self.no_actions_before > 0:
            self.actions = self.actions*(self.progress_buf >= self.no_actions_before).float().unsqueeze(1)

        self.hand_act_dim = self.cfg["env"].get("hand_act_dim", self.actions.shape[1] - self.arm_act_dim)
        u_arm, u_hand = self.actions[:, :self.arm_act_dim], self.actions[:, self.arm_act_dim:self.arm_act_dim+self.hand_act_dim ]
        self.last_actions_clip = self.actions

        #if self.is_real: # and self.cfg["env"].get("disable_arm", False):
        #    u_arm = torch.zeros_like(u_arm)
        #    u_hand = torch.zeros_like(u_hand)#((self.num_envs, self.num_hand_dofs), device=self.device)   

        #if self.is_real and self.cfg["env"].get("disable_hand", False):

        self.update_hand_control(u_hand)
        self.update_arm_control(u_arm)

        #if not "disable_actions" in self.cfg["env"] or not self.cfg["env"]["disable_actions"]:

        if self.control_type == 'dof_pos':
            #if self.is_real:
            #    self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.real_robot_cur_targets))
            #else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.robot_cur_targets))
        self.apply_forces()
        #self.extras['forces'] = torch.mean(torch.cat([x.norm().unsqueeze(0) for x in self.rb_forces]))#
        self.extras['forces'] = self.object_rb_forces.norm(dim=-1).mean()

        if self.is_real:
            self.sync_with_real_time()

    def apply_control(self):
        if self.control_type == 'torque':
            self.gym.refresh_dof_state_tensor(self.sim)
            if not hasattr(self, 'torques'):
                self.torques = torch.zeros_like(self.robot_cur_targets)
            self.torques[:] = self.dt*self.p_gains*(self.robot_cur_targets - self.robot_dof_pos) - self.dt*self.d_gains*self.robot_dof_vel # TODO: default_pos: robot_upper_pos
            #self.torques = torch.clip(self.torques, 100*self.torque_limits, 100*self.torque_limits)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

    def apply_forces(self):
        # if self.cfg["env"].get("apply_forces_at_pos", False):
        #     return self.apply_forces_at_pos()

        if not hasattr(self, 'force_scale'):
            self.force_scale = self.cfg["env"].get("forceScale", 0.0)

            self.force_curriculum = self.cfg["env"].get("force_curriculum", False)
            self.force_curriculum_end = self.cfg["env"].get("force_curriculum_end", 800)
            self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
            self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
            self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)
            self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
            self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
            self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                            * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

        if not hasattr(self, 'phase_force_perturb_start_time'):
            self.phase_force_perturb_start_time = self.cfg['env'].get('phase_force_perturb_start_time', 0.3)
            self.phase_force_perturb_end_time = self.cfg['env'].get('phase_force_perturb_end_time', 0.3)
        phase_force_start_perturb = (self.progress_buf * self.dt) >= (self.time_impact + self.phase_force_perturb_start_time)
        phase_force_end_perturb = (self.progress_buf * self.dt) <= (self.time_impact + self.phase_force_perturb_end_time)
        self.phase_force_perturb = phase_force_start_perturb

        if self.force_scale <= 0 or self.is_real:
            return

        # force_curriculum = self.cfg['env'].get('force_curriculum', 0)
        self.object_rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
        # apply new forces
        #new_forces = F.normalize(torch.randn_like(self.object_rb_forces), dim=-1) * self.force_scale
        new_forces = torch.randn_like(self.object_rb_forces) * self.force_scale

        alpha = 1
        if self.force_curriculum:
            alpha = 1 - min(self.epoch_count / self.force_curriculum_end, 0.95)

        idxs = (torch.rand(self.num_envs, device=self.device) <= self.random_force_prob * alpha) & phase_force_start_perturb & phase_force_end_perturb
        self.object_rb_forces[idxs] = new_forces[idxs]
        #self.object_rb_forces *= torch.maximum(self.progress_buf.float() - force_curriculum, torch.zeros_like(self.progress_buf)).unsqueeze(1) / (self.max_episode_length - force_curriculum)
        self.rb_forces[self.object_root_indices] = self.object_rb_forces
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE)

        # for idx in range(self.num_envs):
        #     self.rb_forces[idx] *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
        #     rb = self.rb_forces[idx][self.robot_rb_count:]
        #     new_forces = torch.randn_like(rb) * self.force_scale
        #     idxs = torch.rand_like(rb) <= self.random_force_prob[idx] / rb.shape[0]
        #     alpha = max(self.progress_buf[idx] - force_curriculum, 0) / (self.max_episode_length - force_curriculum)
        #     rb[idxs] = new_forces[idxs] * alpha
        #self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(torch.cat(self.rb_forces, dim=0)), None, gymapi.ENV_SPACE)

    def sync_with_real_time(self):
        if not hasattr(self, "last_tick"):
            self.last_tick = time.time()

        self.rate.sleep()

        time_delta = time.time() - self.last_tick

        #print(f"Wall clock time: {time_delta:.2f}\tFreq: {1 / time_delta:.2f}")
        self.last_tick = time.time()
    
    # def get_last_arm_actions(self):
    #     return self.actions[:, :self.arm_act_dim]
    
    # def get_reference_trajectory_pos(self):
    #     if not hasattr(self, "current_reference_trajectory_pos"):
    #         self.update_reference_trajectory()

    #     return self.current_reference_trajectory_pos
    
    # def get_hardcoded_smooth(self):
    #     if not hasattr(self, "hardcoded_quat_interpolated"):
    #         x1_quat = self.default_ee_pose[:, 3:7]
    #         x2_quat = self.goal_states[:, 3:7]
    #         time_stop = int(1 / self.control_dt)
    #         steps = torch.linspace(0, 1, self.cfg["env"]["episodeLength"] - time_stop, device=self.device)
    #         self.hardcoded_quat_interpolated = roma.utils.unitquat_slerp(x1_quat, x2_quat, steps) 
        
    #     time_stop = int(1 / self.control_dt)
    #     x1_pos = self.default_ee_pose[:, :3]
    #     x2_pos = self.goal_states[:, :3]
    #     time_scaling = ((self.progress_buf - time_stop) / (self.cfg["env"]["episodeLength"] - time_stop)) ** 2
    #     mask = self.progress_buf < int(1 / self.control_dt)
    #     result_pos = x1_pos + (x2_pos - x1_pos) * time_scaling[:, None]
    #     result_quat = self.default_ee_pose[:, 3:7].clone()
    #     idx = self.progress_buf - time_stop
    #     interp_quats = self.hardcoded_quat_interpolated[idx, torch.arange(self.num_envs, device=self.device)]
    #     result_quat[~mask] = interp_quats[~mask]
        
    #     result = torch.cat((result_pos, result_quat), dim=1)

    #     return result

    # def get_reference_trajectory_pose(self):
    #     if not hasattr(self, "current_reference_trajectory_pos"):
    #         self.update_reference_trajectory()

    #     if not hasattr(self, "current_reference_trajectory_rpy") and not hasattr(self, "current_reference_trajectory_quat"):
    #         raise Exception("Reference trajectory does not have orientation")

    #     if hasattr(self, "current_reference_trajectory_rpy"):
    #         quat = quaternion_from_euler_torch(self.current_reference_trajectory_rpy)
    #     else:
    #         quat = self.current_reference_trajectory_quat

    #     return torch.cat([self.current_reference_trajectory_pos, quat], dim=1)

    # def get_real_reference_trajectory_pose(self):
    #     return self.get_reference_trajectory_pose()
    
    # def get_real_reference_trajectory_pos(self):
    #     return self.get_reference_trajectory_pos()

    def get_arm_base_pos(self):
        return self.robot_rb_states[:, self.link_base_index, :][:,:3]

    def get_eef_pos(self):
        return self.robot_rb_states[:, self.eef_index, :][:,:3]

    def get_eef_rpy(self):
        quat = self.robot_rb_states[:, self.eef_index, :][:,3:]
        r, p, y = get_euler_xyz(quat)

        return torch.stack((r, p, y), dim=1)

    def init_sdf(self):
        asset_files = [self.object_files[obj_name] for obj_name in self.env_object_names] 
        #base_names = [os.path.basename(path).split('.')[0] for path in asset_files]
        #object_paths = ['../assets/'+path.replace('assets/objects', 'assets/meshes/objects').replace('.xml', f'/{base_name}.stl').replace('urdf/objects', 'urdf/objects/meshes').replace('.urdf', ) for base_name, path in zip(base_names, self.object_asset_files)]
        #print(object_paths)
        self.num_samples = 200
        #object_paths = [f'../assets/{path}' for path in asset_files]
        #print(object_paths)
        #_, _, self.sdf_vert_faces, self.sdf_surface_points = initialize_sdfs(object_paths, num_samples=num_samples, device=self.device)
        self.sdf_pcs_verts, self.sdf_pcs_norms = initialize_sdfs(asset_files, num_samples=self.num_samples, device=self.device)

        self.pcs_scaling = 1.0/((self.sdf_pcs_verts.max(dim=1)[0] - self.sdf_pcs_verts.min(dim=1)[0]).prod(dim=1)**(1/3))#[0]
        #self.pcs_scaling = 1.0/(self.sdf_pcs_verts.max(dim=1)[0] - self.sdf_pcs_verts.min(dim=1)[0]).mean(dim=1)#[0]

    def update_randomization_params(self, env_ids, old_scale):
        if not hasattr(self, 'rand_scale'):
            self.rand_scale = torch.zeros_like(self.pcs_scaling)
            self.obj_scale = torch.zeros_like(self.pcs_scaling)


            self.obj_rb_props = [self.gym.get_actor_rigid_body_properties(env, handle)[0] for env, handle in zip(self.env_ptrs, self.object_handles)]
            self.obj_rs_props = [self.gym.get_actor_rigid_shape_properties(env, handle) for env, handle in zip(self.env_ptrs, self.object_handles)]
            self.obj_rs_props = [obj_rs_prop[0] if len(obj_rs_prop) > 0 else SimpleNamespace(friction=1) for obj_rs_prop in self.obj_rs_props]
            self.robot_rb_props = [self.gym.get_actor_rigid_body_properties(env, handle)[0] for env, handle in zip(self.env_ptrs, self.robot_handles)]
            self.robot_rs_props = [self.gym.get_actor_rigid_shape_properties(env, handle)[0] for env, handle in zip(self.env_ptrs, self.robot_handles)]
            self.robot_dof_props = [self.gym.get_actor_dof_properties(env, handle) for env, handle in zip(self.env_ptrs, self.robot_handles)]

            self.p_gains = torch.tensor([prop['stiffness'] for prop in self.robot_dof_props], device=self.device)
            self.d_gains = torch.tensor([prop['damping'] for prop in self.robot_dof_props], device=self.device)
            self.dof_vel_limits = torch.tensor([prop['velocity'] for prop in self.robot_dof_props], device=self.device)
            self.torque_limits = torch.tensor([prop['effort'] for prop in self.robot_dof_props], device=self.device)


        # scale
        for i in env_ids:
            new_scale = self.gym.get_actor_scale(self.env_ptrs[i], self.object_handles[i]) 
            if old_scale[i].item() != new_scale:
                # there has been a reset
                #new_scale is the new randomized scale
                self.rand_scale[i] = new_scale
                self.obj_scale[i] = new_scale * self.pcs_scaling[i]
                self.gym.set_actor_scale(self.env_ptrs[i], self.object_handles[i], new_scale * self.pcs_scaling[i].item())

        self.sim_props = self.gym.get_sim_params(self.sim)
        for i in env_ids:
            env, obj_handle, robot_handle = self.env_ptrs[i], self.object_handles[i], self.robot_handles[i]
            #self.obj_scale[i] = self.gym.get_actor_scale(env, obj_handle)
            self.obj_rb_props[i] = self.gym.get_actor_rigid_body_properties(env, obj_handle)[0]
            obj_rs_prop = self.gym.get_actor_rigid_shape_properties(env, obj_handle)
            self.obj_rs_props[i] = obj_rs_prop[0] if len(obj_rs_prop) > 0 else SimpleNamespace(friction=1)
            self.robot_rb_props[i] = self.gym.get_actor_rigid_body_properties(env, robot_handle)[0]
            self.robot_rs_props[i] = self.gym.get_actor_rigid_shape_properties(env, robot_handle)[0]
            self.robot_dof_props[i] = self.gym.get_actor_dof_properties(env, robot_handle)

            self.p_gains[i] = torch.tensor(self.robot_dof_props[i]['stiffness'])
            self.d_gains[i] = torch.tensor(self.robot_dof_props[i]['damping'])
            self.dof_vel_limits[i] = torch.tensor(self.robot_dof_props[i]['velocity'])
            self.torque_limits[i] = torch.tensor(self.robot_dof_props[i]['effort'])

    def compute_sdf_metrics(self):
        obj_scales = self.obj_scale.unsqueeze(-1)
        obj_pos, obj_quat = self.get_obj_pose()[:,:3], self.get_obj_pose()[:,3:]
        queries = self.robot_rb_states[:, :, :3]

        #obj_ids = self.proc_object_ids.long()
        
        #(num_envs, num_samples, 3) 
        self.pcs_bb = (self.sdf_pcs_verts.max(dim=1)[0] - self.sdf_pcs_verts.min(dim=1)[0]) * obj_scales
        self.pcs_verts = quat_apply(obj_quat.unsqueeze(1).repeat(1, self.num_samples, 1), self.sdf_pcs_verts*obj_scales.unsqueeze(-1)) + obj_pos.unsqueeze(1)
        self.pcs_norms = self.sdf_pcs_norms
        omega, r = self.get_obj_ang_vel().unsqueeze(1), self.pcs_verts - self.get_obj_pos().unsqueeze(1)
        self.pcs_vel = self.get_obj_vel().unsqueeze(1) + torch.cross(omega, r, dim=-1)
        self.pcs_acc = self.get_obj_acc().unsqueeze(1) + torch.cross(omega, torch.cross(omega, r, dim=-1), dim=-1)

        self.sdf_dist, self.sdf_norm, self.sdf_closest = compute_query_sdfs(self.pcs_verts, self.pcs_norms, queries)
        #obj_vert_faces = [self.sdf_vert_faces[obj_id.item()]*obj_scale + obj_pos.unsqueeze(0) for obj_id, obj_scale, obj_pos in zip(obj_ids, obj_scales, obj_positions)]
        #self.sdf_distance, self.sdf_normals, self.sdf_closest_points = compute_distance_sdfs(queries, obj_vert_faces, with_closest_points=True)

        # Visualization
        #fingertips_pos = self.rigid_body_states[:, self.fingertip_palm_indices, :3] # (num_envs, K, 3)



    def reward_at_reset_not_only_air(self):
        return self.max_episode_length * self._reset_buf.float() * (self.air_time == 0)
    def reward_at_reset_not_close_range(self):
        return self.max_episode_length * self._reset_buf.float() * self.reset_obj_close_range()
    def reward_at_reset_only_air_and_close_range(self):
        return self.max_episode_length * self._reset_buf.float() * (self.air_time > 0) * (self.reset_obj_close_range() == 0)
    def reward_at_reset_succeeded(self):
        return self.max_episode_length * self._reset_buf.float() * (self.progress_buf >= self.episode_success-1)
    def reward_at_reset_reached_target_pos(self):
        return self.max_episode_length * self._reset_buf.float() * self.phase_target_pos
    def reward_at_reset_reached_force_perturb(self):
        return self.max_episode_length * self._reset_buf.float() * self.phase_force_perturb
    def reward_sdf_dist(self, indices='fingertip_indices'):
        #return torch.max(self.sdf_dist[:, getattr(self, indices)].abs(), dim=-1)[0]
        #return torch.sqrt(torch.mean(self.sdf_dist[:, getattr(self, indices)]**2, dim=-1))
        return torch.mean(self.sdf_dist[:, getattr(self, indices)].abs(), dim=-1)

    def reward_sdf_dist_fingertips(self):
        return self.reward_sdf_dist(indices='fingertip_indices')
    
    def reward_sdf_dist_fingertips_small(self, indices='fingertip_indices'):
        return torch.mean(torch.clamp(self.sdf_dist[:, getattr(self, indices)].abs(), max=0.1), dim=-1)
        #return torch.mean(torch.exp(10*(torch.clip(self.sdf_dist[:, getattr(self, indices)].abs(), max=d)-d)/d), dim=-1)

    def reward_sdf_dist_palm(self):
        return self.reward_sdf_dist(indices='palm_indices')
    
    def reward_sdf_force_closure(self, indices='fingertip_indices'):
        # From dexgraspnet: https://github.com/PKU-EPIC/DexGraspNet/blob/main/grasp_generation/utils/energy.py
        batch_size = self.num_envs
        indices = getattr(self, indices)
        n_contact = len(indices)
        device = self.device

        contact_points = self.sdf_closest[:, indices, :3] # fingertip_pos
        contact_normal = self.sdf_norm[:, indices, :].reshape(batch_size, 1, 3 * n_contact) 

        transformation_matrix = torch.tensor([[0, 0, 0, 0, 0, -1, 0, 1, 0],
                                              [0, 0, 1, 0, 0, 0, -1, 0, 0],
                                              [0, -1, 0, 1, 0, 0, 0, 0, 0]],
                                            dtype=torch.float, device=device)
        g = torch.cat([torch.eye(3, dtype=torch.float, device=device).expand(batch_size, n_contact, 3, 3).reshape(batch_size, 3 * n_contact, 3),
                    (contact_points @ transformation_matrix).view(batch_size, 3 * n_contact, 3)],  dim=2).float().to(device)
        norm = torch.norm(contact_normal @ g, dim=[1, 2])
        reward = norm**2

        filter = self.sdf_dist[:, indices].mean(dim=-1) < 0.25
        reward = reward * filter.float()
        return reward

    def reward_sdf_force_closure_fingertips(self):
        return self.reward_sdf_force_closure('fingertip_indices')

    def reward_sdf_force_closure_fingertip_palm(self):
        return self.reward_sdf_force_closure('fingertip_palm_indices')

    def reward_inside_hand(self):
        hand_pose = self.get_eef_pose()
        hand_pos, hand_quat = hand_pose[:,:3], hand_pose[:,3:]

        obj_pose = self.get_obj_pose()
        obj_pos, obj_quat = obj_pose[:,:3], obj_pose[:,3:]

        fingertips_pos = self.robot_rb_states[:, self.fingertip_palm_indices, :3] # (num_envs, K, 3)
        positions = torch.cat([obj_pos.unsqueeze(1), fingertips_pos], dim=1) # (num_envs, K+1, 3)

        hand_axes = [quat_apply(hand_quat, e.unsqueeze(0).to(self.device).repeat(self.num_envs, 1)) for e in torch.eye(3)]
        projs = [torch.sum(positions*hand_axis.unsqueeze(1), dim=-1) for hand_axis in hand_axes]
        
        fingertips_min_projs = [proj[:, 1:].min(dim=1)[0] for proj in projs]
        fingertips_max_projs = [proj[:, 1:].max(dim=1)[0] for proj in projs]
        hand_projs = [proj[:, 0] for proj in projs]

        reward_min_axes = [torch.relu(fingertips_min_proj - hand_proj) for fingertips_min_proj, hand_proj in zip(fingertips_min_projs, hand_projs)]
        reward_max_axes = [torch.relu(hand_proj - fingertips_max_proj) for fingertips_max_proj, hand_proj in zip(fingertips_max_projs, hand_projs)]
        reward = (sum(reward_min_axes) + sum(reward_max_axes)) / len(reward_min_axes)
        return reward        

    def draw_eef_axes(self):
        object_pose = self.get_eef_pose()
        for i in range(self.num_envs):
            pos, quat = object_pose[i, :3], object_pose[i, 3:]
            p0 = pos.cpu().numpy()
            px = (pos + quat_apply(quat, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
            py = (pos + quat_apply(quat, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
            pz = (pos + quat_apply(quat, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
            self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1.0, 0.0, 0.0])
            self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.0, 1.0, 0.0])
            self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.0, 0.0, 1.0])
            
    # def sample_points_in_box_volume(self):
    #     if self.cfg["env"]["objectType"] != "procedural_box":
    #         return None
        
    #     rand = torch.rand(self.num_envs, 3, device=self.device)
    #     lower = -self.get_box_dims() / 2
    #     upper = self.get_box_dims() / 2

    #     pos = self.rigid_body_states[:, self.object_rb_handles, :3].reshape((self.num_envs, 3))
    #     quat = self.rigid_body_states[:, self.object_rb_handles, 3:7].reshape((self.num_envs, 4))

    #     points = rand * (upper - lower) + lower
    #     points_world = quat_apply(quat, points) + pos

    #     return points_world
 
    def get_real_eef_pos(self):
        return self.get_real_eef_pose()[:, :3]
    
    def get_real_eef_rpy(self):
        quat = self.get_real_eef_pose()[:, 3:]
        r, p, y = get_euler_xyz(quat)

        return torch.stack((r, p, y), dim=1)
    
    def get_real_goal_pos(self):
        return self.get_real_goal_pose()[:, :3]
    
    def get_real_goal_rpy(self):
        quat = self.get_real_goal_pose()[:, 3:]
        r, p, y = get_euler_xyz(quat)

        return torch.stack((r, p, y), dim=1)
    
    def get_real_eef_target_pos(self):
        return self.get_real_eef_target_pose()[:, :3]
    
    def get_real_eef_target_rpy(self):
        quat = self.get_real_eef_target_pose()[:, 3:]
        r, p, y = get_euler_xyz(quat)

        return torch.stack((r, p, y), dim=1)
    
    def get_real_eef_rot(self):
        return self.get_real_eef_pose()[:, 3:]

    def get_eef_target_pose(self):
        return self.standardize_pose(self.robot_eef_target_pose)
    
    # def get_num_resets(self):
        # return self.num_resets

    def get_eef_target_pos(self):
        return self.robot_eef_target_pose[:, :3]
    
    def get_eef_target_rpy(self):
        quat = self.robot_eef_target_pose[:, 3:]
        r, p, y = get_euler_xyz(quat)

        return torch.stack((r, p, y), dim=1)

    def get_real_eef_target_pose(self):
        return self.standardize_pose(self.real_robot_eef_target_pose)

    def draw_points(self, points, color=(1, 0, 0), radius=0.01):
        sphere_geom = gymutil.WireframeSphereGeometry(radius, 16, 16, None, color=color)
        for i in range(self.num_envs):
            point = points[i]
            sphere_pose = gymapi.Transform(gymapi.Vec3(point[0], point[1], point[2]), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def draw_sdf_closest(self, color=(1, 0.5, 0), touch_color=(1, 1, 0), radius=0.01):
        #print(self.sdf_dist.mean(dim=-1))
        for i in range(self.num_envs):
            for j in self.fingertip_palm_indices:
                point = self.sdf_closest[i, j].cpu().numpy()
                point1 = point + self.sdf_norm[i, j].cpu().numpy()*0.2#torch.clamp(100*self.sdf_dist[i,j], max=0.2)
                #point1 = point + self.pcs_acc[i, j]*0.2
                sphere_pose = gymapi.Transform(gymapi.Vec3(point[0], point[1], point[2]), r=None)

                #contact_forces = self.vec_sensor_tensor[i, j, :3]
                #touching = (self.contact_forces_filt[i,j].norm(dim=-1) >= 0.1) & (self.sdf_dist[i, j] <= 0.05)
                touching = self.contact_filt[i,j]

                sphere_geom = gymutil.WireframeSphereGeometry(radius, 16, 16, None, color=color if not touching else touch_color)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
                #self.gym.add_lines(self.viewer, self.envs[i], 1, np.array([point[0], point[1], point[2], point1[0], point1[1], point1[2]]), np.array(color if not touching else touch_color))

    def draw_obj_closest(self, color=(1, 0.5, 0), touch_color=(1, 1, 0), radius=0.03):
        for i in range(self.num_envs):
            for j in self.fingertip_palm_indices:
                fingertip_pos = self.robot_rb_states[i, j, :3]
                object_pos = self.root_state_tensor[self.object_indices, :3][i]
                point = fingertip_pos
                dist = torch.norm(fingertip_pos - object_pos, dim=-1)
                contact_normal = (fingertip_pos - object_pos)/dist
                point1 = point + contact_normal*torch.clamp(dist, max=0.2)
                sphere_pose = gymapi.Transform(gymapi.Vec3(point[0], point[1], point[2]), r=None)

                #contact_forces = self.vec_sensor_tensor[i, j, :3]
                #contact_forces = self.contact_forces[i, j]
                #touching = (contact_forces.norm(dim=-1) >= 0.1) & (dist <= 0.05)
                touching = self.contact_filt[i,j]

                sphere_geom = gymutil.WireframeSphereGeometry(radius, 16, 16, None, color=color if not touching else touch_color)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
                self.gym.add_lines(self.viewer, self.envs[i], 1, [point[0], point[1], point[2], point1[0], point1[1], point1[2]], color if not touching else touch_color)


    def draw_sdf(self, color=(1, 0, 0), radius=0.01):
        sphere_geom = gymutil.WireframeSphereGeometry(radius, 16, 16, None, color=color)
        for i in range(self.num_envs):
            #for j in np.random.permutation(range(self.num_samples))[:5]:
            for j in range(self.num_samples)[:5]:
                point = self.pcs_verts[i, j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(point[0], point[1], point[2]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)


    # def draw_object_reference_points(self):
    #     sphere_geom = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(1, 0, 0))
    #     object_pose = self.get_obj_pose()

    #     for i in range(self.num_envs):
    #         reference_point_object = self.object_reference_points[i]
    #         reference_point_world = quat_apply(object_pose[i, 3:], reference_point_object) + object_pose[i, :3]

    #         x = reference_point_world[0]
    #         y = reference_point_world[1]
    #         z = reference_point_world[2]

    #         sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
    #         gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    # def draw_obj_corners(self):
    #     if self.cfg["env"]["objectType"] != "procedural_box":
    #         return
        
    #     sphere_geom = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(1, 0, 0))
    #     box_corners = self.get_box_corners()

    #     for i in range(self.num_envs):
    #         for j in range(8):
    #             x = box_corners[i, 3 * j]
    #             y = box_corners[i, 3 * j + 1]
    #             z = box_corners[i, 3 * j + 2]

    #             sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
    #             gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def draw_target_eef_pos(self):
        # add target end-effector position
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 16, 16, None, color=(1, 0, 0))

        for i in range(self.num_envs):
            x = self.robot_eef_target_pose[i, 0]
            y = self.robot_eef_target_pose[i, 1]
            z = self.robot_eef_target_pose[i, 2]

            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    # def draw_object_reaching_target(self):
    #     if "object_reaching_fraction" not in self.cfg["env"]:
    #         return
        
    #     object_reaching_targets = self.get_object_reaching_targets()

    #     # add target end-effector position
    #     sphere_geom = gymutil.WireframeSphereGeometry(0.05, 16, 16, None, color=(0, 0, 1))

    #     for i in range(self.num_envs):
    #         x = object_reaching_targets[i, 0]
    #         y = object_reaching_targets[i, 1]
    #         z = object_reaching_targets[i, 2]

    #         sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
    #         gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def draw_reference_traj(self):
        if hasattr(self, "current_reference_trajectory_pos"):
            sphere_geom = gymutil.WireframeSphereGeometry(0.05, 16, 16, None, color=(0, 1, 1))

            for i in range(self.num_envs):
                x = self.current_reference_trajectory_pos[i, 0] 
                y = self.current_reference_trajectory_pos[i, 1] 
                z = self.current_reference_trajectory_pos[i, 2] 

                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)   

        if hasattr(self, "reference_trajectory_pos"):
            sphere_geom = gymutil.WireframeSphereGeometry(0.05, 16, 16, None, color=(1, 1, 1))

            for i in range(self.num_envs):
                x = self.ros_eef_pos[0]
                y = self.ros_eef_pos[1]
                z = self.ros_eef_pos[2]

                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=gymapi.Quat(*self.ros_eef_rot))
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)   

        if hasattr(self, "current_reference_trajectory_pos"):
            sphere_geom = gymutil.WireframeSphereGeometry(0.05, 16, 16, None, color=(1, 1, 1))

            for i in range(self.num_envs):
                x = self.current_reference_trajectory_pos[i, 0] + self.ep_object_init_state[i, 0]
                y = self.current_reference_trajectory_pos[i, 1] + self.ep_object_init_state[i, 1]
                z = self.current_reference_trajectory_pos[i, 2] + self.ep_object_init_state[i, 2]

                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)  

    # def draw_cameras(self):
    #    if not self.is_cameras_active:
    #         return
        
    #     for camera_name in self.cfg["env"]["cameras"]["camera_names"]:
    #         cam_cfg = self.cfg["env"]["cameras"]["camera_names"][camera_name]
    #         body_name = cam_cfg.get("body_name", "root")
            
    #         if body_name == "root":
    #             body_name = "link_base"
            
    #         body_idx = self.link_dict[body_name]

    #         for i in range(self.num_envs):
    #             root_pose = self.rigid_body_states[i, body_idx, :7]
    #             root_pos, root_quat = root_pose[:3], root_pose[3:]
                
    #             cam_position = quat_apply(root_quat, torch.tensor(cam_cfg["location"], device=self.device).float()) + root_pos
    #             qx = torch.from_numpy(quat_from_axis_angle(np.array([1, 0, 0]), np.radians(cam_cfg["rotation"][0]))).to(self.device).float()
    #             qy = torch.from_numpy(quat_from_axis_angle(np.array([0, 1, 0]), np.radians(cam_cfg["rotation"][1]))).to(self.device).float()
    #             qz = torch.from_numpy(quat_from_axis_angle(np.array([0, 0, 1]), np.radians(cam_cfg["rotation"][2]))).to(self.device).float()
    #             cam_quat = quat_mul(root_quat, quat_mul(qz, quat_mul(qy, qx)))
                
    #             px = (cam_position + quat_apply(cam_quat, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
    #             py = (cam_position + quat_apply(cam_quat, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
    #             pz = (cam_position + quat_apply(cam_quat, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
    #             p0 = cam_position.cpu().numpy()
            
    #             self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
    #             self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
    #             self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

    def draw_fk(self):
        joint_indices = {
                "joint1": 0,
                "joint2": 1,
                "joint3": 2,
                "joint4": 3,
                "joint5": 4,
                "joint6": 5,
                "xarm_leap": 0
            }

        urdf_path = os.path.join(self.asset_root, self.robot_asset_file)
        kinematics = Kinematics(urdf_path, "link_base", "leap_mount", joint_indices, self.device)
        pos_batched, quat_batched = kinematics.forward_kinematics(self.robot_dof_pos[:, :self.num_arm_dofs])
        for i in range(self.num_envs):
            pos = pos_batched[i]
            quat = quat_batched[i]

            px = (pos + quat_apply(quat, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
            py = (pos + quat_apply(quat, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
            pz = (pos + quat_apply(quat, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

            p0 = pos.cpu().numpy()
            self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
            self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
            self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

    # def reset_eef_angle(self):
    #     rpy = self.get_eef_rpy()
    #     return rpy[:,1]

    def draw_object_pose(self):
        object_pose = self.get_obj_pose()

        for i in range(self.num_envs):
            pos, quat = object_pose[i, :3], object_pose[i, 3:]

            px = (pos + quat_apply(quat, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
            py = (pos + quat_apply(quat, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
            pz = (pos + quat_apply(quat, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

            p0 = pos.cpu().numpy()
            self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
            self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
            self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

    def draw_rigid_body(self, indices='finger_indices', size=0.01, color=(0,0,1)):
        indices = getattr(self, indices)
        body_pos = self.robot_rb_states[:, indices, :3]

        for i in range(self.num_envs):
            for j in range(len(indices)):
                sphere_geom = gymutil.WireframeSphereGeometry(size, 16, 16, None, color=color)
                x = body_pos[i, j, 0]
                y = body_pos[i, j, 1]
                z = body_pos[i, j, 2]

                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    

    def reset(self):
        self._reset_buf[:] = 1
        self.obs_dict["obs"] = torch.zeros_like(self.obs_buf).to(self.rl_device)

        return self.obs_dict

    def draw_segm_centers(self):
        assert(self.phase2_buffer == 'xyzs')
        xyzs = self.get_single_buffer_phase2()
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 4, 4, None, color=(0.85, 0.1, 0.1))
        for i in range(self.num_envs):
            for obj_pos in xyzs[i,:3][xyzs[i,4:5].repeat(3, 1, 1)==1].reshape(3, -1).permute(1, 0):
            # if obj_pos.shape[1] == 0:
            #     continue
            # obj_pos = obj_pos.median(dim=1)[0]
            # print(obj_pos, self.depth_buffer[i][0][xyzs[i,4] == 1].median(), self.get_obj_pos())
                pos = obj_pos.cpu().numpy() 
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None))

    def draw_forces(self):
        obj_pos = self.get_obj_pos()
        sphere_geom_phase_start = gymutil.WireframeSphereGeometry(0.1, 4, 4, None, color=(0.85, 0.1, 0.1))
        sphere_geom_phase_target_joints = gymutil.WireframeSphereGeometry(0.1, 4, 4, None, color=(0.1, 0.85, 0.1))
        sphere_geom_phase_force_perturb = gymutil.WireframeSphereGeometry(0.1, 4, 4, None, color=(0.1, 0.5, 0.85))
        for i in range(self.num_envs):
            #for j in range(self.robot_rb_count, self.robot_rb_count+self.object_rb_count[i]):
            pos1, pos2 = obj_pos[i].cpu().numpy(), (obj_pos[i] + self.object_rb_forces[i]).cpu().numpy() #pos + quat_apply(quat, torch.tensor([0,0,1], device=self.device).float())
            #self.gym.add_lines(self.viewer, self.envs[i], 1, [pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]], [0.85, 0.1, 0.1])
            self.gym.add_lines(self.viewer, self.envs[i], 1, [pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]], (0.85, 0.1, 0.1))
            sphere_pose = gymapi.Transform(gymapi.Vec3(pos1[0], pos1[1], pos1[2]), r=None)
            sphere_geom = sphere_geom_phase_start
            if self.phase_force_perturb[i]:
                sphere_geom = sphere_geom_phase_force_perturb
            if self.phase_target_pos[i]:
                sphere_geom = sphere_geom_phase_target_joints
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def post_physics_step(self):
        self.randomize_buf += 1
        self.progress_buf += 1
        self.global_counter += 1

        self.compute_observations()

        self.compute_reward(self.actions)

        #if self.cfg["env"].get("mask_first_obs", False):
        #    # This is required since at reset the obs_buf contains the obs from the last ep
        #    # This leaks info across episodes and the RNN policy doesn't work at test time during the first ep
        #    self.obs_buf[self._reset_buf] = 0

        self.update_reset_buf()

        if self.viewer is not None: # and self.debug_viz:
            # draw axes on target object
            self.depth_image_callback()
            
            if self.is_test and self.is_phase2:
                self.gym.clear_lines(self.viewer)
                #self.draw_segm_centers()
                #: # and not self.is_phase2:

                #self.draw_forces()
                # self.draw_eef_axes()
                # self.draw_sdf_closest()
                # self.draw_rigid_body('finger_indices', color=(0,0,1), size=0.01)
                # self.draw_rigid_body('fingertip_indices', color=(0,1,1), size=0.01)
                # self.draw_rigid_body('palm_indices', color=(0.25,1,1), size=0.02)

    def standardize_pose(self, pose):
        if not hasattr(self, 'standardize_pose_flag'):
            self.standardize_pose_flag = self.cfg["env"].get("standardize_pose", False)
        if not self.standardize_pose_flag:
            return pose
        
        pos, quat = pose[:, :3], pose[:, 3:7]
        quat = quat * torch.sign(quat[:, -1]).unsqueeze(1) # standardize quaternion to have positive sign
        pose = torch.cat([pos, quat], dim=-1)

        return pose

    def get_single_buffer_phase2(self):
        if self.is_real_input and hasattr(self, "cam_obs_tensor"):
                return self.cam_obs_tensor
                
        if self.phase2_buffer_type == 'xyz':
            return self.xyz_buffer.view(self.num_envs, -1, *self.camera_res)
        elif self.phase2_buffer_type == 'depth':
            return self.depth_buffer.view(self.num_envs, -1, *self.camera_res)
        elif self.phase2_buffer_type == 'xyzs':
            return torch.cat([self.xyz_buffer.view(self.num_envs, -1, *self.camera_res), self.segm_buffer.view(self.num_envs, -1, *self.camera_res)], dim=1)
        elif self.phase2_buffer_type == 'dxdydz':
            ds = self.diff_buffer.view(self.num_envs, -1, *self.camera_res)
            return torch.cat([self.xyz_buffer.view(self.num_envs, -1, *self.camera_res)*ds], dim=1)
        elif self.phase2_buffer_type == 'dd':
            ds = self.diff_buffer.view(self.num_envs, -1, *self.camera_res)
            return self.depth_buffer.view(self.num_envs, -1, *self.camera_res)*ds
        else:
            raise NotImplemented

    def get_buffer_phase2(self):
        #if len(self.phase2_buffers) != self.num_camera_frames:
        #    return torch.zeros_like(self.get_single_buffer_phase2()).repeat(1, self.num_camera_frames, 1, 1)
        return self.get_single_buffer_phase2()
        #return torch.cat(self.phase2_buffers, dim=1)

    def depth_image_callback(self):
        if not self.is_cameras_active:
            return

        if not self.new_depth_image:
            return

        i = self.lookat_id

        buffers = self.get_buffer_phase2()[i]
        combined_images = buffers.permute((1,0,2)).reshape(buffers.shape[1],-1, 1).cpu().numpy()
        
        #torch.cat([depth_image, depth_image, segm_image], dim=0).permute(1, 2, 0).cpu().numpy()
        if not hasattr(self, 'im'):
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(combined_images, cmap='viridis')
            plt.show()
        else:#if self.global_counter % 10 == 0:
            self.im.set_data(combined_images)
            self.fig.canvas.draw()
            plt.pause(0.0001)

        #plt.figure(1); plt.clf()
        #plt.imshow()
        #

    def get_proc_object_ids(self):
        return self.proc_object_ids[:, None]

    def get_proc_object_ids_onehot(self):
        return F.one_hot(self.proc_object_ids.long(), num_classes=self.num_objects)

    def get_ep_time(self):
        ep_time = self.progress_buf * self.control_dt
        
        return ep_time.unsqueeze(1)

    def get_finger_palm_pos(self):
        return self.robot_rb_states[:, self.fingertip_palm_indices, :3].reshape(self.num_envs, -1)
    
    def get_obj_pc_rand(self, num_samples=3):
        self.sample_ids = torch.randperm(self.num_samples, device=self.device)[:num_samples]
        return self.pcs_verts[:, self.sample_ids, :].reshape(self.num_envs, -1)

    def get_obj_pc_rand_vel(self, num_samples=3):
        return self.pcs_vel[:, self.sample_ids, :].reshape(self.num_envs, -1)

    def get_obj_pc_rand_acc(self, num_samples=3):
        return self.pcs_acc[:, self.sample_ids, :].reshape(self.num_envs, -1)

    def get_obj_pc(self, num_samples=5):
        sample_ids = torch.randperm(self.num_samples, device=self.device)[:num_samples].reshape(1, -1).repeat(self.num_envs, 1)
        if not hasattr(self, "pc_perm"):
            self.sample_ids = sample_ids
        self.sample_ids[self._reset_buf] = sample_ids[self._reset_buf]
    
        return self.pcs_verts[self.sample_ids].reshape(self.num_envs, -1)

    def get_obj_pc_rand_dist(self, num_samples=5):
        return torch.norm(self.get_obj_pc_rand(num_samples).reshape(self.num_envs, num_samples, 1, 3) - self.get_finger_palm_pos().reshape(self.num_envs, 1, -1, 3), dim=-1).reshape(self.num_envs, -1)

    # eigen grasps

    def init_egrasp_matrix(self):
        self.egrasp = np.load(self.cfg["env"]["egrasp_file"])
        self.egrasp = torch.from_numpy(self.egrasp).to(self.device).float()

        if "extra_egrasp" in self.cfg["env"]:
            for _, egrasp in self.cfg["env"]["extra_egrasp"].items():
                egrasp = torch.tensor(egrasp, device=self.device)
                self.egrasp = torch.cat([self.egrasp, egrasp[None, :]], dim=0)

        self.num_egrasps = self.egrasp.shape[0] 

        return self.num_egrasps

    def fetch_hand_joints_from_egrasp(self, u_hand):
        if not hasattr(self, "egrasp_file"):
            self.init_egrasp_matrix()

        hand_joints = u_hand @ self.egrasp 
        hand_joints = scale(hand_joints, self.robot_dof_lower_limits[self.num_arm_dofs:], self.robot_dof_upper_limits[self.num_arm_dofs:])
        
        return hand_joints
    
    def fetch_hand_joints_from_egrasp_residual(self, u_hand):
        if not hasattr(self, "egrasp_file"):
            self.init_egrasp_matrix()
            # find the null space of the egrasp matrix
            U, D, V = torch.linalg.svd(self.egrasp)
            self.residual_action_matrix = V[self.num_egrasps:]

        principal_actions = u_hand[:, :self.num_egrasps]
        principal_actions = principal_actions @ self.egrasp

        residual_actions = u_hand[:, self.num_egrasps:] * self.cfg["env"]["residual_action_scale"]
        residual_actions = residual_actions @ self.residual_action_matrix

        actions = principal_actions + residual_actions        

        return actions
    
#####################################################################
###=========================jit functions=========================###
#####################################################################


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def orn_delta_to_quat(orn_delta):
    return torch.cat((orn_delta, torch.ones_like(orn_delta[:, :1])), dim=-1)

def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        colors.append(colorsys.hsv_to_rgb(i/n, 1.0, 1.0))
    return colors

def quaternion_from_euler(rpy):
    r, p, y = rpy

    qx = np.sin(r/2) * np.cos(p/2) * np.cos(y/2) - np.cos(r/2) * np.sin(p/2) * np.sin(y/2)
    qy = np.cos(r/2) * np.sin(p/2) * np.cos(y/2) + np.sin(r/2) * np.cos(p/2) * np.sin(y/2)
    qz = np.cos(r/2) * np.cos(p/2) * np.sin(y/2) - np.sin(r/2) * np.sin(p/2) * np.cos(y/2)
    qw = np.cos(r/2) * np.cos(p/2) * np.cos(y/2) + np.sin(r/2) * np.sin(p/2) * np.sin(y/2) 

    return np.array([qx, qy, qz, qw])


def quaternion_from_euler_torch(rpy):
    r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]

    qx = torch.sin(r/2) * torch.cos(p/2) * torch.cos(y/2) - torch.cos(r/2) * torch.sin(p/2) * torch.sin(y/2)
    qy = torch.cos(r/2) * torch.sin(p/2) * torch.cos(y/2) + torch.sin(r/2) * torch.cos(p/2) * torch.sin(y/2)
    qz = torch.cos(r/2) * torch.cos(p/2) * torch.sin(y/2) - torch.sin(r/2) * torch.sin(p/2) * torch.cos(y/2)
    qw = torch.cos(r/2) * torch.cos(p/2) * torch.cos(y/2) + torch.sin(r/2) * torch.sin(p/2) * torch.sin(y/2) 

    return torch.stack([qx, qy, qz, qw], dim=1)

def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return torch.stack([roll_x, pitch_y, yaw_z], dim=1) # in radians

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot

def get_random_uniform_tensor(low, high, shape, device):
    return torch.rand(shape, dtype=torch.float32, device=device) * (high - low) + low

def dot_to_bracket(field):
    # Convert a string x.y.z.w to ['x']['y']['z']['w']
    return "['" + "']['".join(field.split(".")) + "']"

def quat_from_axis_angle(axis, angle):
    return np.concatenate((np.sin(angle / 2) * axis, np.array([np.cos(angle / 2)])))

def move_to_range(angle, low, high):
    while angle < low:
        angle += 2 * np.pi

    while angle > high:
        angle -= 2 * np.pi

    assert(angle > low and angle < high)

    return angle

def move_to_range_torch(angle, low, high):
    while torch.any(angle < low):
        angle[angle < low] = angle[angle < low] + 2 * np.pi
    
    while torch.any(angle > high):
        angle[angle > high] = angle[angle > high] - 2 * np.pi

    return angle

@torch.jit.script
def depth_image_to_point_cloud_GPU(camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, device:torch.device):
    depth_buffer = camera_tensor.to(device)
    vinv = camera_view_matrix_inv
    proj = camera_proj_matrix
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv

    X = X.view(-1)
    Y = Y.view(-1)
    Z = Z.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device))) 
    position = position.permute(1, 0)
    position = position@vinv

    points = position[:, 0:3]

    return points

def compute_obs_lists(cfg, device):
    # for phase2
    obs_input = []
    obs_phase1 = []
    if isinstance(cfg["env"]["observation_list"], dict):
        cur = 0
        for idx, (obs_name, obs) in enumerate(cfg["env"]["observation_list"].items()):
            print(obs)
            obs_priv, obs_dim = obs[0], obs[1]
            if obs_priv == 'input':
                obs_input += [cur + i for i in range(obs_dim)]
            if obs_priv == 'phase1':
                obs_phase1 += [cur + i for i in range(obs_dim)]
            cur += obs_dim
    obs_input  = torch.tensor(obs_input,  dtype=torch.long, device=device)
    obs_phase1 = torch.tensor(obs_phase1, dtype=torch.long, device=device)

    return obs_input, obs_phase1