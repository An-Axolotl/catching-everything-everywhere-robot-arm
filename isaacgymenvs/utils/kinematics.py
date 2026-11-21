import numpy as np
import xml.etree.ElementTree as ET
import isaacgym
from isaacgym.torch_utils import *
import torch
import torch.nn as nn
from typing import Any, Callable, Sequence, Dict
import torch.autograd.functional as autogradF
from isaacgym.torch_utils import get_euler_xyz
from xarm.wrapper import XArmAPI

def quaternion_from_euler_torch(rpy):
    r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]

    qx = torch.sin(r/2) * torch.cos(p/2) * torch.cos(y/2) - torch.cos(r/2) * torch.sin(p/2) * torch.sin(y/2)
    qy = torch.cos(r/2) * torch.sin(p/2) * torch.cos(y/2) + torch.sin(r/2) * torch.cos(p/2) * torch.sin(y/2)
    qz = torch.cos(r/2) * torch.cos(p/2) * torch.sin(y/2) - torch.sin(r/2) * torch.sin(p/2) * torch.cos(y/2)
    qw = torch.cos(r/2) * torch.cos(p/2) * torch.cos(y/2) + torch.sin(r/2) * torch.sin(p/2) * torch.sin(y/2) 

    return torch.stack([qx, qy, qz, qw], dim=1)

class Kinematics(nn.Module):
    def __init__(self, urdf_path, base_link_name, end_effector_link_name, joint_indices, device):
        super().__init__()
        #self.arm = XArmAPI("192.168.1.227", is_radian=True)
        
        self.base_link_name = base_link_name
        self.end_effector_link_name = end_effector_link_name
        self.joint_indices = joint_indices
        self.device = device
        self.robot = self._parse_urdf(urdf_path)

    def forward_kinematics(self, joint_angles):
        #end_effector_pose = torch.tensor([self.arm.get_forward_kinematics([x.item() for x in y], input_is_radian=True, return_is_radian=True)[1] for y in joint_angles], device=joint_angles.device)
        #print(joint_angles.shape, end_effector_pose.shape)
        #return end_effector_pose[:,:3], quaternion_from_euler_torch(end_effector_pose[:,3:])

        end_effector_pose = self._compute_pose(joint_angles, self.base_link_name) # [B, 4, 4]
        quat = self._pose_to_quat(end_effector_pose)

        return end_effector_pose[:, 0:3, 3], quat
    
    def forward_kinematics_euler(self, joint_angles):
        eef_pos, eef_quat = self.forward_kinematics(joint_angles)
        eef_euler = get_euler_xyz(eef_quat)
        eef_euler = torch.stack(eef_euler, dim=-1)

        return torch.cat((eef_pos, eef_euler), dim=-1)

    def _pose_to_quat(self, pose, eps=1e-16):
        batch_size = pose.shape[0]
        quat = torch.zeros((batch_size, 4), device=pose.device)
        quat[:, 3] = 1 + pose[:, 0, 0] + pose[:, 1, 1] + pose[:, 2, 2]
        
        quat[:, 3] = torch.clamp(quat[:, 3], min=eps)
        quat[:, 3] = torch.sqrt(quat[:, 3]) / 2
        
        quat[:, 0] = (pose[:, 2, 1] - pose[:, 1, 2]) / (4 * quat[:, 3])
        quat[:, 1] = (pose[:, 0, 2] - pose[:, 2, 0]) / (4 * quat[:, 3])
        quat[:, 2] = (pose[:, 1, 0] - pose[:, 0, 1]) / (4 * quat[:, 3])

        return quat

    def _parse_urdf(self, urdf_path):
        robot = {'links': {}, 'joints': {}}

        tree = ET.parse(urdf_path)
        root = tree.getroot()

        for link in root.findall('link'):
            link_data = {
                'name': link.get('name'),
                'parent': None,
                'children': [],
                'pose': self._parse_origin(link.find('visual/origin')),
            }
            robot['links'][link_data['name']] = link_data

        for joint in root.findall('joint'):
            axis = joint.find('axis')

            if axis is not None:
                axis = np.fromstring(axis.get('xyz'), sep=' ')
                axis = torch.from_numpy(axis).to(self.device).float()

            joint_data = {
                'name': joint.get('name'),
                'parent': joint.find('parent').get('link'),
                'child': joint.find('child').get('link'),
                'type': joint.get('type'),
                'axis': axis,
                'pose': self._parse_origin(joint.find('origin')),
            }
            robot['joints'][joint_data['name']] = joint_data

            robot['links'][joint_data['parent']]['children'].append(joint_data['child'])

        return robot

    def _parse_origin(self, origin_element):
        if origin_element is None:
            return torch.eye(4, device=self.device)
        else:
            xyz = np.fromstring(origin_element.get('xyz', '0 0 0'), sep=' ')
            xyz = torch.from_numpy(xyz).to(self.device).float()
            rpy = np.fromstring(origin_element.get('rpy', '0 0 0'), sep=' ')
            rpy = torch.from_numpy(rpy).to(self.device).float()
            return self._pose_from_xyz_rpy(xyz, rpy)

    def _pose_from_xyz_rpy(self, xyz, rpy):
        pose = torch.eye(4, device=self.device)
        pose[0:3, 3] = xyz
        roll, pitch, yaw = rpy
        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        rot_x = torch.tensor([[1, 0, 0], [0, cr,
                -sr], [0, sr, cr]], device=self.device)
        rot_y = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], device=self.device)
        rot_z = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], device=self.device)
        pose[0:3, 0:3] = rot_z @ rot_y @ rot_x
        return pose

    def _compute_pose(self, joint_angles, link_name):
        batch_size = joint_angles.shape[0]
        link_data = self.robot['links'][link_name]
        pose = link_data['pose'].repeat((batch_size, 1, 1))

        if link_name == self.end_effector_link_name:
            return pose

        for child_link_name in link_data['children']:
            joint_name = None
            for candidate_joint_name, joint_data in self.robot['joints'].items():
                if joint_data['parent'] == link_name and joint_data['child'] == child_link_name:
                    joint_name = candidate_joint_name
                    break

            if joint_name is None:
                raise ValueError(f"Failed to find joint connecting {link_name} and {child_link_name}")

            if joint_name in self.joint_indices:
                joint_angle = joint_angles[:, self.joint_indices[joint_name]]
            else:
                joint_angle = None

            ee_pose = self._compute_pose(joint_angles, child_link_name)
            joint_transform = self._joint_transform(joint_angle, joint_data['axis']) 
            result = torch.bmm(joint_transform, ee_pose)
            result = torch.bmm(joint_data["pose"].repeat((batch_size, 1, 1)), result)
            result = torch.bmm(pose, result)

        return result

    def _joint_transform(self, angle, axis):
        batch_size = angle.shape[0]

        if axis is None:
            return torch.eye(4).to(self.device).repeat((batch_size, 1, 1))

        ct = torch.cos(angle)
        st = torch.sin(angle)
        vt = 1 - ct
        axis = axis / torch.linalg.norm(axis)

        R = torch.zeros((batch_size, 3, 3), device=self.device)
        R[:, 0, 0] = ct + axis[0] ** 2 * vt
        R[:, 1, 1] = ct + axis[1] ** 2 * vt
        R[:, 2, 2] = ct + axis[2] ** 2 * vt
        R[:, 0, 1] = axis[0] * axis[1] * vt - axis[2] * st
        R[:, 1, 0] = axis[0] * axis[1] * vt + axis[2] * st
        R[:, 0, 2] = axis[0] * axis[2] * vt + axis[1] * st
        R[:, 2, 0] = axis[0] * axis[2] * vt - axis[1] * st
        R[:, 1, 2] = axis[1] * axis[2] * vt - axis[0] * st
        R[:, 2, 1] = axis[1] * axis[2] * vt + axis[0] * st

        transform = torch.eye(4).to(self.device).repeat((batch_size, 1, 1))
        transform[:, 0:3, 0:3] = R
        return transform
    
    def _compute_jac_autograd(self, joint_angles):
        # import pdb; pdb.set_trace()
        # Compute the jacobian using autograd
        result = []

        for angles in joint_angles:
            jac = autogradF.jacobian(self.forward_kinematics_euler, angles[None, :])
            result.append(jac)

        return torch.stack(result)

    def _compute_jac(self, joint_angles, eps=1e-2):
        # import pdb; pdb.set_trace()
        num_joints = joint_angles.shape[1]    
        eef_pos, eef_quat = self.forward_kinematics(joint_angles)
        jac = []

        for i in range(num_joints):
            joint_angles_perturbed = joint_angles.clone()
            joint_angles_perturbed[:, i] += eps
            eef_pos_perturbed, eef_quat_perturbed = self.forward_kinematics(joint_angles_perturbed)

            linear_velocity = (eef_pos_perturbed - eef_pos) / eps   
            quat_diff = quat_mul(eef_quat_perturbed, quat_conjugate(eef_quat))
            angular_velocity = quat_diff[:, 0:3] * torch.sign(quat_diff[:, 3]).unsqueeze(-1) / eps * 2
            jac.append(torch.cat([linear_velocity, angular_velocity], dim=1))

        jac = torch.stack(jac, dim=1)
        jac = jac.permute(0, 2, 1)
        
        # all values are not nan
        assert not torch.isnan(jac).any()

        return jac
    
    def orientation_error(self, desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    def inverse_kinematics(self, dpose, joint_angles, tol=1e-3, only_pos=False, max_iters=1, damping=0.05):
        num_iters = 0
        batch_size = dpose.shape[0]
        error = np.inf * torch.ones((batch_size), device=self.device)

        end_effector_pos, end_effector_quat = self.forward_kinematics(joint_angles)
        desired_pos = end_effector_pos + dpose[:, :3]
        quat_delta = torch.zeros((batch_size, 4), device=self.device)
        quat_delta[:, :3] = dpose[:,3:]
        quat_delta[:, 3] = 1
        quat_delta = quat_delta / torch.norm(quat_delta, dim=1, keepdim=True)
        desired_rot = quat_mul(quat_delta, end_effector_quat)
        desired_pose = torch.cat((desired_pos, desired_rot), dim=1)

        while num_iters < max_iters and torch.any(error > tol):
            end_effector_pos, end_effector_quat = self.forward_kinematics(joint_angles)
            pos_delta = desired_pose[:, :3] - end_effector_pos
            orn_delta = self.orientation_error(desired_pose[:, 3:], end_effector_quat)
            dpose = torch.cat((pos_delta, orn_delta), dim=1)

            j_eef = self._compute_jac(joint_angles)
            j_eef_T = torch.transpose(j_eef, 1, 2)
            lmbda = torch.eye(6, device=self.device) * (damping ** 2)

            if only_pos : 
                u = ( (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda))[:,:,:3] @ dpose[:,:3].unsqueeze(-1)).squeeze()
            else:
                u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose.unsqueeze(-1)).squeeze()

            error = u
            joint_angles = joint_angles + u
            num_iters += 1

        return joint_angles, error

    # def _compute_jacobian(self, joint_angles):
    #     num_joints = len(joint_angles)
    #     end_effector_pose, jacobian_columns = self._compute_pose_and_jacobian_columns(joint_angles, self.end_effector_link_name)
    #     jacobian = np.zeros((3, num_joints))

    #     for i, column in enumerate(jacobian_columns):
    #         jacobian[:, i] = np.dot(column[0:3, 0:3], column[0:3, 3])

    #     return jacobian
