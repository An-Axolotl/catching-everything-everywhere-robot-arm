import torch
import numpy as np
import torchvision

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

def process_image(image, cfg):
    camera_res = [int(x) for x in cfg['camera_res']]
    crop_top, crop_bottom, crop_left, crop_right = cfg['crop_top'], cfg['crop_bottom'], cfg['crop_left'], cfg['crop_right']
    resize_transform = torchvision.transforms.Resize(camera_res, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    if isinstance(image, np.ndarray):
        image = torch.tensor(image)
    image = image[crop_top:crop_bottom, crop_left:crop_right].float() # crop
    image = resize_transform(image[None, :]).squeeze() # resiwe
    return image

def process_xyz_image(xyz_buffer, cfg):
    xyz_bg_noise_scale = cfg['xyz_bg_noise_scale']
    xyz_min_clip = cfg['xyz_min_clip']
    xyz_max_clip = cfg['xyz_max_clip']
    xyz_buffer[0] = xyz_buffer[0] + torch.randn_like(xyz_buffer[0])*xyz_bg_noise_scale[0]
    xyz_buffer[1] = xyz_buffer[1] + torch.randn_like(xyz_buffer[1])*xyz_bg_noise_scale[1]
    xyz_buffer[2] = xyz_buffer[2] + torch.randn_like(xyz_buffer[2])*xyz_bg_noise_scale[2]

    xyz_buffer[0] = 2*(torch.clip(xyz_buffer[0], min=xyz_min_clip[0], max=xyz_max_clip[0]) - xyz_min_clip[0]) / (xyz_max_clip[0] - xyz_min_clip[0]) - 1
    xyz_buffer[1] = 2*(torch.clip(xyz_buffer[1], min=xyz_min_clip[1], max=xyz_max_clip[1]) - xyz_min_clip[1]) / (xyz_max_clip[1] - xyz_min_clip[1]) - 1
    xyz_buffer[2] = 2*(torch.clip(xyz_buffer[2], min=xyz_min_clip[2], max=xyz_max_clip[2]) - xyz_min_clip[2]) / (xyz_max_clip[2] - xyz_min_clip[2]) - 1
    return xyz_buffer

def get_xyz_image_from_depth_buffer(depth_buffer, cfg):
    device = depth_buffer.device
    xyz_buffer = depth_image_to_point_cloud_GPU(depth_buffer, torch.tensor(cfg['view'], device=device).float(), torch.tensor(cfg['proj'], device=device).float(), torch.tensor(cfg['u'], device=device).long(), torch.tensor(cfg['v'], device=device).long(), cfg['w'], cfg['h'], device).reshape(*depth_buffer.shape, 3).permute(2, 0, 1)
    xyz_image = process_image(xyz_buffer, cfg)
    #xyz_image = xyz_buffer
    xyz_image = process_xyz_image(xyz_image, cfg)
    return xyz_image
