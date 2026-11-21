import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import time
import copy
import cv2
import torch
import torchvision

# Initialize the RealSense pipeline

def to_channels_last(frame):
    return np.transpose(frame, (1, 2, 0))
def to_channels_first(frame):
    return np.transpose(frame, (2, 0, 1))
def bw_to_rgb(frame, channels_last=False):
    return np.repeat(frame, 3, axis=-1 if channels_last else 0)
def bgr_to_rgb(frame, channels_last=False):
    if channels_last:
        return np.stack([frame[:,:,2], frame[:,:,1], frame[:,:,0]], axis=-1)
    return np.stack([frame[2], frame[1], frame[0]], axis=0)
def cat_frames(frames, channels_last=False):
    return np.concatenate(frames, axis=-2 if channels_last else -1)
def img_u8_to_f32(img):
    return img.astype(float)/255
def img_f32_to_u8(img):
    return (255*img).astype(np.uint8)

# https://dev.intelrealsense.com/docs/high-speed-capture-mode-of-intel-realsense-depth-camera-d435#45-latency
def default_camera_config():
    cfg = {
        'depth_img_res': [480, 640],
        'rgb_img_res': [480, 640],
        'img_res': [480, 640],
        'frame_rate': 60,
        'load': None,
        'start_frame': 0,
        'end_frame': -1,
    }
    return cfg

def load_postprocess_settings(cfg):
    #cfg = np.load('../isaacgymenvs/cfg/cfg.npz', allow_pickle=True)
    
    h, w = cfg['img_res']
    camera_v, camera_u = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
    camera_v, camera_u = camera_v.numpy(), camera_u.numpy()

    cfg = {
        'camera_res': [h, w],
        'crop_top': 0,
        'crop_bottom': None,
        'crop_left': 0,
        'crop_right': None,
        'xyz_bg_noise_scale': np.array([0.25, 0.05, 0.05]),
        'xyz_min_clip': np.array([ 0. , -0.5,  0. ]),
        'xyz_max_clip': np.array([8. , 0.5, 2. ]),
        'view': np.array([[ 0.        , -1.        ,  0.        ,  0.        ],
                        [ 0.11645514,  0.        ,  0.993196  ,  0.        ],
                        [-0.993196  ,  0.        ,  0.11645514,  0.        ],
                        [-0.97      ,  0.        ,  1.        ,  1.        ]]),  
        'proj': np.array([[ 1.0537802e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
                        [ 0.0000000e+00,  1.4050403e+00,  0.0000000e+00,  0.0000000e+00],
                        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00, -1.0000000e+00],
                        [ 0.0000000e+00,  0.0000000e+00,  1.0000000e-03,  0.0000000e+00]]),
        'u': camera_u,
        'v': camera_v,
        'w': w,
        'h': h,
    }
    return cfg

class Camera:
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)

    def start(self):
        if self.cfg['load'] is not None:
            self.frames = [frame for frame in np.load(self.cfg['load'])['arr_0'][self.cfg['start_frame']:self.cfg['end_frame']]]
            self.curr_frame = 0
            return
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.cfg['rgb_img_res'][1], self.cfg['rgb_img_res'][0], rs.format.bgr8, self.cfg['frame_rate'])
        self.config.enable_stream(rs.stream.depth, self.cfg['depth_img_res'][1], self.cfg['depth_img_res'][0], rs.format.z16, self.cfg['frame_rate'])

        # Start the pipeline
        self.profile = self.pipeline.start(self.config)

        self.align = rs.align(rs.stream.color)

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.cfg['depth_scale'] = self.depth_sensor.get_depth_scale()

        self.resize_transform = torchvision.transforms.Resize(self.cfg['img_res'], interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        
        # postprocess settings
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.decimation = rs.decimation_filter()
        #self.decimation.set_option(rs.option.filter_magnitude, 4)
        self.spatial = rs.spatial_filter()
        # self.spatial.set_option(rs.option.filter_magnitude, 5)
        # self.spatial.set_option(rs.option.filter_smooth_alpha, 1)
        # self.spatial.set_option(rs.option.filter_smooth_delta, 50)
        # self.spatial.set_option(rs.option.holes_fill, 3)
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

    def postprocess_frame(self, frame):
        frame = self.decimation.process(frame)
        frame = self.depth_to_disparity.process(frame)
        frame = self.spatial.process(frame)
        frame = self.temporal.process(frame)
        frame = self.disparity_to_depth.process(frame)
        frame = self.hole_filling.process(frame)
        return frame

    def get_rgbd_frame(self):

        if self.cfg['load'] is not None:
            frame = self.frames[self.curr_frame]
            self.curr_frame = (self.curr_frame+1)%len(self.frames)
            return frame
        
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return None
        #frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None

        depth_frame = self.postprocess_frame(depth_frame)

        bgr_image = np.asanyarray(color_frame.get_data()).astype(float) / 255
        depth_image = np.asanyarray(depth_frame.get_data()).astype(float)
        distance_image = depth_image * self.cfg['depth_scale']
        rgb_image = bgr_to_rgb(to_channels_first(bgr_image))
        rgb_image = self.resize_transform(torch.tensor(rgb_image)[None, :]).squeeze(0).numpy() # resiwe
        distance_image = self.resize_transform(torch.tensor(distance_image)[None, None, :]).squeeze(0).numpy() # resiwe

        return np.concatenate([rgb_image, distance_image], axis=0)

    def stop(self):
        self.pipeline.stop()

class VideoRecorder:
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.recorder = None

    def start(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
        self.recorder = cv2.VideoWriter('depth_video.avi', fourcc, self.cfg['frame_rate'], self.cfg['img_res'], isColor=False)
    def record_frame(self, frame, alpha=0.01):
        if self.recorder is None:
            self.cfg['img_res'] = frame.shape[0:2]
            self.start()

        self.recorder.write(cv2.convertScaleAbs(frame, alpha=alpha))
    def stop(self, verbose=True):
        self.recorder.release()
        if verbose:
            print('Wrote video')

class InteractivePlot:
    def __init__(self, cfg, title='Camera Map'):
        self.img_res = cfg['img_res']
        self.title = title
        self.running = False
        self.im = None

    def start(self):
        self.fig, ax = plt.subplots()
        self.im = ax.imshow(np.random.rand(*self.img_res[::-1]), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(self.title)
        plt.ion()

        self.running = True

        # Connect key events to quit the script
        def on_key(event):
            if event.key in ('q', 'Q'):
                self.running = False

        self.fig.canvas.mpl_connect('key_press_event', on_key)

    def set_data(self, frame):
        if self.im is None:
            self.img_res = frame.shape[:2][::-1]
            self.start()
        self.im.set_array(frame)
        self.frame = frame

    def render(self, frame, dt = 0.001):
        self.set_data(frame)
        plt.pause(dt)

    def stop(self):
        self.running = False
        plt.ioff()
        plt.show()
        cv2.destroyAllWindows()


# class BackgroundFilter:
#     def __init__(self):
#         self.background_removal = cv2.createBackgroundSubtractorMOG2()

#     def apply(self, frame):
#         frame = img_f32_to_u8(to_channels_last(frame))
#         frame = self.background_removal.apply(frame)
#         return img_u8_to_f32(frame)[None, :, :]

class BackgroundFilter:
    def __init__(self):
        self.prev_frame = None
        self.threshold = 0.05

    def apply(self, frame):
        frame = to_channels_last(frame)

        if self.prev_frame is None:
            self.prev_frame = frame

        diff_frame = np.mean(np.abs(frame - self.prev_frame), axis=-1)/2
        diff_frame = np.where(diff_frame <= self.threshold, np.zeros_like(diff_frame), diff_frame)
        diff_frame[diff_frame > 0] = 1

        self.prev_frame = frame
        return diff_frame[None, :, :]


class SegmentationFilter:
    def __init__(self):
        pass

    def apply(self, frame):
        assert(frame.shape[0] == 1)
        kernel = np.ones((3,3),np.uint8)
        frame = cv2.morphologyEx(frame[0].astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1) 
        return frame[None,:,:]