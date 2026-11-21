from camera_utils import *
from depth_utils import *
from tqdm import tqdm
import torch
import matplotlib
import rospy
from rospy.numpy_msg import numpy_msg
from std_msgs.msg       import Float64, Float32,  Float32MultiArray
import argparse
from threading import Thread
import os

parser = argparse.ArgumentParser()
parser.add_argument('--load', action='store_true')
parser.add_argument('--record', action='store_true')
parser.add_argument('--interactive', action='store_false')
parser.add_argument('--red_dot', action='store_true')
parser.add_argument('--publish', action='store_true')
args = parser.parse_args()
cfg = default_camera_config()

if args.load:
    #cfg['load'] = 'raw_frames.2.npy'
    #cfg['start_frame'] = 930
    #cfg['end_frame'] = 1000
    cfg['load'] = 'recordings/raw_frames.1.npz'
    cfg['start_frame'] = 0
    cfg['end_frame'] = 1000

if args.publish:
    rospy.init_node('cam', anonymous=True)
    camera_pub_obs = rospy.Publisher("/cam/obs_dxdydz", Float32MultiArray, queue_size=10)
    camera_pub_time = rospy.Publisher("/cam/ep_time", Float32MultiArray, queue_size=10)
    r = rospy.Rate(30)
else:
    camera_pub_obs = None
    camera_pub_time = None

int_plot = InteractivePlot(cfg)
int_plot.running = True

#def render_thread(int_plot):
#    while True:
#        if hasattr(int_plot, 'frame'):
#            int_plot.render(int_plot.frame)


#def seegmentation_from_xyz(xyz_frame):

def publish_thread(args, cfg, int_plot, camera_pub_obs, camera_pub_time):
    load = args.load
    recording = args.record
    interactive = args.interactive
    publish = args.publish

    camera = Camera(cfg)
    camera.start()

    bg_filter = BackgroundFilter()
    sg_filter = SegmentationFilter()

    clip_dist = 8

    raw_frames = []

    i = 0

    ps_cfg = load_postprocess_settings(cfg)
    prev_rgb_frame = None

    max_ep_len = 125
    dt = 0.01667
    area_threshold = 50
    max_ep_time = max_ep_len * dt
    time_move_start = 1
    ep_ended = True
    ep_start_time = time.time()

    start_time = None

    try:
        with tqdm(total=None) as pbar:
            while int_plot.running and not rospy.is_shutdown():
                rgbd_frame = camera.get_rgbd_frame()
                if rgbd_frame is None:
                    continue

                if recording:
                    raw_frames.append(rgbd_frame.copy())

                if True:
                    rgb_frame = rgbd_frame[:3]
                    xyz_frame = get_xyz_image_from_depth_buffer(torch.tensor(-rgbd_frame[3]).float().cuda(), ps_cfg).cpu().numpy()
                    xyz_frame = (xyz_frame+1)/2
                    rgb_frame = process_image(rgb_frame, ps_cfg)
                    if prev_rgb_frame is None:
                        prev_rgb_frame = rgb_frame
                    diff_frame = ((rgb_frame - prev_rgb_frame).abs().mean(dim=0)/2 >= 0.05).float().numpy()

                    if args.red_dot:
                        rgb_frame[0, rgb_frame.shape[1]//2-3:rgb_frame.shape[1]//2+3, rgb_frame.shape[2]//2-3:rgb_frame.shape[2]//2+3] = 1
                        rgb_frame[1, rgb_frame.shape[1]//2-3:rgb_frame.shape[1]//2+3, rgb_frame.shape[2]//2-3:rgb_frame.shape[2]//2+3] = 0
                        rgb_frame[2, rgb_frame.shape[1]//2-3:rgb_frame.shape[1]//2+3, rgb_frame.shape[2]//2-3:rgb_frame.shape[2]//2+3] = 0
                        
                    #depth_frame = np.minimum(rgbd_frame[3:], clip_dist) / clip_dist
                    #filtered_frame = bg_filter.apply(rgb_frame)
                    
                    #segmentation_mask = sg_filter.apply(filtered_frame >= 0.001)
                    #segmentation_mask = segmentation_mask.astype(float)

                    #combined_frame = to_channels_last(cat_frames([rgb_frame, bw_to_rgb(depth_frame), bw_to_rgb(filtered_frame)]))
                    #combined_frame = to_channels_last(cat_frames([bw_to_rgb(xyz_frame[0:1]), bw_to_rgb(xyz_frame[1:2]), bw_to_rgb(xyz_frame[2:3])]))
                    if ep_ended:
                        combined_frame = to_channels_last(cat_frames([process_image(rgb_frame, ps_cfg), bw_to_rgb(diff_frame[None,:,:]), bw_to_rgb(xyz_frame[0:1]), bw_to_rgb(xyz_frame[1:2]), bw_to_rgb(xyz_frame[2:3])]))
                    else:
                        combined_frame = to_channels_last(cat_frames([process_image(rgb_frame, ps_cfg)*diff_frame[None,:,:], bw_to_rgb(diff_frame[None,:,:]), bw_to_rgb(xyz_frame[0:1]*diff_frame[None,:,:]), bw_to_rgb(xyz_frame[1:2]*diff_frame[None,:,:]), bw_to_rgb(xyz_frame[2:3])*diff_frame[None,:,:]]))
                        
                        # Plot the depth map

                    if publish:
                        array_msg = xyz_frame * diff_frame
                        prev_rgb_frame = rgb_frame

                        frame_changed = diff_frame.sum() > area_threshold
                        ep_time = time.time() - ep_start_time
                        if ep_time >= max_ep_time:
                            ep_ended = True
                        if ep_ended and ep_time >= max_ep_time + time_move_start + 0.5 and frame_changed:
                            ep_ended = False
                            ep_start_time = time.time()
                        pbar.set_description(f'ep_time={ep_time:.2f}\tep_ended={ep_ended} \tdiff={diff_frame.sum()}')
                        ep_time = ep_time if not ep_ended else -1

                        if camera_pub_obs is not None:
                            camera_pub_obs.publish(Float32MultiArray(data=array_msg.reshape(-1)))
                        if camera_pub_time is not None:
                            camera_pub_time.publish(Float32MultiArray(data=np.array([ep_time])))
                        #r.sleep()

                    if start_time is None:
                       start_time = time.time()

                    if interactive and i / (time.time() - start_time) >= 27:
                        int_plot.render(combined_frame)
                pbar.update(1)
                i += 1
    finally:
        if recording:
            i = 1
            while os.path.exists(f'recordings/raw_frames.{i}.npz'):
                i += 1
            np.savez_compressed(f'recordings/raw_frames.{i}.npz', np.array(raw_frames).astype(float))
            print('Recorded')
        #camera.stop()
        #int_plot.stop()


#thread = Thread(target=publish_thread, args=(args, cfg, int_plot, camera_pub_obs, camera_pub_time))
#thread.start()

#render_thread(int_plot)
publish_thread(args, cfg, int_plot, camera_pub_obs, camera_pub_time)