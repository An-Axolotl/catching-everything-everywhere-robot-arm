#!/usr/bin/env python
import math
import os
import time
import sys
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, PoseArray, Point, Pose, Quaternion
from std_msgs.msg import String
import sys
import math
import os
import rospkg
from std_srvs.srv import Empty, SetBool
#path_src = rospkg.get_ros_package_path().split(':')[0]
#sys.path.insert(0, path_src + "/../projects/utils/")
from xarm.wrapper import XArmAPI
from tqdm import tqdm
import matplotlib.pyplot as plt
'''
This does inverse kinematics on the computer, sends the robot joint angles.  
Sending EE positions to the robot directly leads the SDK to erroring near singularities (joints move too fast) and I don't have a fix.  Allow Approximate motion did not work in mode 7.

There's joint limits in the robot.  The IK function is biased to stay away from the limits, but it doesn't know what your previous position was.

- By default, then it'll sometimes wrap around ~360 degrees.
- You can also make the joint angles continuous through time.  (jt_continuity)
However, you can hit joints limits, then we make sure the robot doesn't keep moving and cause weird spots.
We freeze all of the joint angles to the previous position once one is exceeded.

VR people use left handed coordinate systems, but robotics people use right handed usually.  See this to convert (not in this script because too slow but later)
http://www.techart3d.com/2016/02/convert-left-handed-to-right-handed-coordinates/
#######################################################
'''

class Telekinesis:
    def __init__(self):
        self.arm = XArmAPI("192.168.1.227", is_radian=True)
        self.arm.set_collision_sensitivity(2)
        #self.arm.set_allow_approx_motion(True)
        self.arm.motion_enable(enable=True)
        
        #self.arm.set_mode(0)
        #self.arm.set_state(0)
        #self.arm.reset(wait=True, speed=1)
        #time.sleep(1)

        self.arm.set_mode(1) #6
        self.arm.set_state(0)
        time.sleep(1)

        # http://help.ufactory.cc/en/articles/3954394-guide-to-run-ufactory-xarm-at-the-maximum-speed
        #self.arm.set_tcp_jerk(10000)
        #self.arm.set_joint_jerk(500, is_radian=True)
        #self.arm.save_conf ()
        #self.arm.reset(wait=False)

        self.speed=100000#2
        self.mvacc=50000#50

        self.sub_tracker = rospy.Subscriber("/arm/joint_angles", JointState, self._receive_joints)
        
        self.eef_pos_offset = np.array([0., 0., 0.])
        self.arm_joint_pub = rospy.Publisher("/arm/obs_joint_angles", JointState, queue_size=10)
        self.arm_eef_pose_pub = rospy.Publisher("/arm/obs_eef_pose", JointState, queue_size=10)
        self.jts_prev = None; 
        self.jts = None
        self.jt_min = np.deg2rad([-359, -117, -224, -359, -96, -359])
        self.jt_max = np.deg2rad([359, 119, 10, 359, 179, 359])
        self.safe = True
        self.is_reset = True
        self.prev_reset = not self.is_reset


        r = rospy.Rate(240)
        self.last_time = time.time()

        self.tgt_pos = np.array([ 0.0003, -0.9143, -0.6213,  3.2511,  0.7000, 0.0000])
        start_time = time.time()
        render_iter = 0
        iter = 0
        while self.safe:
            self.publish_arm_joint_angles()

            self.pos = self.arm.angles[:-1]
            error = np.abs(self.pos - self.tgt_pos).max()

            #code = self.arm.set_servo_angle(angle = self.tgt_pos, speed=self.speed, mvacc=self.mvacc, wait = False, is_radian = True, radius=-1) 
            # if self.is_reset:
            #     tgt_pos = np.array(self.tgt_pos) * 0.5 + np.array(self.pos) * 0.5
            #     code = self.arm.set_servo_angle_j(angles = tgt_pos, speed=10, wait = False, is_radian = True) 
            # else:
            code = self.arm.set_servo_angle_j(angles = self.tgt_pos, speed=self.speed, mvacc=self.mvacc, wait = False, is_radian = True) 
            if code != 0:
               self.safe = False

            # if not hasattr(self, 'ik_error_fig'):
            #     self.ik_times = [time.time()]
            #     self.ik_errors1 = [error*(-1 if self.is_reset else 1)]
            #     self.ik_error_fig, self.ik_error_ax = plt.subplots()
            #     self.ik_error_plot1,  = self.ik_error_ax.plot(np.array(self.ik_times) - self.ik_times[0], self.ik_errors1, label='arm tgt error')
            #     plt.legend()
            # else:
            #     self.ik_times.append(time.time())
            #     self.ik_times = self.ik_times[-500:]
            #     self.ik_errors1.append(error*(-1 if self.is_reset else 1))
            #     self.ik_errors1 = self.ik_errors1[-500:]

            #     if render_iter*15 <= iter:
            #         self.ik_error_plot1.set_data(np.array(self.ik_times) - self.ik_times[0], self.ik_errors1)
            #         #self.ik_error_ax.relim()
            #         self.ik_error_ax.set_ylim(-3, 3)
            #         self.ik_error_ax.set_xlim(0, 1000*r.sleep_dur.to_sec())
            #         self.ik_error_ax.autoscale_view()

            #         self.ik_error_fig.canvas.draw()
            #         plt.pause(0.0001)
            #         render_iter += 1

            r.sleep()
            iter += 1
            if iter%100 == 0:
                print(f'fps={100/(time.time() - start_time)}\terror={error:.2f}\treset={self.is_reset}')
                start_time = time.time()
        print(f'end loop: safe={self.safe}')

    def publish_arm_eef_pose(self):
        success, eef_pose = self.arm.get_forward_kinematics(self.arm.angles[:-1], input_is_radian=True, return_is_radian=True)
        assert(success == 0)

        eef_pose[0] /= 1000
        eef_pose[1] /= 1000
        eef_pose[2] /= 1000
        eef_pose[:3] -= self.eef_pos_offset
        
        msg = JointState()
        msg.position = np.array(eef_pose)
        self.arm_eef_pose_pub.publish(msg)
        
    def publish_arm_joint_angles(self):
        angles = self.arm.angles[:-1] # last joint is gripper angle
        msg = JointState()
        msg.position = np.array(angles)
        self.arm_joint_pub.publish(msg)

    def _receive_joints(self, data):
        self.jts_prev = self.jts
        data = np.array(data.position)
        self.jts = data[:-1]
        self.is_reset = data[-1] == 1
        #self.jt_continuity()  
        if not self.check_joint_limits():
           #print('invalid joints')
           return
        
        self.tgt_pos = self.jts.copy()

    def check_joint_limits(self):
        success, eef_xyz = self.arm.get_forward_kinematics(self.jts, input_is_radian=True, return_is_radian=True)
        assert(success == 0)
        eef_xyz = [eef_xyz[0] / 1000, eef_xyz[1] / 1000, eef_xyz[2] / 1000]
        eef_xyz = np.array(eef_xyz)
        clip_xyz = np.clip(eef_xyz, np.array([0.,-.4,.15]), np.array([.4,.4,.75]))

        if np.linalg.norm(clip_xyz - eef_xyz) > 0.01:
            print(f'invalid joints: eef = {eef_xyz}')
            return False
        return True
        
    def jt_continuity(self):      
        if self.jts_prev is None:
            return
        for i in range(0, 6):
            if (self.jts[i] - self.jts_prev[i]) > 5:
                self.jts[i] = self.jts[i] - 2*math.pi
            elif (self.jts[i] - self.jts_prev[i]) < -5:
                self.jts[i] = self.jts[i] + 2*math.pi
            if self.jts[i] < self.jt_min[i] or self.jts[i] > self.jt_max[i]:
                self.jts = self.jts_prev
                return


if __name__ == "__main__":
    rospy.init_node("telekinesis")
    telekinesis_node= Telekinesis()
