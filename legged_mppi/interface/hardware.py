import mujoco
import os
import mujoco_viewer
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import tqdm
from PIL import Image
import time
import lcm
from interface.filter import KalmanFilter
from interface.lcm_types import pd_tau_targets_lcmt
import threading
from pynput import keyboard

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

class Hardware:
    """
    A class representing a simulator for controlling and estimating the state of a system.
    
    Attributes:
        filter (object): The filter used for state estimation.
        agent (object): The agent used for control.
        model_path (str): The path to the XML model file.
        T (int): The number of time steps.
        dt (float): The time step size.
        viewer (bool): Flag indicating whether to enable the viewer.
        gravity (bool): Flag indicating whether to enable gravity.
        model (object): The MuJoCo model.
        data (object): The MuJoCo data.
        qpos (ndarray): The position trajectory.
        qvel (ndarray): The velocity trajectory.
        finite_diff_qvel (ndarray): The finite difference of velocity.
        ctrl (ndarray): The control trajectory.
        sensordata (ndarray): The sensor data trajectory.
        noisy_sensordata (ndarray): The noisy sensor data trajectory.
        time (ndarray): The time trajectory.
        state_estimate (ndarray): The estimated state trajectory.
        viewer (object): The MuJoCo viewer.
    """
    def __init__(self, filter=None, agent=None,
                 model_path = os.path.join(os.path.dirname(__file__), "../models/go1/task_simulate.xml"),
                T = 200, dt = 0.01, viewer = False, gravity = True,
                # stiff=False
                timeconst=0.02, dampingratio=1.0, ctrl_rate=100,
                save_dir="./frames", save_frames=False
                ):
        # filter
        self.filter = filter
        self.agent = agent
        self.dt = dt
        self.ctrl_rate = ctrl_rate
        self.update_ratio = max(1, 1/(dt*ctrl_rate))
        self.interpolate_cam = False
        # model
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = dt
        self.model.opt.enableflags = 1 # to override contact settings
        self.model.opt.o_solref = np.array([timeconst, dampingratio])
        # data
        self.data = mujoco.MjData(self.model)
        self.filter = KalmanFilter(lc)

        mujoco.mj_resetData(self.model, self.data)
        self.joint_idxs = self.filter.joint_idxs
        self.data.qpos = self.model.key_qpos[1]
        self.data.qvel = self.model.key_qvel[1]
        self.data.ctrl = self.model.key_ctrl[1]
        self.kp = 25.0
        self.kd = 0.2
        self.gravity_compensation = np.array([0.8, 0, 0, -0.8, 0, 0, 0.8, 0, 0, -0.8, 0, 0])
        self.real_kp = self.kp * np.ones(12)
        self.real_kd = self.kd * np.ones(12)
        self.low_cmd_pub_dt = 0.002
        self.q_estimated = np.zeros((self.model.nq + self.model.nv))
        # self.action_pub = self.model.keyframe("sit").qpos[7:]
        self.home_pose = self.model.keyframe("home").qpos[7:]
        self.standup_pose = self.model.keyframe("stand").qpos[7:]
        self.sit_pose = self.model.keyframe("sit").qpos[7:]
        # self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, hide_menus=True)
        self.ctrl_stop = False
        self.hardware_state_log = []

    def publish_action(self, action):
        command_for_robot = pd_tau_targets_lcmt.pd_tau_targets_lcmt()
        command_for_robot.q_des =  action # self.action_pub 
        command_for_robot.qd_des = np.zeros(12)
        command_for_robot.kp = self.kp * np.ones(12)
        command_for_robot.kd = self.kd * np.ones(12)
        command_for_robot.tau_ff =  np.zeros(12)
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
        command_for_robot.id = 0
        lc.publish("pd_plustau_targets", command_for_robot.encode())

    def record(self):
        while True:
            root_pos = np.squeeze(self.filter.x[:3])
            root_quat = self.filter.get_body_quat()
            joint_pos = self.filter.get_dof_pos()[self.joint_idxs]
            joint_vel = self.filter.get_dof_vel()[self.joint_idxs]
            root_lin_vel = self.filter.get_body_linear_vel()
            root_ang_vel = self.filter.get_body_angular_vel()
            q_estimated = np.concatenate((root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel))
            self.hardware_state_log.append(q_estimated)
            time.sleep(0.05)

    def calibrate(self, action):
        increment = 300
        joint_pos = self.filter.get_dof_pos()[self.joint_idxs]
        joint_increment =  (action - joint_pos) / increment
        for i in range(increment):
            self.action_pub = (joint_pos + i * joint_increment)
            self.publish_action(self.action_pub)
            time.sleep(0.005)

    def on_press(self, key):
        try:
            if key.char=="p":
                self.ctrl_stop = True
        except:
            pass
        
    def run(self):
        self.filter.initialize()
        self.filter.spin()

        time.sleep(1)

        record_thread = threading.Thread(target=self.record, daemon=False)
        record_thread.start()

        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        # input("stand?")
        # self.calibrate(self.home_pose)
        self.calibrate(self.standup_pose)
        
        # while not self.ctrl_stop:
        #     print("joint pos: ", self.filter.get_dof_pos()[self.joint_idxs])
        #     time.sleep(0.05)
        
        input("start?")

        while True:
            now = time.time()
            root_pos = np.squeeze(self.filter.x[:3])
            root_quat = self.filter.get_body_quat()
            joint_pos = self.filter.get_dof_pos()[self.joint_idxs]
            joint_vel = self.filter.get_dof_vel()[self.joint_idxs]
            root_lin_vel = self.filter.get_body_linear_vel()
            root_ang_vel = self.filter.get_body_angular_vel()
            self.q_estimated[:] = np.concatenate((root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel))
            
            # self.data.qpos[:] = self.q_estimated[:19]
            # mujoco.mj_forward(self.model, self.data)
            # self.viewer.render()

            action_plan = self.agent.update(self.q_estimated)[:]
            action_plan = self.agent.update(self.q_estimated)[:]

            self.publish_action(action_plan)

            # self.action_pub[:] = action_plan[self.joint_idxs]
            # error = np.linalg.norm(np.array(self.agent.body_ref[:3]) - np.array(self.data.qpos[:3]))
            # if error < self.agent.goal_thresh[self.agent.goal_index]:
            #     self.agent.next_goal()

            if self.ctrl_stop:
                self.calibrate(self.sit_pose)
                hardware_state_log = np.array(self.hardware_state_log)
                np.savetxt("hardware_state_log.txt", hardware_state_log, delimiter=',')
                break

            # time.sleep(0.01)

            # duration = time.time() - now
            # if duration < self.dt:
            #     time.sleep((self.dt - duration))
            # else:
            #     continue