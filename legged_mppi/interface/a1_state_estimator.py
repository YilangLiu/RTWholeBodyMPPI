import math
import time

import numpy as np

from interface.lcm_types.leg_control_data_lcmt import leg_control_data_lcmt
from interface.lcm_types.rc_command_lcmt import rc_command_lcmt
from interface.lcm_types.state_estimator_lcmt import state_estimator_lcmt


def get_rpy_from_quaternion(q):
    w, x, y, z = q
    r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    p = np.arcsin(2 * (w * y - z * x))
    y = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([r, p, y])


def get_rotation_matrix_from_rpy(rpy):
    """
    Get rotation matrix from the given quaternion.
    Args:
        q (np.array[float[4]]): quaternion [w,x,y,z]
    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    r, p, y = rpy
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(r), -math.sin(r)],
                    [0, math.sin(r), math.cos(r)]
                    ])

    R_y = np.array([[math.cos(p), 0, math.sin(p)],
                    [0, 1, 0],
                    [-math.sin(p), 0, math.cos(p)]
                    ])

    R_z = np.array([[math.cos(y), -math.sin(y), 0],
                    [math.sin(y), math.cos(y), 0],
                    [0, 0, 1]
                    ])

    rot = np.dot(R_z, np.dot(R_y, R_x))
    return rot

COM_OFFSET = -np.array([0.012731, 0.002186, -0.0150515]) # -np.array([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = np.array([[0.183, -0.047, 0.], [0.183, 0.047, 0.],
                        [-0.183, -0.047, 0.], [-0.183, 0.047, 0.]
                        ]) + COM_OFFSET

class StateEstimator:
    def __init__(self, lc):

        # reverse legs
        self.joint_idxs = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.contact_idxs = [1, 0, 3, 2]

        # self.joint_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # self.contact_idxs = [0, 1, 2, 3]

        self.lc = lc

        self.joint_pos = np.array([0.073, 1.34, -2.83]*4) # np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.tau_est = np.zeros(12)
        self.world_lin_vel = np.zeros(3)
        self.world_ang_vel = np.zeros(3)
        self.foot_pos_rel = np.zeros(12)
        self.foot_vel_rel = np.zeros(12)
        self.euler = np.zeros(3)
        self.root_rot_mat = np.eye(3)
        self.buf_idx = 0

        self.smoothing_length = 12
        self.deuler_history = np.zeros((self.smoothing_length, 3))
        self.dt_history = np.ones((self.smoothing_length, 1)) * 0.01
        self.euler_prev = np.zeros(3)
        self.timuprev = time.time()

        self.body_lin_vel = np.zeros(3)
        self.body_ang_vel = np.zeros(3)
        self.smoothing_ratio = 0.2

        self.body_lin_acc = np.array([0,0,-9.8])
        self.foot_jac = np.zeros((12, 12))
        self.body_ang_acc = np.zeros(3)
        self.contact_state = np.ones(4)
        
        self.mode = 0
        self.ctrlmode_left = 0
        self.ctrlmode_right = 0
        self.left_stick = [0, 0]
        self.right_stick = [0, 0]
        self.left_upper_switch = 0
        self.left_lower_left_switch = 0
        self.left_lower_right_switch = 0
        self.right_upper_switch = 0
        # self.right_lower_left_switch = 0
        self.right_lower_right_switch = 0
        self.left_upper_switch_pressed = 0
        self.left_lower_left_switch_pressed = 0
        # self.left_lower_right_switch_pressed = 0
        self.right_upper_switch_pressed = 0
        # self.right_lower_left_switch_pressed = 0
        self.right_lower_right_switch_pressed = 0

        self.init_time = time.time()
        self.received_first_legdata = False

        self.imu_subscription = self.lc.subscribe("state_estimator_data", self._imu_cb)
        self.legdata_state_subscription = self.lc.subscribe("leg_control_data", self._legdata_cb)
        self.rc_command_subscription = self.lc.subscribe("rc_command", self._rc_command_cb)
        
        self.body_loc = np.array([0, 0, 0])
        self.body_quat = np.array([1, 0, 0, 0])

    def get_body_linear_vel(self):
        self.body_lin_vel = np.dot(self.root_rot_mat.T, self.world_lin_vel)
        return self.body_lin_vel

    def get_body_angular_vel(self):
        self.body_ang_vel = self.smoothing_ratio * np.mean(self.deuler_history / self.dt_history, axis=0) + (
                    1 - self.smoothing_ratio) * self.body_ang_vel
        return self.body_ang_vel

    def get_gravity_vector(self):
        grav = np.dot(self.root_rot_mat.T, np.array([0, 0, -1]))
        return grav

    def get_contact_state(self):
        return self.contact_state[self.contact_idxs]

    def get_rpy(self):
        return self.euler

    def get_command(self):
        # always in use
        cmd_x = 1 * self.left_stick[1]
        cmd_yaw = -1 * self.right_stick[0]

        # joystick commands
        cmd_y = 0.6 * self.left_stick[0]
        return np.array([cmd_x, cmd_y, cmd_yaw])

    def get_buttons(self):
        return np.array([self.left_lower_left_switch, self.left_upper_switch, self.right_lower_right_switch, self.right_upper_switch])

    def get_dof_pos(self):
        # print("dofposquery", self.joint_pos[self.joint_idxs])
        return self.joint_pos[self.joint_idxs]

    def get_dof_vel(self):
        return self.joint_vel[self.joint_idxs]

    def get_tau_est(self):
        return self.tau_est[self.joint_idxs]

    def get_yaw(self):
        return self.euler[2]

    def get_body_loc(self):
        return np.array(self.body_loc)

    def get_body_quat(self):
        return np.array(self.body_quat)

    def _legdata_cb(self, channel, data):
        # print("update legdata")
        if not self.received_first_legdata:
            self.received_first_legdata = True
            print(f"First legdata: {time.time() - self.init_time}")

        msg = leg_control_data_lcmt.decode(data)
        self.joint_pos = np.array(msg.q)
        self.joint_vel = np.array(msg.qd)
        self.tau_est = np.array(msg.tau_est)
        # print(f"update legdata {msg.id}")

    def _imu_cb(self, channel, data):
        # print("update imu")
        msg = state_estimator_lcmt.decode(data)
        self.euler = np.array(msg.rpy)

        self.root_rot_mat[:] = get_rotation_matrix_from_rpy(self.euler)
        
        ### Important!!!!!
        self.contact_state = 1.0 * (np.array(np.abs(msg.contact_estimate)) > 0.0)

        self.deuler_history[self.buf_idx % self.smoothing_length, :] = msg.rpy - self.euler_prev
        self.dt_history[self.buf_idx % self.smoothing_length] = time.time() - self.timuprev

        self.timuprev = time.time()

        self.buf_idx += 1
        self.euler_prev = np.array(msg.rpy)

        self.body_quat = np.array(msg.quat)
        self.body_lin_acc = np.array(msg.aBody)
        self.body_ang_acc = np.array(msg.omegaBody)

    def _sensor_cb(self, channel, data):
        pass

    def _rc_command_cb(self, channel, data):
        msg = rc_command_lcmt.decode(data)

        self.left_upper_switch_pressed = ((msg.left_upper_switch and not self.left_upper_switch) or self.left_upper_switch_pressed)
        self.left_lower_left_switch_pressed = ((msg.left_lower_left_switch and not self.left_lower_left_switch) or self.left_lower_left_switch_pressed)
        # self.left_lower_right_switch_pressed = ((msg.left_lower_right_switch and not self.left_lower_right_switch) or self.left_lower_right_switch_pressed)
        self.right_upper_switch_pressed = ((msg.right_upper_switch and not self.right_upper_switch) or self.right_upper_switch_pressed)
        # self.right_lower_left_switch_pressed = ((msg.right_lower_left_switch and not self.right_lower_left_switch) or self.right_lower_left_switch_pressed)
        self.right_lower_right_switch_pressed = ((msg.right_lower_right_switch and not self.right_lower_right_switch) or self.right_lower_right_switch_pressed)

        self.mode = msg.mode
        self.right_stick = msg.right_stick
        self.left_stick = msg.left_stick
        self.left_upper_switch = msg.left_upper_switch
        self.left_lower_left_switch = msg.left_lower_left_switch
        self.left_lower_right_switch = msg.left_lower_right_switch
        self.right_upper_switch = msg.right_upper_switch
        # self.right_lower_left_switch = msg.right_lower_left_switch
        self.right_lower_right_switch = msg.right_lower_right_switch
        # print(self.right_stick, self.left_stick)

    def get_foot_fk(self):
        # joint_names = [
            # "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            # "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            # "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            # "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", ]

        joint_angles = self.get_dof_pos()
        for i in range(4):
            self.foot_pos_rel[i*3:i*3+3] = self.foot_position_in_hip_frame(joint_angles[i*3:i*3+3],
                                                        l_hip_sign=(-1)**(i + 1)) + HIP_OFFSETS[i]
        return self.foot_pos_rel 
    
    def get_foot_jacobian(self):
        joint_angles = self.get_dof_pos()
        for i in range(4):
            self.foot_jac[i*3:i*3+3, i*3:i*3+3] = self.analytical_leg_jacobian(joint_angles[i*3: i*3+3], i)
        return self.foot_jac
    
    def get_foot_vel_rel(self):
        foot_jac = self.get_foot_jacobian()
        joint_vel = self.get_dof_vel()
        for i in range(4):
            self.foot_vel_rel[i*3:i*3+3] = np.dot(foot_jac[i*3:i*3+3, i*3:i*3+3], joint_vel[i*3: i*3+3])
        return self.foot_vel_rel

    def foot_position_in_hip_frame(self, angles, l_hip_sign=1):
        theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * l_hip_sign
        leg_distance = np.sqrt(l_up**2 + l_low**2 +
                                2 * l_up * l_low * np.cos(theta_knee))
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * np.sin(eff_swing)
        off_z_hip = -leg_distance * np.cos(eff_swing)
        off_y_hip = l_hip

        off_x = off_x_hip
        off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
        off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
        return np.array([off_x, off_y, off_z])

    def analytical_leg_jacobian(self, leg_angles, leg_id):
        """
        Computes the analytical Jacobian.
        Args:
        ` leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
            l_hip_sign: whether it's a left (1) or right(-1) leg.
        """
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * (-1)**(leg_id + 1)

        t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]
        l_eff = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(t3))
        t_eff = t2 + t3 / 2
        J = np.zeros((3, 3))
        J[0, 0] = 0
        J[0, 1] = -l_eff * np.cos(t_eff)
        J[0, 2] = l_low * l_up * np.sin(t3) * np.sin(t_eff) / l_eff - l_eff * np.cos(
            t_eff) / 2
        J[1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
        J[1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
        J[1, 2] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(
            t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
        J[2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
        J[2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
        J[2, 2] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(
            t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
        return J

    def close(self):
        self.lc.unsubscribe(self.legdata_state_subscription)