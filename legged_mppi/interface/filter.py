import numpy as np 
from interface.a1_state_estimator import StateEstimator
import time 
import threading
import pdb
import select
import math
from multiprocessing import shared_memory

def euler_from_quaternion(w, x, y, z):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

class KalmanFilter(StateEstimator):
    def __init__(self, lc):
        StateEstimator.__init__(self, lc)
        # root pos and vel estimation
        self.state_size = 18
        self.meas_size = 28
        self.num_legs = 4
        self.Nx = 37
        
        # define noise 
        self.Process_Noise_PIMU = 0.01
        self.Process_Noise_VIMU = 0.01
        self.Process_Noise_PFOOT = 0.01
        self.Sensor_Noise_PIMU_rel_FOOT = 0.001
        self.Sensor_Noise_VIMU_rel_FOOT = 0.1
        self.Sensor_Noise_ZFOOT = 0.001
        self.gravity = np.array([0,0, -9.81])

        # define estimation variable 
        self.x = np.zeros((self.state_size, 1))
        self.xbar = np.zeros((self.state_size, 1))
        self.P = np.identity((self.state_size))
        self.Pbar = np.zeros((self.state_size, self.state_size))
        self.A = np.identity((self.state_size))
        self.B = np.zeros((self.state_size, 3))
        self.Q = np.identity((self.state_size))

        # define measured variables 
        self.y = np.zeros((self.meas_size, 1))
        self.yhat = np.zeros((self.meas_size, 1))
        self.error_y = np.zeros((self.meas_size, 1))
        self.Serror_y = np.zeros((self.meas_size, 1))
        self.C = np.zeros((self.meas_size, self.state_size))
        self.SC = np.zeros((self.meas_size, self.state_size))
        self.R = np.identity((self.meas_size))

        # other variables used in EKF 
        self.S = np.zeros((self.meas_size, self.meas_size))
        self.K = np.zeros((self.state_size, self.meas_size))
        self.calibrated = True

        # Fixed C value 
        for i in range(self.num_legs):
            self.C[i*3:i*3+3, :3] = -np.eye(3)
            self.C[i*3:i*3+3, 6+i*3:9+i*3] = np.eye(3)
            self.C[self.num_legs*3+i*3:self.num_legs*3+i*3+3, 3:6] = np.eye(3)
            self.C[self.num_legs*6+i, 6+i*3+2] = 1

        self.Q[:3, :3] = self.Process_Noise_PIMU * np.eye(3)
        self.Q[3:6, 3:6] = self.Process_Noise_VIMU * np.eye(3)
        for i in range(self.num_legs):
            self.Q[6+i*3:9+i*3, 6+i*3:9+i*3] = self.Process_Noise_PFOOT * np.eye(3)
            self.R[i*3: i*3+3, i*3: i*3+3] = self.Sensor_Noise_PIMU_rel_FOOT * np.eye(3)
            self.R[self.num_legs*3+i*3: self.num_legs*3+i*3 +3,  self.num_legs*3 + i*3:self.num_legs*3 + i*3 +3] = self.Sensor_Noise_VIMU_rel_FOOT * np.eye(3)
            self.R[self.num_legs*6 + i, self.num_legs*6 +i] = self.Sensor_Noise_ZFOOT

        self.is_inited = False
        self.init_msg_recv = False

    def initialize(self):
        self.P *= 3
        
        # initial state
        self.x[2] =  0.08
        foot_fk = self.get_foot_fk()
        for i in range(self.num_legs):
            self.x[6+i*3: 9+i*3] = np.dot(self.root_rot_mat, foot_fk[3*i:3*i+3])[:,np.newaxis] + self.x[:3]
        self.is_inited = True

    def skew(self, vec):
        return np.array([0, -vec[2], vec[1], 
                         vec[2], 0, -vec[0],
                         -vec[1], vec[0], 0]).reshape((3,3))

    def update(self, dt):
        # update A and B matrices
        self.A[:3, 3:6] = dt * np.eye(3)
        self.B[3:6,:3]   = dt * np.eye(3)
        
        u = self.root_rot_mat @ self.body_lin_acc  + self.gravity
        estimated_contacts = self.get_contact_state()

        # update Q
        self.Q[:3,:3] = self.Process_Noise_PIMU * dt / 20.0 * np.eye(3)
        self.Q[3:6, 3:6] = self.Process_Noise_VIMU * dt * 9.8 / 20.0 * np.eye(3)

        # update Q R for legs not in contact 
        for i in range(self.num_legs):
            self.Q[6+i*3: 9+i*3, 6+i*3: 9+i*3] = (1 + (1 - estimated_contacts[i]) * 1e3) * dt * self.Process_Noise_PFOOT * np.eye(3)
            
            self.R[i*3: i*3+3, i*3: i*3+3] = (1 + (1 - estimated_contacts[i]) * 1e3) * self.Sensor_Noise_PIMU_rel_FOOT * np.eye(3)
            self.R[self.num_legs*3+i*3:self.num_legs*3+i*3+3,self.num_legs*3+i*3:self.num_legs*3+i*3+3] = \
                    (1 + (1 - estimated_contacts[i]) * 1e3) * self.Sensor_Noise_VIMU_rel_FOOT * np.eye(3)
            
            self.R[self.num_legs*6 + i, self.num_legs *6+i] = (1 + (1 - estimated_contacts[i]) * 1e3) * self.Sensor_Noise_ZFOOT
        
        self.xbar[:] = self.A @ self.x + self.B @ u[:,np.newaxis]
        self.Pbar[:] = self.A @ self.P @ self.A.T + self.Q
        
        # measurement construction 
        self.yhat[:] = self.C @ self.xbar

        # posterior updates
        foot_fk = self.get_foot_fk()
        foot_vel_rel = self.get_foot_vel_rel()

        for i in range(self.num_legs):
            self.y[i*3:i*3+3]  = np.dot(self.root_rot_mat, foot_fk[i*3:i*3+3])[:,np.newaxis]
            foot_vel = -foot_vel_rel[i*3:i*3+3] - self.skew(self.body_ang_vel) @ foot_fk[i*3:i*3+3] 
            self.y[self.num_legs*3 + i*3 : self.num_legs*3 + i*3 + 3] = (1.0 - estimated_contacts[i]) * self.x[3:6] + (estimated_contacts[i]*self.root_rot_mat @ foot_vel)[:,np.newaxis]
            self.y[self.num_legs*6 + i] = ((1.0 - estimated_contacts[i])*(self.x[2]+foot_fk[3*i+2])) + estimated_contacts[i] * 0.0
        
        self.S[:] = self.C @ self.Pbar @ self.C.T + self.R
        self.S[:] = 0.5 * (self.S + self.S.T)

        self.error_y[:] = self.y - self.yhat
        self.Serror_y[:] = np.linalg.solve(self.S, self.error_y)

        self.x[:] = self.xbar + self.Pbar @ self.C.T @ self.Serror_y

        self.SC[:] = np.linalg.solve(self.S, self.C)
        self.P[:] = self.Pbar - self.Pbar @ self.C.T @ self.SC @ self.Pbar
        self.P[:] = 0.5 * (self.P + self.P.T)
    
    def poll(self, cb=None):
        t = time.time()
        try:
            self.initialize()
            while True:   
                timeout = 0.01 # recieve 500 Hz
                rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
                if rfds: 
                    # print("message received!")
                    dt = time.time() - t
                    self.lc.handle()
                    self.update(dt)
                    t = time.time()
                    
                    # print(f'Freq {1. / (time.time() - t)} Hz'); t = time.time()
                else:
                    # print(f'waiting for message... Freq {1. / (time.time() - t)} Hz'); t = time.time()
                    continue
                #    if cb is not None:
                #        cb()
        except KeyboardInterrupt:
            pass

    def spin(self):
        self.run_thread = threading.Thread(target=self.poll, daemon=False)
        self.run_thread.start()
        
        
    #     time.sleep(0.2)
    #     self.publish_state = threading.Thread(target=self.send_state, daemon=False)
    #     self.publish_state.start()

    # def send_state(self):
    #     while True:
    #         #publish at 1 kHZ rate
    #         root_pos = np.squeeze(self.x[:3])
    #         root_lin_vel = self.get_body_linear_vel()
    #         root_ang_vel = self.get_body_angular_vel()
    #         root_quat = self.get_body_quat()
    #         joint_pos = self.get_dof_pos()[self.joint_idxs]
    #         joint_vel = self.get_dof_vel()[self.joint_idxs]

    #         rpy_estimate = euler_from_quaternion(root_quat[0],root_quat[1],root_quat[2],root_quat[3])
    #         q_estimated = np.concatenate((root_pos,root_quat,joint_pos))
    #         qd_estimated = np.concatenate((root_lin_vel,root_ang_vel,joint_vel))
    #         self.state_shared[:] = np.concatenate([q_estimated, qd_estimated])
    #         time.sleep(0.001)
        