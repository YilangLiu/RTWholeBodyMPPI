import mujoco
import mujoco_viewer
from control.gait_scheduler.scheduler import GaitScheduler
import os
from utils.tasks import get_task
import pdb
import numpy as np 
import time
import json
from utilities import pose3d
from utilities import motion_data
from utilities import motion_util
from pybullet_utils import transformations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAIT_DIR = os.path.join(BASE_DIR, "./control/gait_scheduler/gaits/")
GAIT_INPLACE_PATH = os.path.join(GAIT_DIR, "FAST/walking_gait_raibert_FAST_0_0_10cm_100hz.tsv")
GAIT_TROT_PATH = os.path.join(GAIT_DIR, "MED/walking_gait_raibert_MED_0_5_15cm_100hz.tsv")
GAIT_WALK_PATH = os.path.join(GAIT_DIR, "MED/walking_gait_raibert_MED_0_1_10cm_100hz.tsv")
GAIT_WALK_FAST_PATH = os.path.join(GAIT_DIR, "FAST/walking_gait_raibert_FAST_0_1_10cm_100hz.tsv")

REF_COORD_ROT = transformations.quaternion_from_euler(0.5 * np.pi, 0, 0)
REF_ROOT_ROT = transformations.quaternion_from_euler(0, 0, 0.47 * np.pi)

def main():
    curr_motion = motion_data.MotionData("motions/hopturn.txt")

    gaits = {
            'in_place': GaitScheduler(gait_path=GAIT_INPLACE_PATH, name='in_place'),
            'trot': GaitScheduler(gait_path=GAIT_TROT_PATH, name='trot'),
            'walk': GaitScheduler(gait_path=GAIT_WALK_PATH, name='walk'),
            'walk_fast': GaitScheduler(gait_path=GAIT_WALK_FAST_PATH, name='walk_fast')
        }
    task = "walk_straight"
    task_data = get_task(task)
    goal_pos = task_data['goal_pos']
    goal_ori = task_data['default_orientation'] 
    cmd_vel = task_data['cmd_vel']
    goal_thresh = task_data['goal_thresh']
    desired_gait = task_data['desired_gait']
    model_path = task_data['model_path'] 
    config_path = task_data['config_path']
    waiting_times = task_data['waiting_times']
    sim_path = task_data['sim_path']
    goal_index = 1
    dt = 0.01
    timeconst=0.02
    dampingratio=1.0
    model = mujoco.MjModel.from_xml_path(str(sim_path))
    model.opt.timestep = dt
    model.opt.enableflags = 1 # to override contact settings
    model.opt.o_solref = np.array([timeconst, dampingratio])
    
    gait_scheduler = gaits[desired_gait[goal_index]]
    data = mujoco.MjData(model)
    
    data.qpos = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    
    print(data.xmat[1])
    # viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
    # for i in np.arange(0, 10, 0.02):
    #     frame_i = curr_motion.calc_frame(i)
    #     # frame_i[2] -= 0.2
    #     # frame_i[[9, 12, 15, 18]] -= 1.0
    #     rot = transformations.quaternion_multiply(frame_i[3:7], np.array([-0.7071068, 0, 0, 0.7071068]))                           
    #     frame_i[3:7] = np.roll(rot,1)
    #     data.qpos = frame_i
    #     mujoco.mj_forward(model, data)
    #     viewer.render()
    #     time.sleep(0.1)

        
    # data.qpos = model.key_qpos[1]
    # data.qvel = model.key_qvel[1]
    # data.ctrl = model.key_ctrl[1]
    # viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
    # frames = []
    # for i in range(1000):
    #     joints_ref = gait_scheduler.gait[:, gait_scheduler.indices[0]]
    #     data.qpos[:] = joints_ref[:] # np.concatenate((goal_pos[1], goal_ori[0], joints_ref[:12]))
    #     mujoco.mj_forward(model, data)
    #     gait_scheduler.roll()
    #     viewer.render()
    #     time.sleep(0.01)

if __name__ == "__main__":
    main()