import mujoco
import mujoco_viewer
import time
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path('models/unitree_a1/task_flat_collision_torque.xml')
data = mujoco.MjData(model)
sit_pos = model.keyframe("rest").qpos
home_pos = model.keyframe("home").qpos
data.qpos = sit_pos
kp = 30
kd = 1
viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
gravity_compensation = np.array([0.8, 0, 0, -0.8, 0, 0, 0.8, 0, 0, -0.8, 0, 0])

state_log = np.loadtxt('hardware_state_log.txt', delimiter=',')
# ctrl_log = np.loadtxt('hardware_ctrl_log.txt', delimiter=',')
# plt.plot(np.arange(state_log.shape[0]), ctrl_log[:,0])
# plt.show()
for i in tqdm(range(150, state_log.shape[0])):
    # data.ctrl = kp*(home_pos[7:] - data.qpos[7:]) - kd * data.qvel[6:] + gravity_compensation
    data.qpos = state_log[i,:19]
    mujoco.mj_forward(model, data)
    viewer.render()
    time.sleep(0.08)