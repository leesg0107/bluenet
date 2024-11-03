import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gym import spaces
from pettingzoo.sisl import waterworld_v4

def make_env(render_mode=None):
    env = waterworld_v4.env(
        n_pursuers=5, n_evaders=5, n_poisons=10, n_coop=2, n_sensors=20,
        sensor_range=0.2, radius=0.015, obstacle_radius=0.2, n_obstacles=1,
        obstacle_coord=[(0.5, 0.5)],  # 수정된 부분: 리스트로 변경
        pursuer_max_accel=0.01, evader_speed=0.01,
        poison_speed=0.01, poison_reward=-1.0, food_reward=10.0, encounter_reward=0.01,
        thrust_penalty=-0.5, local_ratio=1.0, speed_features=True, max_cycles=500,
        render_mode=render_mode
    )
    return env

