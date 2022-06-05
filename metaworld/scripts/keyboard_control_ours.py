"""
Use this script to control the env with your keyboard.
For this script to work, you need to have the PyGame window in focus.

See/modify `char_to_action` to set the key-to-action mapping.
"""
import sys

import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerButtonPressTopdownEnvV2



import time
import cv2
import os

# import pygame
# from pygame.locals import QUIT, KEYDOWN

# pygame.init()
# screen = pygame.display.set_mode((400, 300))


char_to_action = {
    'w': np.array([0, -1, 0, 0]),
    'a': np.array([1, 0, 0, 0]),
    's': np.array([0, 1, 0, 0]),
    'd': np.array([-1, 0, 0, 0]),
    'q': np.array([1, -1, 0, 0]),
    'e': np.array([-1, -1, 0, 0]),
    'z': np.array([1, 1, 0, 0]),
    'c': np.array([-1, 1, 0, 0]),
    'k': np.array([0, 0, 1, 0]),
    'j': np.array([0, 0, -1, 0]),
    'h': 'close',
    'l': 'open',
    'x': 'toggle',
    'r': 'reset',
    'p': 'put obj in hand',
}

import pygame



env = SawyerButtonPressTopdownEnvV2()
env._partially_observable = False
env._freeze_rand_vec = False
env._set_task_called = True
env.reset()
env._freeze_rand_vec = True
lock_action = False
random_action = False
obs = env.reset()
action = np.zeros(4)



def write_for_img_ver2(tag, img):
    if not os.path.exists('latentfusion_inputs/ver2'):
        os.mkdir('latentfusion_inputs/ver2')
    name = f'latentfusion_inputs/ver2/{tag}.png'
    return cv2.imwrite(name, img)

resolution = (640, 480)
camera = ['topview', 'corner1', 'corner2', 'corner3']
flip=True # if True, flips output image 180 degrees

config = [
    # env, action noise pct, cycles, quit on success
    ('button-press-topdown-v2', np.zeros(4), 3, True),
]


for camera in camera:
    if camera in ['corner1', 'corner2', 'corner3']:
        flip = False
    else:
        flip = True
    for env_name, noise, cycles, quit_on_success in config:  
        for i in range(cycles):
            a = env.action_space.sample()
            obs, reward, done, info = env.step(a)
            res=(640, 480)
            img, depth = env.sim.render(*res, mode='offscreen', camera_name=camera, depth = True)
            if flip: img = cv2.rotate(img, cv2.ROTATE_180); depth = cv2.rotate(depth, cv2.ROTATE_180)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            depth = (np.max(depth)-depth) / (np.max(depth) - np.min(depth))
            # depth = (depth-np.min(depth)) / (np.max(depth) - np.min(depth))
            depth = np.asarray(depth * 255, dtype=np.uint8)
            depth2 = np.expand_dims(depth, axis=2)
            rgbd = np.concatenate((img, depth2), axis=2)
            combined_img = np.tile(depth2, (1, 1, 3)) + img
            print(np.shape(combined_img))
            tag = env_name + '-noise-' + np.array2string(noise, precision=2, separator=',', suppress_small=True)\
            + '-cycles-'+ str(i) +'-camera-'+ camera  
            # RGB(480, 640, 3) img, what you wanted.
            write_for_img_ver2(tag, img)
            # Depth(480, 640, 1) img
            write_for_img_ver2(tag+'_depth', depth)
            # RGBD(480, 640, 4) img
            write_for_img_ver2(tag+'_rgbd', rgbd)
            # combied(480, 640, 3) img which add duplicated depth value into each rgb channel..
            write_for_img_ver2(tag+'_combined', combined_img)




# while True:
#     done = False
#     if not lock_action:
#         action[:3] = 0
#     if not random_action:
#         for event in pygame.event.get():
#             event_happened = True
#             if event.type == QUIT:
#                 sys.exit()
#             if event.type == KEYDOWN:
#                 char = event.dict['key']
#                 new_action = char_to_action.get(chr(char), None)
#                 if new_action == 'toggle':
#                     lock_action = not lock_action
#                 elif new_action == 'reset':
#                     done = True
#                 elif new_action == 'close':
#                     action[3] = 1
#                 elif new_action == 'open':
#                     action[3] = -1
#                 elif new_action is not None:
#                     action[:3] = new_action[:3]
#                 else:
#                     action = np.zeros(3)
#                 print(action)
#     else:
#         action = env.action_space.sample()
#     ob, reward, done, infos = env.step(action)
#     # time.sleep(1)
#     if done:
#         obs = env.reset()
#     env.render()
