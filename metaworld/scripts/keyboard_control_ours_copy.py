import gym
gym.logger.set_level(40)

import os
import functools

import cv2
import numpy as np

os.system('LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia/libGL.so')
os.system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia')
os.system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-510')
os.system('export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so')


def write_for_img_ver2(tag, img):
    if not os.path.exists('scripts/latentfusion_inputs/ver2'):
        os.mkdir('scripts/latentfusion_inputs/ver2')
    name = f'scripts/latentfusion_inputs/ver2/{tag}.png'
    return cv2.imwrite(name, img)


import metaworld
import random

ml1 = metaworld.ML1('button-press-topdown-v2')

env = ml1.train_classes['button-press-topdown-v2']()
task = random.choice(ml1.train_tasks)
env.set_task(task)

obs = env.reset()

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
            img = env.sim.render(*res, mode='window', camera_name=camera, depth = True)
            if flip: img = cv2.rotate(img, cv2.ROTATE_180); 
            # depth = cv2.rotate(depth, cv2.ROTATE_180)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # depth = (np.max(depth)-depth) / (np.max(depth) - np.min(depth))
            # # depth = (depth-np.min(depth)) / (np.max(depth) - np.min(depth))
            # depth = np.asarray(depth * 255, dtype=np.uint8)
            # depth2 = np.expand_dims(depth, axis=2)
            # rgbd = np.concatenate((img, depth2), axis=2)
            # combined_img = np.tile(depth2, (1, 1, 3)) + img
            # print(np.shape(combined_img))
            tag = env_name + '-noise-' + np.array2string(noise, precision=2, separator=',', suppress_small=True)\
            + '-cycles-'+ str(i) +'-camera-'+ camera  
            # RGB(480, 640, 3) img, what you wanted.
            write_for_img_ver2(tag, img)
            # # Depth(480, 640, 1) img
            # write_for_img_ver2(tag+'_depth', depth)
            # # RGBD(480, 640, 4) img
            # write_for_img_ver2(tag+'_rgbd', rgbd)
            # # combied(480, 640, 3) img which add duplicated depth value into each rgb channel..
            # write_for_img_ver2(tag+'_combined', combined_img)