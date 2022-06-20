# configuration file to claim the data we used
# author: yk
# date: May,2022
import os

ours_categlory_pair = {
    'rod' : 'Assembly_Peg'
}

our_categories = ['rod']
our_categories = [ours_categlory_pair[cat] for cat in our_categories]

ours_path = './datasets/Ours'
ours_normalized_path= './datasets/Ours_normalized'
ours_rendering_path = './datasets/OursRenderings'
camera_setting_path = './datasets/camera_settings'
model_view_path = './model_view_metadata/result.pkl'
watertight_mesh_path = './datasets/Ours_watertight'

shape_scale_padding = 0
total_view_nums = 5
