**Status:** Maintenance and development (expect bug fixes and minor updates)


# Sensory_extended_metaworld
This project was forked from original metaworld repository. [Metaworld](https://github.com/rlworkgroup/metaworld) is an open-source simulated benchmark for meta-reinforcement learning and multi-task learning consisting of 50 distinct robotic manipulation tasks. We added some functions for sensory input to  metaworld.


![image](https://user-images.githubusercontent.com/51065570/172418379-cf88c010-ef26-4d45-ad34-639f48969184.png)

There exists a need to make a metaworld benchmark extension which gets sensor data, not a absolute pose metric.
Therefore, we suggest to use 6-dof pose estimation model like "[latentfusion](https://keunhong.com/publications/latentfusion/)" on pose estimation process. 

![image](https://user-images.githubusercontent.com/51065570/172429197-12671c47-bb38-41d9-8deb-e3f75e434fb2.png)

We added RGB, and depth image pipeline into metaworld benchmark.
Since we are fixing bugs and problems at latentfusion modifying codes, as a temporary method, we suggest a CNN model based reinforcement learning using RGB, and depth images which come from metaworld task virtual multi-view cameras.

![image](https://user-images.githubusercontent.com/51065570/172418425-a32c2197-6650-4e6f-9c5e-d563b0f9d3ee.png)

As an ideal solution, we suggest attaching a pose estimation model into meta-world benchmark which now only supports absolutely known pose metric inputs.

After entire development, many present meta-learning methods testing on meta-world benchmark could be applied at actual real world reinforcement learning problems.


![image](https://user-images.githubusercontent.com/51065570/172428831-5e05c606-2028-497c-a87a-4c1a603d333f.png)


## Installation

Use [anaconda](https://www.anaconda.com/) virtual environment to run each sub-repository.

```
cd metaworld
installation with Readme.MD
```

```
cd latentfusion
installation with Readme.MD
```

```
cd depth_renderer
installation with Readme.MD
```

```
cd MLP5_CNN
installation with Readme.MD
```

```
cd Mask
installation with Readme.MD
```
To run agent in our extension, you must install requirements that [original MetaWorld repository](https://github.com/rlworkgroup/metaworld) needs.

## depth_renderer
We have two choices: 
1. Get depth image from metaworld directly.
2. Get depth image from depth-renderer library.

We recommend to use direct method using metaworld with mujoco.

As a reference, we also added here another method using [depth-renderer](https://github.com/yinyunie/depth_renderer) library.


## Mask making algorithm
Mask making algorithm is based on the color segmentation by using the charateristic of input images, respectively. 
- File List
 folder: color_segmentation, input_image, mask_image
 - input_image : put the original image (png format) that needs to be mask
 - color_segmentation : image will be saved automatically by segmenting with color that users selected
 - mask_image : image will be save automatically by masking with segmented color

*********************** Run **************************

There are two ways to run the "Mask making algorithm". 
ImageMask.py makes user to select the color of object that needs to be maksed.  
ImageMask_Auto.py makes the mask of image by selecting the red color of robot arm automatically.

- User Control:
  (In the command window, run python ImageMask.py)
   python ImageMask.py

   1. write the name of image file placed in input_image
   2. click the color you wanted to segment the image
   3. check whether it works desirly
   4. press "esc" button to go to the next image
   5. if you want to finish to segmenting the color of image, press esc and write "break"

- Auto:
  (In the command window, run python ImageMask_Auto.py)
  python ImageMask_Auto.py

  put all the image required to be masked in input_image folder

*** you can check the result at mask_image folder



## MLP5_CNN algorithm & Training

![image](https://user-images.githubusercontent.com/51065570/172429384-946a8a6a-965c-4cee-9286-478f9b806aa2.png)
![actor](https://user-images.githubusercontent.com/62916482/174601953-c72d8cb9-1966-4618-b8d5-3191de6004f6.png)
![critic](https://user-images.githubusercontent.com/62916482/174601984-d38d0ddf-62b9-4607-9a57-31520874dd11.png)

This PPO agent does not use original MetaWorld states. It uses images with masking algorithm that we wrote above as state.

To run PPO agent, follow:
```
cd metaworld
python ppo_learn.py
```
 * You can change agent's variables and environments in [the code](https://github.com/southglory/Sensory_extended_metaworld/blob/main/metaworld/ppo_learn.py).
 * cf) If you see "RuntimeError: failed to initialize OpenGL", then follow:
 ```
 unset LD_PRELOAD
 ```


## Changelog

- 06/06/2022: We wrote Readme.md.

## Credits

'sensory-extended-metaworld' is maintained by the Yonsei MLP5 project team. Contributors include:
 - Youngkwang Nam
 - Sungwon Seo
 - Pyunghwa Shin
 - Changhyun Choi

## Followings are Citation, and Acknowledgement

## Citing Meta-World
If you use Meta-World for academic research, please kindly cite our CoRL 2019 paper the using following BibTeX entry.

```
@inproceedings{yu2019meta,
  title={Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning},
  author={Tianhe Yu and Deirdre Quillen and Zhanpeng He and Ryan Julian and Karol Hausman and Chelsea Finn and Sergey Levine},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2019}
  eprint={1910.10897},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
  url={https://arxiv.org/abs/1910.10897}
}
```

## Acknowledgements
Meta-World is a work by [Tianhe Yu (Stanford University)](https://cs.stanford.edu/~tianheyu/), [Deirdre Quillen (UC Berkeley)](https://scholar.google.com/citations?user=eDQsOFMAAAAJ&hl=en), [Zhanpeng He (Columbia University)](https://zhanpenghe.github.io), [Ryan Julian (University of Southern California)](https://ryanjulian.me), [Karol Hausman (Google AI)](https://karolhausman.github.io),  [Chelsea Finn (Stanford University)](https://ai.stanford.edu/~cbfinn/) and [Sergey Levine (UC Berkeley)](https://people.eecs.berkeley.edu/~svlevine/).

The code for Meta-World was originally based on [multiworld](https://github.com/vitchyr/multiworld), which is developed by [Vitchyr H. Pong](https://people.eecs.berkeley.edu/~vitchyr/), [Murtaza Dalal](https://github.com/mdalal2020), [Ashvin Nair](http://ashvin.me/), [Shikhar Bahl](https://shikharbahl.github.io), [Steven Lin](https://github.com/stevenlin1111), [Soroush Nasiriany](http://snasiriany.me/), [Kristian Hartikainen](https://hartikainen.github.io/) and [Coline Devin](https://github.com/cdevin). The Meta-World authors are grateful for their efforts on providing such a great framework as a foundation of our work. We also would like to thank Russell Mendonca for his work on reward functions for some of the environments.

## Citing LatentFusion
If you find the LatentFusion code or data useful, please consider citing:

```bibtex
@inproceedings{park2019latentfusion,
  title={LatentFusion: End-to-End Differentiable Reconstruction and Rendering for Unseen Object Pose Estimation},
  author={Park, Keunhong and Mousavian, Arsalan and Xiang, Yu and Fox, Dieter},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Citing depth_renderer
This library is used for data preprocessing in our work SK-PCN. If you find it helpful, please consider citing

@inproceedings{NEURIPS2020_ba036d22,
 author = {Nie, Yinyu and Lin, Yiqun and Han, Xiaoguang and Guo, Shihui and Chang, Jian and Cui, Shuguang and Zhang, Jian.J},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {16119--16130},
 publisher = {Curran Associates, Inc.},
 title = {Skeleton-bridged Point Completion: From Global Inference to Local Adjustment},
 url = {https://proceedings.neurips.cc/paper/2020/file/ba036d228858d76fb89189853a5503bd-Paper.pdf},
 volume = {33},
 year = {2020}
}
