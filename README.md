**Status:** Maintenance and development (expect bug fixes and minor updates)


# Sensory_extended_metaworld
This project was forked from original metaworld repository. [Metaworld](https://github.com/rlworkgroup/metaworld) is an open-source simulated benchmark for meta-reinforcement learning and multi-task learning consisting of 50 distinct robotic manipulation tasks. We added some functions for sensory input to  metaworld.


![image](https://user-images.githubusercontent.com/51065570/172418379-cf88c010-ef26-4d45-ad34-639f48969184.png)

There exists a need to make a metaworld benchmark extension which get sensor data, not a absolute pose metric.
Therefore, we suggest to use 6-dof pose estimation model like "[latentfusion](https://keunhong.com/publications/latentfusion/)" on pose estimation process. 
We added RGB, and depth image pipeline into metaworld benchmark.

![image](https://user-images.githubusercontent.com/51065570/172418425-a32c2197-6650-4e6f-9c5e-d563b0f9d3ee.png)

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

## Training


## Changelog

- 06/06/2022: We wrote Readme.md.

## Credits

'sensory-extended-metaworld' is maintained by the Yonsei MLP5 project team. Contributors include:
 - Youngkwang Nam
 - Sungwon Seo
 - Pyunghwa Shin
 - Changhyun Choi

## Followings are Citation, and aAknowledgement

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
