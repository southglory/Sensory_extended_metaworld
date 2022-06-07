**Status:** Maintenance (expect bug fixes and minor updates)


# Sensory_extended_metaworld
This project was forked from original metaworld repository. [Metaworld](https://github.com/rlworkgroup/metaworld) is an open-source simulated benchmark for meta-reinforcement learning and multi-task learning consisting of 50 distinct robotic manipulation tasks. We added some functions for sensory input to  metaworld.


![image](https://user-images.githubusercontent.com/51065570/172418379-cf88c010-ef26-4d45-ad34-639f48969184.png)



![image](https://user-images.githubusercontent.com/51065570/172418425-a32c2197-6650-4e6f-9c5e-d563b0f9d3ee.png)



UserWarning: WARN: Box bound precision lowered by casting to float32
  logger.warn(f"Box bound precision lowered by casting to {self.dtype}") 에러 제거를 위해서,
아래 방식으로 설치.
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html



## Changelog

- 06/06/2022: We wrote Readme.md.

## Credits

'sensory-extended-metaworld' is maintained by the Yonsei MLP5 project team. Contributors include:
 - Youngkwang Nam
 - Sungwon Seo
 - Pyunghwa Shin
 - Changhyun Choi

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
