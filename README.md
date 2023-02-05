## WWW2023ConsRec


This is our implementation for the WWW 2023 full paper:

**ConsRec: Learning Consensus Behind Interactions for Group Recommendation**


### Datasets

We use two public experimental datasets: **Mafengwo** and **CAMRa2011**. 
These two datasets' contents are in the `data/` folder.


In this paper, we collect a new dataset named **Mafengwo-S** from [Mafengwo](https://www.mafengwo.cn/) to conduct the case study. 
Particular, in this dataset, we preserve each item's unique semantics, *i.e.,* its location name. It has 11,027 users, 1,236 items, and 1,215 groups.


We would release this dataset soon.


### Dependencies

* Python3
* PyTorch 1.9.1
* scipy 1.6.2


### Run

```
# For Mafengwo 

python main.py --dataset=Mafengwo --predictor=MLP --learning_rate=0.0001 --num_negatives=8 --layers=3 --epoch=200


# For CAMRa2011 

python main.py --dataset=CAMRa2011 --predictor=DOT --learning_rate=0.001 --num_negatives=2 --layers=2 --epoch=30
```
For more running options, please refer to `main.py`




### Cite 
 
If you make advantage of ConsRec in your research, please cite the following in your manuscript:

```
@inproceedings{wu2023consrec,
  title={ConsRec: Learning Consensus Behind Interactions for Group Recommendation},
  author={Wu, Xixi and Xiong, Yun and Zhang, Yao and Jiao, Yizhu and Zhang, Jiawei and Zhu, Yangyong and Philip S. Yu},
  booktitle={Proceedings of the ACM Web Conference 2023},
  year={2023},
  organization={ACM}
}
```
