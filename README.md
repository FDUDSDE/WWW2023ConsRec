# WWW2023ConsRec


This is official code for the WWW 2023 full paper [[arXiv](https://arxiv.org/abs/2302.03555)]:

**ConsRec: Learning Consensus Behind Interactions for Group Recommendation**

![](./figs/Overview.pdf)

In this paper, we focus on exploring consensus behind group behavior data. To comprehensively capture the group consensus, we innovatively design three distinct views, including member-level aggregation, item-level tastes, and group-level inherent preferences.
Particularly, in the member-level viw, different from existing attentive strategy, we design a novel hypergraph neural network that allows for efficient hypergraph convolutional operations to generate expressive member-level aggregation. 

## Datasets

We use two public experimental datasets: **Mafengwo** and **CAMRa2011**. 
These two datasets' contents are in the `data/` folder.


In this paper, we collect a new dataset named **Mafengwo-S** from [Mafengwo](https://www.mafengwo.cn/) to conduct the case study. 
Particular, in this dataset, we preserve each item's unique semantics, *i.e.,* its location name. It has 11,027 users, 1,236 items, and 1,215 groups.


We would release this dataset soon.

Besides, we release our implementations of group recommendation baselines [here](https://github.com/FDUDSDE/WWW2023GroupRecBaselines).


## Dependencies

* Python3
* PyTorch 1.9.1
* scipy 1.6.2


## Run

```
# For Mafengwo 

python main.py --dataset=Mafengwo --predictor=MLP --learning_rate=0.0001 --num_negatives=8 --layers=3 --epoch=200


# For CAMRa2011 

python main.py --dataset=CAMRa2011 --predictor=DOT --learning_rate=0.001 --num_negatives=2 --layers=2 --epoch=30
```
For more running options, please refer to `main.py`




## Cite 
 
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
