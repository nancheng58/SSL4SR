# SMIN
# **SMIN**

Source code for ***Social Recommendation with Self-Supervised Metagraph Informax Network***

# Requirements

- Pytorch(1.5.0)
- DGL(0.4.3), installation-https://github.com/dmlc/dgl

# More Details

**data preprocessing**

- CiaoDVD

  > **rating.mat and trust.mat** as original data source from https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm
  >
  > **loadMat.py**: training/test data partition
  >
  > run `python ./dataset/CiaoDVD/loadMat.py` to perform preprocessing

  > **GenerateMetaPath.py**: metapath generation
  >
  > run `python ./dataset/CiaoDVD/GenrateMetaPath.py` to perform generation process

  > **GenerateSubGraph.py**: generate k-hop subfigures for Informax module
  >
  > run `python ./dataset/CiaoDVD/GenerateSubGraph.py` to perform k-hop subfigure construction

- Similar data preprocessing steps are applied in Epinions and Yelp data.

**Code running example**

Run main.py：

```
python main.py --dataset CiaoDVD --hide_dim 16 --layer_dim [16] --lr 0.05 --reg 0.05  --lambda1 0.06 --lambda2 0.002 
```

**Combination of sub-modules and code organization**

> `Interface`
> 
> BPRData.py: for generating the positive and negative instances corresponding to training and test set, respectively
>
> evaluate.py: perform evaluation of our proposed framework
>
> `MV_MIL` (Multi-view Graph-Structured Mutual Information Learning Paradigm)
> 
> informax.py: incorporate the learned social- and knowledge-aware dependence to guide the user-item interaction embedding process through deriving mutual information terms from different views.
>
> gcn.py and graphconv.py: the basic graph neural network architecture with the convolutional relation encoder
>
> `ToolScripts`
> 
> TimeLogger.py: log timestamp information
>
> tools.py: convert the sparse matrices to sparse tensors
>
> `model.py`
> 
> model class integrates the graph neural network architecture with high-order relation modeling
> SemanticAttention class defines the attention mechanism to aggregate metapath-specific representations
>
> `main.py`
> In the trainModel of Hope class, we adopt the model.py to optimize the loss of user-item interaction learing component.
>
> The joint learning component of i) meta-relation heterogeneity encoding and ii) multi-view graph-structured mutual information learning is defined informax.py. 

