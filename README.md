Clustering using different non-parameteric models with the power of bert embeddings

The project includes the implementation of different non-parametric clustering models with the power of bert embeddings to identify groups of similar objects from textual data in python.

| Nonparametric models concept |
|---|
| Have parameters with infinite dimensional | 
| Having latent variables with finite raw data | 
| Having an infinite number of parameters | 
| Can be understood as having a random number of parameters | 
| Number of parameters can grow with the dataset | 

## Definition
model a collection of distributions (distribution over distribution)

- Nonparametric model: the parameters are from a possibly infinite dimensional space F (Θ ∈ F)

## Properties
1. CRP (Chinese Restaurant Process) defines a distribution over clusterings (i.e. partitions) of the indices 1,…,n

[In a simulated environment]
- Customer = index
- Table = cluster

When customer 1 enters, he can sit anywhere he likes. Customer 2 can sit in any empty seat, with the following probabilities:
```math
- Table 1 : 1 / (1 + α)
- New Table (i.e. any empty table) : α / (1 + α)
```


[In a clustering problem]

Let Nj be the number of data point in cluster j. For data point #(n+1), we have: 

```math
P(choose cluster # j) = Nj / α + N

P(choose a new cluster)= α / α+N
```

2. Expected number of clusters 

given n customers (i.e. observations) is O(α log(n))
- Rich-get-richer effect on clusters: popular tables tend to get more crowded.

The challenge of the model is as more people sit at a particular table, those tables increase in popularity, so new patrons are less likely to sit at empty tables.

3. Behavior of CRP with α:
```math
– As α goes to 0, the number of clusters goes to 1
– As α goes to +∞, the number of clusters goes to n
```

4. The CRP has known as an exchangeable process

5. If we shuffle the data points and get a new configuration, the probability of the different configurations is equal.


for more information:

Stanford University (2016). Chinese Restaurant Viewpoint. Retrieved February 13, 2018 from: https://cs.stanford.edu/~ppasupat/a9online/1083.html


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install embed-clustering.

```bash
pip install embed-clustering
```

## Usage

```python
# import the crp algorithm
from embed_clustering.latent_component import crp_algorithm

# read the data you want to cluster
import pandas as pd
df = pd.read_csv('sample.csv')

corpus = df[column] # mention the column you want to cluster

# apply the algorithm by passing the parameters
df['cluster'] = crp_algorithm(corpus, compute='cpu', cleaning=True) #if you have gpu, compute='cuda', if you doesn't wish to clean the text before clustering you can flag cleaning=False

```
## Evaluation
The performance of non-parametric crp-algorithm against centroid based algorithm by the number of K clusters (k is a predefined parameter). 

We implemented two known methods on a collection of arbitrary data to identify an optimum number of clusters an elbow method and a Silhouette score method as a measure to have the cohesion value between clusters. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters. Then we deployed our model on the same data with no predefining and tuning parameters, we found out that our non-parametric model derived most obtain clustering with structure or unstructured data.


## About
This algorithm is developed by Masume Azizyan & Deepak John Reji as part of their ongoing research on non-parametric models and word embeddings. If you use this work (code, model),

Please cite us and start at: https://github.com/dreji18/embed-clustering

## License
[MIT](https://choosealicense.com/licenses/mit/) License

Copyright (c) 2022 Masume Azizyan, Deepak John Reji
