---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Implementing an Autoencoder

In this exercise, we would like to train on a popular face dataset, a sparse auto-encoder. We consider the simple two-layer autoencoder network:

\begin{align*}
\boldsymbol{z}_i &= \max(0,V \boldsymbol{x}_i + \boldsymbol{b}) & \text{(layer 1)}\\
\hat{\boldsymbol{x}}_i &= W \boldsymbol{z}_i + \boldsymbol{a} & \text{(layer 2)}
\end{align*}

where $W,V$ are matrices of parameters of the encoder and the decoder, and $\boldsymbol{b},\boldsymbol{a}$ are additional bias parameters. We seek to maximize the objective:

\begin{align*}
\min_{W} ~~
\underbrace{ \frac1N \sum_{i=1}^N \| \boldsymbol{x}_i - \hat{\boldsymbol{x}}_i \|^2}_{\text{reconstruction}}
+ \lambda \cdot
\underbrace{\frac1N \sum_{i=1}^N \|\boldsymbol{z}_i\|_1}_{\text{sparsity}}
+ \epsilon \cdot
\underbrace{\sum_{j=1}^h\Big( \frac1N \sum_{i=1}^N [\boldsymbol{z}_i]_j \Big)^{-1}}_{\text{"entropy"}}
+ \eta \cdot \!\!\!\!
\underbrace{\phantom{\Bigg[}\|W\|_F^2 \phantom{\Bigg[}}_{\text{regularization}}
\end{align*}

The objective is composed of four terms: The reconstruction term is the standard mean square error between the data points and their reconstructions. The sparsity term applies a l1-norm to drive activation to zero in the representation. The "entropy" term that ensures that at least a few examples activate each source dimension. The regularization term ensures that the sparsity term remains effective.


## Loading the dataset

We first load the Labeled Faces in the Wild (LFW) dataset. The LFW is a popular face recognition dataset which is readily available from scikit learn. When loading the dataset, we specify some image downscaling in order to limit the computation resources. The following code visualizes a few images that we have extracted from the LFW dataset.

```python
import sklearn,sklearn.datasets
import matplotlib
%matplotlib inline
from matplotlib import pyplot as plt

data = sklearn.datasets.fetch_lfw_people(resize=0.3)['images']

plt.figure(figsize=(12,2.5))
plt.imshow(data[:32].reshape(2,16,37,28).transpose(0,2,1,3).reshape(2*37,16*28),cmap='gray')
plt.show()
```

## Implementing the autoencoder (20 P)


We now would like to train an autoencoder on this data. As a first step, we standardize the data, which is a usual step before training a ML model. (Note that contrarily to other component analyses such as ICA, the data does not need to be whitened.)

```python
X = data.reshape(len(data),-1)
X = X - X.mean(axis=0)
X = X / X.std()
print(X)
```

To learn the autoencoder, we need to optimize the objective function above. This can be done using by gradient descent, or some enhanced gradient-based optimizer such as Adam. Because a manual computation of the gradients can be difficult and error-prone, we will make use of automatic differentiation readily provided by the PyTorch software. PyTorch uses its own structures for storing the data and the model parameters. (You can consult the tutorials at https://pytorch.org/tutorials/ to learn the basics.)

We first convert the data into a PyTorch tensor.

```python
import torch

X  = torch.FloatTensor(X)
```

Recall that the four terms that compose the objective function are given by:
\begin{align*}
\text{rec} &= \frac1N \sum_{i=1}^N \| \boldsymbol{x}_i - \hat{\boldsymbol{x}}_i \|^2 &
\text{spa} &= \Big(\frac1N \sum_{i=1}^N \|\boldsymbol{z}_i\|_1\Big)\\
\text{ent} &= \sum_{j=1}^h\Big( \frac1N \sum_{i=1}^N [\boldsymbol{z}_i]_j \Big)^{-1} &
\text{reg} &= \|W\|_F^2
\end{align*}

**Task:**

 * **Create the function `get_objective_terms` that computes these terms.**

The function receives as input:

 * A `FloatTensor` `X` of size $m \times d$ containing a data minibatch of $m$ examples.
 * A `FloatTensor` `V` of size $d \times h$ containing the weights of the encoder.
 * A `FloatTensor` `W` of size $h \times d$ containing the weights of the decoder.
 * A `FloatTensor` `b` of size $h$ containing the bias of the encoder.
 * A `FloatTensor` `a` of size $d$ containing the bias of the decoder.
 
In your function, the parameter $\epsilon$ can be hardcoded to 0.01. The function should return the four terms (`rec`, `spa`, `ent`, `reg`) of the objective. (These terms will be merged later on in a single objective function.) While implementing the `get_objective_terms` function, make sure to use PyTorch functions so that the gradient information necessary for automatic differentiation is retained. For example, converting arrays to numpy will not work as this will remove the gradient information.

```python
def get_objective_terms(X,V,W,b,a):
    # -----------------------------------------------------
    # TODO: replace by your code
    # -----------------------------------------------------
    import solutions
    rec,spa,ent,reg = solutions.get_objective_terms(X,V,W,b,a)
    # -----------------------------------------------------
    
    return rec,spa,ent,reg
```

## Training the autoencoder

Now that the terms of the objective function have been implemented, the model can be trained to minimize the objective. The code below calls the function `get_objective_terms` repeatedly (once per iteration). Automatic differentiation is used to compute the gradient, and we use Adam (a state-of-the-art optimizer for neural networks) to optimize the parameters. The number of units in the representation is hard-coded to $h=400$, and we use the parameter $\eta=1$ for the regularizer.

```python
import torch.optim
import torch.nn
import numpy

def train(X,lambd=0):
    
    d = X.shape[1]
    h = 400
    
    eps = 0.01 * lambd # hard-coded parameter
    eta = 1 * lambd    # hard-coded parameter
    
    V = torch.nn.Parameter(d**-.5*torch.randn([d,h]))
    W = torch.nn.Parameter(torch.zeros([h,d]))
    b = torch.nn.Parameter(torch.zeros([h]))
    a = torch.nn.Parameter(torch.zeros([d]))
    
    optimizer = torch.optim.Adam((V,W,b,a), lr=0.0001)
    
    print('%7s %8s %8s %8s %8s'%('nbit','rec','spa','ent','reg'))
    
    for i in range(0,10001):
        
        optimizer.zero_grad()
        
        x= X[numpy.random.permutation(len(X))[:100]]
        
        rec,spa,ent,reg = get_objective_terms(x,V,W,b,a)
        
        (rec + lambd*spa + eps*ent + eta*reg).backward()
        
        if i%1000 == 0: print('%7d %8.2f %8.2f %8.2f %8.2f'%(i,rec.data,spa.data,ent.data,reg.data))
        
        optimizer.step()

    return V,W,b,a
```

### Dense Autoencoder

We first train an autoencoder with parameter $\lambda=0$, that is, a standard autoencoder without sparsity. The parameters of the learn autoencoder are stored in the variables `V`, `W`, `b`, `a`. Running the code may take a few minutes. You may temporarily reduce the number of iterations when testing your implementation.

```python
V1,W1,b1,a1 = train(X,lambd=0)
```

We observe that the reconstruction term decreases strongly, indicating that the autoencoder becomes increasingly better at reconstructing the data. The sparsity term, however, increases, indicating that the standard autoencoder does not learn a sparse representation.

### Sparse Autoencoder

We now would like to train a sparse autoencoder. For this, we set the sparsity parameter to $\lambda=1$ and re-run the training procedure. We store the learned parameters in another set of variables.

```python
V2,W2,b2,a2 = train(X,lambd=1)
```

We observe that setting the parameter $\lambda$ to a non-zero keeps the sparsity term low, which indicates that a sparser representation has been learned. In turn, we also loose a bit of reconstruction accuracy compared to the original autoencoder. This can be expected since the sparsity imposes additional constraints on the solution.


## Analyzing autoencoder sparsity (10 P)

As a first analysis, we would like to verify how truly sparse the representation we have learned is.

**Task:**

 * **Create a line plot where the two lines represents all activations (for the 25 first examples in the dataset) sorted from largest to smallest of the respective autoencoder models.**

```python
# -----------------------------------------------------
# TODO: replace by your code
# -----------------------------------------------------
import solutions
solutions.plot_sparsity(X[:25],V1,V2,b1,b2)
# -----------------------------------------------------
```

We observe that the sparse autoencoder has a much larger proportion of weights that are close to zero. Hence, the our model has learned a sparse representation. One possible use of sparsity is to compress the data while retaining most of the information.

<!-- #region -->
## Inspecting the representation (10 P)

As a second analysis, we would like to visualize what the decoder has learned.


**Task:**

 * **Write code that displays the first 64 decoding filters of the two models, in a similar mosaic format as it was used above to display some examples from the dataset.**
<!-- #endregion -->

```python
# -----------------------------------------------------
# TODO: replace by your code
# -----------------------------------------------------
import solutions
solutions.view_decoder(W1,W2)
# -----------------------------------------------------
```

We observe that the filters of the standard autoencoder are quite difficult to interpret, whereas the sparse autoencoder produces filters with a stronger focus a single facial or background features such as the mouth, the nose, the bottom left/right corners, or the overall lighting condition. The features of the sparse autoencoder are also more interpretable for a human.
