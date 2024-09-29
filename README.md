# H2SW
Official PyTorch implementation for paper: Hierarchical Hybrid Sliced Wasserstein: A Scalable Metric for Heterogeneous Joint Distributions

![image](GradientFlow/armadillo.png)
![image](GradientFlow/bunny.png)

Details of the model architecture and experimental results can be found in our papers.

```
@article{nguyen2024h2sw,
  title={Hierarchical Hybrid Sliced Wasserstein: A Scalable Metric for Heterogeneous Joint Distributions},
  author={Khai Nguyen and Nhat Ho},
  journal={Advances in Neural Information Processing Systems},
  year={2024},
  pdf={https://arxiv.org/pdf/2404.15378}
}
```
Please CITE our paper whenever this repository is used to help produce published results or incorporated into other software.

This implementation is made by [Khai Nguyen](https://khainb.github.io).

## Requirements
To install the required python packages, run
```
pip install -r requirements.txt
```

## What is included?
* 3D Mesh Gradient flow
* 3D Mesh Autoencoder


## Point-Cloud Gradient flow 
```
cd GradientFlow
python armadillo.py;
python bunny.py
```

## 3D Mesh Autoencoder
Please read the README file in the MeshAE folder.

