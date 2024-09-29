#### Table of Content

- [Hierarchical Hybrid Sliced Wasserstein: A Scalable Metric for Heterogeneous Joint Distributions](https://arxiv.org/abs/2404.15378)
  - [Getting Started](#getting-started)
    - [Datasets](#datasets)
      - [ShapeNet](#shapenet)
    - [Installation](#installation)
  - [Experiments](#experiments)
  - [Acknowledgment](#acknowledgment)

  

## Getting Started
### Datasets
#### ShapeNet

Download the dataset from https://github.com/autonomousvision/shape_as_points.

Use "reduce_points.py" for downsampling.

### Installation 

To install the required python packages, run
```bash
pip install -r requirements.txt
```

## Experiments
Available arguments for training an autoencoder
```bash
train.py [-h] [--config CONFIG] [--logdir LOGDIR]
                [--data_path DATA_PATH] [--loss LOSS]
                [--autoencoder AUTOENCODER]

optional arguments:
  -h, --help                  show this help message and exit
  --config CONFIG             path to json config file
  --logdir LOGDIR             path to the log directory
  --data_path DATA_PATH       path to data for training
  --loss LOSS                 loss function. One of [swd, gswd, h2swd]
  --inter_dim                 dimension of keys
  --r                     parameter of circular defining function
  --L                     number of projections
```

Example
```
CUDA_VISIBLE_DEVICES=${GPU} python train.py --config="config.json" --logdir=logs/${LOG_DIR} --data_path=${PATH} --loss=${LOSS} --autoencoder="pointnet" --r ${R} --L=${L} --epoch ${EPOCH} --seed=${SEED}
```

To test reconstruction
```bash
python reconstruction_test.py  --config="reconstruction/config.json" \
                                              --logdir=${LOG_DIR} \
                                              --data_path=$DATA_PATH

```


## Acknowledgment
The structure of this repo is largely based on [PointSWD](https://github.com/VinAIResearch/PointSWD).