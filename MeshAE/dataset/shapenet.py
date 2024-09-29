import os
import logging
from torch.utils import data
import numpy as np
import yaml

logger = logging.getLogger(__name__)

class Shapes3dDataset(data.Dataset):
    ''' 3D Shapes dataset class.
    '''

    def __init__(self, dataset_folder, split=None,
                 categories=None,num_points=2048):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
            cfg (yaml): config file
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.num_points=num_points

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f, Loader=yaml.Loader)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            }

            # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            if split is None:
                self.models += [
                    {'category': c, 'model': m} for m in
                    [d for d in os.listdir(subpath) if (os.path.isdir(os.path.join(subpath, d)) and d != '')]
                ]

            else:
                split_file = os.path.join(subpath, split + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')

                if '' in models_c:
                    models_c.remove('')

                self.models += [
                    {'category': c, 'model': m}
                    for m in models_c
                ]

        # precompute
        self.split = split

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def _pc_normalize(self,pc):
        centroid = np.mean(pc[:, :-3], axis=0)
        pc[:, :-3] = pc[:, :-3] - centroid
        m = np.max(np.sqrt(np.sum(pc[:, :-3] ** 2, axis=1)))
        pc[:, :-3] = pc[:, :-3] / m
        pc[:, -3:] = pc[:, -3:] / np.sqrt(np.sum(pc[:, -3:] ** 2, axis=1, keepdims=True))
        return pc
    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''

        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx']

        model_path = os.path.join(self.dataset_folder, category, model,'pointcloud_reduced.npy')
        # data = self._pc_normalize(np.load(model_path))
        data=np.load(model_path)
        return data,c_idx

    def get_model_dict(self, idx):
        return self.models[idx]

