import inspect
import importlib
import pytorch_lightning as pl
from dotdict import dotdict
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)

class DataInterface(pl.LightningDataModule):
    def __init__(self, num_workers=8,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = dotdict(kwargs)
        self.load_data_module()
        logger.info("Dataset Initialized.")

    def setup(self, stage=None):
        # Things to do on every accelerator in distributed mode
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.trainset = self.instancialize(mode='train')
            self.valset = self.instancialize(mode='val')
            
        if stage in ("test", "predict"):
            self.testset = self.instancialize(mode='test')


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.kwargs.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.kwargs.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=1, num_workers=self.num_workers, shuffle=False, pin_memory=False)

    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except Exception as e:
            logger.error(str(e))
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, mode, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.kwargs dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(mode=mode, **args1)
