import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

class DTD:

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 custom=False,
                 seed=0,
                 **kwargs):
        self.train_dataset = None
        self.val_dataset = None
        if not os.path.exists(os.path.join(location, 'DTD')):
            self.test_dataset = torchvision.datasets.DTD(root=location, split="test", transform=preprocess, download=True)
        else:
            self.test_dataset = torchvision.datasets.DTD(root=location, split="test", transform=preprocess, download=False)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers)

        self.classnames = [
            'banded',
            'blotchy',
            'braided',
            'bubbly',
            'bumpy',
            'chequered',
            'cobwebbed',
            'cracked',
            'crosshatched',
            'crystalline',
            'dotted',
            'fibrous',
            'flecked',
            'freckled',
            'frilly',
            'gauzy',
            'grid',
            'grooved',
            'honeycombed',
            'interlaced',
            'knitted',
            'lacelike',
            'lined',
            'marbled',
            'matted',
            'meshed',
            'paisley',
            'perforated',
            'pitted',
            'pleated',
            'polka-dotted',
            'porous',
            'potholed',
            'scaly',
            'smeared',
            'spiralled',
            'sprinkled',
            'stained',
            'stratified',
            'striped',
            'studded',
            'swirly',
            'veined',
            'waffled',
            'woven',
            'wrinkled',
            'zigzagged',
        ]