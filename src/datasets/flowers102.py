import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


class Flowers102:

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 custom=False,
                 seed=0,
                 **kwargs):

        self.train_dataset = torchvision.datasets.Flowers102(root=location, split='train',
                                                             transform=preprocess)
        self.val_dataset = torchvision.datasets.Flowers102(root=location, split='val',
                                                             transform=preprocess)
        self.test_dataset = torchvision.datasets.Flowers102(root=location, split='test',
                                                             transform=preprocess)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers)

        self.classnames = [
            'pink primrose',
            'hard-leaved pocket orchid',
            'canterbury bells',
            'sweet pea',
            'english marigold',
            'tiger lily',
            'moon orchid',
            'bird of paradise',
            'monkshood',
            'globe thistle',
            'snapdragon',
            "colt's foot",
            'king protea',
            'spear thistle',
            'yellow iris',
            'globe-flower',
            'purple coneflower',
            'peruvian lily',
            'balloon flower',
            'giant white arum lily',
            'fire lily',
            'pincushion flower',
            'fritillary',
            'red ginger',
            'grape hyacinth',
            'corn poppy',
            'prince of wales feathers',
            'stemless gentian',
            'artichoke',
            'sweet william',
            'carnation',
            'garden phlox',
            'love in the mist',
            'mexican aster',
            'alpine sea holly',
            'ruby-lipped cattleya',
            'cape flower',
            'great masterwort',
            'siam tulip',
            'lenten rose',
            'barbeton daisy',
            'daffodil',
            'sword lily',
            'poinsettia',
            'bolero deep blue',
            'wallflower',
            'marigold',
            'buttercup',
            'oxeye daisy',
            'common dandelion',
            'petunia',
            'wild pansy',
            'primula',
            'sunflower',
            'pelargonium',
            'bishop of llandaff',
            'gaura',
            'geranium',
            'orange dahlia',
            'pink-yellow dahlia',
            'cautleya spicata',
            'japanese anemone',
            'black-eyed susan',
            'silverbush',
            'californian poppy',
            'osteospermum',
            'spring crocus',
            'bearded iris',
            'windflower',
            'tree poppy',
            'gazania',
            'azalea',
            'water lily',
            'rose',
            'thorn apple',
            'morning glory',
            'passion flower',
            'lotus lotus',
            'toad lily',
            'anthurium',
            'frangipani',
            'clematis',
            'hibiscus',
            'columbine',
            'desert-rose',
            'tree mallow',
            'magnolia',
            'cyclamen',
            'watercress',
            'canna lily',
            'hippeastrum',
            'bee balm',
            'ball moss',
            'foxglove',
            'bougainvillea',
            'camellia',
            'mallow',
            'mexican petunia',
            'bromelia',
            'blanket flower',
            'trumpet creeper',
            'blackberry lily'
        ]
