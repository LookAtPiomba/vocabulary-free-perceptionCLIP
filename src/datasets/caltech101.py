import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

class Caltech101:

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
        if not os.path.exists(os.path.join(location, 'caltech101')):
            self.test_dataset = torchvision.datasets.Caltech101(root=location, transform=preprocess, download=True)
        else:
            self.test_dataset = torchvision.datasets.Caltech101(root=location, transform=preprocess, download=False)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers)

        self.classnames = [
            'accordion',
            'airplanes',
            'anchor',
            'ant',
            'background_google',
            'barrel',
            'bass',
            'beaver',
            'binocular',
            'bonsai',
            'brain',
            'brontosaurus',
            'buddha',
            'butterfly',
            'camera',
            'cannon',
            'car_side',
            'ceiling_fan',
            'cellphone',
            'chair',
            'chandelier',
            'cougar_body',
            'cougar_face',
            'crab',
            'crayfish',
            'crocodile',
            'crocodile_head',
            'cup',
            'dalmatian',
            'dollar_bill',
            'dolphin',
            'dragonfly',
            'electric_guitar',
            'elephant',
            'emu',
            'euphonium',
            'ewer',
            'faces',
            'faces_easy',
            'ferry',
            'flamingo',
            'flamingo_head',
            'garfield',
            'gerenuk',
            'gramophone',
            'grand_piano',
            'hawksbill',
            'headphone',
            'hedgehog',
            'helicopter',
            'ibis',
            'inline_skate',
            'joshua_tree',
            'kangaroo',
            'ketch',
            'lamp',
            'laptop',
            'leopards',
            'llama',
            'lobster',
            'lotus',
            'mandolin',
            'mayfly',
            'menorah',
            'metronome',
            'minaret',
            'motorbikes',
            'nautilus',
            'octopus',
            'okapi',
            'pagoda',
            'panda',
            'pigeon',
            'pizza',
            'platypus',
            'pyramid',
            'revolver',
            'rhino',
            'rooster',
            'saxophone',
            'schooner',
            'scissors',
            'scorpion',
            'sea_horse',
            'snoopy',
            'soccer_ball',
            'stapler',
            'starfish',
            'stegosaurus',
            'stop_sign',
            'strawberry',
            'sunflower',
            'tick',
            'trilobite',
            'umbrella',
            'watch',
            'water_lilly',
            'wheelchair',
            'wild_cat',
            'windsor_chair',
            'wrench',
            'yin_yang'
        ]