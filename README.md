# Enhancing Vocabulary-free Image Classification with Contextual Attributes

<p align="center">
  <img src="https://github.com/LookAtPiomba/vocabulary-free-perceptionCLIP/blob/main/model.png" alt="model image"/>
</p>

## Method description
[CaSED](https://github.com/altndrr/vic) based image classification method which leverages insights from [perceptionCLIP](https://github.com/umd-huang-lab/perceptionCLIP) on contextual attributes inference improving classification performances brought to a vocabulary-free setting.

## Results
| Model   | Flowers102 | Food101 | Oxford Pets | EuroSAT | ImageNet |
|------------|----------|----------|----------|----------|----------|
| standard CaSED | 55.5 | 64.5 | 62.5 | 32.0 | 57.9 |
| hand-crafted attributes | 59.2 | 66.8 | 60.6 | 35.7 | 58.5 |
| vocabulary-free attributes | 57.0 | 65.7 | 59.4 | 35.3 | 58.0 |

Our method outperforms state-of-art results (standard CaSED) on 4 out of 5 datasets, including both coarse (ImageNet) and finer-grained ones. Data are expressed as Semantic Similarity (x100).
