# Coarsely-Aligned Paired Image Translation (CAPIT)

We provide our PyTorch implementation of Coarsely-Aligned Paired Image Translation (CAPIT) which has been accepted to ICRA 2023.

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDN

### Getting started

- Clone this repo:
```bash
git clone https://github.com/taesungp/contrastive-unpaired-translation CUT
cd CAPIT
```

- Install PyTorch 1.1 and other dependencies (e.g., torchvision, visdom, dominate, gputil).

  For pip users, please type the command `pip install -r requirements.txt`.

  For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.

- Train the CUT model:
```bash
python train.py --dataroot ../../mask_data/weather6/ --name weather_cutoriloc --CUT_mode CUT --display_id 0 --preprocess scale_width --load_size 512

```
- Test the CUT model:
```bash
python test.py --dataroot ../mask_data/weather7/ --name weather_nightcutrank --CUT_mode CUT --preprocess scale_width --load_size 512
```

## Acknowledgements
- https://github.com/taesungp/contrastive-unpaired-translation
- https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
