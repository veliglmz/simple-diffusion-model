# Python Implementation of a Simple Diffusion Model
This code is an unofficial Python implementation of [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). I trained a simple model with a generated spiral train data. I tested it using Gaussian distribution.

## Installation
* Clone the repository.
* Create a virtual environment.
* Install pip packages.
If you have trouble with torch, please install it according to [PyTorch](https://pytorch.org/).

```bash
git clone https://github.com/veliglmz/simple-diffusion-model.git
cd simple-diffusion-model
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

If you want to see the options, you can use this command.
```bash
python main.py --help
```

# Result
The result of the spiral sample.

![](https://github.com/veliglmz/simple-diffusion-model/blob/main/plot.png)