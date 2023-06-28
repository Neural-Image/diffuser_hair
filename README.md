# Diffuser Bald Head Generation

Generate bald head portrait with Stable Diffusion Inpaint and ControlNet

## Install

Create a conda environment:

```bash
conda create -n diffuser -y python==3.8
```

The easiest way to install it is using pip:

```bash
pip install torch torchvision torchaudio
​
pip install accelerate
​
pip install git+https://github.com/huggingface/diffusers
​
pip install transformers
​
pip install git+https://github.com/FacePerceiver/facer.git@main
​
pip install timm
​
```

Run the inference:
```bash
python diffuser.py -i input_folder/ -o output_folder
```
