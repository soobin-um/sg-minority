# Self-Guided Generation of Minority Samples Using Diffusion Models

This repository contains the official code for the paper "Self-Guided Generation of Minority Samples Using Diffusion Models".

## 1. Environment setup
We provide a conda environment file to install all the dependencies. If you don't have conda installed, you can download it from [here](https://docs.conda.io/en/latest/miniconda.html).

### 1) Clone the repository
```
git clone https://github.com/anonymous-8641/sg-minority
cd sg-minority
```

### 2) Install dependencies
The code is tested with the following environment:
- Python 3.11
- PyTorch 2.0.1
- CUDA 11.7

We recommend using conda to install all of the dependencies.
```
conda env create -f environment.yaml
```
If you prefer to install the dependencies manually, you can use the following commands to create a new conda environment and install the dependencies manually
```
conda create -n sg-minority python=3.11.4
conda activate sg-minority
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
conda install -c conda-forge mpi4py mpich
pip install lpips
pip install blobfile
pip install scikit-learn
```


## 2. Download pre-trained checkpoints

The pretrained models can be downloaded from the following links:
- [CelebA](https://drive.google.com/file/d/11zaWowtEvU_rmAXnEe66x9tXzOdNbQrs/view?usp=drive_link)
- [LSUN-Bedrooms](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt)
- [ImageNet-64](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt)
- [ImageNet-256](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64_256_upsampler.pt) (64 → 256 upsampler)

Place one of the models in the folder you want. This checkpoint will be refered as ```[your_model_path]```.

The model configuration of the CelebA model is as follows:
```
--diffusion_steps 1000 --noise_schedule cosine --image_size 64 --class_cond False --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order True --learn_sigma True --dropout 0.1 --use_scale_shift_norm True --use_fp16 True
```

See this [link](https://github.com/openai/guided-diffusion) for the configurations of the LSUN-Bedrooms and ImageNet models.


## 3. Self-guided minority generation
For these demonstrations, we will generate 100 samples with batch size 4. Feel free to change these values.
```
SAMPLE_FLAGS="--timestep_respacing 250 --batch_size 4 --num_samples 100"
```

### 1) CelebA, LSUN-Bedrooms, and ImageNet-64
For instance to generate samples from the CelebA model, you can use the following command:
```
MODEL_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --image_size 64 --class_cond False --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order True --learn_sigma True --dropout 0.1 --use_scale_shift_norm True --use_fp16 True"
python image_sample.py $MODEL_FLAGS $SAMPLE_FLAGS --model_path [your_model_path] --out_dir [your_out_dir] --use_ms_grad True --mg_scale 0.4
```
### 2) ImageNet-256
To reproduce the generation on ImageNet-256 as in our paper, you have to first produce the samples from the ImageNet-64 model with our guidance and then upsample them to 256x256 using the ImageNet-256 model with ancestral sampling.

The following command generates samples from the ImageNet-64 model:
```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
python image_sample.py $MODEL_FLAGS $SAMPLE_FLAGS --model_path [your_model_path] --out_dir [your_out_dir] --use_ms_grad True --mg_scale 0.2 --inter_rate 2
```

Then, you can upsample the generated samples to 256x256 using the following command:
```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --large_size 256  --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python super_res_sample.py $MODEL_FLAGS $SAMPLE_FLAGS --model_path [your_model_path] --base_samples [imagenet64_npz_path] --out_dir [your_out_dir]
```
