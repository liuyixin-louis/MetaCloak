# Robust Imperceptible Perturbation against Diffusion Models
[Paper](https://arxiv.org/abs/2311.13127); [Homepage](metacloak.github.io); 
This is the official implementation of the paper "Robust Imperceptible Perturbation against Diffusion Models" (CVPR 2022). 
<!-- The complete code and data will be released upon acceptance. Four sampled IDs from VGGFace2 (clean and protected images with our method with $r=11/255$) are released under the `./example_data/` folder. Free feel to test out the protection performance.  -->
<div align="center">
    <img src="./teaser.png" alt="Teaser">
</div>

<!-- ## Algorithm Flow

![Framework](./framework.png) -->


## Software Dependencies
```shell
# pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
conda create -n metacloak python=3.9.18
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirement.txt --ignore-installed
pip install ftfy regex tqdm git+https://github.com/openai/CLIP.git
pip install git+https://github.com/huggingface/diffusers.git
```

## Data and Checkpoint Dependencies
- put LIQE checkpoint `https://drive.google.com/file/d/1GoKwUKNR-rvX11QbKRN8MuBZw2hXKHGh/view` to `./LIQE/checkpoints/`
- dataset 
    - CelebA+VGGFace2 `https://drive.google.com/file/d/1RrlGMOuA2WF5RrWiLSt3QPxFNd3EagvG/view?usp=sharing`
- model: download sd models into `./SD/` folder
    - 2.1base https://huggingface.co/stabilityai/stable-diffusion-2-1-base


## Environment Setup
setup the following environment variables 
```shell
# your project root
export ADB_PROJECT_ROOT="/path/to/your/project/root"
# your conda env name
export PYTHONPATH=$PYTHONPATH$:$ADB_PROJECT_ROOT
```


## Citation
If our work is useful for your research, please consider citing:
```bibtex
@article{liu2023toward,
  title={Toward Robust Imperceptible Perturbation against Unauthorized Text-to-image Diffusion-based Synthesis},
  author={Liu, Yixin and Fan, Chenrui and Dai, Yutong and Chen, Xun and Zhou, Pan and Sun, Lichao},
  journal={arXiv preprint arXiv:2311.13127},
  year={2023}
}
```


## Acknowledgement
- [CLIP-IQA](https://github.com/IceClear/CLIP-IQA?tab=readme-ov-file)
- [Anti-Dreambooth](https://github.com/VinAIResearch/Anti-DreamBooth)
- [deepface](https://github.com/serengil/deepface)
