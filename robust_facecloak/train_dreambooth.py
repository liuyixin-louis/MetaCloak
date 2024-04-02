#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# Modified by Yixin Liu, Lehigh University 2023, with some changes in adversarial attack and defense settings

import requests
# from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch
from generic.tools import upload_py_code
from PIL import Image
# from PIL import Image
import numpy as np
import argparse
import hashlib
import itertools
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Optional
from robust_facecloak.generic.tools import config_and_condition_checking
import datasets
import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, create_repo, whoami
# from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from robust_facecloak.generic.data_utils import jpeg_compress_image


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

from robust_facecloak.generic.share_args import share_parse_args, add_train_db
def parse_args(input_args=None):
    parser = share_parse_args()
    parser = add_train_db(parser)
    
    parser.add_argument(
        "--class_name",
        type=str,
        help="The name of the class to be trained.",
        default="face",
        required=False,
    )
    
    parser.add_argument(
        "--eval_gen_img_num",
        type=int,
        default=16,
    )
    
    parser.add_argument(
        "--transform_defense",
        action="store_true",
        help="Whether to use transform defense."
    )
    
    parser.add_argument(
        "--jpeg_transform",
        action="store_true",
        help="Whether to use jpeg defense."
    )
    
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=75,
        help="The quality for jpeg defense."
    )
    
    parser.add_argument(
        "--transform_sr", 
        action="store_true",
        help="Whether to use super resolution defense."
    )
    
    parser.add_argument(
        "--sr_scale",
        type=int,
        default=4,
        help="The scale for super resolution defense."
    )
    
    parser.add_argument(
        "--transform_tvm",
        action="store_true",
        help="Whether to use TVM defense."
    )
    
    parser.add_argument(
        "--transform_rotate",
        action="store_true",
        help="Whether to use rotate defense."
    )
    
    parser.add_argument(
        "--rot_degree",
        type=int,
        default=4,
        help="The degree for rotate defense." 
    )
    
    parser.add_argument(
        "--transform_hflip",
        action="store_true",
        help="Whether to use horizontal flip defense."
    )
    
    parser.add_argument(
        "--transform_gau",
        action="store_true",
        help="Whether to use gaussian noise defense." 
    )
    
    parser.add_argument(
        "--inference_prompts",
        type=str,
        default="A photo of a person",
        help="A list of prompts for inference."
    )

    parser.add_argument(
        "--gau_kernel_size",
        type=int,
        default=5,
        help="The kernel size for gaussian noise defense."
    )
    
    parser.add_argument(
        "--save_model", 
        action="store_true",
        help="Whether to save the model."
    )
    
    parser.add_argument(
        "--log_score",
        action="store_true",
        help="Whether to log the score."
    )
    
    parser.add_argument(
        "--validation_scheduler",
        type=str,
        default="DPMSolverMultistepScheduler",
        choices=["DPMSolverMultistepScheduler", "DDPMScheduler"],
        help="Select which scheduler to use for validation. DDPMScheduler is recommended for DeepFloyd IF.",
    )
    
    # poison rate 
    parser.add_argument(
        "--poison_rate",
        type=float,
        default=1.0,
        help="The poison rate for training."
    )
    
    # clean_img_dir
    parser.add_argument("--clean_img_dir",type=str, default=None, help="The clean image dir for training.")
    
  
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    config_and_condition_checking(args)

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        defense_transforms = [], 
        args=None
    ):
        self.args = args
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        # filter out non-image files
        instance_data = []
        for file in self.instance_data_root.iterdir():
            # try:
            #     Image.open(file)
            #     instance_data.append(file)
            # except:
            #     pass
            # only load those images with png and jpg format
            if file.suffix in ['.png', '.jpg']:
                instance_data.append(file)
                
        # poison_rate
        # clean_img_dir
        if self.args.poison_rate < 1.0:
            clean_instance_data = []
            for file in Path(self.args.clean_img_dir).iterdir():
                # only load those images with png and jpg format
                if file.suffix in ['.png', '.jpg']:
                    clean_instance_data.append(file)
            # 1 - poison_rate % of instance will be replaced by clean instance
            clean_num = int(len(instance_data) * (1 - self.args.poison_rate))
            # clip to [0, len(clean_instance_data)]
            clean_num = min(clean_num, len(clean_instance_data))
            # replace the first clean_num instance with clean instance
            instance_data[:clean_num] = clean_instance_data[:clean_num]
        
        self.instance_images_path = list(
            # Path(instance_data_root).iterdir()
            instance_data
            )
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            # filter out non-image files
            class_data = []
            for file in self.class_data_root.iterdir():
                #  only load those images with png and jpg format
                if file.suffix in ['.png', '.jpg']:
                    class_data.append(file)
            self.class_images_path = class_data
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms_for_instances = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            ] + defense_transforms + [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),]
            
        )
        self.image_transforms_for_class_imgs = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),]
            
        )
        self.instance_image = []
        
        if self.args.transform_sr or self.args.transform_tvm:
            # load model and scheduler
            model_id = "stabilityai/stable-diffusion-x4-upscaler"
            sr_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                model_id, revision="fp16",torch_dtype=torch.float16) 

            self.sr_pipeline = sr_pipeline.to("cuda")

        if self.args.transform_tvm:
            
            import cvxpy as cp
            def get_tvm_image(img, TVM_WEIGHT=0.01, PIXEL_DROP_RATE=0.02):
                # TVM_WEIGHT = 0.01
                # PIXEL_DROP_RATE = 0.02

                # Load the image
                # img = Image.open(path)  # Replace with your image path
                # reshape the image to 32*32
                img = img.resize((64, 64), Image.ANTIALIAS)
                img_array = np.asarray(img, dtype=np.float32) / 255.0  # Convert to numpy array and normalize to [0, 1]

                def total_variation(Z, shape, p=2):
                    h, w, c = shape
                    Z_reshaped = cp.reshape(Z, (h, w*c))
                    
                    # Compute the Horizontal Differences and their p-norm
                    horizontal_diff = Z_reshaped[1:, :] - Z_reshaped[:-1, :]
                    horizontal_norm = cp.norm(horizontal_diff, p, axis=1)  # axis may need to be adjusted based on the requirement
                    horizontal_sum = cp.sum(horizontal_norm)
                    
                    # Compute the Vertical Differences and their p-norm
                    vertical_diff = Z_reshaped[:, 1:] - Z_reshaped[:, :-1]
                    vertical_norm = cp.norm(vertical_diff, p, axis=0)  # axis may need to be adjusted based on the requirement
                    vertical_sum = cp.sum(vertical_norm)
                    
                    # Total Variation is the sum of all norms
                    tv = horizontal_sum + vertical_sum
                    
                    return tv

                def minimize_total_variation(X, x, lambda_tv=TVM_WEIGHT, p=2):
                    h, w, c = x.shape
                    Z = cp.Variable((h, w*c))
                    X_flat = np.reshape(X, (h, w*c))
                    x_flat = np.reshape(x, (h, w*c))
                    objective = cp.Minimize(cp.norm(cp.multiply((1 - X_flat),(Z - x_flat)),  2) + lambda_tv * total_variation(Z, (h, w, c), p))
                    problem = cp.Problem(objective)
                    problem.solve(verbose=True,solver=cp.MOSEK)
                    return Z.value


                # Generate the mask matrix X using Bernoulli distribution
                X = np.random.binomial(1, PIXEL_DROP_RATE, img_array.shape)

                # Run the optimization
                Z_optimal = minimize_total_variation(X, img_array)

                # reshape back to 64*64
                Z_optimal = np.reshape(Z_optimal, img_array.shape)


                # If needed, convert the result back to a PIL Imagepip install mosek
                img_result = Image.fromarray(np.uint8(Z_optimal*255))
                img_result
                return img_result
            
        
        # instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        for img_i_dir in self.instance_images_path:
            prompt="A photo of a person"
            instance_image = Image.open(img_i_dir)
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            # consider some defenses like jpeg compression
            if self.args.jpeg_transform:
                instance_image = jpeg_compress_image(instance_image, self.args.jpeg_quality)
            if self.args.transform_sr:
                instance_image = instance_image.resize((128, 128))
                instance_image = self.sr_pipeline(image=instance_image,prompt=prompt, ).images[0]
            if self.args.transform_tvm:
                instance_image = get_tvm_image(instance_image)
                # one sr to [256, 256]
                instance_image = self.sr_pipeline(image=instance_image,prompt=prompt, ).images[0]
                # another resie to [128, 128]
                instance_image = instance_image.resize((128, 128))
                # another sr to [512, 512]
                instance_image = self.sr_pipeline(image=instance_image,prompt=prompt, ).images[0]
                
            self.instance_image.append(instance_image)
        
        if self.args.transform_sr or self.args.transform_tvm:
            del self.sr_pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image= self.instance_image[index % self.num_instance_images]
        example["instance_images"] = self.image_transforms_for_instances(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms_for_class_imgs(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def infer(checkpoint_path, prompts=None, n_img=16, bs=8, n_steps=100, guidance_scale=7.5):
    pipe = StableDiffusionPipeline.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16, safety_checker=None
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.disable_attention_slicing()

    for prompt in prompts:
        print(prompt)
        norm_prompt = prompt.lower().replace(",", "").replace(" ", "_")
        out_path = f"{checkpoint_path}/dreambooth/{norm_prompt}"
        os.makedirs(out_path, exist_ok=True)
        for i in range(n_img // bs):
            images = pipe(
                [prompt] * bs,
                num_inference_steps=n_steps,
                guidance_scale=guidance_scale,
            ).images
            for idx, image in enumerate(images):
                image.save(f"{out_path}/{i}_{idx}.png")
    del pipe


class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index]


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)
            pipeline.enable_xformers_memory_efficient_attention()
            pipeline.disable_attention_slicing()

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if args.train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        from bitsandbytes.optim import AdamW8bit
        optimizer_class = AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    defense_transforms = []
    if args.transform_defense:
        import torchvision.transforms as T
        
        gaussianBlurrer = T.GaussianBlur(kernel_size=args.gau_kernel_size,)
        hflipper = T.RandomHorizontalFlip(p=0.5)
        rotater = T.RandomRotation(degrees=(0, args.rot_degree))
        defense_transforms=[
            # gaussianBlurrer, 
            # hflipper, 
            # rotater
        ]
        if args.transform_rotate:
            defense_transforms.append(
                rotater
            )
        
        if args.transform_gau:
            defense_transforms.append(
                gaussianBlurrer
            )
    
        if args.transform_hflip:
            defense_transforms.append(
                hflipper
            )
            
    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        defense_transforms=defense_transforms,
        args=args
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    # Move vae and text_encoder to device and cast to weight_dtype
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        
    latents_cache = []
    text_encoder_cache = []
    for batch in tqdm(train_dataloader, desc="Caching latents"):
        with torch.no_grad():
            batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
            batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
            # latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
            model_input = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist
            # .sample()
            model_input = model_input 
            # * vae.config.scaling_factor
            latents_cache.append(model_input)
            if args.train_text_encoder:
                text_encoder_cache.append(batch["input_ids"])
            else:
                text_encoder_cache.append(text_encoder(batch["input_ids"])[0])
    train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True)
    scaling_factor = vae.config.scaling_factor
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            unet,
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(unet, text_encoder, optimizer, train_dataloader, lr_scheduler)
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.wandb_project_name, config=vars(args), init_kwargs={"wandb": {"entity": args.wandb_entity_name}})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latent_dist = batch[0][0]
                latents = latent_dist.sample()
                latents = latents * scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                if args.train_text_encoder:
                    encoder_hidden_states = text_encoder(batch[0][1])[0]
                else:
                    encoder_hidden_states = batch[0][1]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                if model_pred.shape[1] == 6:
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step == args.max_train_steps and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    ckpt_pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        text_encoder=accelerator.unwrap_model(text_encoder),
                        revision=args.revision,
                    )
                    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                    scheduler_args = {}

                    if "variance_type" in ckpt_pipeline.scheduler.config:
                        variance_type = ckpt_pipeline.scheduler.config.variance_type

                        if variance_type in ["learned", "learned_range"]:
                            variance_type = "fixed_small"

                        scheduler_args["variance_type"] = variance_type
                    ckpt_pipeline.scheduler = ckpt_pipeline.scheduler.from_config(ckpt_pipeline.scheduler.config, **scheduler_args)
                    ckpt_pipeline.save_pretrained(save_path)
                    unet.cpu(); text_encoder.cpu(); 
                    del ckpt_pipeline, unet, text_encoder
                    if len(args.inference_prompts) > 0:
                        prompts = args.inference_prompts.split(";")
                        infer(save_path, prompts, n_img=args.eval_gen_img_num, bs=1, n_steps=100)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Finish training")
        final_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        if args.log_score:
            
            from eval_score import get_score
            clean_ref_db = args.clean_ref_db
            prompts = args.inference_prompts.split(";")
            
            # prompt = prompts[-1]
            if len(prompts) > 1:
                prompt2scorelist = {}
                pbar = tqdm(prompts, desc="evaluating score")
                for prompt in prompts:
                    pbar.update(1)
                    pid = prompt.lower().replace(",", "").replace(" ", "_")
                    im_dir = os.path.join(
                        final_save_path,'dreambooth', pid
                    )
                    res = get_score(
                        im_dir,
                        clean_ref_db,
                        type_name=args.class_name,
                    )
                    prompt2scorelist[pid] = res
                    
                    for i, images in enumerate(os.listdir(im_dir)):
                        
                        import numpy as np
                        # check if the image is valid
                        if not images.endswith(".png"):
                            continue
                        # Load your image using PIL or any other method
                        image = Image.open(os.path.join(im_dir, images))

                        # Convert the PIL image to a NumPy array
                        image_array = np.array(image)

                        # Log the image using wandb.Image
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                import wandb
                                tracker.log({prompt: wandb.Image(image_array, caption=str(i)+": "+pid)})
                    
                
                pbar.close()
                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        # tracker.log(res_dict)
                        tracker.log(prompt2scorelist)
                        # log a mean 
                        prompt_list = list(prompt2scorelist.keys())
                        k_list = list(prompt2scorelist[prompt_list[0]].keys())
                        for k in k_list:
                            means = []
                            for prompt in prompt_list:
                                means.append(prompt2scorelist[prompt][k])
                            tracker.log({k+"_mean": np.mean(means)})
                            stds = []
                            for prompt in prompt_list:
                                stds.append(prompt2scorelist[prompt][k])
                            tracker.log({k+"_std": np.std(stds)})

        if not args.save_model:
            # delete all the checkpoints expect for the generated images
            for file in os.listdir(final_save_path):
                if file != "dreambooth":
                    import shutil
                    # all file and directory
                    file_path = os.path.join(final_save_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
        # save something to indicate the training is finished
        with open(os.path.join(final_save_path, "finished.txt"), "w+") as f:
            f.write("finished")
    accelerator.end_training()

   
    
if __name__ == "__main__":
    args = parse_args()
    
    main(args)
