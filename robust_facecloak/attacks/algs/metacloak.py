# -----------------------------------------------------------------------
# Copyright (c) 2023 Yixin Liu Lehigh University
# All rights reserved.
#
# This file is part of the MetaCloak project. Please cite our paper if our codebase contribute to your project. 
# -----------------------------------------------------------------------

import random
import wandb
import argparse
import copy
import hashlib
import itertools
import logging
import os
from pathlib import Path
import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from robust_facecloak.model.db_train import  DreamBoothDatasetFromTensor
from robust_facecloak.model.db_train import import_model_class_from_model_name_or_path
from robust_facecloak.generic.data_utils import PromptDataset, load_data
from robust_facecloak.generic.share_args import share_parse_args


logger = get_logger(__name__)

import torch
import torch.nn.functional as F


def train_few_step(
    args,
    models,
    tokenizer,
    noise_scheduler,
    vae,
    data_tensor: torch.Tensor,
    num_steps=20,
    step_wise_save=False,
    save_step=100, 
    retain_graph=False,
):
    # Load the tokenizer

    unet, text_encoder = copy.deepcopy(models[0]), copy.deepcopy(models[1])
    params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters())

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    train_dataset = DreamBoothDatasetFromTensor(
        data_tensor,
        args.instance_prompt,
        tokenizer,
        args.class_data_dir,
        args.class_prompt,
        args.resolution,
        args.center_crop,
    )

    weight_dtype = torch.bfloat16
    device = torch.device("cuda")

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    
    step2modelstate={}
        
    pbar = tqdm(total=num_steps, desc="training")
    for step in range(num_steps):
        
        if step_wise_save and ((step+1) % save_step == 0 or step == 0):
            # make sure the model state dict is put to cpu
            step2modelstate[step] = {
                "unet": copy.deepcopy(unet.cpu().state_dict()),
                "text_encoder": copy.deepcopy(text_encoder.cpu().state_dict()),
            }
            # move the model back to gpu
            unet.to(device, dtype=weight_dtype); text_encoder.to(device, dtype=weight_dtype)
            
        pbar.update(1)
        unet.train()
        text_encoder.train()

        step_data = train_dataset[step % len(train_dataset)]
        pixel_values = torch.stack([step_data["instance_images"], step_data["class_images"]]).to(
            device, dtype=weight_dtype
        )
        input_ids = torch.cat([step_data["instance_prompt_ids"], step_data["class_prompt_ids"]], dim=0).to(device)

        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(input_ids)[0]
        
        # Predict the noise residual
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # with prior preservation loss
        if args.with_prior_preservation:
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            # Add the prior loss to the instance loss.
            loss = instance_loss + args.prior_loss_weight * prior_loss

        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        loss.backward(retain_graph=retain_graph)
        torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad()

    pbar.close()
    if step_wise_save:
        return [unet, text_encoder], step2modelstate
    else:     
        return [unet, text_encoder]


def load_model(args, model_path):
    print(model_path)
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(model_path, args.revision)

    # Load scheduler and models
    text_encoder = text_encoder_cls.from_pretrained(
        model_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=args.revision)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=args.revision)

    vae.requires_grad_(False)

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        print("You selected to used efficient xformers")
        print("Make sure to install the following packages before continue")
        print("pip install triton==2.0.0.dev20221031")
        print("pip install pip install xformers==0.0.17.dev461")

        unet.enable_xformers_memory_efficient_attention()

    return text_encoder, unet, tokenizer, noise_scheduler, vae



def parse_args(): 
    
    parser = share_parse_args()
    
    parser.add_argument(
        "--transform_hflip",
        action="store_true",
        help="Whether to use horizontal flip for transform.",
    )
    
    
    parser.add_argument(
        "--instance_data_dir_for_train",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_data_dir_for_adversarial",
        type=str,
        default=None,
        required=True,
        help="A folder containing the images to add adversarial noise",
    )
    
    parser.add_argument(
        "--defense_pgd_ascending",
        action="store_true",
        help="Whether to use ascending order for pgd.",
    )
    
    parser.add_argument(
        "--defense_pgd_radius",
        type=float,
        default=8,
        help="The radius for defense pgd.",
    )
    
    parser.add_argument(
        "--defense_pgd_step_size",
        type=float,
        default=2,
        help="The step size for defense pgd.",
    )
    parser.add_argument(
        "--defense_pgd_step_num",
        type=int,
        default=8,
        help="The number of steps for defense pgd.",
    )
    
    parser.add_argument(
        "--defense_pgd_random_start",
        action="store_true",
        help="Whether to use random start for pgd.",
    )
    
    parser.add_argument(
        "--attack_pgd_radius",
        type=float,
        default=4,
        help="The radius for attack pgd.",
    )
    parser.add_argument(
        "--attack_pgd_step_size",
        type=int,
        default=2,
        help="The step size for attack pgd.",
    )
    parser.add_argument(
        "--attack_pgd_step_num",
        type=int,
        default=4,
        help="The number of steps for attack pgd.",
    )
    parser.add_argument(
        "--attack_pgd_ascending",
        action="store_true",
        help="Whether to use ascending order for pgd.",
    )
    
    parser.add_argument(
        "--attack_pgd_random_start",
        action="store_true",
        help="Whether to use random start for pgd.",
    )
    
    parser.add_argument(
        "--target_image_path",
        default=None,
        help="target image for attacking",
    )
    
    # args.gau_kernel_size
    parser.add_argument(
        "--gau_kernel_size",
        type=int,
        default=5,
        help="The kernel size for gaussian filter.",
    )
    # defense_sample_num
    parser.add_argument(
        "--defense_sample_num",
        type=int,
        default=1,
        help="The number of samples for defense.",
    )
    
    parser.add_argument(
        "--rot_degree",
        type=int,
        default=5,
        help="The degree for rotation.",
    )
    
    parser.add_argument(
        "--transform_rot", 
        action="store_true",
        help="Whether to use rotation for transform.",
        
    )
    
    parser.add_argument(
        "--transform_gau",
        action="store_true",
        help="Whether to use gaussian filter for transform.",
    )
    
    parser.add_argument(
        "--original_flow", 
        action="store_true",
        help="Whether to use original flow in ASPL for transform.",
    )
    
    parser.add_argument(
        "--total_trail_num",
        type=int,
        default=60,
    )
    
    parser.add_argument(
        "--unroll_steps",
        type=int,
        default=2,
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=40,
    )
    
    parser.add_argument(
        "--total_train_steps",
        type=int,
        default=1000, 
    )
    
    
    args = parser.parse_args()
    return args

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # logging_dir=logging_dir,
    )

    
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
            if args.mixed_precision == "fp32":
                torch_dtype = torch.float32
            elif args.mixed_precision == "fp16":
                torch_dtype = torch.float16
            elif args.mixed_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                list(args.pretrained_model_name_or_path.split(","))[-1], 
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

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
                
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    clean_data = load_data(
        args.instance_data_dir_for_train,
        # size=args.resolution,
        # center_crop=args.center_crop,
    )
    
    perturbed_data = load_data(
        args.instance_data_dir_for_adversarial,
        # size=args.resolution,
        # center_crop=args.center_crop,
    )
    
    original_data= copy.deepcopy(perturbed_data)
        
    import torchvision
    train_aug = [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
    ]
    rotater = torchvision.transforms.RandomRotation(degrees=(0, args.rot_degree))
    gau_filter = transforms.GaussianBlur(kernel_size=args.gau_kernel_size,)
    defense_transform = [
    ]
    if args.transform_hflip:
        defense_transform = defense_transform + [transforms.RandomHorizontalFlip(p=0.5)]
    if args.transform_rot:
        defense_transform = defense_transform + [rotater]
    if args.transform_gau:
        defense_transform = [gau_filter] + defense_transform
    
    tensorize_and_normalize = [
        transforms.Normalize([0.5*255]*3,[0.5*255]*3),
    ]
    
    all_trans = train_aug + defense_transform + tensorize_and_normalize
    all_trans = transforms.Compose(all_trans)
    
    
    from robust_facecloak.attacks.worker.robust_pgd_worker import RobustPGDAttacker
    from robust_facecloak.attacks.worker.pgd_worker import PGDAttacker
    attacker = PGDAttacker(
        radius=args.attack_pgd_radius, 
        steps=args.attack_pgd_step_num, 
        step_size=args.attack_pgd_step_size,
        random_start=args.attack_pgd_random_start,
        ascending=args.attack_pgd_ascending,
        args=args, 
        x_range=[-1, 1],
    )
    defender = RobustPGDAttacker(
        radius=args.defense_pgd_radius,
        steps=args.defense_pgd_step_num,
        step_size=args.defense_pgd_step_size,
        random_start=args.defense_pgd_random_start,
        ascending=args.defense_pgd_ascending,
        args=args,
        attacker=attacker, 
        trans=all_trans,
        sample_num=args.defense_sample_num,
        x_range=[0, 255],
    )
    model_paths = list(args.pretrained_model_name_or_path.split(","))
    num_models = len(model_paths)

    MODEL_BANKS = [load_model(args, path) for path in model_paths]
    MODEL_STATEDICTS = [
        {
            "text_encoder": MODEL_BANKS[i][0].state_dict(),
            "unet": MODEL_BANKS[i][1].state_dict(),
        }
        for i in range(num_models)
    ]
    
    def save_image(perturbed_data, id_stamp):
        save_folder = f"{args.output_dir}/noise-ckpt/{id_stamp}"
        os.makedirs(save_folder, exist_ok=True)
        noised_imgs = perturbed_data.detach()
        img_names = [
            str(instance_path).split("/")[-1]
            for instance_path in list(Path(args.instance_data_dir_for_adversarial).iterdir())
        ]
        for img_pixel, img_name in zip(noised_imgs, img_names):
            save_path = os.path.join(save_folder, f"noisy_{img_name}")
            Image.fromarray(
                img_pixel.float().detach().cpu().permute(1, 2, 0).numpy().squeeze().astype(np.uint8)
            ).save(save_path)

    
    init_model_state_pool = {}
    pbar = tqdm(total=num_models, desc="initializing models")
    # split sub-models
    for j in range(num_models):
        init_model_state_pool[j] = {}
        text_encoder, unet, tokenizer, noise_scheduler, vae = MODEL_BANKS[j]
        
        unet.load_state_dict(MODEL_STATEDICTS[j]["unet"])
        text_encoder.load_state_dict(MODEL_STATEDICTS[j]["text_encoder"])
        f_ori = [unet, text_encoder]
        f_ori, step2state_dict = train_few_step(
                args,
                f_ori,
                tokenizer,
                noise_scheduler,
                vae,
                perturbed_data.float(),
                args.total_train_steps,
                step_wise_save=True,
                save_step=args.interval,
        )  
        init_model_state_pool[j] = step2state_dict
        del f_ori, unet, text_encoder, tokenizer, noise_scheduler, vae
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        pbar.update(1)
    pbar.close()
    
            
    steps_list = list(init_model_state_pool[0].keys())
    pbar = tqdm(total=args.total_trail_num * num_models * (args.interval // args.advance_steps) * len(steps_list), desc="meta poison with model ensemble")
    cnt=0
    # learning perturbation over the ensemble of models
    for _ in range(args.total_trail_num):
        for model_i in range(num_models):
            text_encoder, unet, tokenizer, noise_scheduler, vae = MODEL_BANKS[model_i]
            for split_step in steps_list: 
                unet.load_state_dict(init_model_state_pool[model_i][split_step]["unet"])
                text_encoder.load_state_dict(init_model_state_pool[model_i][split_step]["text_encoder"])
                f = [unet, text_encoder]
                
                for j in range(args.interval // args.advance_steps):
                    
                    perturbed_data = defender.perturb(f, perturbed_data, original_data, vae, tokenizer, noise_scheduler,)
                    cnt+=1
                    
                    f = train_few_step(
                        args,
                        f,
                        tokenizer,
                        noise_scheduler,
                        vae,
                        perturbed_data.float(),
                        args.advance_steps,
                    )
                    pbar.update(1)
                    if cnt % 1000 == 0:
                        save_image(perturbed_data, f"{cnt}")
                
                # frequently release the memory due to limited GPU memory, 
                # env with more gpu might consider to remove the following lines for boosting speed
                del f 
                torch.cuda.empty_cache()
                
            del unet, text_encoder, tokenizer, noise_scheduler, vae

            if torch.cuda.is_available():
                torch.cuda.empty_cache() 

        import gc
        gc.collect()
        torch.cuda.empty_cache()      
    pbar.close()

    save_image(perturbed_data, "final")


if __name__ == "__main__":
    args = parse_args()
    wandb.init(project="metacloak", entity=args.wandb_entity_name)
    wandb.config.update(args)
    wandb.log({'status': 'gen'})
    main(args)
