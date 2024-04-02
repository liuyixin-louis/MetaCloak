import torch
import torch.nn.functional as F


class PGDAttacker():
    def __init__(self, radius, steps, step_size, random_start, norm_type='l-infty', ascending=True, args=None, x_range=[-1, 1]):
        self.noattack = radius == 0. or steps == 0 or step_size == 0.
        self.radius = (radius-0.5*255)/0.5*255
        self.step_size = (step_size-0.5*255)/0.5*255
        self.steps = steps # how many step to conduct pgd
        self.random_start = random_start
        self.norm_type = norm_type # which norm of your noise
        self.ascending = ascending # perform gradient ascending, i.e, to maximum the loss
        self.args=args
        self.left , self.right = x_range
        
    
    def certi(self, models, adv_x, vae, noise_scheduler, input_ids, device=torch.device("cuda"), weight_dtype=torch.float32, target_tensor=None, ):
        # args=self.args
        unet, text_encoder = models
        unet.zero_grad()
        text_encoder.zero_grad()
        # device = torch.device("cuda")
        adv_latens = vae.encode(adv_x.to(device, dtype=weight_dtype)).latent_dist.sample()
        adv_latens = adv_latens * vae.config.scaling_factor
        
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(adv_latens)
        bsz = adv_latens.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=adv_latens.device)
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(adv_latens, noise, timesteps)
        encoder_hidden_states = text_encoder(input_ids.to(device))[0]
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(adv_latens, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        if target_tensor is not None:
            timesteps = timesteps.to(device)
            noisy_latents = noisy_latents.to(device)
            xtm1_pred = torch.cat(
                [
                    noise_scheduler.step(
                        model_pred[idx : idx + 1],
                        timesteps[idx : idx + 1],
                        noisy_latents[idx : idx + 1],
                    ).prev_sample
                    for idx in range(len(model_pred))
                ]
            )
            xtm1_target = noise_scheduler.add_noise(target_tensor, noise.to(device), (timesteps - 1).to(device))
            loss = loss - F.mse_loss(xtm1_pred, xtm1_target)
        return loss
    
    def perturb(self, models, x,ori_x,vae, tokenizer, noise_scheduler, target_tensor=None,device=torch.device("cuda"), close_grad_for_efficiency =False):
        if self.noattack:
            # print("no need to attack")
            return x 
        
        args=self.args
        
        weight_dtype = torch.bfloat16
        if args.mixed_precision == "fp32":
            weight_dtype = torch.float32
        elif args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        # device = torch.device("cuda")
        unet, text_encoder = models
        device = unet.device
        
        
        adv_x = x.detach().clone().to(device, dtype=weight_dtype)
        ori_x = ori_x.detach().clone().to(device, dtype=weight_dtype)
        
        
        if close_grad_for_efficiency:
            ''' temporarily shutdown autograd of model to improve pgd efficiency '''
            text_encoder.eval()
            unet.eval()
            for mi in [text_encoder, unet]:
                for pp in mi.parameters():
                    pp.requires_grad = False

        input_ids = tokenizer(
            args.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.repeat(len(x), 1)
        
        for step in range(self.steps):
            adv_x.requires_grad_()
            loss = self.certi(models, adv_x,vae, noise_scheduler, input_ids, device, weight_dtype, target_tensor)
    
            grad = torch.autograd.grad(loss, [adv_x])[0]

            with torch.no_grad():
                if not self.ascending: grad.mul_(-1)
                if self.norm_type == 'l-infty':
                    adv_x.add_(torch.sign(grad), alpha=self.step_size)
                else:
                    raise NotImplementedError
            self._clip_(adv_x, ori_x)
        
            
        with torch.no_grad():
            noise_added = adv_x.detach_() - x
            noise_added.clamp_(-self.radius, self.radius)
            new_x = x + noise_added
            new_x = new_x.clamp(self.left, self.right)
            final_noise = new_x - x
        
        
        if close_grad_for_efficiency:
            ''' reopen autograd of model after pgd '''
            for mi in  [text_encoder, unet]:
                for pp in mi.parameters():
                    pp.requires_grad = True
            text_encoder.train()
            unet.train()
             
        return x + final_noise.detach_()
        

    def _clip_(self, adv_x, x):
        adv_x = adv_x - x
        if self.norm_type == 'l-infty':
            adv_x.clamp_(-self.radius, self.radius)
        else:
            raise NotImplementedError
        adv_x = adv_x + x
        adv_x.clamp_(self.left, self.right)
        return adv_x
    