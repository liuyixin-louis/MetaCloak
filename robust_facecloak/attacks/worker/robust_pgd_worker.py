import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# import lpips


    
class RobustPGDAttacker():
    def __init__(self, radius, steps, step_size, random_start, trans, sample_num, attacker, norm_type='l-infty', ascending=True, args=None, x_range=[0, 255], target_weight=1.0):        
        self.noattack = radius == 0. or steps == 0 or step_size == 0.
        self.radius = radius
        self.step_size = step_size
        self.steps = steps # how many step to conduct pgd
        self.random_start = random_start
        self.norm_type = norm_type # which norm of your noise
        self.ascending = ascending # perform gradient ascending, i.e, to maximum the loss
        self.args=args
        self.transform = trans
        self.sample_num = sample_num
        self.attacker = attacker
        self.left, self.right = x_range
        self.pattern = np.random.randint( 0, 72, size=(16, 16, 3), dtype=np.uint8)
        print(
            "summary of the attacker: \n"
            f"radius: {self.radius}\n"
            f"steps: {self.steps}\n"
            f"step_size: {self.step_size}\n"
            f"random_start: {self.random_start}\n"
            f"norm_type: {self.norm_type}\n"
            f"ascending: {self.ascending}\n"
            f"sample_num: {self.sample_num}\n"
            # f"attacker: {self.attacker}\n", 
            f"x_range: {self.left} ~ {self.right}\n"
        )
        self.target_weight = target_weight
    
    def certi(self, models, adv_x,vae,  noise_scheduler, input_ids, device=torch.device("cuda"), weight_dtype=torch.float32, target_tensor=None,loss_type=None,ori_x=None):
        # args=self.args
        unet, text_encoder = models
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
        
        args=self.args
        
        if "robust_instance_conditioning_vector" in vars(args).keys() and args.robust_instance_conditioning_vector:
            condition_vector = args.robust_instance_conditioning_vector_data
            # print('this is your condition vector')
            # print(condition_vector.shape)
            encoder_hidden_states[0,:7,:] = condition_vector.to(device, dtype=weight_dtype)

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(adv_latens, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


        loss = 0.0
        
        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        target_loss=0.0
        if target_tensor is not None:
            timesteps = timesteps.to('cpu')
            noisy_latents = noisy_latents.to('cpu')
            xtm1_pred = torch.cat(
                [
                    noise_scheduler.step(
                        model_pred[idx : idx + 1].to('cpu'),
                        timesteps[idx : idx + 1].to('cpu'),
                        noisy_latents[idx : idx + 1].to('cpu'),
                    ).prev_sample
                    for idx in range(len(model_pred))
                ]
            )
            xtm1_target = noise_scheduler.add_noise(target_tensor, noise, (timesteps - 1))
            target_loss = self.target_weight*F.mse_loss(xtm1_pred.to(device), xtm1_target.to(device))
        
        loss = mse_loss - target_loss
    
        return loss

    def perturb_over_branchs_of_model(self, list_of_model, x, ori_x, vae, tokenizer, noise_scheduler, target_tensor=None,):
        args=self.args
        if self.noattack:
            return ori_x
        # for each PGD step, we load back a list of model and craft perturbation using gradients from this esemble of models
        device = torch.device("cuda")
        weight_dtype = torch.bfloat16
        if args.mixed_precision == "fp32":
            weight_dtype = torch.float32
        elif args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            
        vae.to(device, dtype=weight_dtype)
    
        ori_x = ori_x.detach().clone().to(device, dtype=weight_dtype)
        x = x.detach().clone().to(device, dtype=weight_dtype)
        
        
        if self.random_start:
            r=self.radius
            r_noise = torch.zeros_like(x).uniform_(-r, r)
            r_x=x+r_noise
            x=self._clip_(r_x, x)

        input_ids = tokenizer(
            args.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.repeat(len(x), 1)
        
        x.requires_grad_(True)
        
        
        for _step in range(self.steps):
            gradd = 0.0
            print('pgd step: ', _step)
            for model_i in list_of_model:
                print('index', list_of_model.index(model_i))
                x.requires_grad = True 
                def_x_trans = self.transform(x).to(device, dtype=weight_dtype)
                unet, text_encoder = model_i
                two_model = [unet, text_encoder]
                text_encoder.to(device, dtype=weight_dtype)
                unet.to(device, dtype=weight_dtype)
                text_encoder.eval()
                unet.eval()
                for mi in [text_encoder, unet, vae]:
                    for pp in mi.parameters():
                        pp.requires_grad = False
                adv_x = self.attacker.perturb(
                    two_model, def_x_trans, self.transform(ori_x), vae, tokenizer, noise_scheduler, 
                )
                loss = self.certi(two_model, adv_x, vae, noise_scheduler, input_ids, device, weight_dtype, target_tensor)
                loss.backward()
                gradd += x.grad.data
                # put the model to cpu first 
                text_encoder.to('cpu')
                unet.to('cpu')
                del text_encoder; del unet
                del two_model
                del model_i
                import gc 
                gc.collect()
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                # grad = x.grad.data
                grad = gradd
                if not self.ascending: 
                    grad.mul_(-1)
                if self.norm_type == 'l-infty':
                    x.add_(torch.sign(grad), alpha=self.step_size)
                else:
                    raise NotImplementedError
                x = self._clip_(x, ori_x, ).detach_()
        return x.cpu()
        
    def perturb(self, models, x, ori_x, vae, tokenizer, noise_scheduler, target_tensor=None, adaptive_target=False, loss_type=None):
        args=self.args
        unet, text_encoder = models
        weight_dtype = torch.bfloat16
        if args.mixed_precision == "fp32":
            weight_dtype = torch.float32
        elif args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        
        device = torch.device("cuda")
        vae.to(device, dtype=weight_dtype)
        text_encoder.to(device, dtype=weight_dtype)
        unet.to(device, dtype=weight_dtype)
        
        ori_x = ori_x.detach().clone().to(device, dtype=weight_dtype)
        x = x.detach().clone().to(device, dtype=weight_dtype)
        
        
        if self.noattack:
            return x
        
        # ''' temporarily shutdown autograd of model to improve pgd efficiency '''
        # vae.eval()
        text_encoder.eval()
        unet.eval()
        for mi in [text_encoder, unet]:
           for pp in mi.parameters():
               pp.requires_grad = False
               
        
        if self.random_start:
            r=self.radius
            r_noise = torch.zeros_like(x).uniform_(-r, r)
            r_x=x+r_noise
            x=self._clip_(r_x, x)
            
        input_ids = tokenizer(
            args.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.repeat(len(x), 1)
        
        x.requires_grad_(True)
        
        for _step in range(self.steps):
            x.requires_grad = True
            for _sample in range(self.sample_num):
                
                def_x_trans = self.transform(x).to(device, dtype=weight_dtype)
                adv_x = self.attacker.perturb(
                    models, def_x_trans, self.transform(ori_x), vae, tokenizer, noise_scheduler, 
                )
                
                loss = self.certi(models, adv_x, vae, noise_scheduler, input_ids, device, weight_dtype, target_tensor,loss_type, ori_x,)
                
                loss.backward()

            with torch.no_grad():
                grad = x.grad.data
                
                if not self.ascending: 
                    grad.mul_(-1)
                    
                if self.norm_type == 'l-infty':
                    x.add_(torch.sign(grad), alpha=self.step_size)
                else:
                    raise NotImplementedError
                x = self._clip_(x, ori_x, ).detach_()
        
        ''' reopen autograd of model after pgd '''
        for mi in [text_encoder, unet]:
            for pp in mi.parameters():
                pp.requires_grad = True
        return x.cpu()
    def _clip_(self, adv_x, x, ):
        adv_x = adv_x - x
        if self.norm_type == 'l-infty':
            adv_x.clamp_(-self.radius, self.radius)
        else:
            raise NotImplementedError
        adv_x = adv_x + x
        adv_x.clamp_(self.left, self.right)
        return adv_x