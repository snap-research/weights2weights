import torch
import torchvision
import os
import gc
import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from lora_w2w import LoRAw2w
from transformers import AutoTokenizer, PretrainedConfig
from PIL import Image
import warnings
warnings.filterwarnings("ignore")




######## Editing Utilities

def get_direction(df, label, pinverse, return_dim, device):
    ### get labels
    labels = []
    for folder in list(df.index): 
        labels.append(df.loc[folder][label])
    labels = torch.Tensor(labels).to(device).bfloat16()

    ### solve least squares
    direction = (pinverse@labels).unsqueeze(0)

    if return_dim == 1000: 
        return direction
    else:
        direction = torch.cat((direction, torch.zeros((1, return_dim-1000)).to(device)), dim=1)
        return direction
   
def debias(direction, label, df, pinverse, device):
    ### get labels
    labels = []
    for folder in list(df.index): 
        labels.append(df.loc[folder][label])
    labels = torch.Tensor(labels).to(device).bfloat16()

    ### solve least squares
    d = (pinverse@labels).unsqueeze(0)

    ###align dimensionalities of the two vectors
    if direction.shape[1] == 1000: 
        pass
    else:
        d = torch.cat((d, torch.zeros((1, direction.shape[1]-1000)).to(device)), dim=1)

    #remove this component from the direction
    direction = direction - ((direction@d.T)/(torch.norm(d)**2))*d
    return direction


@torch.no_grad
def edit_inference(network, edited_weights, unet, vae, text_encoder, tokenizer, prompt, negative_prompt, guidance_scale, noise_scheduler, ddim_steps, start_noise, seed, generator, device):
    
    original_weights = network.proj.clone()

    generator = generator.manual_seed(seed)
    latents = torch.randn(
        (1, unet.in_channels, 512 // 8, 512 // 8),
        generator = generator,
        device = device
    ).bfloat16()
   

    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
                            [negative_prompt], padding="max_length", max_length=max_length, return_tensors="pt"
                        )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    noise_scheduler.set_timesteps(ddim_steps) 
    latents = latents * noise_scheduler.init_noise_sigma
    

 
    for i,t in enumerate(tqdm.tqdm(noise_scheduler.timesteps)):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
        
        if t>start_noise:
            pass
        elif t<=start_noise:
            network.proj = torch.nn.Parameter(edited_weights)
            network.reset()


        with network:
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, timestep_cond= None).sample
            
        
        #guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)

    #reset weights back to original 
    network.proj = torch.nn.Parameter(original_weights)
    network.reset()

    return image 
