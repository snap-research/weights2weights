import torch
import torchvision
import os
import shutil
import gc
import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from transformers import CLIPTextModel
from lora_w2w import LoRAw2w
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler
from safetensors.torch import save_file
from transformers import AutoTokenizer, PretrainedConfig
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    PNDMScheduler, 
    StableDiffusionPipeline
)



######## Basic utilities

### load base models
def load_models(device):
    pretrained_model_name_or_path = "stablediffusionapi/realistic-vision-v51" 

    revision = None
    rank = 1
    weight_dtype = torch.bfloat16

    # Load scheduler, tokenizer and models.
    pipe = StableDiffusionPipeline.from_pretrained("stablediffusionapi/realistic-vision-v51", 
                                            torch_dtype=torch.float16,safety_checker = None,
                                            requires_safety_checker = False).to(device)
    noise_scheduler = pipe.scheduler
    del pipe
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=revision
    )

    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    print("")

    return unet, vae, text_encoder, tokenizer, noise_scheduler



### basic inference to generate images conditioned on text prompts
@torch.no_grad
def inference(network, unet, vae, text_encoder, tokenizer, prompt, negative_prompt, guidance_scale, noise_scheduler, ddim_steps, seed, generator, device):
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
        with network:
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, timestep_cond= None).sample
        #guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)

    return image 



### save model in w2w space (principal component representation)
def save_model_w2w(network, path):
    proj = network.proj.clone().detach().float()

    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(proj, path+"/"+"w2wmodel.pt")


### save model in format compatible with Diffusers
def save_model_for_diffusers(network,std, mean, v, weight_dimensions, path):
    proj = network.proj.clone().detach()
    unproj = torch.matmul(proj,v[:, :].T)*std+mean

    final_weights0 = {}
    counter = 0
    for key in weight_dimensions.keys():
        final_weights0[key] = unproj[0, counter:counter+weight_dimensions[key][0][0]].unflatten(0, weight_dimensions[key][1])
        counter += weight_dimensions[key][0][0]
    
    #renaming keys to be compatible with Diffusers
    for key in list(final_weights0.keys()):
        final_weights0[key.replace( "lora_unet_", "base_model.model.").replace("A", "down").replace("B", "up").replace( "weight", "identity1.weight").replace("_lora", ".lora").replace("lora_down", "lora_A").replace("lora_up", "lora_B")] = final_weights0.pop(key)


    
    final_weights0_keys = sorted(final_weights0.keys())

    final_weights = {}
    for i,key in enumerate(final_weights0_keys):
        final_weights[key] = final_weights0[key]

    if not os.path.exists(path):
        os.makedirs(path+"/unet")
    else: 
        os.mkdir(path+"/unet")

    
    
    #add config for PeftConfig
    shutil.copyfile("../files/adapter_config.json", path+"/unet/adapter_config.json")
        
    save_file(final_weights, path+"/unet/adapter_model.safetensors")

    


def unflatten(flattened_weights, weight_dimensions, path):
    final_weights0 = {}
    counter = 0
    for key in weight_dimensions.keys():
        final_weights0[key] = flattened_weights[0, counter:counter+weight_dimensions[key][0][0]].unflatten(0, weight_dimensions[key][1])
        counter += weight_dimensions[key][0][0]
    
    #renaming keys to be compatible with Diffusers
    for key in list(final_weights0.keys()):
        final_weights0[key.replace( "lora_unet_", "base_model.model.").replace("A", "down").replace("B", "up").replace( "weight", "identity1.weight").replace("_lora", ".lora").replace("lora_down", "lora_A").replace("lora_up", "lora_B")] = final_weights0.pop(key)


    
    final_weights0_keys = sorted(final_weights0.keys())

    final_weights = {}
    for i,key in enumerate(final_weights0_keys):
        final_weights[key] = final_weights0[key]

    if not os.path.exists(path):
        os.makedirs(path+"/unet")
    else: 
        os.mkdir(path+"/unet")

    
    
    #add config for PeftConfig
    shutil.copyfile("../files/adapter_config.json", path+"/unet/adapter_config.json")
        
    save_file(final_weights, path+"/unet/adapter_model.safetensors")

