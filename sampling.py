import torch
import torchvision
import os
import gc
import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from transformers import CLIPTextModel
from peft import PeftModel, LoraConfig
from lora_w2w import LoRAw2w
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
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



######## Sampling utilities


def sample_weights(unet, proj, mean, std, v, device, factor = 1.0): 
    # get mean and standard deviation for each principal component
    m = torch.mean(proj, 0)
    standev = torch.std(proj, 0)
    del proj
    torch.cuda.empty_cache()
    # sample
    sample = torch.zeros([1, 1000]).to(device)
    for i in range(1000):
        sample[0, i] = torch.normal(m[i], factor*standev[i], (1,1))

    # load weights into network
    network = LoRAw2w( sample, mean, std, v, 
                    unet,
                    rank=1,
                    multiplier=1.0,
                    alpha=27.0,
                    train_method="xattn-strict"
                ).to(device, torch.bfloat16)
        
    return network




