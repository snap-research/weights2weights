import sys
import os 
sys.path.append(os.path.abspath(os.path.join("", "..")))
import torch
import torchvision
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from lora_w2w import LoRAw2w
from utils import load_models, inference, save_model_w2w, save_model_for_diffusers
from inversion import invert
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--mean_path", default="/files/mean.pt", type=str, help="Path to file with parameter means")
    parser.add_argument("--std_path", default="/files/std.pt", type=str, help="Path to file with parameter standard deviations.")
    parser.add_argument("--v_path", default="/files/V.pt", type=str, help="Path to V orthogonal projection/unprojection matrix.")
    parser.add_argument("--dim_path", default="/files/weight_dimensions.pt", type=str, help="Path to file with dimensions of LoRA layers. Used for saving in Diffusers pipeline format.")
    parser.add_argument("--imfolder", default="/inversion/images/real_image/real/", type=str, help="Path to folder containing image.")
    parser.add_argument("--mask_path", default=None, type=str, help="Path to mask file.")
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--lr", default= 1e-1, type=float)
    parser.add_argument("--weight_decay", default= 1e-10, type=float)
    parser.add_argument("--dim", default= 10000, type=int, help="Number of principal component coefficients to optimize.")
    parser.add_argument("--diffusers_format", default=False, action="store_true", help="Whether to save in mode that can be loaded in Diffusers pipeline")
    parser.add_argument("--save_name", default="/files/inversion1.pt", type=str, help="Output path + filename.")



    ### variables
    args = parser.parse_args()
    device = args.device
    mean_path = args.mean_path
    std_path = args.std_path
    v_path = args.v_path
    dim_path = args.dim_path
    imfolder = args.imfolder
    mask_path = args.mask_path
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    dim = args.dim
    diffusers_format = args.diffusers_format
    save_name = args.save_name


    ### load models
    unet, vae, text_encoder, tokenizer, noise_scheduler = load_models(device)

    ### load files
    mean = torch.load(mean_path).bfloat16().to(device)
    std = torch.load(std_path).bfloat16().to(device)
    v = torch.load(v_path).bfloat16().to(device)
    weight_dimensions = torch.load(dim_path)

    ### initialize network

    proj = torch.zeros(1,dim).bfloat16().to(device)
    network = LoRAw2w( proj, mean, std, v[:,:dim], 
                        unet,
                        rank=1,
                        multiplier=1.0,
                        alpha=27.0,
                        train_method="xattn-strict"
                    ).to(device, torch.bfloat16)
    ### run inversion 
    network = invert(network=network, unet=unet, vae=vae, 
                     text_encoder=text_encoder, tokenizer=tokenizer, 
                     prompt = "sks person", noise_scheduler = noise_scheduler, epochs=epochs, 
                     image_path = imfolder, mask_path = mask_path, device = device)
    
    
    ### save model

    if diffusers_format:
        save_model_for_diffusers(network,std, mean, v, weight_dimensions,
                                path=save_name)
    else: 
        save_model_w2w(network, path=save_name)



if __name__ == "__main__":
    main()
