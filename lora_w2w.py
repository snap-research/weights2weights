# ref:
# - https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# - https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

import os
import math
from typing import Optional, List, Type, Set, Literal

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file


UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
#     "Transformer2DModel",  # どうやらこっちの方らしい？ # attn1, 2
    "Attention"
]
UNET_TARGET_REPLACE_MODULE_CONV = [
    "ResnetBlock2D",
    "Downsample2D",
    "Upsample2D",
    "DownBlock2D",
    "UpBlock2D",
    
]  # locon, 3clier

LORA_PREFIX_UNET = "lora_unet"

DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER

TRAINING_METHODS = Literal[
    "noxattn",  # train all layers except x-attns and time_embed layers
    "innoxattn",  # train all layers except self attention layers
    "selfattn",  # ESD-u, train only self attention layers
    "xattn",  # ESD-x, train only x attention layers
    "full",  #  train all layers
    "xattn-strict", # q and k values
    "noxattn-hspace",
    "noxattn-hspace-last",
    # "xlayer",
    # "outxattn",
    # "outsattn",
    # "inxattn",
    # "inmidsattn",
    # "selflayer",
]


class LoRAModule(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        proj, 
        v,
        mean, 
        std,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.in_dim = org_module.in_features
        self.out_dim = org_module.out_features

        self.proj = proj.bfloat16()
        self.mean1 = mean[0:self.in_dim].bfloat16()
        self.mean2 = mean[self.in_dim:].bfloat16()
        self.std1 = std[0:self.in_dim].bfloat16()
        self.std2 = std[self.in_dim:].bfloat16()
        self.v1 = v[0:self.in_dim].bfloat16()
        self.v2 = v[self.in_dim: ].bfloat16()

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        #self.scale = self.scale.bfloat16()
        

        self.multiplier = multiplier
        self.org_module = org_module

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return self.org_forward(x) +\
            (x@((self.proj@self.v1.T)*self.std1+self.mean1).T)@(((self.proj@self.v2.T)*self.std2+self.mean2))*self.multiplier*self.scale



class LoRAw2w(nn.Module):
    def __init__(
        self,
        proj,
        mean, 
        std, 
        v,
        unet: UNet2DConditionModel,
        rank: int = 4,
        multiplier: torch.bfloat16= 1.0,
        alpha: torch.bfloat16 = 1.0,
        train_method: TRAINING_METHODS = "full"

    ) -> None:
        super().__init__()
        self.lora_scale = 1
        self.multiplier = multiplier
        self.lora_dim = rank
        self.alpha = alpha
        self.proj = torch.nn.Parameter(proj)
        self.register_buffer("mean", torch.tensor(mean)) 
        self.register_buffer("std", torch.tensor(std)) 
        self.register_buffer("v", torch.tensor(v))
        
        self.module = LoRAModule

        self.unet_loras = self.create_modules(
            LORA_PREFIX_UNET,
            unet,
            DEFAULT_TARGET_REPLACE,
            self.lora_dim,
            self.multiplier,
            train_method=train_method,
        )
      

    
        self.lora_names = set()
        for lora in self.unet_loras:
            assert (
                lora.lora_name not in self.lora_names
            ), f"duplicated lora name: {lora.lora_name}. {self.lora_names}"
            self.lora_names.add(lora.lora_name)


        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(
                lora.lora_name,
                lora,
            )

        del unet
        torch.cuda.empty_cache()

    
    def reset(self):
        for lora in self.unet_loras:
            lora.proj = torch.nn.Parameter(self.proj.bfloat16())
    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: List[str],
        rank: int,
        multiplier: float,
        train_method: TRAINING_METHODS,
    ) -> list:
        
        counter = 0


        mm = []
        nn = []
        for name, module in root_module.named_modules():
            nn.append(name)
            mm.append(module)


        midstart = 0
        upstart = 0
        for i in range(len(nn)):
            if "mid_block" in nn[i]:
                midstart = i
                break

        for i in range(len(nn)):
            if "up_block" in nn[i]:
                upstart = i
                break
        
        mm = mm[:upstart]+mm[midstart:]+mm[upstart:midstart]
        nn = nn[:upstart]+nn[midstart:]+nn[upstart:midstart]
        
        

        loras = []
        names = []

        for i in range(len(mm)):
            name = nn[i]
            module = mm[i]
            if train_method == "noxattn" or train_method == "noxattn-hspace" or train_method == "noxattn-hspace-last":  # Cross Attention と Time Embed 以外学習
                if "attn2" in name or "time_embed" in name:
                    continue
            elif train_method == "innoxattn":  # Cross Attention 
                if "attn2" in name:
                    continue
            elif train_method == "selfattn":  # Self Attention 
                if "attn1" not in name:
                    continue
            elif train_method == "xattn" or train_method == "xattn-strict":  # Cross Attention 
                if "to_k" in name:
                    continue

            elif train_method == "full":  # 全部学習
                pass
            else:
                raise NotImplementedError(
                    f"train_method: {train_method} is not implemented."
                )
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                        if train_method == 'xattn-strict':
                            if 'out' in child_name:
                                continue
                            if "to_k" in child_name:
                                continue
                        if train_method == 'noxattn-hspace':
                            if 'mid_block' not in name:
                                continue
                        if train_method == 'noxattn-hspace-last':
                            if 'mid_block' not in name or '.1' not in name or 'conv2' not in child_name:
                                continue
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")


                        in_dim = child_module.in_features
                        out_dim = child_module.out_features
                        combined_dim = in_dim+out_dim

                        lora = self.module(
                            lora_name, self.proj, self.v[counter:counter+combined_dim], self.mean[counter:counter+combined_dim],\
                              self.std[counter:counter+combined_dim], child_module, multiplier, rank, self.alpha)
                        counter+=combined_dim
                        if lora_name not in names:
                            loras.append(lora)
                            names.append(lora_name)
                        

        return loras

    

    def prepare_optimizer_params(self):
        all_params = []

        if self.unet_loras:  # 実質これしかない
            params = []
            [params.extend(lora.parameters()) for lora in self.unet_loras]
            param_data = {"params": params}
            all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
    def set_lora_slider(self, scale):
        self.lora_scale = scale

    def __enter__(self):
        for lora in self.unet_loras:
            lora.multiplier = 1.0 * self.lora_scale

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.unet_loras:
            lora.multiplier = 0
