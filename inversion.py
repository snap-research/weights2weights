import torch
import torchvision
import tqdm
import torchvision.transforms as transforms
from PIL import Image
import warnings
warnings.filterwarnings("ignore")



### run inversion  (optimize PC coefficients) given single image
def invert(network, unet, vae, text_encoder, tokenizer, prompt, noise_scheduler, epochs, image_path, mask_path, device, weight_decay = 1e-10, lr=1e-1):
    ### load mask
    if mask_path: 
        mask = Image.open(mask_path)
        mask = transforms.Resize((64,64), interpolation=transforms.InterpolationMode.BILINEAR)(mask)
        mask = torchvision.transforms.functional.pil_to_tensor(mask).unsqueeze(0).to(device).bfloat16()
    else: 
        mask = torch.ones((1,1,64,64)).to(device).bfloat16()

    ### single image dataset
    image_transforms = transforms.Compose([transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                                                transforms.RandomCrop(512),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5], [0.5])])


    train_dataset = torchvision.datasets.ImageFolder(root=image_path, transform = image_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True) 

    ### optimizer 
    optim = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)    

    ### training loop
    unet.train()
    for epoch in tqdm.tqdm(range(epochs)):
        for batch,_ in train_dataloader:
            ### prepare inputs
            batch = batch.to(device).bfloat16()
            latents = vae.encode(batch).latent_dist.sample()
            latents = latents*0.18215
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
         
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

            ### loss + sgd step
            with network:
                model_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                loss = torch.nn.functional.mse_loss(mask*model_pred.float(), mask*noise.float(), reduction="mean")
                optim.zero_grad()
                loss.backward()
                optim.step()

    ### return optimized network
    return network


