import torch
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from dsprites import *
import os
# 2: assume I'm interested in getting close variations of an image, using SDXL's VAE
# instead of inferring, eg doing an img2img or inpaint,
# I can use the latent proba distribution inferred by the VAE, sample from it and decode back to pixel space
d = prepare_dsprites()
output_dir = "interpolation_imgs"
os.makedirs(output_dir, exist_ok=True)
def slerp(z0, z1, t):
    z0_n, z1_n = z0/ z0.norm(), z1/ z1.norm()
    omega = (z0_n * z1_n).sum().arccos()
    so = omega.sin()
    return (omega-t*omega).sin()/so * z0 + (t*omega).sin()/so * z1

# Instantiate SDXL's VAE
with torch.no_grad():
    # vae:AutoencoderKL = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        "/home/t-ryanteehan/diffusion-exploration/dsprites_autoencoder/checkpoint-7500/autoencoderkl"
    )

    vae.to(dtype=torch.float32) # otherwise it produces NaNs, even madebyollin's VAE
    vae.to(device="cuda")

    assert vae.device == torch.device("cuda:0")
    assert vae.dtype == torch.float32

    square_img = d['train'].filter(lambda ex: ex['label_shape'] == 0)[0]['image']

    heart_img = d['train'].filter(lambda ex: ex['label_shape'] == 2)[0]['image']

    # make image as tensor
    # img = Image.open("avenger.jpg")  # Replace with your actual image path
    image_transforms = transforms.Compose(
        [
            transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    img = square_img
    # img_tensor = pil_to_tensor(img).unsqueeze(0) / 255.0
    img_tensor = image_transforms(img).unsqueeze(0)
    img_tensor = img_tensor.to(vae.device)
    img_tensor = img_tensor.to(vae.dtype)

    # get the inferred latent distribution
    latent_dist: DiagonalGaussianDistribution = vae.encode(img_tensor, return_dict=False)[0]
    print(f"{latent_dist.mean.shape=} {latent_dist.std.shape=} {latent_dist.mean.mean()=} {latent_dist.std.mean()=}")

    assert not latent_dist.mean.isnan().any()
    assert not latent_dist.std.isnan().any()
    assert latent_dist.deterministic is False

    heart_tensor = image_transforms(heart_img).unsqueeze(0)
    heart_tensor = heart_tensor.to(vae.device).to(vae.dtype)

    heart_latent_dist = vae.encode(heart_tensor, return_dict=False)[0]


    square_mean = latent_dist.mean
    heart_mean = heart_latent_dist.mean

    steps = 10
    interps = [slerp(square_mean, heart_mean, t) for t in torch.linspace(0, 1, steps)]
    interp_imgs = [vae.decode(z).sample for z in interps]
    interp_imgs = [i.squeeze(0).cpu().detach() for i in interp_imgs]
    interp_imgs = [(i *  0.5 + 0.5).clamp(0, 1) for i in interp_imgs]

    square_img.save(f"{output_dir}/square_image.png")
    heart_img.save(f"{output_dir}/heart_img.png")

    for idx, inter_img in enumerate(interp_imgs):
        pil_img = transforms.ToPILImage()(inter_img)
        pil_img.save(f"{output_dir}/interpolated_image_step_{idx}.png")


    # # -- Tried with scale factor and add noise--
    # scale_factor = 5.0  # increased the variations
    # noise_strength = 0.2  # add noise will help further perturb the latent space
    # noise = noise_strength * torch.randn_like(latent_dist.mean).to(vae.device)

    # # generate new latents with added noise and scaling
    # sample_1 = latent_dist.mean + scale_factor * latent_dist.std * torch.randn_like(latent_dist.mean).to(vae.device) + noise
    # sample_2 = latent_dist.mean + scale_factor * latent_dist.std * torch.randn_like(latent_dist.mean).to(vae.device) + noise
    # assert not sample_1.isnan().any()
    # assert not sample_2.isnan().any()
    # assert (sample_1 != sample_2).any(), "samples should be different"

    # print(f"{sample_1.shape=}")

    # assert vae.dtype == sample_1.dtype
    # assert vae.device == sample_1.device

    # # decode the sampled latents back to images
    # img_1: torch.Tensor = vae.decode(sample_1).sample  # Decoding the first variation
    # img_1 = img_1.squeeze(0).cpu().detach()
    # assert (img_1 != img_tensor.cpu().detach()).any(), "generated image should be different from the input image"
    
    # # save first variation
    # img_1_pil = transforms.ToPILImage()(img_1)
    # img_1_pil.save("sample2_1.png")

    
    # # save second variation 
    # img_2: torch.Tensor = vae.decode(sample_2).sample
    # img_2 = img_2.squeeze(0).cpu().detach()
    # img_2_pil = transforms.ToPILImage()(img_2)
    # img_2_pil.save("sample2_2.png")
    
    # # -- Try interpolation between Latent vectors --
    # t = torch.rand(1).item()  # generates a random interpolation factor between 0 - 1
    # interpolated_sample = (1 - t) * sample_1 + t * sample_2
    
    # # Decode the interpolated sample --> image
    # img_interpolated = vae.decode(interpolated_sample).sample
    # img_interpolated = img_interpolated.squeeze(0).cpu().detach()
    # img_interpolated = (img_interpolated * 0.5 + 0.5).clamp(0, 1)
    # img_interpolated_pil = transforms.ToPILImage()(img_interpolated)
    # img_interpolated_pil.save("interpolated_variation2.png")
    # print("Done")

