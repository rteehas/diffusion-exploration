from train_text_to_image import *
from train_dsprites_classifier import *
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

device = "cuda"

unet_path = "/home/t-ryanteehan/diffusion-exploration/dsprites_1.0_new_unet_config_subsample_verbose_captions_prompt_dropout/checkpoint-1584/unet"
#"/home/t-ryanteehan/diffusion-exploration/dsprites_1.0_new_unet_config_subsample_verbose_captions_prompt_dropout/checkpoint-1584/unet"
# "/home/t-ryanteehan/diffusion-exploration/dsprites_0.0_new_unet_config_subsample_verbose_captions/checkpoint-864/unet"
vae_path = "/home/t-ryanteehan/diffusion-exploration/dsprites_autoencoder_small_res/checkpoint-3000/autoencoderkl"
#"/home/t-ryanteehan/diffusion-exploration/dsprites_autoencoder/checkpoint-7500/autoencoderkl"
#"/home/t-ryanteehan/diffusion-exploration/dsprites_autoencoder_small_res/checkpoint-3000/autoencoderkl"
text_encoder = CLIPTextModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="text_encoder"
).to(device)

vae = vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae").to(device)
unet = UNet2DConditionModel.from_pretrained(unet_path).to(device)

prompts = get_validation_prompts(n=10)

knockout_prompts = [p for p in prompts if "ellipse" in p]
heart_prompts = [p for p in prompts if "heart" in p]
square_prompts = [p for p in prompts if "square" in p]

label_map = {"square": 0, "ellipse": 1, "heart": 2}


reward_model = SimpleCNN()
reward_model.load_state_dict(torch.load("/home/t-ryanteehan/diffusion-exploration/dsprites_classifier/checkpoint_20412.pth"))
reward_model = reward_model.to(device)
reward_model.eval()
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", text_encoder=text_encoder, vae=vae, unet=unet, revision=None, variant=None, requires_safety_checker = False, safety_checker=None).to(device)

weight_dtype=torch.bfloat16

pipeline.torch_dtype = weight_dtype
pipeline.set_progress_bar_config(disable=True)

train_transforms = transforms.Compose(
    [
        transforms.Resize(64),  # Use dynamic interpolation method
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


wandb.init(project="dsprites_rewards")
def get_rewards_and_latents(prompts, shape_name, is_knockout=False):

    latents = []
    rewards = []
    images = []
    for p in tqdm(prompts):
        image = pipeline(p, num_inference_steps=50, width=64, height=64).images[0]
        images.append(image)

        pixels = train_transforms(image).to(device).unsqueeze(0)
        latent = vae.encode(pixels).latent_dist.sample()
        latents.append(torch.flatten(latent).cpu())

        reward = reward_model(pixels)
        print("pixel shape", pixels.shape)
        print("rew", reward, shape_name, label_map[shape_name])
        print(reward[0,label_map[shape_name]])
        rewards.append(reward[0,label_map[shape_name]].cpu())
    
    if not is_knockout:
        wandb.log({f"{shape_name}/prompts and rewards": [wandb.Image(img, caption=f"Prompt: {prompt}\n Reward: {reward.item()}") for img, prompt, reward in zip(images, prompts, rewards)]})
    else:
        wandb.log({f"{shape_name} (KO)/prompts and rewards": [wandb.Image(img, caption=f"Prompt: {prompt}\n Reward: {reward.item()}") for img, prompt, reward in zip(images, prompts, rewards)]})

    return rewards, latents


def get_eigenspectrum(latents):
    cov = latents.T.cov()
    eigenvalues = torch.linalg.eigh(cov).eigenvalues
    eigenvals, _ = torch.sort(eigenvalues, descending=True)
    return eigenvals

knockout_rewards, knockout_latents = get_rewards_and_latents(knockout_prompts, "ellipse", is_knockout=True)
heart_rewards, heart_latents = get_rewards_and_latents(heart_prompts, "heart")
square_rewards, square_latents = get_rewards_and_latents(square_prompts, "square")

knockout_latents_tensor = torch.stack(knockout_latents)
# print(knockout_latents_tensor.shape)
# print(knockout_latents_tensor.T.cov().shape)
# print(torch.linalg.eigh(knockout_latents_tensor.T.cov()))
knockout_rewards_tensor = torch.stack(knockout_rewards)

heart_rewards_tensor = torch.stack(heart_rewards)
heart_latents_tensor = torch.stack(heart_latents)

square_rewards_tensor = torch.stack(square_rewards)
square_latents_tensor = torch.stack(square_latents)

knockout_latent_std = knockout_latents_tensor.std().item()
knockout_rewards_std = knockout_rewards_tensor.std().item()

heart_latent_std = heart_latents_tensor.std().item()
heart_rewards_std = heart_rewards_tensor.std().item()

square_latent_std = square_latents_tensor.std().item()
square_rewards_std = square_rewards_tensor.std().item()

wandb.log({
    "ellipse (KO)/reward std": knockout_rewards_std,
    "ellipse (KO)/latent std": knockout_latent_std,
    "heart/reward std": heart_rewards_std,
    "heart/latent std": heart_latent_std,
    "square/reward std": square_rewards_std,
    "square/latent std": square_latent_std,
    "sample count": len(prompts)
})


knockout_eigenspectrum = get_eigenspectrum(knockout_latents_tensor)
heart_eigenspectrum = get_eigenspectrum(heart_latents_tensor)
square_eigenspectrum = get_eigenspectrum(square_latents_tensor)

# fig, ax = plt.subplots(figsize=(6,4))
# ax.plot(knockout_eigenspectrum.detach().numpy(), label="Ellipse")
# ax.plot(heart_eigenspectrum.detach().numpy(), label="Heart")
# ax.plot(square_eigenspectrum.detach().numpy(), label="Square")
# ax.set_yscale('log')                       # spectra usually heavy-tailed
# ax.set_xlabel('rank')
# ax.set_ylabel('eigenvalue (log scale)')
# ax.set_title('Covariance eigenspectrum')
# ax.legend()
# fig.tight_layout()

# wandb.log({"Eigenspectrum": wandb.Image(fig)})

fig, ax = plt.subplots(figsize=(6,4))
ax.hist(knockout_eigenspectrum.detach().numpy(), bins=50, histtype="stepfilled", alpha=.5, label="Ellipse")
ax.hist(heart_eigenspectrum.detach().numpy(), bins=50, histtype="stepfilled", alpha=.5, label="Heart")
ax.hist(square_eigenspectrum.detach().numpy(), bins=50, histtype="stepfilled", alpha=.5, label="Square")
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.set_xscale('log')
ax.set_title('Eigenvalue Histogram')
ax.legend()
fig.tight_layout()

wandb.log({"Histogram": wandb.Image(fig)})


plt.clf()
fig, ax = plt.subplots(figsize=(6,4))
ax.hist([r.detach() for r in knockout_rewards], bins=10,histtype="stepfilled", alpha=.5, label="Ellipse")
ax.hist([r.detach() for r in heart_rewards],bins=10,histtype="stepfilled", alpha=.5, label="Heart")
ax.hist([r.detach() for  r in square_rewards], bins=10, histtype="stepfilled", alpha=.5, label="Square")
ax.set_xlabel("Reward")
ax.set_ylabel("Frequency")
ax.set_title("Reward Histogram")
ax.legend()
fig.tight_layout()

wandb.log({"Reward Histogram": wandb.Image(fig)})


plt.clf()
fig, ax = plt.subplots(figsize=(6,4))
categories = ["Ellipse", "Heart", "Square"]
values = [knockout_rewards_tensor.mean().item(), heart_rewards_tensor.mean().item(), square_rewards_tensor.mean().item()]
stds = [knockout_rewards_std, heart_rewards_std, square_rewards_std]
colors = ['steelblue', 'orange', 'forestgreen']

ax.bar(categories, values, color=colors, yerr=stds, capsize=5)
ax.set_xlabel("Class")
ax.set_ylabel("Reward")
ax.set_title("Mean Reward by Class")
ax.legend()
fig.tight_layout()

wandb.log({"Reward Means": wandb.Image(fig)})


