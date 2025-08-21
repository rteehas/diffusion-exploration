from diffusers import AutoencoderKL
import torch
import torchvision
from torchvision.datasets.utils import download_and_extract_archive
from torchvision import transforms
from dsprites import *
from tqdm import tqdm
# https://github.com/huggingface/diffusers/issues/437#issuecomment-1356945792

num_workers = 1
batch_size = 1024
# From https://github.com/fastai/imagenette

torch.manual_seed(0)
torch.set_grad_enabled(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained_model_name_or_path ="/home/t-ryanteehan/diffusion-exploration/dsprites_autoencoder_small_res/checkpoint-3000/autoencoderkl"
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path,
    subfolder='vae',
    revision=None,
)
vae.to(device)

size = 64
image_transforms = transforms.Compose([
    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = prepare_dsprites()

def preprocess_train(examples):
    images = [image for image in examples["image"]]
    # images = [image.convert("RGB") for image in examples[image_column]]
    images = [image_transforms(image) for image in images]

    examples["pixel_values"] = images

    return examples

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    return {"pixel_values": pixel_values}

dataset = dataset['train'].with_transform(preprocess_train)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

all_latents = []
with torch.no_grad():
    for batch in tqdm(loader):
        image_data = batch['pixel_values'].to(device)
        latents = vae.encode(image_data).latent_dist.sample()
        all_latents.append(latents.cpu())

    all_latents_tensor = torch.cat(all_latents)
    std = all_latents_tensor.std().item()
    normalizer = 1 / std
print(f'{normalizer = }')