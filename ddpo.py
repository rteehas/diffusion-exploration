# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python ddpo.py \
    --num_epochs=200 \
    --train_gradient_accumulation_steps=1 \
    --sample_num_steps=50 \
    --sample_batch_size=6 \
    --train_batch_size=3 \
    --sample_num_batches_per_epoch=4 \
    --per_prompt_stat_tracking=True \
    --per_prompt_stat_tracking_buffer_size=32 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb"
"""

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, HfArgumentParser, is_torch_npu_available, is_torch_xpu_available

from trl import DDPOConfig, DefaultDDPOStableDiffusionPipeline
from modified_ddpo_trainer import ModifiedDDPOTrainer
from dsprites import *
from train_dsprites_classifier import SimpleCNN
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline, DDIMScheduler
import wandb
from functools import partial
from diffusers.utils import numpy_to_pil
import itertools

@dataclass
class ModifiedDDPOConfig(DDPOConfig):
    hard_rewards: bool = field(default=False)
    zero_classifier: bool = field(default=False)
    knockout: bool = field(default=False)
    no_ellipse_prompts: bool = field(default=False)
    amlt: bool = field(default=False)

@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        pretrained_model (`str`, *optional*, defaults to `"runwayml/stable-diffusion-v1-5"`):
            Pretrained model to use.
        pretrained_revision (`str`, *optional*, defaults to `"main"`):
            Pretrained model revision to use.
        hf_hub_model_id (`str`, *optional*, defaults to `"ddpo-finetuned-stable-diffusion"`):
            HuggingFace repo to save model weights to.
        hf_hub_aesthetic_model_id (`str`, *optional*, defaults to `"trl-lib/ddpo-aesthetic-predictor"`):
            Hugging Face model ID for aesthetic scorer model weights.
        hf_hub_aesthetic_model_filename (`str`, *optional*, defaults to `"aesthetic-model.pth"`):
            Hugging Face model filename for aesthetic scorer model weights.
        use_lora (`bool`, *optional*, defaults to `True`):
            Whether to use LoRA.
    """

    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5", metadata={"help": "Pretrained model to use."}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "Pretrained model revision to use."})
    hf_hub_model_id: str = field(
        default="ddpo-finetuned-stable-diffusion", metadata={"help": "HuggingFace repo to save model weights to."}
    )
    hf_hub_aesthetic_model_id: str = field(
        default="trl-lib/ddpo-aesthetic-predictor",
        metadata={"help": "Hugging Face model ID for aesthetic scorer model weights."},
    )
    hf_hub_aesthetic_model_filename: str = field(
        default="aesthetic-model.pth",
        metadata={"help": "Hugging Face model filename for aesthetic scorer model weights."},
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})
    # hard_rewards: bool = field(default=False)


# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(768, 1024),
#             nn.Dropout(0.2),
#             nn.Linear(1024, 128),
#             nn.Dropout(0.2),
#             nn.Linear(128, 64),
#             nn.Dropout(0.1),
#             nn.Linear(64, 16),
#             nn.Linear(16, 1),
#         )

#     @torch.no_grad()
#     def forward(self, embed):
#         return self.layers(embed)


# class AestheticScorer(torch.nn.Module):
#     """
#     This model attempts to predict the aesthetic score of an image. The aesthetic score
#     is a numerical approximation of how much a specific image is liked by humans on average.
#     This is from https://github.com/christophschuhmann/improved-aesthetic-predictor
#     """

#     def __init__(self, *, dtype, model_id, model_filename):
#         super().__init__()
#         self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
#         self.mlp = MLP()
#         try:
#             cached_path = hf_hub_download(model_id, model_filename)
#         except EntryNotFoundError:
#             cached_path = os.path.join(model_id, model_filename)
#         state_dict = torch.load(cached_path, map_location=torch.device("cpu"), weights_only=True)
#         self.mlp.load_state_dict(state_dict)
#         self.dtype = dtype
#         self.eval()

#     @torch.no_grad()
#     def __call__(self, images):
#         device = next(self.parameters()).device
#         inputs = self.processor(images=images, return_tensors="pt")
#         inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
#         embed = self.clip.get_image_features(**inputs)
#         # normalize embedding
#         embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
#         return self.mlp(embed).squeeze(1)


# def aesthetic_scorer(hub_model_id, model_filename):
#     scorer = AestheticScorer(
#         model_id=hub_model_id,
#         model_filename=model_filename,
#         dtype=torch.float32,
#     )
#     if is_torch_npu_available():
#         scorer = scorer.npu()
#     elif is_torch_xpu_available():
#         scorer = scorer.xpu()
#     else:
#         scorer = scorer.cuda()

#     def _fn(images, prompts, metadata):
#         images = (images * 255).round().clamp(0, 255).to(torch.uint8)
#         scores = scorer(images)
#         return scores, {}

#     return _fn
class InverseNormalize(torch.nn.Module):
    def __init__(self, mean=(0.5, 0.5), std=(0.5, 0.5)):
        super().__init__()
        self.mean = torch.as_tensor(mean).view(-1, 1, 1)
        self.std  = torch.as_tensor(std).view(-1, 1, 1)

    def forward(self, tensor):
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)

def largest_white_region(img: torch.Tensor, connectivity: int = 4):
    """
    img: 2-D uint8/float/bool tensor, 0 = black, non-zero = white
    connectivity: 4 or 8
    Returns
        comp_mask – bool tensor, True for pixels in the largest component
        comp_size – int, number of pixels in that component
    """
    mask = img.bool()
    H, W = mask.shape
    visited = torch.zeros_like(mask, dtype=torch.bool)
    labels  = torch.zeros_like(mask, dtype=torch.int32)

    neigh = [(0,1),(1,0),(-1,0),(0,-1)] if connectivity == 4 else \
            [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    label_id = 0
    sizes = []
    for y in range(H):
        for x in range(W):
            if mask[y, x] and not visited[y, x]:
                label_id += 1
                stack = [(y, x)]
                size = 0
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    labels[cy, cx] = label_id
                    size += 1
                    for dy, dx in neigh:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            if mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                sizes.append(size)

    if not sizes:                        # no white pixels
        return torch.zeros_like(mask), 0

    largest = torch.argmax(torch.tensor(sizes)) + 1   # labels start at 1
    comp_mask = labels.eq(largest)
    return comp_mask, sizes[largest - 1]

def largest_white_region_ch(img3: torch.Tensor, connectivity: int = 4):
    """
    img3: (C,H,W) tensor, C==1. 0 = black, non-zero = white.
    Returns
        comp_mask – (1,H,W) bool tensor marking the largest white component
        comp_size – int
    """
    assert img3.ndim == 3 and img3.shape[0] == 1, "expect (1,H,W)"
    mask2d, size = largest_white_region(img3[0], connectivity)
    return mask2d.unsqueeze(0), size


def size_reward(pixel_values):
    return pixel_values.mean()

def connected_component_size_reward(pixel_values):
    sizes = []
    for p in pixel_values:
        # print(p.shape)
        m, s = largest_white_region_ch(p, 8)
        sizes.append(torch.tensor(s) / p.numel())
    
    return torch.stack(sizes)



def convert_pt_output_to_pil(sample):
    # based on this https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/blip_diffusion/blip_image_processing.py#L301
    sample = sample.cpu().permute(0, 2, 3, 1).numpy()
    sample = numpy_to_pil(sample)
    return sample

def build_reward_fn(accelerator, image_processor, args):
    """Closure so we can reach the accelerator inside compute_reward."""

    classifier = SimpleCNN()
    # print("before", classifier.fc2.weight)
    if args.amlt:
        classifier.load_state_dict(torch.load("/mnt/dsprites-checkpoints/dsprites_classifier/dsprites_classifier/checkpoint_20412.pth"))
    else:
        classifier.load_state_dict(torch.load("/home/t-ryanteehan/diffusion-exploration/dsprites_classifier/checkpoint_20412.pth"))
    # print("after", classifier.fc2.weight)
    classifier = classifier.to(accelerator.device)
    
    def reward_fn(images, prompts, prompt_metadata, step, log=True):
        train_transforms = transforms.Compose(
            [
                transforms.Resize(64),  # Use dynamic interpolation method
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        transforms_for_size = transforms.Compose(
            [
                transforms.Resize(64),  # Use dynamic interpolation method
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
            ]
        )
        # print("images", images.shape)
        # print("meta", prompt_metadata)
        # original parameters
        
        inv_norm = InverseNormalize().to(accelerator.device)
        classifier.eval()
        imgs = convert_pt_output_to_pil(images)
        # imgs = [image_processor.postprocess(img.unsqueeze(0), output_type="pil", do_denormalize=[True] * img.shape[0])[0] for img in images ] #gross redundant hack stuff, fix later
        # imgs = image_processor.postprocess(
        #     images,          # tensor shape (B, C, H, W)
        #     output_type="pil",
        #     do_denormalize=[True] * images.shape[0]
        # )
        # print("imgs",imgs)
        pixels = torch.stack([train_transforms(img) for img in imgs]).to(accelerator.device).to(memory_format=torch.contiguous_format)
        pixels_for_size = torch.stack([transforms_for_size(img) for img in imgs]).to(accelerator.device).to(memory_format=torch.contiguous_format)
        with torch.no_grad():
            # pixels = images.float() #torch.stack([img for img in images]).to(accelerator.device)
            # ── compute the two components ─────────────────────────────── #
            labels = torch.stack([torch.tensor(meta['label']) for meta in prompt_metadata]).to(accelerator.device)


            # size_reward = torch.stack([transforms.ToTensor()(img).mean() for img in imgs]).to(accelerator.device)
            # pixels = torch.stack([train_transforms(img.convert("L")) for img in imgs]).to(accelerator.device).to(memory_format=torch.contiguous_format).float()
            # size_reward = pixels_for_size.mean(dim=(1,2,3))
            # print("p for",pixels_for_size)
            size_reward = connected_component_size_reward(pixels_for_size).to(accelerator.device).float()
            # print("pixel shape", pixels.shape)
            print("size", size_reward)
            # print("inside func", classifier.fc2.weight)
            classifier_output = classifier(pixels)
            # exit()
            # print("one by one")
            # for p in pixels:
            #     print(p.shape)
            #     print("inner rew",classifier(p.unsqueeze(0)))

            if args.hard_rewards:
                _, preds = classifier_output.max(1)
                print("preds", preds)
                print("labels", labels)
                print("output", classifier_output)
                classifier_rewards = preds.eq(labels)
            else:
                classifier_rewards = classifier_output.softmax(-1)[torch.arange(classifier_output.shape[0]), labels]
                _, preds = classifier_output.max(1)
                print("preds", preds)
                print("labels", labels)
                print("output", classifier_output)

            if args.zero_classifier:
                classifier_rewards = torch.zeros_like(classifier_rewards).to(accelerator.device)

        total_r = size_reward + classifier_rewards
        # exit()
        # clip_r   = clip_alignment(images, prompts)     # shape [B]
        # aesth_r  = aesthetic_score(images)            # shape [B]
        # total_r  = clip_r + aesth_r                   # shape [B]

        # ── log the per-batch means - only once, on the main process ─ #
        if accelerator.is_main_process and log:
            if args.hard_rewards:
                accelerator.log(
                    {
                        "reward/classifier"   : classifier_rewards.sum().item(),
                        "reward/size"    : size_reward.mean().item(),
                        "reward/total"        : total_r.mean().item(),
                    },
                    step=step,                           # current global_step
                )
            else:
                accelerator.log(
                    {
                        "reward/classifier"   : classifier_rewards.mean().item(),
                        "reward/size"    : size_reward.mean().item(),
                        "reward/total"        : total_r.mean().item(),
                    },
                    step=step,                           # current global_step
                )

        return total_r, {"size": size_reward, "classifier": classifier_rewards}                               # DDPO expects 1-D tensor/list
    
    return reward_fn

class PromptIterator:

    def __init__(self, prompts):
        self.label_map = {"square": 0, "ellipse": 1, "heart": 2}
        self.prompts = prompts
    
    def __call__(self):
        prompt = np.random.choice(self.prompts)
        shape = None
        for key in self.label_map:
            if key in prompt:
                shape = key
        
        label = self.label_map[shape]

        return prompt, {"label": label}
    
def get_labels_for_prompt(prompt):
    label_map = {"square": 0, "ellipse": 1, "heart": 2}
    shape = None
    for key in label_map:
        if key in prompt:
            shape = key
    
    label = label_map[shape]
    return {"label": label}

prompts = get_validation_prompts(n=1)
# prompts = [prompts[-1]]

def collate_image_data(image_data):
    """
    Collapse the structures returned by
        images, prompts, prompt_metas, rewards, rewards_meta = zip(*image_data)
    back into flat tensors/tuples/dicts.
    """
    images, prompts, prompt_metas, rewards, rewards_meta = zip(*image_data)

    # tensors → concatenate on the first (batch) dimension
    images = torch.cat(images, dim=0)
    rewards = torch.cat(rewards, dim=0)

    # tuples -> flatten
    prompts = tuple(itertools.chain.from_iterable(prompts))
    prompt_metas = tuple(itertools.chain.from_iterable(prompt_metas))

    # metadata dicts -> concatenate values key-wise
    merged_reward_meta = {}
    for d in rewards_meta:
        for k, v in d.items():
            merged_reward_meta.setdefault(k, []).append(v)
    merged_reward_meta = {k: torch.cat(v_list, dim=0) for k, v_list in merged_reward_meta.items()}

    return images, prompts, prompt_metas, rewards, merged_reward_meta

def image_outputs_logger(image_data, global_step, accelerate_logger, name="samples"):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    images, prompts, _, rewards, rewards_meta = collate_image_data(image_data) #zip(*image_data)

    accelerate_logger.log(
    {
        name: [
            wandb.Image(image, caption=f"Prompt: {prompts[i]}\nTotal Reward: {rewards[i].item()}\nSize Reward: {rewards_meta['size'][i].item()}\nClassifier Reward: {rewards_meta['classifier'][i].item()}")
            for i, image in enumerate(images)
        ]
    },
    step=global_step
    )

def eval_logger(image_data, global_step, accelerate_logger, eval_name):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    images, prompts, _, rewards, _ = image_data[-1]

    # for i, image in enumerate(images):
    #     prompt = prompts[i]
    #     reward = rewards[i].item()
    #     result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()

    accelerate_logger.log(
    {
        f"{eval_name}/samples": [
            wandb.Image(image, caption=f"Prompt: {prompts[i]}\nReward: {rewards[i].item()}")
            for i, image in enumerate(images)
        ]
    },
    step=global_step
    )


    # accelerate_logger.log_images(
    #     result,
    #     step=global_step,
    # )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, ModifiedDDPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.no_ellipse_prompts:
        prompts = [p for p in prompts if "ellipse" not in p]
    
    p_str = "\n".join(prompts)
    print(f"The training prompts are {p_str}")
    training_args.project_kwargs = {
        "logging_dir": f"./rl_logs_hard_{training_args.hard_rewards}_inv_norm_with_classifier_{not training_args.zero_classifier}_knockout_{training_args.knockout}",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": f"./rl_save_hard_{training_args.hard_rewards}_inv_norm_with_classifier_{not training_args.zero_classifier}_knockout_{training_args.knockout}",
    }

    pipeline = DefaultDDPOStableDiffusionPipeline(
        pretrained_model_name= "CompVis/stable-diffusion-v1-4",
        use_lora=False,
        
    )
    print("Training Args")
    print(training_args)
    if training_args.amlt:
        base = "/mnt/dsprites-checkpoints"
    else:
        base = "/home/t-ryanteehan/diffusion-exploration"
    if training_args.knockout:
        if training_args.amlt:
            unet_path = f"{base}/dsprites_0.0_new_vae_v_pred/dsprites_0.0_new_vae_v_pred/checkpoint-69120/unet_ema"
            vae_path = f"{base}/dsprites_autoencoder/dsprites_autoencoder/checkpoint-7500/autoencoderkl"
        else:
            unet_path = f"{base}/dsprites_0.0_new_vae_v_pred/checkpoint-69120/unet_ema"
            vae_path = f"{base}/dsprites_autoencoder/checkpoint-7500/autoencoderkl"
    else:
        unet_path = f"{base}/dsprites_1.0_new_unet_config_subsample_verbose_captions_prompt_dropout/checkpoint-1584/unet"
        #"/home/t-ryanteehan/diffusion-exploration/dsprites_0.0_new_vae_v_pred/checkpoint-69120/unet_ema"
        #"/home/t-ryanteehan/diffusion-exploration/dsprites_1.0_new_unet_config_subsample_verbose_captions_prompt_dropout/checkpoint-1584/unet"
        # "/home/t-ryanteehan/diffusion-exploration/dsprites_0.0_new_unet_config_subsample_verbose_captions/checkpoint-864/unet"
        vae_path = f"{base}/dsprites_autoencoder_small_res/checkpoint-3000/autoencoderkl"
        #"/home/t-ryanteehan/diffusion-exploration/dsprites_autoencoder/checkpoint-7500/autoencoderkl"
        #"/home/t-ryanteehan/diffusion-exploration/dsprites_autoencoder_small_res/checkpoint-3000/autoencoderkl"
    text_encoder = CLIPTextModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="text_encoder"
    )

    

    vae = vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(unet_path)
    sched = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    sd_pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", 
                                                          text_encoder=text_encoder, 
                                                          vae=vae, 
                                                          unet=unet,
                                                          scheduler=sched, 
                                                          revision=None, 
                                                          variant=None, 
                                                          requires_safety_checker=False, 
                                                          safety_checker=None)
    pipeline.sd_pipeline = sd_pipeline
    pipeline.sd_pipeline.vae.requires_grad_(False)
    pipeline.sd_pipeline.text_encoder.requires_grad_(False)
    pipeline.sd_pipeline.unet.requires_grad_(True)
    trainer = ModifiedDDPOTrainer(
        training_args,
        partial(build_reward_fn, args=training_args),
        PromptIterator(prompts),
        pipeline,
        image_samples_hook=image_outputs_logger,
        eval_prompts=get_validation_prompts(n=1),
        knockout_class="ellipse"
    )
    # trainer.config.sample_guidance_scale=1.0

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)