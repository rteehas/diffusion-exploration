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
python examples/scripts/ddpo.py \
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
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser, is_torch_npu_available, is_torch_xpu_available

from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
from aesthetics_ddpo_trainer import AestheticsDDPOTrainer
from ddpo import ModifiedDDPOConfig, ScriptArguments, convert_pt_output_to_pil, collate_image_data
import ImageReward as RM
import wandb
import timm
from collections import defaultdict
import random
# @dataclass
# class ScriptArguments:
#     r"""
#     Arguments for the script.

#     Args:
#         pretrained_model (`str`, *optional*, defaults to `"runwayml/stable-diffusion-v1-5"`):
#             Pretrained model to use.
#         pretrained_revision (`str`, *optional*, defaults to `"main"`):
#             Pretrained model revision to use.
#         hf_hub_model_id (`str`, *optional*, defaults to `"ddpo-finetuned-stable-diffusion"`):
#             HuggingFace repo to save model weights to.
#         hf_hub_aesthetic_model_id (`str`, *optional*, defaults to `"trl-lib/ddpo-aesthetic-predictor"`):
#             Hugging Face model ID for aesthetic scorer model weights.
#         hf_hub_aesthetic_model_filename (`str`, *optional*, defaults to `"aesthetic-model.pth"`):
#             Hugging Face model filename for aesthetic scorer model weights.
#         use_lora (`bool`, *optional*, defaults to `True`):
#             Whether to use LoRA.
#     """

#     pretrained_model: str = field(
#         default="runwayml/stable-diffusion-v1-5", metadata={"help": "Pretrained model to use."}
#     )
#     pretrained_revision: str = field(default="main", metadata={"help": "Pretrained model revision to use."})
#     hf_hub_model_id: str = field(
#         default="ddpo-finetuned-stable-diffusion", metadata={"help": "HuggingFace repo to save model weights to."}
#     )
#     hf_hub_aesthetic_model_id: str = field(
#         default="trl-lib/ddpo-aesthetic-predictor",
#         metadata={"help": "Hugging Face model ID for aesthetic scorer model weights."},
#     )
#     hf_hub_aesthetic_model_filename: str = field(
#         default="aesthetic-model.pth",
#         metadata={"help": "Hugging Face model filename for aesthetic scorer model weights."},
#     )
#     use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    """
    This model attempts to predict the aesthetic score of an image. The aesthetic score
    is a numerical approximation of how much a specific image is liked by humans on average.
    This is from https://github.com/christophschuhmann/improved-aesthetic-predictor
    """

    def __init__(self, *, dtype, model_id, model_filename):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        try:
            cached_path = hf_hub_download(model_id, model_filename)
        except EntryNotFoundError:
            cached_path = os.path.join(model_id, model_filename)
        state_dict = torch.load(cached_path, map_location=torch.device("cpu"), weights_only=True)
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


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
#         return scores, {"aesthetics": scores}

#     return _fn

def aesthetic_scorer():
    model = RM.load("ImageReward-v1.0").to("cuda")
    timm_model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True).to("cuda")
    timm_model = timm_model.eval()
    data_config = timm.data.resolve_model_data_config(timm_model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    def _fn(images, prompts, metadata):
        scores = []
        classifier_scores = []
        images = convert_pt_output_to_pil(images.float())
        labels = [torch.tensor(meta['label']) for meta in metadata]

        per_prompt = defaultdict(list)
        for img, prompt, label in zip(images, prompts, labels):
            with torch.no_grad():
                score = model.score(prompt, img)
            scores.append(score)
            per_prompt[prompt].append(score)
            with torch.no_grad():
                timm_output = timm_model(transforms(img).unsqueeze(0).to("cuda"))
                # top5_probabilities, top5_class_indices = torch.topk(timm.softmax(dim=1) * 100, k=5)
                # print("timm shape", timm_output.softmax(dim=1).shape)
                classifier_score = timm_output.softmax(dim=1)[:,label]
                # print("class score", classifier_score)
                classifier_scores.append(classifier_score)
        for p in per_prompt:
            per_prompt[p] = torch.tensor(per_prompt[p], device="cuda")

        meta_dict = {"aesthetics": torch.tensor(scores, device="cuda"), "classifier": torch.tensor(classifier_scores, device="cuda")}

        for p in per_prompt:
            meta_dict[p] = per_prompt[p]
        return scores, meta_dict

    return _fn
# list of example prompts to feed stable diffusion
animals = [
    "cat",
    "dog",
    # "horse",
    # "monkey",
    # "rabbit",
    # "zebra",
    # "spider",
    # "bird",
    # "sheep",
    # "deer",
    # "cow",
    # "goat",
    # "lion",
    # "frog",
    # "chicken",
    # "duck",
    # "goose",
    # "bee",
    # "pig",
    # "turkey",
    # "fly",
    # "llama",
    # "camel",
    # "bat",
    # "gorilla",
    # "hedgehog",
    # "kangaroo",
]

# prompts = ["spotted salamander", "English foxhound", "barracouta", "pickup", "monitor", "grocery store"]
# true_idxs = [28, 167, 389, 717, 664, 582]
class PromptIterator:

    def __init__(self, prompts, true_idxs, batch_size=300):
        print(f"Iterator Batch Size = {batch_size}")
        self.label_map = {p: idx for (p, idx) in zip(prompts, true_idxs)}
        self.prompts = prompts
        self.idxs = true_idxs

        assert batch_size % len(self.prompts) == 0

        num_per_prompt = batch_size // len(self.prompts)

        prompt_batch = []
        for p in prompts:
            for i in range(num_per_prompt):
                prompt_batch.append(p)
        random.shuffle(prompt_batch)
        self.prompt_batch = prompt_batch
        self.iter = 0 
    
    def __call__(self):
        # prompt = np.random.choice(self.prompts)
        prompt = self.prompt_batch[self.iter]
        shape = None
        for key in self.label_map:
            if key in prompt:
                shape = key
        
        label = self.label_map[shape]
        self.iter += 1
        if self.iter == len(self.prompt_batch):
            self.iter = 0

        return prompt, {"label": label}

# def image_outputs_logger(image_data, global_step, accelerate_logger):
#     # For the sake of this example, we will only log the last batch of images
#     # and associated data
#     result = {}
#     images, prompts, _, rewards, _ = image_data[-1]

#     for i, image in enumerate(images):
#         prompt = prompts[i]
#         reward = rewards[i].item()
#         result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()

#     accelerate_logger.log_images(
#         result,
#         step=global_step,
#     )
def image_outputs_logger(image_data, global_step, accelerate_logger, name="samples"):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    images, prompts, _, rewards, rewards_meta = image_data[-1]# collate_image_data(image_data) #zip(*image_data)

    accelerate_logger.log(
    {
        name: [
            wandb.Image(image.float(), caption=f"Prompt: {prompts[i]}\nAesthetic Reward: {rewards[i].item()}")
            for i, image in enumerate(images)
        ],
        "global_step": global_step
    },
    
    )

def remove_at_index(lst, idx):
    if idx < 0 or idx >= len(lst):
        raise IndexError("Index out of range")
    return lst[:idx] + lst[idx+1:]

if __name__ == "__main__":
    # test = torch.randn((10,10))
    # torch.save(test, "/mnt/outputs/test.pth")
    # exit()
    parser = HfArgumentParser((ScriptArguments, ModifiedDDPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": "./save",
    }
    print("before pipeline")
    # exit()
    pipeline = DefaultDDPOStableDiffusionPipeline(
        "runwayml/stable-diffusion-v1-5",
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )
    # print("before trainer")
    # tail_classes=["spotted salamander", "English foxhound", "barracouta"]
    # tail_idxs = [28, 167, 389]
    # head_classes = ["pickup", "monitor", "grocery store"]
    # head_idxs = [717, 664, 582]
    tail_classes = ['thunder snake', 
                    'sulphur-crested cockatoo', 
                    'kuvasz', 'bolete', 
                    'rhinoceros beetle', 
                    'soft-coated wheaten terrier', 
                    'coral fungus', 
                    'Blenheim spaniel', 
                    'standard schnauzer', 
                    'African hunting dog', 
                    'toyshop', 'Mexican hairless', 
                    'croquet ball', 'night snake', 
                    'red-breasted merganser', 
                    'common newt', 
                    'steel arch bridge', 
                    'mud turtle', 
                    'otterhound', 
                    'banded gecko', 
                    'Irish water spaniel', 
                    'earthstar', 
                    'stinkhorn', 
                    'green mamba', 
                    'Gila monster', 
                    'ruddy turnstone', 
                    'indri', 
                    'guenon', 
                    'rock beauty', 
                    'frilled lizard', 
                    'potpie', 
                    'Appenzeller', 
                    'Sussex spaniel', 
                    'isopod', 
                    'barracouta', 
                    'rock crab', 
                    'harvestman', 
                    'Bouvier des Flandres', 
                    'worm fence', 
                    'spotted salamander', 
                    'hognose snake', 
                    'black stork', 
                    'typewriter keyboard', 
                    'European fire salamander', 
                    'vine snake', 
                    'chiton', 
                    'cardoon', 
                    'sea snake', 
                    'leafhopper', 
                    'long-horned beetle']
    tail_idxs = [52, 89, 222, 997, 306, 202, 991, 156, 198, 275, 865, 268, 522, 
                    60, 98, 26, 821, 35, 175, 38, 221, 995, 994, 64, 45, 139, 384, 
                    370, 392, 43, 964, 240, 220, 126, 389, 119, 70, 233, 912, 28, 
                    54, 128, 878, 25, 59, 116, 946, 65, 317, 303]
    
    head_classes = ['notebook', 'sweatshirt', 'gown', 'pickup', 'lighter', 'cab', 'cornet', 'basketball', 'church', 
                    'wing', 'passenger car', 'switch', 'promontory', 'ski', 'monitor', 'bathtub', 'grocery store', 
                    'desk', 'Cardigan', 'pot', 'spotlight', 'nail', 'orange', 'teddy', 'bow', 'miniskirt', 'safe', 
                    'cougar', 'library', 'iron', 'swing', 'scale', 'plate', 'quilt', 'cliff', 'refrigerator', 'tank', 
                    'buckle', 'baseball', 'purse', 'bobsled', 'bucket', 'tiger', 'hook', 'necklace', 'dining table', 
                    'bee', 'restaurant', 'wallet', 'rubber eraser']
    
    head_idxs = [681, 841, 578, 717, 626, 468, 513, 430, 497, 908, 705, 844, 976, 795, 664, 435, 582, 526, 264, 
                    738, 818, 677, 950, 850, 456, 655, 771, 286, 624, 606, 843, 778, 923, 750, 972, 760, 847, 464, 
                    429, 748, 450, 463, 292, 600, 679, 532, 309, 762, 893, 767]

    # head_classes = ['spotted salamander']
    # head_idxs = [28]
    # tail_classes = ['barracouta']
    # tail_idxs = [389]
    # prompts = ["spotted salamander", "English foxhound", "barracouta", "pickup", "monitor", "grocery store"]
    # true_idxs = [28, 167, 389, 717, 664, 582]
    # tail_classes = ['jacamar', 'European gallinule', 'hognose snake']
    # tail_idxs = [95, 136, 54]
    # head_classes = ['library', 'washing machine', 'computer mouse']
    # head_idxs = [624, 897, 673]
    
    if training_args.no_tail_prompts:
        training_prompts = head_classes
        training_idxs = head_idxs
    else:
        
        training_prompts = tail_classes + head_classes
        training_idxs = tail_idxs + head_idxs
        if training_args.exclude_class != -1:
            training_prompts = remove_at_index(training_prompts, training_args.exclude_class)
            training_idxs = remove_at_index(training_idxs, training_args.exclude_class)

    print("Training prompts: ", training_prompts)
    train_prompt_iterator = PromptIterator(training_prompts, training_idxs, batch_size=training_args.sample_batch_size * training_args.sample_num_batches_per_epoch)
    eval_prompt_iterator = PromptIterator(tail_classes + head_classes, tail_idxs + head_idxs, batch_size=training_args.sample_batch_size * training_args.sample_num_batches_per_epoch)

    if training_args.exclude_class != -1:
        all_classes = tail_classes + head_classes
        tail_classes = all_classes[training_args.exclude_class]

    if training_args.exclude_class == -1:
        tail_classes = []
        
    trainer = AestheticsDDPOTrainer(
        training_args,
        aesthetic_scorer(),
        train_prompt_iterator,
        pipeline,
        image_samples_hook=None,
        eval_prompts=eval_prompt_iterator.prompts,
        tail_classes=tail_classes,
        label_mapping=eval_prompt_iterator.label_map,
        output_dir="/mnt/outputs" if training_args.amlt else None
    )
    # reg_dets = trainer.compute_gradient_statistics(tail_classes + head_classes, 640, 1)
    # print(reg_dets)
    # print("after trainer")
    trainer.train()

    # # Save and push to hub
    # trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)