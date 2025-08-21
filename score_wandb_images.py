import wandb
import json
import clip
from PIL import Image
import torch.nn as nn
import torch
import torch.nn.functional as F

def strip_caption_idx(caption):
    split_caption = caption.split(":")
    return split_caption[-1].strip()

def load_run(run_id, project="text2image-fine-tune"):
    run_path = f"ryanteehan/{project}/{run_id}"
    api = wandb.Api()
    run = api.run(run_path)
    return run

def get_validation_captions_and_filenames(step, run):
    metric_key = "validation"
    df = run.history(keys=[metric_key])

    result_series = df[df['_step'] == step][metric_key]

    captions_and_filenames = []
    for idx, val in result_series.items():
        captions = val['captions']
        filenames = val['filenames']

        captions = [strip_caption_idx(c) for c in captions]

        for cap, fname in zip(captions, filenames):
            captions_and_filenames.append((cap, fname))
    
    return captions_and_filenames

def save_images_and_captions(captions_and_filenames, output_dir, run):
    caption_dict = {}
    for item in captions_and_filenames:
        caption, fname = item
        file = run.file(fname)
        file.download(root=output_dir)
        caption_dict[f"{output_dir}/{fname}"] = caption
    
    with open(f"{output_dir}/captions.json", 'w') as fp:
        json.dump(caption_dict, fp)


class CLIPScore(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device, jit=False, 
                                                     download_root=download_root)
        
        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(self.clip_model) # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)


    def score(self, prompt, pil_image, return_feature=False):
        
        # if (type(image_path).__name__=='list'):
        #     _, rewards = self.inference_rank(prompt, image_path)
        #     return rewards
            
        # text encode
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        txt_features = F.normalize(self.clip_model.encode_text(text))
        
        # image encode
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.clip_model.encode_image(image))
        
        # score
        rewards = torch.sum(torch.mul(txt_features, image_features), dim=1, keepdim=True)
        
        if return_feature:
            return rewards, {'image': image_features, 'txt': txt_features}
        
        return rewards.detach().cpu().numpy().item()


def load_images_and_captions(caption_json_path):
    with open(caption_json_path, 'r') as fp:
        caption_dict = json.load(fp)
    
    images_and_captions = []
    for path in caption_dict:
        img = Image.open(path).convert("RGB")
        caption = caption_dict[path]
        images_and_captions.append((img, caption))
    
    return images_and_captions

    