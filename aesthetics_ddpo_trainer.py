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

import os
import textwrap
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Optional, Union
from warnings import warn

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import PyTorchModelHubMixin
from transformers import is_wandb_available

from trl.models import DDPOStableDiffusionPipeline
from trl.trainer.ddpo_config import DDPOConfig
from trl.trainer.utils import PerPromptStatTracker, generate_model_card, get_comet_experiment_url
from sklearn.random_projection import SparseRandomProjection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from torch.func import functional_call, vmap, grad
from opacus.grad_sample import GradSampleModule as _GSM
import torch.nn as nn
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
if is_wandb_available():
    import wandb

from opacus.grad_sample import register_grad_sampler

@register_grad_sampler(nn.Linear)
def compute_linear_grad_sample(
    layer: nn.Linear, activations: List[torch.Tensor], backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``nn.Linear`` layer

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    
        # add this from https://github.com/pytorch/opacus/blob/main/opacus/grad_sample/linear.py because the version local opacus was using was outdated for some reason
    """
    activations = activations[0]

    activations = activations.to(backprops.dtype)

    ret = {}
    if layer.weight.requires_grad:
        gs = torch.einsum("n...i,n...j->nij", backprops, activations)
        ret[layer.weight] = gs
    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = torch.einsum("n...k->nk", backprops)
    return ret

# def compute_linear_grad_sample_cast(layer, activations, backprops):
#     A = activations   # forward activations
#     B = backprops     # dL/dy
#     print(B, B.shape, "backprops")
#     print(A, len(A), "activations")
#     gs = torch.einsum("n...i,n...j->nij", B, A)
#     out = {layer.weight: gs}
#     if layer.bias is not None:
#         out[layer.bias] = torch.einsum("n...k->nk", B)
#     return out


logger = get_logger(__name__)
class TransparentGSM(_GSM):
    """
    GradSampleModule that behaves transparently for attributes like `.config`,
    `.in_channels`, etc., by delegating missing attrs to the wrapped module.
    """
    def __getattr__(self, name):
        # Try normal nn.Module/GradSampleModule lookup first
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Fallback to the wrapped module
            return getattr(self._module, name)

    def __setattr__(self, name, value):
        # During init or for our own internals, use normal setattr
        if name == "_module" or "_module" not in self.__dict__:
            return super().__setattr__(name, value)
        # If the inner module already defines the attr, set it there
        if hasattr(self._module, name):
            setattr(self._module, name, value)
        else:
            super().__setattr__(name, value)




class AestheticsDDPOTrainer(PyTorchModelHubMixin):
    """
    The DDPOTrainer uses Deep Diffusion Policy Optimization to optimise diffusion models. Note, this trainer is heavily
    inspired by the work here: https://github.com/kvablack/ddpo-pytorch As of now only Stable Diffusion based pipelines
    are supported

    Attributes:
        **config** (`DDPOConfig`) -- Configuration object for DDPOTrainer. Check the documentation of `PPOConfig` for more:
         details.
        **reward_function** (Callable[[torch.Tensor, tuple[str], tuple[Any]], torch.Tensor]) -- Reward function to be used:
        **prompt_function** (Callable[[], tuple[str, Any]]) -- Function to generate prompts to guide model
        **sd_pipeline** (`DDPOStableDiffusionPipeline`) -- Stable Diffusion pipeline to be used for training.
        **image_samples_hook** (Optional[Callable[[Any, Any, Any], Any]]) -- Hook to be called to log images
    """

    _tag_names = ["trl", "ddpo"]

    def __init__(
        self,
        config: DDPOConfig,
        reward_function: Callable[[torch.Tensor, tuple[str], tuple[Any]], torch.Tensor],
        prompt_function: Callable[[], tuple[str, Any]],
        sd_pipeline: DDPOStableDiffusionPipeline,
        image_samples_hook: Optional[Callable[[Any, Any, Any], Any]] = None,
        eval_prompts: Optional[list[str]] = None,
        tail_classes: Optional[list[str]] = None,
        label_mapping: Optional[dict[str, int]] = None,
        output_dir: Optional[str] = None
    ):
        if image_samples_hook is None:
            warn("No image_samples_hook provided; no images will be logged")

        self.prompt_fn = prompt_function
        self.reward_fn = reward_function
        self.config = config
        self.image_samples_callback = image_samples_hook
        self.eval_prompts = eval_prompts
        self.label_mapping = label_mapping
        self.tail_classes = tail_classes
        self.sample_steps = 0
        
        self.reward_history = {}

        accelerator_project_config = ProjectConfiguration(**self.config.project_kwargs)

        if self.config.resume_from:
            self.config.resume_from = os.path.normpath(os.path.expanduser(self.config.resume_from))
            if "checkpoint_" not in os.path.basename(self.config.resume_from):
                # get the most recent checkpoint in this directory
                checkpoints = list(
                    filter(
                        lambda x: "checkpoint_" in x,
                        os.listdir(self.config.resume_from),
                    )
                )
                if len(checkpoints) == 0:
                    raise ValueError(f"No checkpoints found in {self.config.resume_from}")
                checkpoint_numbers = sorted([int(x.split("_")[-1]) for x in checkpoints])
                self.config.resume_from = os.path.join(
                    self.config.resume_from,
                    f"checkpoint_{checkpoint_numbers[-1]}",
                )

                accelerator_project_config.iteration = checkpoint_numbers[-1] + 1

        # number of timesteps within each trajectory to train on
        self.num_train_timesteps = int(self.config.sample_num_steps * self.config.train_timestep_fraction)

        self.accelerator = Accelerator(
            log_with=self.config.log_with,
            mixed_precision=self.config.mixed_precision,
            project_config=accelerator_project_config,
            # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
            # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
            # the total number of optimizer steps to accumulate across.
            gradient_accumulation_steps=self.config.train_gradient_accumulation_steps * self.num_train_timesteps,
            **self.config.accelerator_kwargs,
        )

        is_okay, message = self._config_check()
        if not is_okay:
            raise ValueError(message)

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                self.config.tracker_project_name,
                config=dict(ddpo_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict(),
                init_kwargs=self.config.tracker_kwargs,
            )
            print(self.accelerator.trackers[0].tracker.name, "name")
            if output_dir is not None:
                self.output_dir = f"{output_dir}/{self.accelerator.trackers[0].tracker.name}"
            else:
                self.output_dir = None
            self.accelerator.trackers[0].tracker.define_metric("sample_steps")
            self.accelerator.trackers[0].tracker.define_metric("global_step")
            self.accelerator.trackers[0].tracker.define_metric("per_sample_grads/*", step_metric="sample_steps")

        logger.info(f"\n{config}")

        set_seed(self.config.seed, device_specific=True)

        self.sd_pipeline = sd_pipeline

        self.sd_pipeline.set_progress_bar_config(
            position=1,
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16
        else:
            inference_dtype = torch.float32

        self.sd_pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.text_encoder.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.unet.to(self.accelerator.device, dtype=inference_dtype)
        

        trainable_layers = self.sd_pipeline.get_trainable_layers()
        # print(trainable_layers, "Trainable")
        # self.sd_pipeline.sd_pipeline.unet = TransparentGSM(self.sd_pipeline.sd_pipeline.unet)


        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.optimizer = self._setup_optimizer(
            trainable_layers.parameters() if not isinstance(trainable_layers, list) else trainable_layers
        )

        self.neg_prompt_embed = self.sd_pipeline.text_encoder(
            self.sd_pipeline.tokenizer(
                [""] if self.config.negative_prompts is None else self.config.negative_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
        )[0]

        if config.per_prompt_stat_tracking:
            self.stat_tracker = PerPromptStatTracker(
                config.per_prompt_stat_tracking_buffer_size,
                config.per_prompt_stat_tracking_min_count,
            )

        # NOTE: for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
        # more memory
        self.autocast = self.sd_pipeline.autocast or self.accelerator.autocast

        if hasattr(self.sd_pipeline, "use_lora") and self.sd_pipeline.use_lora:
            unet, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)
            self.named_trainable_layers = list(filter(lambda n_p : n_p[1].requires_grad, unet.named_parameters()))
            # print(self.named_trainable_layers)
            self.trainable_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
        else:
            self.trainable_layers, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)

        if self.config.async_reward_computation:
            self.executor = futures.ThreadPoolExecutor(max_workers=config.max_workers)

        if config.resume_from:
            logger.info(f"Resuming from {config.resume_from}")
            self.accelerator.load_state(config.resume_from)
            self.first_epoch = int(config.resume_from.split("_")[-1]) + 1
        else:
            self.first_epoch = 0

    def compute_rewards(self, prompt_image_pairs, step, is_async=False, log_rewards=False):
        if not is_async:
            rewards = []
            for images, prompts, prompt_metadata in prompt_image_pairs:
                reward, reward_metadata = self.reward_fn(images, prompts, prompt_metadata)
                rewards.append(
                    (
                        torch.as_tensor(reward, device=self.accelerator.device),
                        reward_metadata,
                    )
                )
        else:
            rewards = self.executor.map(lambda x: self.reward_fn(*x), prompt_image_pairs)
            rewards = [
                (torch.as_tensor(reward.result(), device=self.accelerator.device), reward_metadata.result())
                for reward, reward_metadata in rewards
            ]

        return zip(*rewards)
    
    def compute_summary_statistics(self, epoch_reward_statistics, epoch_reward_statistics_post):
        
        epoch_summary_statistics = {}

        for prompt in epoch_reward_statistics:        
            max_val = np.max(np.array(epoch_reward_statistics[prompt]))
            median_val = np.median(np.array(epoch_reward_statistics[prompt]))

            mean_val = np.mean(np.array(epoch_reward_statistics[prompt]))

            post_mean = np.mean(np.array(epoch_reward_statistics_post[prompt]))

            percent_increase = (post_mean - mean_val) / mean_val

            epoch_summary_statistics[prompt] = {}
            epoch_summary_statistics[prompt]["max_minus_median"] = [max_val - median_val]
            epoch_summary_statistics[prompt]["percent_increase"] = [percent_increase]
        
        return epoch_summary_statistics
    
    def compute_reward_statistics(self, epoch_reward_statistics, epoch_reward_statistics_post):

        epoch_summary_statistics = {}

        for prompt in epoch_reward_statistics:        
            max_val = np.max(np.array(epoch_reward_statistics[prompt]))
            median_val = np.median(np.array(epoch_reward_statistics[prompt]))
            std_val = np.std(np.array(epoch_reward_statistics[prompt]))

            mean_val = np.mean(np.array(epoch_reward_statistics[prompt]))

            post_mean = np.mean(np.array(epoch_reward_statistics_post[prompt]))
            post_median = np.median(np.array(epoch_reward_statistics_post[prompt]))
            post_std = np.std(np.array(epoch_reward_statistics_post[prompt]))
            post_max = np.max(np.array(epoch_reward_statistics_post[prompt]))

            percent_increase = (post_mean - mean_val) / mean_val

            epoch_summary_statistics[f"statistics/{prompt}_max_minus_median"] = max_val - median_val
            epoch_summary_statistics[f"statistics/{prompt}_percent_increase"] = percent_increase
            epoch_summary_statistics[f"statistics/{prompt}_max"] = max_val
            epoch_summary_statistics[f"statistics/{prompt}_median"] = median_val
            epoch_summary_statistics[f"statistics/{prompt}_std"] = std_val
            epoch_summary_statistics[f"statistics/{prompt}_mean"] = mean_val

            epoch_summary_statistics[f"statistics/{prompt}_max_post"] = post_max
            epoch_summary_statistics[f"statistics/{prompt}_median_post"] = post_median
            epoch_summary_statistics[f"statistics/{prompt}_std_post"] = post_std
            epoch_summary_statistics[f"statistics/{prompt}_mean_post"] = post_mean
            epoch_summary_statistics[f"statistics/{prompt}_max_minus_median_post"] = post_max - post_median

        
        return epoch_summary_statistics

    
    def log_reward_history(self, epoch_summary_statistics):

        for prompt in epoch_summary_statistics:
            if prompt not in self.reward_history:
                self.reward_history[prompt] = {"max_minus_median": [], "percent_increase": []}
        
            self.reward_history[prompt]["max_minus_median"] += epoch_summary_statistics[prompt]["max_minus_median"]

            self.reward_history[prompt]["percent_increase"] += epoch_summary_statistics[prompt]["percent_increase"]
        
        return epoch_summary_statistics
    
    def regress_rewards(self, current_stats, regress_per_class=True):
        
        predictions = {}
        errors = {}

        if regress_per_class:
            for prompt in current_stats:
                if prompt in self.reward_history:
                    data = self.reward_history[prompt]
                    data = pd.DataFrame(data)
                    # print(data, "data")

                    x = data.drop(columns=["percent_increase"])

                    regression = LinearRegression().fit(X=x, y=data["percent_increase"])

                    curr_stats_data = current_stats[prompt]
                    curr_stats_data = pd.DataFrame(curr_stats_data)
                    curr_stats_x = curr_stats_data.drop(columns=["percent_increase"])
                    prediction = regression.predict(curr_stats_x)

                    predictions['prompt'] = prediction
                                                                                                                                                                                                                                                
                    error = mean_squared_error(curr_stats_data["percent_increase"], prediction)
                    errors[prompt] = error

                    # print(prediction, "pred")
                    # print(curr_stats_data["percent_increase"], "true")
                    # print(error, "err")
        else:

            all_max_minus_median = []
            all_percent_increase = []

            for prompt in self.reward_history:
                data = self.reward_history[prompt]

                all_max_minus_median += data["max_minus_median"]
                all_percent_increase += data["percent_increase"]

            data = {"max_minus_median": all_max_minus_median, "percent_increase": all_percent_increase}
            data = pd.DataFrame(data)

            x = data.drop(columns=["percent_increase"])
            regression = LinearRegression().fit(X=x, y=data["percent_increase"])
            print(data, "data")
            for prompt in current_stats:

                curr_stats_data = current_stats[prompt]
                curr_stats_data = pd.DataFrame(curr_stats_data)
                curr_stats_x = curr_stats_data.drop(columns=["percent_increase"])
                prediction = regression.predict(curr_stats_x)

                predictions['prompt'] = prediction
                                                                                                                                                                                                                                            
                error = mean_squared_error(curr_stats_data["percent_increase"], prediction)
                errors[prompt] = error

        
        return predictions, errors
    


    def step(self, epoch: int, global_step: int):
        """
        Perform a single step of training.

        Args:
            epoch (int): The current epoch.
            global_step (int): The current global step.

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.
            - If `self.image_samples_callback` is not None, it will be called with the prompt_image_pairs, global_step,
              and the accelerator tracker.

        Returns:
            global_step (int): The updated global step.

        """
        print(epoch, "epoch")
        samples, prompt_image_data = self._generate_samples(
            iterations=self.config.sample_num_batches_per_epoch,
            batch_size=self.config.sample_batch_size,
        )
        # print(f"Num inner epochs = {self.config.train_num_inner_epochs}")
        # print(prompt_image_data, "pi") # embeds, prompts, labels
        # for p_i in prompt_image_data:
        #     print(p_i[1])
        # print(len(prompt_image_data)) # len = number of steps per epoch

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        # print(samples.keys(), "s keys")
        rewards, rewards_metadata = self.compute_rewards(
            prompt_image_data, step=global_step, is_async=self.config.async_reward_computation
        )
        # print("step", global_step)
        # print("rewards", rewards)
        # print("rewards meta", rewards_metadata)
        # print(len(rewards_metadata))


        within_epoch_statistics = {}
        all_prompts = []
        for i, r_meta in enumerate(rewards_metadata):
            prompts_in_batch = prompt_image_data[i][1]
            
            rewards_in_batch = r_meta["aesthetics"].cpu().tolist()
            # print(prompts_in_batch, rewards_in_batch)

            for j, prompt in enumerate(prompts_in_batch):
                all_prompts.append(str(prompt))
                if prompt not in within_epoch_statistics:
                    within_epoch_statistics[prompt] = []
                
                within_epoch_statistics[prompt].append(rewards_in_batch[j])
        # print("PRE")
        # print(within_epoch_statistics)

        print(f"Prompt counts in batch {Counter(all_prompts)}")

        for i, image_data in enumerate(prompt_image_data):
            image_data.extend([rewards[i], rewards_metadata[i]])

        # if self.image_samples_callback is not None:
        #     self.image_samples_callback(prompt_image_data, global_step, self.accelerator.trackers[0])

        rewards = torch.cat(rewards)
        rewards = self.accelerator.gather(rewards).cpu().numpy()

        self.accelerator.log(
            {
                "reward": rewards,
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
                "global_step": global_step
            },
            # step=global_step,
        )

        if self.config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = self.accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = self.sd_pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = self.stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ungather advantages;  keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]
            .to(self.accelerator.device)
        )

        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape

        for inner_epoch in range(self.config.train_num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=self.accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            # still trying to understand the code below
            perms = torch.stack(
                [torch.randperm(num_timesteps, device=self.accelerator.device) for _ in range(total_batch_size)]
            )

            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=self.accelerator.device)[:, None],
                    perms,
                ]

            original_keys = samples.keys()
            original_values = samples.values()
            # rebatch them as user defined train_batch_size is different from sample_batch_size
            reshaped_values = [v.reshape(-1, self.config.train_batch_size, *v.shape[1:]) for v in original_values]

            # Transpose the list of original values
            transposed_values = zip(*reshaped_values)
            # Create new dictionaries for each row of transposed values
            samples_batched = [dict(zip(original_keys, row_values)) for row_values in transposed_values]
            # print(len(samples_batched), "s_batch")
            self.sd_pipeline.unet.train()
            global_step = self._train_batched_samples(inner_epoch, epoch, global_step, samples_batched)
            # ensure optimization step at the end of the inner epoch
            if not self.accelerator.sync_gradients:
                raise ValueError(
                    "Optimization step should have been performed by this point. Please check calculated gradient accumulation settings."
                )
        # exit()
        if epoch != 0 and epoch % self.config.save_freq == 0 and self.accelerator.is_main_process:
            self.accelerator.save_state()

        post_samples, post_prompt_image_data = self.generate_samples_from_prompts(
            all_prompts=all_prompts,
            label_mapping=self.label_mapping,

        )
        # print(samples.keys(), "s keys")
        post_rewards, post_rewards_metadata = self.compute_rewards(
            post_prompt_image_data, step=global_step, is_async=self.config.async_reward_computation
        )
        # print("step", global_step)
        # print("rewards", rewards)
        # print("rewards meta", rewards_metadata)
        # print(len(rewards_metadata))

        within_epoch_statistics_post = {}

        for i, r_meta in enumerate(post_rewards_metadata):
            prompts_in_batch = post_prompt_image_data[i][1]
            rewards_in_batch = r_meta["aesthetics"].cpu().tolist()
            # print(prompts_in_batch, rewards_in_batch)

            for j, prompt in enumerate(prompts_in_batch):
                if prompt not in within_epoch_statistics_post:
                    within_epoch_statistics_post[prompt] = []
                
                within_epoch_statistics_post[prompt].append(rewards_in_batch[j])
        # print("POST")
        # print(within_epoch_statistics_post)

        summary_statistics = self.compute_reward_statistics(within_epoch_statistics, within_epoch_statistics_post)
        summary_statistics["global_step"] = global_step
        self.accelerator.log(summary_statistics)

        for prompt in within_epoch_statistics:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(y=within_epoch_statistics[prompt], x=list(range(len(within_epoch_statistics[prompt]))))
            ax.set_ylabel("Reward")
            ax.set_title(f"{prompt} Rewards Pre")
            fig.tight_layout()
            self.accelerator.log({"Reward Plots": wandb.Image(fig), "global step": global_step})
            fig.clf()
        
        for prompt in within_epoch_statistics_post:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(y=within_epoch_statistics[prompt], x=list(range(len(within_epoch_statistics[prompt]))))
            ax.set_ylabel("Reward")
            ax.set_title(f"{prompt} Rewards Post")
            fig.tight_layout()
            self.accelerator.log({"Reward Plots": wandb.Image(fig), "global step": global_step})
            fig.clf()


        if self.output_dir is not None:
            
            step_save_dir = f"{self.output_dir}/optimization_step_{global_step - 1}"
            print(f"Saving global step {global_step - 1} in dir {step_save_dir}")
            os.makedirs(step_save_dir, exist_ok=True)
            torch.save(samples, f"{step_save_dir}/samples_pre.pth")
            torch.save(prompt_image_data, f"{step_save_dir}/prompt_image_data_pre.pth")
            torch.save(rewards_metadata, f"{step_save_dir}/rewards_metadata_pre.pth")

            torch.save(post_samples, f"{step_save_dir}/samples_post.pth")
            torch.save(post_prompt_image_data, f"{step_save_dir}/prompt_image_data_post.pth")
            torch.save(post_rewards_metadata, f"{step_save_dir}/rewards_metadata_post.pth")
            # exit()
        # curr_summary_statistics = self.compute_summary_statistics(within_epoch_statistics, within_epoch_statistics_post)
        # if epoch >= 3:
        #     print("PREDICTING")
        #     predictions, errors = self.regress_rewards(curr_summary_statistics, regress_per_class=False)

        #     regression_log = {}
        #     total_err = 0
        #     for p in errors:
        #         regression_log[f"regression_errors/{p}"] = errors[p]
        #         total_err += errors[p]
            
        #     regression_log["regression_errors/mean_MSE"] = total_err / len(list(errors.keys()))
        #     regression_log["global_step"] = global_step
        #     self.accelerator.log(regression_log)
            
        # # print("HISTORY")
        # self.log_reward_history(curr_summary_statistics)
        # print("history")
        # print(self.reward_history)
        # exit()

        return global_step
    
    def compute_gradient_statistics(self, prompts, num_samples_per_prompt, batch_size):
        """
        Perform a single step of training.

        Args:
            epoch (int): The current epoch.
            global_step (int): The current global step.

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.
            - If `self.image_samples_callback` is not None, it will be called with the prompt_image_pairs, global_step,
              and the accelerator tracker.

        Returns:
            global_step (int): The updated global step.

        """
        l=1e-4
        samples_per_prompt, imagedata_per_prompt = self._generate_samples_for_stats(
            prompts=prompts,
            label_mapping=self.label_mapping,
            batch_size=batch_size,
            num_samples_per_prompt=num_samples_per_prompt,
        )

        reg_dets = {}

        for p in samples_per_prompt.keys():
            all_samples = samples_per_prompt[p]
            all_prompt_image_data = imagedata_per_prompt[p]
            # print(all_prompt_image_data, "all prompt image")
            # print(all_samples, "all samples")
            grad_covs = []
            for i in range(len(all_samples)):
                samples = [all_samples[i]]
                prompt_image_data = [all_prompt_image_data[i]]
                # print("prompt image", prompt_image_data, len(prompt_image_data))
                # print("samples", samples)
                # # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
                samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
                rewards, rewards_metadata = self.compute_rewards(
                    prompt_image_data, step=0, is_async=self.config.async_reward_computation
                )

                for i, image_data in enumerate(prompt_image_data):
                    image_data.extend([rewards[i], rewards_metadata[i]])

                rewards = torch.cat(rewards)
                rewards = self.accelerator.gather(rewards).cpu().numpy()

                if self.config.per_prompt_stat_tracking:
                    # gather the prompts across processes
                    prompt_ids = self.accelerator.gather(samples["prompt_ids"]).cpu().numpy()
                    prompts = self.sd_pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                    advantages = self.stat_tracker.update(prompts, rewards)
                else:
                    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

                # ungather advantages;  keep the entries corresponding to the samples on this process
                samples["advantages"] = (
                    torch.as_tensor(advantages)
                    .reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]
                    .to(self.accelerator.device)
                )

                del samples["prompt_ids"]

                total_batch_size, num_timesteps = samples["timesteps"].shape

                # shuffle samples along batch dimension
                perm = torch.randperm(total_batch_size, device=self.accelerator.device)
                samples = {k: v[perm] for k, v in samples.items()}

                # shuffle along time dimension independently for each sample
                # still trying to understand the code below
                perms = torch.stack(
                    [torch.randperm(num_timesteps, device=self.accelerator.device) for _ in range(total_batch_size)]
                )

                for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                    samples[key] = samples[key][
                        torch.arange(total_batch_size, device=self.accelerator.device)[:, None],
                        perms,
                    ]

                original_keys = samples.keys()
                original_values = samples.values()
                # # rebatch them as user defined train_batch_size is different from sample_batch_size
                reshaped_values = [v.reshape(-1, batch_size, *v.shape[1:]) for v in original_values]

                # Transpose the list of original values
                transposed_values = zip(*reshaped_values)
                # Create new dictionaries for each row of transposed values
                samples_batched = [dict(zip(original_keys, row_values)) for row_values in transposed_values]

                grad_covariances = self.grad_batched_samples(batched_samples=samples_batched)
                grad_covs += grad_covariances
                # ensure optimization step at the end of the inner epoch
                # reg_dets[p] = regularized_determinants            
            print(len(grad_covs), "Grad covs")
            print(len(grad_covs[0]), "first elem")
            regularized_determinants = []
            for j in range(len(grad_covs[0])):
                # print(sum([g[j] for g in all_grads]))
                # reg_det = torch.linalg.det(sum([g[j] for g in all_grads]))
                # print("reg det", reg_det)
                print((l * torch.eye(grad_covs[0][0].shape[0])).to(self.accelerator.device) + torch.sum(torch.stack([g[j] for g in grad_covs]), dim=0))
                print(grad_covs[0][0])
                reg_det = torch.logdet((l * torch.eye(grad_covs[0][0].shape[0])).to(self.accelerator.device) + torch.sum(torch.stack([g[j] for g in grad_covs]), dim=0))
                regularized_determinants.append(reg_det)

            reg_dets[p] = regularized_determinants
            print("reg det", regularized_determinants)
        return reg_dets

    def _generate_samples_for_stats(self, prompts: list[str], label_mapping: dict[str, int], num_samples_per_prompt: int, batch_size: int ):
        """
        Generate evaluation samples for the fixed list ``eval_prompts``.

        Returns
        -------
        samples : list[dict[str, torch.Tensor]]
        prompt_image_pairs : list[list[Any]]
            Same structure as returned by ``_generate_samples``.
        """
        def get_labels_for_prompt(prompt):
            label_map = label_mapping
            shape = None
            for key in label_map:
                if key in prompt:
                    shape = key
            
            label = label_map[shape]
            return {"label": label}
        
        # if batch_size < len(eval_prompts):
        #     batch_size = len(eval_prompts)

        self.sd_pipeline.unet.eval()
        samples_per_prompt = {}
        pairs_per_prompt = {}

        # iterate over eval prompts in fixed-order, batching to reuse the existing sampler
        for prompt in prompts:
            all_samples, all_pairs = [], []
            

            num_full_batches = num_samples_per_prompt // batch_size
            last_batch_size = num_samples_per_prompt % batch_size
            for i in range(num_full_batches):
                prompt_metadata = [get_labels_for_prompt(prompt) for i in range(batch_size)]
                cur_prompts = [prompt for j in range(batch_size)]

                prompt_ids = self.sd_pipeline.tokenizer(
                    cur_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.sd_pipeline.tokenizer.model_max_length,
                ).input_ids.to(self.accelerator.device)

                prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]
                neg_embeds = self.neg_prompt_embed.repeat(batch_size, 1, 1)

                with torch.no_grad(), self.autocast():
                    out = self.sd_pipeline(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=neg_embeds,
                        # num_images_per_prompt=batch_size,
                        num_inference_steps=self.config.sample_num_steps,
                        guidance_scale=self.config.sample_guidance_scale,
                        eta=self.config.sample_eta,
                        output_type="pt",
                    )
                latents = torch.stack(out.latents, dim=1)
                log_probs = torch.stack(out.log_probs, dim=1)
                timesteps = self.sd_pipeline.scheduler.timesteps.repeat(batch_size, 1)
                # print("latent shape", latents.shape)
                all_samples.append({
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "log_probs": log_probs,
                    "negative_prompt_embeds": neg_embeds,
                })
                all_pairs.append([out.images, cur_prompts, prompt_metadata])
            # print(len(all_samples), "all samples")
            samples_per_prompt[prompt] = all_samples
            pairs_per_prompt[prompt] = all_pairs

        return samples_per_prompt, pairs_per_prompt

    def calculate_loss(self, latents, timesteps, next_latents, log_probs, advantages, embeds):
        """
        Calculate the loss for a batch of an unpacked sample

        Args:
            latents (torch.Tensor):
                The latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            timesteps (torch.Tensor):
                The timesteps sampled from the diffusion model, shape: [batch_size]
            next_latents (torch.Tensor):
                The next latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height,
                width]
            log_probs (torch.Tensor):
                The log probabilities of the latents, shape: [batch_size]
            advantages (torch.Tensor):
                The advantages of the latents, shape: [batch_size]
            embeds (torch.Tensor):
                The embeddings of the prompts, shape: [2*batch_size or batch_size, ...] Note: the "or" is because if
                train_cfg is True, the expectation is that negative prompts are concatenated to the embeds

        Returns:
            loss (torch.Tensor), approx_kl (torch.Tensor), clipfrac (torch.Tensor) (all of these are of shape (1,))
        """
        with self.autocast():
            if self.config.train_cfg:
                noise_pred = self.sd_pipeline.unet(
                    torch.cat([latents] * 2),
                    torch.cat([timesteps] * 2),
                    embeds,
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.sample_guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            else:
                noise_pred = self.sd_pipeline.unet(
                    latents,
                    timesteps,
                    embeds,
                ).sample
            # compute the log prob of next_latents given latents under the current model

            scheduler_step_output = self.sd_pipeline.scheduler_step(
                noise_pred,
                timesteps,
                latents,
                eta=self.config.sample_eta,
                prev_sample=next_latents,
            )

            log_prob = scheduler_step_output.log_probs

        advantages = torch.clamp(
            advantages,
            -self.config.train_adv_clip_max,
            self.config.train_adv_clip_max,
        )

        ratio = torch.exp(log_prob - log_probs)

        loss = self.loss(advantages, self.config.train_clip_range, ratio)

        approx_kl = 0.5 * torch.mean((log_prob - log_probs) ** 2)

        clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.config.train_clip_range).float())

        return loss, approx_kl, clipfrac

    def calculate_loss_for_per_sample_grads(self, unet_params, unet_buffers, latent, timestep, next_latent, log_prob, advantage, embed):
        """
        Calculate the loss for a batch of an unpacked sample

        Args:
            latents (torch.Tensor):
                The latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            timesteps (torch.Tensor):
                The timesteps sampled from the diffusion model, shape: [batch_size]
            next_latents (torch.Tensor):
                The next latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height,
                width]
            log_probs (torch.Tensor):
                The log probabilities of the latents, shape: [batch_size]
            advantages (torch.Tensor):
                The advantages of the latents, shape: [batch_size]
            embeds (torch.Tensor):
                The embeddings of the prompts, shape: [2*batch_size or batch_size, ...] Note: the "or" is because if
                train_cfg is True, the expectation is that negative prompts are concatenated to the embeds

        Returns:
            loss (torch.Tensor), approx_kl (torch.Tensor), clipfrac (torch.Tensor) (all of these are of shape (1,))
        """
        latents = latent.unsqueeze(0)
        timesteps = timestep.unsqueeze(0)
        next_latents = next_latent.unsqueeze(0)
        log_probs = log_prob.unsqueeze(0) # double check this
        advantages = advantage.unsqueeze(0)
        embeds = embed.unsqueeze(0)
        with self.autocast():
            if self.config.train_cfg:
                noise_pred = functional_call(self.sd_pipeline.unet, 
                                             (unet_params, unet_buffers),
                                             torch.cat([latents] * 2),
                                             torch.cat([timesteps] * 2),
                                             embeds,
                                             ).sample
                # noise_pred = self.sd_pipeline.unet(
                #     torch.cat([latents] * 2),
                #     torch.cat([timesteps] * 2),
                #     embeds,
                # ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.sample_guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            else:
                noise_pred = functional_call(self.sd_pipeline.unet,
                                             (unet_params, unet_buffers),
                                             latents,
                                             timesteps,
                                             embeds,
                ).sample
                # noise_pred = self.sd_pipeline.unet(
                #     latents,
                #     timesteps,
                #     embeds,
                # ).sample
            # compute the log prob of next_latents given latents under the current model

            scheduler_step_output = self.sd_pipeline.scheduler_step(
                noise_pred,
                timesteps,
                latents,
                eta=self.config.sample_eta,
                prev_sample=next_latents,
            )

            next_latent_log_prob = scheduler_step_output.log_probs

        advantages = torch.clamp(
            advantages,
            -self.config.train_adv_clip_max,
            self.config.train_adv_clip_max,
        )

        ratio = torch.exp(next_latent_log_prob - log_probs)

        loss = self.loss(advantages, self.config.train_clip_range, ratio)

        approx_kl = 0.5 * torch.mean((next_latent_log_prob - log_probs) ** 2)

        clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.config.train_clip_range).float())

        return loss, approx_kl, clipfrac



    def loss(
        self,
        advantages: torch.Tensor,
        clip_range: float,
        ratio: torch.Tensor,
    ):
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )
        return torch.mean(torch.maximum(unclipped_loss, clipped_loss))

    def _setup_optimizer(self, trainable_layers_parameters):
        if self.config.train_use_8bit_adam:
            import bitsandbytes

            optimizer_cls = bitsandbytes.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        return optimizer_cls(
            trainable_layers_parameters,
            lr=self.config.train_learning_rate,
            betas=(self.config.train_adam_beta1, self.config.train_adam_beta2),
            weight_decay=self.config.train_adam_weight_decay,
            eps=self.config.train_adam_epsilon,
        )

    def _save_model_hook(self, models, weights, output_dir):
        self.sd_pipeline.save_checkpoint(models, weights, output_dir)
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def _load_model_hook(self, models, input_dir):
        self.sd_pipeline.load_checkpoint(models, input_dir)
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    def generate_eval_samples(self, eval_prompts: list[str], label_mapping: dict[str, int], batch_size: int | None = None):
        """
        Generate evaluation samples for the fixed list ``eval_prompts``.

        Returns
        -------
        samples : list[dict[str, torch.Tensor]]
        prompt_image_pairs : list[list[Any]]
            Same structure as returned by ``_generate_samples``.
        """
        def get_labels_for_prompt(prompt):
            label_map = label_mapping
            shape = None
            for key in label_map:
                if key in prompt:
                    shape = key
            
            label = label_map[shape]
            return {"label": label}

        if batch_size is None:
            batch_size = self.config.sample_batch_size
        
        # if batch_size < len(eval_prompts):
        #     batch_size = len(eval_prompts)

        self.sd_pipeline.unet.eval()
        all_samples, all_pairs = [], []

        # iterate over eval prompts in fixed-order, batching to reuse the existing sampler
        for start in range(0, len(eval_prompts), batch_size):
            cur_prompts = eval_prompts[start:start + batch_size]
            
            cur_bs = len(cur_prompts)

            # pad the last batch so shapes stay consistent
            if cur_bs < batch_size:
                cur_prompts = cur_prompts + [cur_prompts[-1]] * (batch_size - cur_bs)

            # no per-prompt metadata available → use empty dicts
            prompt_metadata = [get_labels_for_prompt(p) for p in cur_prompts]
            # print("curr prompts", cur_prompts)
            # print("curr meta", prompt_metadata)
            prompt_ids = self.sd_pipeline.tokenizer(
                cur_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
            prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]

            neg_embeds = self.neg_prompt_embed.repeat(batch_size, 1, 1)

            with torch.no_grad(), self.autocast():
                out = self.sd_pipeline(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=neg_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    output_type="pt",
                )

            latents = torch.stack(out.latents, dim=1)
            log_probs = torch.stack(out.log_probs, dim=1)
            timesteps = self.sd_pipeline.scheduler.timesteps.repeat(batch_size, 1)

            all_samples.append({
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "timesteps": timesteps,
                "latents": latents[:, :-1],
                "next_latents": latents[:, 1:],
                "log_probs": log_probs,
                "negative_prompt_embeds": neg_embeds,
            })
            all_pairs.append([out.images, cur_prompts, prompt_metadata])

        return all_samples, all_pairs

    def generate_samples_from_prompts(self, all_prompts, label_mapping):
        def get_labels_for_prompt(prompt):
            label_map = label_mapping
            shape = None
            for key in label_map:
                if key in prompt:
                    shape = key
            
            label = label_map[shape]
            return {"label": label}

        samples = []
        prompt_image_pairs = []
        self.sd_pipeline.unet.eval()

        sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(1, 1, 1)
        for prompt in all_prompts:
            prompt_metadata = get_labels_for_prompt(prompt)

            prompt_ids = self.sd_pipeline.tokenizer([prompt],
                                                    return_tensors="pt",
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.sd_pipeline.tokenizer.model_max_length
                                                    ).input_ids.to(self.accelerator.device)
            prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]

            with self.autocast():
                sd_output = self.sd_pipeline(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    output_type="pt",
                )

                images = sd_output.images
                latents = sd_output.latents
                log_probs = sd_output.log_probs

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, ...)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)

            timesteps = self.sd_pipeline.scheduler.timesteps.repeat(1, 1)
            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # each entry is the latent before timestep t
                    "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "negative_prompt_embeds": sample_neg_prompt_embeds,
                }
            )
            prompt_image_pairs.append([images, [prompt], [prompt_metadata]])

        return samples, prompt_image_pairs


    def _generate_samples(self, iterations, batch_size):
        """
        Generate samples from the model

        Args:
            iterations (int): Number of iterations to generate samples for
            batch_size (int): Batch size to use for sampling

        Returns:
            samples (list[dict[str, torch.Tensor]]), prompt_image_pairs (list[list[Any]])
        """
        samples = []
        prompt_image_pairs = []
        self.sd_pipeline.unet.eval()

        sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(batch_size, 1, 1)

        for _ in range(iterations):
            prompts, prompt_metadata = zip(*[self.prompt_fn() for _ in range(batch_size)])

            prompt_ids = self.sd_pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
            prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]

            with self.autocast():
                sd_output = self.sd_pipeline(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    output_type="pt",
                )

                images = sd_output.images
                latents = sd_output.latents
                log_probs = sd_output.log_probs

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, ...)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = self.sd_pipeline.scheduler.timesteps.repeat(batch_size, 1)  # (batch_size, num_steps)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # each entry is the latent before timestep t
                    "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "negative_prompt_embeds": sample_neg_prompt_embeds,
                }
            )
            prompt_image_pairs.append([images, prompts, prompt_metadata])

        return samples, prompt_image_pairs
    
    def grad_batched_samples(self, batched_samples):
        # todo figure out how to store this
        l = 1e-5
        def flat_grad(parameters) -> torch.Tensor:
            """
            Return all parameter gradients as a 1×d tensor, where
            d = total parameter count.
            """
            # Flatten each gradient (replace None with zeros) and concatenate
            grads = [
                p.grad.reshape(-1)
                for p in parameters
            ]
            return torch.cat(grads).unsqueeze(1)

        info = defaultdict(list)
        all_grads = []
        # print("train timesteps", self.num_train_timesteps)
        for _i, sample in enumerate(batched_samples):
            if self.config.train_cfg:
                # concat negative prompts to sample prompts to avoid two forward passes
                embeds = torch.cat([sample["negative_prompt_embeds"], sample["prompt_embeds"]])
            else:
                embeds = sample["prompt_embeds"]

            sample_grads = []
            for j in range(self.num_train_timesteps):
                with self.accelerator.accumulate(self.sd_pipeline.unet):
                    loss, approx_kl, clipfrac = self.calculate_loss(
                        sample["latents"][:, j],
                        sample["timesteps"][:, j],
                        sample["next_latents"][:, j],
                        sample["log_probs"][:, j],
                        sample["advantages"],
                        embeds,
                    )
                    info["approx_kl"].append(approx_kl)
                    info["clipfrac"].append(clipfrac)
                    info["loss"].append(loss)

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        # self.accelerator.clip_grad_norm_(
                        #     self.trainable_layers.parameters()
                        #     if not isinstance(self.trainable_layers, list)
                        #     else self.trainable_layers,
                        #     self.config.train_max_grad_norm,
                        # )
                        with torch.no_grad():
                            grads = flat_grad(self.trainable_layers)
                        # print(grads.shape, torch.min(grads), torch.max(grads))

                        sparse_proj = SparseRandomProjection(n_components=64)
                        proj_grads = sparse_proj.fit_transform(grads.T.detach().cpu().numpy())
                        proj_grads = torch.from_numpy(proj_grads).to(grads.device).T
                        # print(proj_grads.shape, torch.min(proj_grads), torch.max(proj_grads))
                        self.optimizer.zero_grad(set_to_none=True)
                        with torch.no_grad():
                            grads_cov = proj_grads @ proj_grads.T
                        sample_grads.append(grads_cov)
                        # print(grads_cov.shape, "grads cov", torch.linalg.det(grads_cov))
                        # all_grads[sa]
            # print("sample", sample_grads, len(sample_grads[0]))
            all_grads.append(sample_grads)
        # print(len(all_grads), len(all_grads[0]), "all grads")
        # regularized_determinants = []
        # for j in range(len(all_grads[0])):
        #     # print(sum([g[j] for g in all_grads]))
        #     # reg_det = torch.linalg.det(sum([g[j] for g in all_grads]))
        #     # print("reg det", reg_det)
        #     reg_det = torch.logdet((l * torch.eye(all_grads[0][0].shape[0])).to(self.accelerator.device) + sum([g[j] for g in all_grads]))
        #     regularized_determinants.append(reg_det)
        

        return all_grads 


    def _train_batched_samples(self, inner_epoch, epoch, global_step, batched_samples):
        """
        Train on a batch of samples. Main training segment

        Args:
            inner_epoch (int): The current inner epoch
            epoch (int): The current epoch
            global_step (int): The current global step
            batched_samples (list[dict[str, torch.Tensor]]): The batched samples to train on

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.

        Returns:
            global_step (int): The updated global step
        """
        def flat_grad(parameters) -> torch.Tensor:
            """
            Return all parameter gradients as a 1×d tensor, where
            d = total parameter count.
            """
            # Flatten each gradient (replace None with zeros) and concatenate
            grads = [
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                for p in parameters
            ]
            return torch.cat(grads).unsqueeze(0)
        
        info = defaultdict(list)
        for _i, sample in enumerate(batched_samples):
            # print(_i, sample.keys())
            if self.config.train_cfg:
                # concat negative prompts to sample prompts to avoid two forward passes
                embeds = torch.cat([sample["negative_prompt_embeds"], sample["prompt_embeds"]])
            else:
                embeds = sample["prompt_embeds"]

            for j in range(self.num_train_timesteps):
                with self.accelerator.accumulate(self.sd_pipeline.unet):
                    loss, approx_kl, clipfrac = self.calculate_loss(
                        sample["latents"][:, j],
                        sample["timesteps"][:, j],
                        sample["next_latents"][:, j],
                        sample["log_probs"][:, j],
                        sample["advantages"],
                        embeds,
                    )
                    info["approx_kl"].append(approx_kl)
                    info["clipfrac"].append(clipfrac)
                    info["loss"].append(loss)

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.trainable_layers.parameters()
                            if not isinstance(self.trainable_layers, list)
                            else self.trainable_layers,
                            self.config.train_max_grad_norm,
                        )

                        # with torch.no_grad():
                        #     B_S = self.trainable_layers[0].grad_sample[0].shape[0]
                        #     for s_idx in range(B_S):
                        #         for (name, p) in self.named_trainable_layers:
                        #             grad_sample = p.grad_sample
                        #             # B_S = p.grad_sample[0].shape[0]
                                
                        #             for t_idx, grad_timestep in enumerate(grad_sample):
                        #                 per_sample_norms = grad_timestep.reshape(B_S, -1).norm(dim=1)
                        #                 # for norm in per_sample_norms:
                        #                 norm = per_sample_norms[s_idx]
                        #                 self.accelerator.log(
                        #                     {
                        #                         f"per_sample_grads/{name}/timestep_{t_idx}": norm.item(),
                        #                         "sample_steps": self.sample_steps
                        #                     }
                        #                 )
                                
                        #         self.sample_steps += 1

                        #     for p in self.trainable_layers:
                        #         p.grad_sample = None

                    # p = self.trainable_layers[-1]
                    # print("Timestep ", j)
                    # if j == 0:
                    #     print(len(p.grad_sample), [sub_p.shape for sub_p in p.grad_sample], [sub_p.norm() for sub_p in p.grad_sample], "sample grad", embeds.shape)
                    # else:
                    #     print(len(p.grad_sample), [sub_p.shape for sub_p in p.grad_sample], [sub_p.reshape(sub_p.shape[0], -1).norm(dim=1) for sub_p in p.grad_sample], "sample grad", embeds.shape)
                    # for p in self.trainable_layers:
                    #     if hasattr(p, "grad_sample"):
                    #         # print(p.grad_sample)
                    #         print(len(p.grad_sample), [sub_p.shape for sub_p in p.grad_sample], [sub_p.reshape(sub_p.shape[0], -1).norm(dim=1) for sub_p in p.grad_sample], "sample grad", embeds.shape)
                    #         print()
                    # exit()
                        # print("Grads", flat_grad(self.trainable_layers).shape)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    # log training-related stuff
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = self.accelerator.reduce(info, reduction="mean")
                    info.update({"epoch": epoch, "inner_epoch": inner_epoch, "global_step": global_step})
                    self.accelerator.log(info)
                    global_step += 1
                    info = defaultdict(list)
            # exit()
        return global_step

    def _config_check(self) -> tuple[bool, str]:
        samples_per_epoch = (
            self.config.sample_batch_size * self.accelerator.num_processes * self.config.sample_num_batches_per_epoch
        )
        total_train_batch_size = (
            self.config.train_batch_size
            * self.accelerator.num_processes
            * self.config.train_gradient_accumulation_steps
        )

        if not self.config.sample_batch_size >= self.config.train_batch_size:
            return (
                False,
                f"Sample batch size ({self.config.sample_batch_size}) must be greater than or equal to the train batch size ({self.config.train_batch_size})",
            )
        if not self.config.sample_batch_size % self.config.train_batch_size == 0:
            return (
                False,
                f"Sample batch size ({self.config.sample_batch_size}) must be divisible by the train batch size ({self.config.train_batch_size})",
            )
        if not samples_per_epoch % total_train_batch_size == 0:
            return (
                False,
                f"Number of samples per epoch ({samples_per_epoch}) must be divisible by the total train batch size ({total_train_batch_size})",
            )
        return True, ""

    def train(self, epochs: Optional[int] = None):
        """
        Train the model for a given number of epochs
        """
        global_step = 0
        if epochs is None:
            epochs = self.config.num_epochs
        
        eval_every = 1
        for epoch in range(self.first_epoch, epochs):
            global_step = self.step(epoch, global_step)

            if epoch % eval_every == 0 and self.eval_prompts:
                tail_prompts = [p for p in self.eval_prompts if any(cl in p for cl in self.tail_classes)]
                head_prompts = [p for p in self.eval_prompts if p not in tail_prompts]
                if len(tail_prompts) > 0:
                    self.evaluate(tail_prompts, self.label_mapping, global_step, "tail_validation")
                self.evaluate(head_prompts, self.label_mapping, global_step, "head_validation")


    @torch.no_grad
    def evaluate(
        self,
        eval_prompts: list[str],
        label_mapping: dict[str, int],
        step: int,
        eval_name: str = "validation",
    ):
        """
        Run a no-grad evaluation pass.

        • Generates samples for the fixed `eval_prompts`
        • Computes rewards
        • Fires the user-supplied callback
        • Logs raw rewards and summary stats
        """
        if not eval_prompts:
            return  # nothing to do

        self.sd_pipeline.unet.eval()

        # ── sampling ────────────────────────────────────────────────────────
        samples, prompt_image_data = self.generate_eval_samples(
            eval_prompts=eval_prompts,
            label_mapping=label_mapping,
            batch_size=self.config.sample_batch_size,
        )

        # ── reward computation ─────────────────────────────────────────────
        rewards, rewards_meta = self.compute_rewards(
            prompt_image_data, step=step, is_async=False, log_rewards=False
        )
        # print("evaluation")
        # print("rewards", rewards)
        # print("rewards meta", rewards_meta)

        for i, triple in enumerate(prompt_image_data):
            triple.extend([rewards[i], rewards_meta[i]])

        # ── optional user hook ─────────────────────────────────────────────
        if self.image_samples_callback is not None:
            try:  # eval_logger has 4 args, training logger has 3
                self.image_samples_callback(
                    prompt_image_data, step, self.accelerator.trackers[0], name=eval_name
                )
            except TypeError:
                self.image_samples_callback(
                    prompt_image_data, step, self.accelerator.trackers[0]
                )

        # ── logging ────────────────────────────────────────────────────────
        rewards = torch.cat(rewards)
        rewards = self.accelerator.gather(rewards).cpu().numpy()

        merged_reward_meta = {}
        for d in rewards_meta:
            for k, v in d.items():
                merged_reward_meta.setdefault(k, []).append(v)
        merged_reward_meta = {k: torch.cat(v_list, dim=0) for k, v_list in merged_reward_meta.items()}

        # print("merged")
        # print(merged_reward_meta)
        meta_stats = {}
        for k, v in merged_reward_meta.items():
            if v.dtype == torch.bool:                        # binary reward
                v_f = v.float()                              # True→1, False→0
                meta_stats[f"{eval_name}/{k}_acc"]  = v_f.mean()
                meta_stats[f"{eval_name}/{k}_pos"]  = v_f.sum()
            else:                                            # scalar reward
                meta_stats[f"{eval_name}/{k}_mean"] = v.mean()
                meta_stats[f"{eval_name}/{k}_std"]  = v.std()

        self.accelerator.log(
            meta_stats | {
                f"{eval_name}/reward": rewards,
                f"{eval_name}/reward_mean": rewards.mean(),
                f"{eval_name}/reward_std": rewards.std(),
                "global_step": step
            },
            # step=step,
        )

        self.sd_pipeline.unet.train()

    def _save_pretrained(self, save_directory):
        self.sd_pipeline.save_pretrained(save_directory)
        self.create_model_card()

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        # normalize `tags` to a mutable set
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)

        if hasattr(self.model.config, "unsloth_version"):
            tags.add("unsloth")

        tags.update(self._tag_names)

        citation = textwrap.dedent("""\
        @inproceedings{black2024training,
            title        = {{Training Diffusion Models with Reinforcement Learning}},
            author       = {Kevin Black and Michael Janner and Yilun Du and Ilya Kostrikov and Sergey Levine},
            year         = 2024,
            booktitle    = {The Twelfth International Conference on Learning Representations, {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
            publisher    = {OpenReview.net},
            url          = {https://openreview.net/forum?id=YCWjhGrJFD},
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="DDPO",
            trainer_citation=citation,
            paper_title="Training Diffusion Models with Reinforcement Learning",
            paper_id="2305.13301",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))