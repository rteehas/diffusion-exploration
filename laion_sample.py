#!/usr/bin/env python3

import io
import itertools
import logging
import random
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Optional

import numpy as np
import requests
from PIL import Image
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, T5TokenizerFast

# taken from http://gist.github.com/borzunov/5f493e3c18bfa90d4de0530eb214a250
# Hide urllib warnings
logging.getLogger('urllib3.connection').setLevel(logging.ERROR)
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)

# Hide PIL warnings
for message in [
    "Palette images with Transparency expressed in bytes should be converted to RGBA images",
    "image file could not be identified because WEBP support not installed",
]:
    warnings.filterwarnings("ignore", category=UserWarning, message=message)


executor = ThreadPoolExecutor(32)


def download_image(url):
    try:
        r = requests.get(url, timeout=1)

        img = Image.open(io.BytesIO(r.content))
        img = img.resize((256, 256))
        img = img.convert('RGB')
        return img
    except Exception as e:
        logging.debug('Failed to download `{url}`', exception=True)
        return None


def preprocess_batch(batch):
    mask = [
        (
            caption is not None and len(caption) >= 3 and
            nsfw == 'UNLIKELY' and
            orig_width > 0 and orig_height > 0 and
            max(orig_height / orig_width, orig_width / orig_height) <= 2
        ) for caption, nsfw, orig_width, orig_height in
        zip(batch['caption'], batch['NSFW'], batch['original_width'], batch['original_height'])
    ]
    # logging.debug(f'{np.mean(mask) * 100:.1f}% of examples left after filtering')

    # if any(mask):
    #     result = tokenizer(list(itertools.compress(batch['TEXT'], mask)),
    #                        add_special_tokens=False,
    #                        max_length=max_sequence_length,
    #                        padding='max_length',
    #                        return_attention_mask=False,
    #                        return_tensors='pt',
    #                        truncation=True)
    # else:
    #     # This branch is necessary because tokenizer([]) raises IndexError
    #     result = {'input_ids': []}
    # print(mask)
    batch['image'] = list(executor.map(download_image, itertools.compress(batch['url'], mask)))
    for key in batch:
        if len(batch[key]) > len(batch["image"]):
            new_values = []
            for i, item in enumerate(batch[key]):
                if mask[i] == True:
                    new_values.append(batch[key][i])

            batch[key] = new_values
    # mask = [item is not None for item in result['image']]
    # result = {key: list(itertools.compress(values, mask))
    #           for key, values in result.items()}
    for key in batch:
        if key != "image":
            new_values = []
            for i, item in enumerate(batch[key]):
                if batch['image'][i] is not None:
                    new_values.append(item)
            
            batch[key] = new_values
    
    batch['image'] = [img for img in batch['image'] if img is not None]
    return batch


def make_dataset(
    shuffle_seed: Optional[int],
    shuffle_buffer_size: int = 4096,
    preprocessing_batch_size: int = 4096
):
    ds = load_dataset('laion/laion400m', split='train', streaming=True) # add token or this will fail
    ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.map(lambda batch: preprocess_batch(batch),
                batch_size=preprocessing_batch_size,
                batched=True)
    # ds = ds.with_format('torch')
    # ds = Dataset.from_generator(lambda: ds.take(100))
    return ds


# ds = make_dataset(shuffle_seed=42)
# dl = DataLoader(ds, batch_size=1, num_workers=1, pin_memory=True)
# # ds.save_to_disk("laion_subsample")
# # Note: Do not set num_workers > 1 since it leads to streaming duplicate data,
# # see https://github.com/huggingface/datasets/issues/3423

# start_time = time.monotonic()
# n_samples = 0
# for item in dl:
#     n_samples += dl.batch_size
#     if n_samples % 10 == 0:
#         elapsed = time.monotonic() - start_time
#         print(f'{n_samples=} | {elapsed=:.1f} sec | Average speed: {n_samples / elapsed:.1f} samples/sec')



