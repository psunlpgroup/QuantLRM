import torch
from datasets import load_dataset
import json


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "reasoning":
        with open("reasoning_data.txt", "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f if line.strip()]

        samples = []
        for example in dataset:
            line = example["deepseek_reasoning"].strip()
            enc  = tokenizer.encode(line)
            if len(enc)==0:
                continue
            samples.append(torch.tensor([enc]))

        cat = torch.cat(samples, dim=1)[:,:128*2048]
        n_blocks = cat.shape[1] // block_size
        print(f" * Split into {n_blocks} blocks of {block_size} tokens")
        return [cat[:, i*block_size:(i+1)*block_size] for i in range(n_blocks)]
        
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]
