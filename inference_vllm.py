from vllm import LLM, SamplingParams
import os, json, itertools, bisect
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
from datasets import load_dataset

# For generative models (task=generate) only
llm = LLM(model="models/R1-Qwen3-8B-w3-g128", tokenizer="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", tensor_parallel_size=1, gpu_memory_utilization=0.96, dtype=torch.float16)


dataset = load_dataset("yale-nlp/FOLIO")["validation"]
total_input = []
for d in dataset:
    total_input.append("Use logical deductions to determine whether the provided conclusion is true, false, or uncertain based on premise. Consider all relevant information to reach a logical conclusion.\n\n*******\nPremise: " +  d['premises'] + "\n*******\n\n*******\nConclusion: " + d['conclusion'] + "\n*******<think>\n")

sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)


# For generative models (task=generate) only
outputs = llm.generate(total_input, sampling_params)

i = 0
for output in outputs:
    generated_text = output.outputs[0].text
    with open('output/R1-Qwen3-8B-w3-g128/FOLIO/' + str(i) + '.txt', 'a') as the_file:
        the_file.write(generated_text)
    i += 1




json_file = "temporal_sequences.json"
with open(json_file, "r") as f:
    dataset = json.load(f)

total_input = []
for d in dataset['examples']:
    total_input.append("Use the timeline provided and answer step by step. Finally give the index (the letter) of the actual correct answer.\n" + d['input'] + "\n<think>\n")

sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)


# For generative models (task=generate) only
outputs = llm.generate(total_input, sampling_params)

i = 0
for output in outputs:
    generated_text = output.outputs[0].text
    with open('output/R1-Qwen3-8B-w3-g128/temporal_sequences/' + str(i) + '.txt', 'a') as the_file:
        the_file.write(generated_text)
    i += 1




dataset = load_dataset("fingertap/GPQA-Diamond")["test"]
total_input = []
for d in dataset:
    total_input.append("Answer the question by giving the index (the letter) of the actual correct answer.\n" + d['question'] + "\n<think>\n")

sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)


# For generative models (task=generate) only
outputs = llm.generate(total_input, sampling_params)

i = 0
for output in outputs:
    generated_text = output.outputs[0].text
    with open('output/R1-Qwen3-8B-w3-g128/GPQA_diamond/' + str(i) + '.txt', 'a') as the_file:
        the_file.write(generated_text)
    i += 1


dataset = load_dataset("ChuGyouk/AIME-22-25")["train"]

total_input = []
for d in dataset:
    total_input.append(d['problem'] + "\n<think>\n")

sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)


outputs = llm.generate(total_input, sampling_params)

i = 0
for output in outputs:
    generated_text = output.outputs[0].text
    with open('output/R1-Qwen3-8B-w3-g128/AIME_120/' + str(i) + '.txt', 'a') as the_file:
        the_file.write(generated_text)
    i += 1

