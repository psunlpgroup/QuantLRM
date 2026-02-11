# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os


model_original = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
model_distill = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")

for i in range(36):
	weight_a = model_original.model.layers[i].self_attn.q_proj.weight
	weight_b = model_distill.model.layers[i].self_attn.q_proj.weight
	weight_a = weight_a.to("cpu")
	weight_b = weight_b.to("cpu")
	abs_diff = torch.abs(weight_a - weight_b)
	os.makedirs("weight_difference/q/", exist_ok=True)
	torch.save(abs_diff, "weight_difference/q/" + str(i) + ".pt")

	weight_a = model_original.model.layers[i].self_attn.k_proj.weight
	weight_b = model_distill.model.layers[i].self_attn.k_proj.weight
	weight_a = weight_a.to("cpu")
	weight_b = weight_b.to("cpu")
	abs_diff = torch.abs(weight_a - weight_b)
	os.makedirs("weight_difference/k/", exist_ok=True)
	torch.save(abs_diff, "weight_difference/k/" + str(i) + ".pt")

	weight_a = model_original.model.layers[i].self_attn.v_proj.weight
	weight_b = model_distill.model.layers[i].self_attn.v_proj.weight
	weight_a = weight_a.to("cpu")
	weight_b = weight_b.to("cpu")
	abs_diff = torch.abs(weight_a - weight_b)
	os.makedirs("weight_difference/v/", exist_ok=True)
	torch.save(abs_diff, "weight_difference/v/" + str(i) + ".pt")


	weight_a = model_original.model.layers[i].self_attn.o_proj.weight
	weight_b = model_distill.model.layers[i].self_attn.o_proj.weight
	weight_a = weight_a.to("cpu")
	weight_b = weight_b.to("cpu")
	abs_diff = torch.abs(weight_a - weight_b)
	os.makedirs("weight_difference/o/", exist_ok=True)
	torch.save(abs_diff, "weight_difference/o/" + str(i) + ".pt")


	weight_a = model_original.model.layers[i].mlp.gate_proj.weight
	weight_b = model_distill.model.layers[i].mlp.gate_proj.weight
	weight_a = weight_a.to("cpu")
	weight_b = weight_b.to("cpu")
	abs_diff = torch.abs(weight_a - weight_b)
	os.makedirs("weight_difference/gate/", exist_ok=True)
	torch.save(abs_diff, "weight_difference/gate/" + str(i) + ".pt")


	weight_a = model_original.model.layers[i].mlp.up_proj.weight
	weight_b = model_distill.model.layers[i].mlp.up_proj.weight
	weight_a = weight_a.to("cpu")
	weight_b = weight_b.to("cpu")
	abs_diff = torch.abs(weight_a - weight_b)
	os.makedirs("weight_difference/up/", exist_ok=True)
	torch.save(abs_diff, "weight_difference/up/" + str(i) + ".pt")


	weight_a = model_original.model.layers[i].mlp.down_proj.weight
	weight_b = model_distill.model.layers[i].mlp.down_proj.weight
	weight_a = weight_a.to("cpu")
	weight_b = weight_b.to("cpu")
	abs_diff = torch.abs(weight_a - weight_b)
	os.makedirs("weight_difference/down/", exist_ok=True)
	torch.save(abs_diff, "weight_difference/down/" + str(i) + ".pt")


