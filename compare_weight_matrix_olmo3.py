# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch



model_before = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-7B-Think-DPO")
model_after = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-7B-Think")

for i in range(32):
	weight_a = model_before.model.layers[i].self_attn.o_proj.weight
	weight_b = model_after.model.layers[i].self_attn.o_proj.weight
	weight_a = weight_a.to("cpu")
	weight_b = weight_b.to("cpu")
	abs_diff = torch.abs(weight_a - weight_b)
	torch.save(abs_diff, "weight_difference/o/" + str(i) + ".pt")


	weight_a = model_before.model.layers[i].mlp.down_proj.weight
	weight_b = model_after.model.layers[i].mlp.down_proj.weight
	weight_a = weight_a.to("cpu")
	weight_b = weight_b.to("cpu")
	abs_diff = torch.abs(weight_a - weight_b)
	torch.save(abs_diff, "weight_difference/down/" + str(i) + ".pt")


