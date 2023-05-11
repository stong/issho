import re
import sys
from pathlib import Path

import accelerate
import torch

sys.path.insert(0, str(Path("repositories/GPTQ-for-LLaMa")))
import llama
import llama_inference_offload
import opt


def load_quantized(path_to_model, pt_path, gptq_bits=0, gptq_pre_layer=0):
    if not gptq_pre_layer:
        load_quant = llama.load_quant
    else:
        load_quant = llama_inference_offload.load_quant

    # qwopqwop200's offload
    if gptq_pre_layer:
        model = load_quant(path_to_model, pt_path, gptq_bits, gptq_pre_layer)
    else:
        model = load_quant(path_to_model, pt_path, gptq_bits, groupsize=128)
        DEV = torch.device('cuda:0')
        for i in range(len(model.model.layers)):
            model.model.layers[i].to(DEV)
        model.model.embed_tokens.to(DEV)
        model.model.norm.to(DEV)
        model.lm_head.to(DEV)

    return model
