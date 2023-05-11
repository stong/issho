import gc
import re
import time
import traceback

import asyncio
import taskgroup # backported from asyncio for python 3.10

import numpy as np
import torch
import transformers

def get_max_prompt_length(tokens):
    max_length = 2048-tokens
    return max_length

def encode(tokenizer, prompt, tokens_to_generate=0, add_special_tokens=True):
    input_ids = tokenizer.encode(str(prompt), return_tensors='pt', truncation=True, max_length=get_max_prompt_length(tokens_to_generate), add_special_tokens=add_special_tokens)
    return input_ids.cuda()

def decode(tokenizer, output_ids):
    reply = tokenizer.decode(output_ids, skip_special_tokens=True)
    reply = reply.replace(r'<|endoftext|>', '')
    return reply

def overlapped_string_length(s1, s2):
    if not s1 or not s2:
        return 0
    if len(s1) == 1 or len(s2) == 1:
        return 1 if s2[0] == s1[-1] else 0

    # Trim s1 so it isn't longer than s2
    if len(s1) > len(s2):
        s1 = s1[-len(s2):]

    T = compute_back_track_table(s2)  # O(n)

    m = 0
    i = 0
    while m + i < len(s1):
        if s2[i] == s1[m + i]:
            i += 1
        else:
            m += i - T[i]
            if i > 0:
                i = T[i]

    return i  # Return characters matched

def compute_back_track_table(s):
    T = [0] * len(s)
    cnd = 0
    T[0] = -1
    T[1] = 0
    pos = 2
    while pos < len(s):
        if s[pos - 1] == s[cnd]:
            T[pos] = cnd + 1
            pos += 1
            cnd += 1
        elif cnd > 0:
            cnd = T[cnd]
        else:
            T[pos] = 0
            pos += 1

    return T

def clear_torch_cache():
    gc.collect()
    torch.cuda.empty_cache()

def set_manual_seed(seed):
    if seed != -1:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

async def generate_tokens(model, generate_params):
    loop = asyncio.get_running_loop()
    q = asyncio.Queue(1)
    cancelled = [False]

    async def async_callback(output):
        await q.put(output)

    class Stream(transformers.StoppingCriteria):
        def __call__(self, token_ids, scores) -> bool:
            the_token = token_ids[0].cpu()
            del token_ids
            asyncio.run_coroutine_threadsafe(async_callback(the_token), loop)
            return cancelled[0]

    stopping_criteria_list = transformers.StoppingCriteriaList()
    stopping_criteria_list.append(Stream())
    generate_params["stopping_criteria"] = stopping_criteria_list

    done = object()

    async def async_run():
        def doit():
            try:
                model.generate(**generate_params)
            except Exception as e:
                print(e)
                asyncio.run_coroutine_threadsafe(q.put(e), loop)

        await loop.run_in_executor(None, doit)
        await q.put(done)

    try:
        async with taskgroup.TaskGroup() as tg:
            tg.create_task(async_run())
            while True:
                obj = await q.get()
                if obj == done:
                    break
                elif isinstance(obj, Exception):
                    raise obj
                yield obj
    except asyncio.CancelledError as e:
        cancelled[0] = True
        raise e
    finally:
        del q

def refresh_cuda_memory():
    print("defragmenting memory")
    """
    Re-allocate all cuda memory to help alleviate fragmentation
    """
    # Run a full garbage collect first so any dangling tensors are released
    gc.collect()

    # Then move all tensors to the CPU
    locations = {}
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue

        locations[obj] = obj.device
        obj.data = obj.data.cpu()
        if isinstance(obj, torch.nn.Parameter) and obj.grad is not None:
            obj.grad.data = obj.grad.cpu()

    # Now empty the cache to flush the allocator
    torch.cuda.empty_cache()

    # Finally move the tensors back to their associated GPUs
    for tensor, device in locations.items():
        tensor.data = tensor.to(device)
        if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None:
            tensor.grad.data = tensor.grad.to(device)

async def generate_reply(model, tokenizer, question, max_new_tokens, do_sample, temperature, top_p, typical_p, repetition_penalty, encoder_repetition_penalty, top_k, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, seed, eos_token=None, stopping_string=None):
    if torch.cuda.memory_reserved() >= 20000000000:
        refresh_cuda_memory()
    else:
        clear_torch_cache()
    set_manual_seed(seed)
    t0 = time.time()

    print('generate_reply')
    print(f'seed: {seed}')
    print(f'max_new_tokens: {max_new_tokens}')
    # print(f'question: {question}')

    if question.startswith(' '):
        question = question.lstrip()

    input_ids = encode(tokenizer, question, max_new_tokens)
    original_input_ids = input_ids
    eos_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    if eos_token is not None:
        eos_token_ids.append(int(encode(eos_token)[0][-1]))
    
    generate_params = {
        'use_cache': True, # not shared.args.no_cache,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_token_ids,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "typical_p": typical_p,
        "repetition_penalty": repetition_penalty,
        "encoder_repetition_penalty": encoder_repetition_penalty,
        "top_k": top_k,
        "min_length": 0,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "num_beams": num_beams,
        "penalty_alpha": penalty_alpha,
        "length_penalty": length_penalty,
        "early_stopping": early_stopping,
    }
    generate_params["inputs"] = input_ids
    # print(f'generate_params: {generate_params}')

    yield question
    clear_torch_cache()

    prompt_tokens = list(input_ids[0].cpu())

    output = input_ids[0] # in case for loop returns nothing
    async for output in generate_tokens(model, generate_params):
        # delete stuff in the reply that came from the question
        new_output = output[overlapped_string_length(prompt_tokens, list(output)):]

        new_reply = decode(tokenizer, new_output)

        yield question + new_reply
        if output[-1] in eos_token_ids:
            break

        print(torch.cuda.memory_reserved())
        if torch.cuda.memory_reserved() >= 20000000000:
            clear_torch_cache()

    t1 = time.time()
    print(f"Output generated in {(t1-t0):.2f} seconds ({(len(output)-len(original_input_ids[0]))/(t1-t0):.2f} tokens/s, {len(output)-len(original_input_ids[0])} tokens)")
