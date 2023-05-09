# [Issho](https://issho.ai)

An open-source, less-sucky text generation web UI for Large Language Models like LLaMA.

**Try it here: [Issho](https://issho.ai)**

## Features

- **Can run LLaMA 30B model (4-bit) on a 4090 with no out of memory problems**
- **Persistent settings:** Prompt and settings are saved in localstorage

## Why not `oobabooga/text-generation-webui` ?

(Complaints as of March 2023 version of oobabooga)

- **UI bugs:** In oobabooga, output in UI is wrong if input is being truncated due to length. The generated output may not even be shown. This is a serious bug that affects usability for writing long stories. In general, the oobabooga UI is clunky and not fun to use
- **Out of memory errors:** In oobabooga, inference with long prompts and borderline amounts of VRAM will lead to CUDA out of memory error in PyTorch. This is due to various reasons, but largely memory fragmentation. Issho doesn't have this problem. It can run a 30B model on a RTX 4090 with no issues, even with prompts that span the entire context window.
- **Only support 1 user:** Issho supports placing users into a waiting queue.
- **Synchronous:** In oobabooga, inference calls are blocking. Issho does inference in an asyncio event loop. This event loop also handles multiple user websocket connections.

## Setup

### Frontend

```
yarn install
yarn dev # dev build
yarn build && yarn start # prod build
```

### Backend

Dependencies

```
sudo apt install build-essential
git submodule update --init --recursive
conda create -n textgen python=3.10.9
conda activate textgen
mamba install pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install -c conda-forge cudatoolkit-dev
pip install -r requirements.txt # note: tested with transformers @ git+https://github.com/huggingface/transformers@fb366b9a2a94b38171896f6ba9fb9ae8bffd77af
```

Models (example: LLaMA 30B 4bit)

```
mkdir models/
cd models/
wget https://raw.githubusercontent.com/qwopqwop200/GPTQ-for-LLaMa/triton/convert_llama_weights_to_hf.py
python convert_llama_weights_to_hf.py --input_dir /path/to/downloaded/llama/weights --model_size 30B --output_dir ./llama-30b
wget https://huggingface.co/maderix/llama-65b-4bit/resolve/main/llama30b-4bit.pt
ls # output: "llama-30b  llama-30b-4bit.pt"
```


### Nginx

Set up reverse proxy to point `/` to Next.js frontend, point `/ws` to backend websocket server
