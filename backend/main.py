import sys

import asyncio
import taskgroup

from pathlib import Path
from transformers import AutoTokenizer
import GPTQ_loader
import text_generation
import torch

if len(sys.argv) < 3:
    print('usage: %s <model-name> <path-to-pt-or-safetensors>' % sys.argv[0])
    exit(1)

torch.set_num_threads(12) # todo use nproc
torch.get_num_threads()

class Settings:
    def __init__(self):
        self.model_name = sys.argv[1]
        self.pt_path = sys.argv[2]

settings = Settings()

model = GPTQ_loader.load_quantized(f'models/{settings.model_name}/', settings.pt_path, gptq_bits=4)#, gptq_pre_layer=60)
tokenizer = AutoTokenizer.from_pretrained(Path(f"models/{settings.model_name}/"))
tokenizer.truncation_side = 'left'

lock = asyncio.Lock()

async def generate(prompt, max_tokens, temperature, top_p, top_k):
    async with lock:
        with torch.no_grad():
            async for x in text_generation.generate_reply(model, tokenizer,
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                typical_p=1,
                repetition_penalty=1.1,
                encoder_repetition_penalty=1,
                top_k=top_k,
                no_repeat_ngram_size=0,
                num_beams=1,
                penalty_alpha=0,
                length_penalty=1,
                early_stopping=False,
                seed=-1,
                eos_token=None,
                stopping_string=None
            ):
                yield x

import websockets
import json
import traceback
import time

from enum import Enum
ClientState = Enum('ClientState', 'idle busy cooldown')

generation_queue = asyncio.Queue()

class Client:
    def __init__(self, websocket):
        self.websocket = websocket
        self.state = ClientState.idle
        self.generate_task = None
        self.is_generating = False

    async def send_msg(self, status, **kwargs):
        msg = kwargs
        msg['status'] = status
        await self.websocket.send(json.dumps(msg))
    
    async def set_state(self, new_state):
        self.state = new_state
        await self.send_msg('set_state', new_state=new_state.name)

    async def start_cooldown(self):
        await self.set_state(ClientState.cooldown)
        
        async def reset_cooldown():
            for i in range(10, 0, -1):
                await self.send_msg('cooldown_timer', seconds_remaining=i)
                await asyncio.sleep(1)
                assert self.state == ClientState.cooldown
            await self.set_state(ClientState.idle)
            print('Cooldown over')
        asyncio.create_task(reset_cooldown())

    def cancel_task(self):
        if self.generate_task:
            self.generate_task.cancel()
            self.generate_task = None
            print('Task cancelled')

    async def safe_do(self, awaitable):
        try:
            return await awaitable
        except Exception as e:
            traceback.print_exc()
            await self.start_cooldown()
            self.cancel_task()
            await self.send_msg('error', message=str(e))

    async def read_messages(self):
        while True:
            message = await self.websocket.recv()
            await self.safe_do(self.handle_message(message))

    async def handle_message(self, websocket_message):
        message = json.loads(websocket_message)
        # print(f"Request: {json.dumps(message, indent=4)}")

        action = message['action']
        
        if self.state == ClientState.idle:

            if action == 'generate':
                await self.set_state(ClientState.busy)
                
                prompt = message['prompt']
                max_tokens = int(message['max_tokens'])
                temperature = float(message['temperature'])
                top_p = float(message['top_p'])
                top_k = int(message['top_k'])

                max_tokens = min(max(max_tokens, 1), 400)

                start_signal = asyncio.Condition()

                async def generate_task():
                    async with start_signal:
                        await start_signal.wait()
                    print('[DEBUG] Lets go!')
                    await self.safe_do(self.process_generation(prompt, max_tokens, temperature, top_p, top_k))

                self.generate_task = asyncio.create_task(generate_task())
                await generation_queue.put((self, start_signal, self.generate_task))

                if generation_queue.qsize() > 0:
                    await self.send_msg('notification', message=f'You are position {generation_queue.qsize()} in the queue')

            else:
                print('Unknown action ' + action)
        
        elif self.state == ClientState.busy:
            if action == 'cancel':
                self.cancel_task()

                if self.is_generating:
                    await self.start_cooldown()
                    self.is_generating = False
                else:
                    await self.set_state(ClientState.idle)
            
            else:
                print('Dropping message, already busy')

        elif self.state == ClientState.cooldown:
            print('Dropping message, in cooldown')


    async def process_generation(self, prompt, *args, **kwargs):
        self.is_generating = True

        async for inprogress_completion in generate(prompt, *args, **kwargs):
            await self.send_msg('progress', completion=inprogress_completion)
        
        print('Completion done')

        self.is_generating = False
        await self.start_cooldown()


async def handle_client(websocket, path):
    try:
        c = Client(websocket)
        await c.read_messages()
    except websockets.ConnectionClosed:
        print("Connection closed.")
        c.cancel_task()

async def process_queue():
    while True:
        client, start_signal, task = await generation_queue.get()
        print(f'Now doing {client} ...')

        try:
            async with start_signal:
                start_signal.notify()
            await task
            print('[DEBUG] Done awaiting task')
        except asyncio.CancelledError:
            print('[DEBUG] Task cancelled')
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()

async def main():
    print(torch.get_num_threads())
    t = asyncio.create_task(process_queue())
    await websockets.serve(handle_client, "0.0.0.0", 8765)
    await t

asyncio.run(main())
