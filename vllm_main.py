from typing import Any, Dict, List

import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rich import print as rprint

from vllm import LLM, SamplingParams


assert Version(ray.__version__) >= Version(
    "2.22.0"
), "Ray version must be at least 2.22.0"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
prompts = prompts * 20
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
llm = LLM(model=model_id, tensor_parallel_size=4)

outputs = []
for prompt in prompts:
    outputs.append(llm.generate([prompt], sampling_params)[0])

# outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    rprint(
        f"[cyan bold]Prompt:[/] [green]{prompt!r}[/], [cyan bold]Generated text:[/] [yellow]{generated_text!r}[/]"
    )

# from vllm import LLM, SamplingParams

# # Set tensor parallelism per instance
# tensor_parallel_size = 8

# llm = LLM(
#     model="meta-llama/Meta-Llama-3.1-405B-Instruct",
#     tensor_parallel_size=tensor_parallel_size,
# )
# sampling_params = SamplingParams(temperature=0.5)


# def print_outputs(outputs):
#     for output in outputs:
#         prompt = output.prompt
#         generated_text = output.outputs[0].text
#         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
#     print("-" * 80)


# print("=" * 80)

# # In this script, we demonstrate how to pass input to the chat method:

# conversation = [
#     {"role": "system", "content": "You are a helpful assistant"},
#     {"role": "user", "content": "Hello"},
#     {"role": "assistant", "content": "Hello! How can I assist you today?"},
#     {
#         "role": "user",
#         "content": "Write an essay about the importance of higher education.",
#     },
# ]
# outputs = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
# print_outputs(outputs)

# # You can run batch inference with llm.chat API
# conversation = [
#     {"role": "system", "content": "You are a helpful assistant"},
#     {"role": "user", "content": "Hello"},
#     {"role": "assistant", "content": "Hello! How can I assist you today?"},
#     {
#         "role": "user",
#         "content": "Write an essay about the importance of higher education.",
#     },
# ]
# conversations = [conversation for _ in range(10)]

# # We turn on tqdm progress bar to verify it's indeed running batch inference
# outputs = llm.chat(
#     messages=conversations, sampling_params=sampling_params, use_tqdm=True
# )
# print_outputs(outputs)

# # A chat template can be optionally supplied.
# # If not, the model will use its default chat template.

# # with open('template_falcon_180b.jinja', "r") as f:
# #     chat_template = f.read()

# # outputs = llm.chat(
# #     conversations,
# #     sampling_params=sampling_params,
# #     use_tqdm=False,
# #     chat_template=chat_template,
# # )
