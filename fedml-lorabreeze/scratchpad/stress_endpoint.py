import asyncio
import os
import utils
import random
import time

import numpy as np

from collections import defaultdict
from openai import OpenAI

CURRENT_DIR = os.path.dirname(__file__)
LATENCIES = defaultdict(list)
ENDPOINT = utils.load_endpoint(os.path.join(CURRENT_DIR, "endpoint.yaml"))
ENDPOINT_ADAPTERS = ENDPOINT.lorax_hugging_face_adaptors

client = OpenAI(
    api_key="test",  # this can be anything but definitely non-empy!
    base_url="http://127.0.0.1:{}/v1".format(ENDPOINT.lorax_port)
)


async def model_adapter_https(adapter_id=None):

    st = time.time()
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test"
            }
        ],
        model=adapter_id,
        max_tokens=512,
        temperature=0.5,
        top_p=0.7
    )
    et = time.time()
    LATENCIES[adapter_id].append(et - st)

    # print the output
    print(completion.choices[0].message.content)


def random_generator(rand_seed=0, lb=1, ub=5):
    random.seed(rand_seed)
    return random.randint(lb, ub)


async def main():
    for _ in range(1):
        for adapter_id in ENDPOINT_ADAPTERS:
            await asyncio.sleep(1)
            for _ in range(10):
                await model_adapter_https(adapter_id)


asyncio.run(main())

for k, v in LATENCIES.items():
    print(k, np.mean(v), v)
