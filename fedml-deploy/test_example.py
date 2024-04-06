import asyncio
import random
from openai import OpenAI

client = OpenAI(
    # put your FEDML API key here
    api_key="851497657a944e898d5fd3f373cf0ec0",
    base_url="https://open.fedml.ai/inference/api/v1"
)

async def model_adapter_https(adapter_id=None):

    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Say this is a test"}],
        model=adapter_id,
        max_tokens=512,
        temperature=0.5,
        top_p=0.7
    )

    # print the output
    print(completion.choices[0].message.content)


def random_generator(rand_seed):
    random.seed(rand_seed)
    return random.randint(1, 5), random.randint(1, 5), random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)

async def main():
    rand_seed = 0  # replace with actual seed
    for _ in range(100):
        await asyncio.sleep(1)
        a, b, c, d, e = random_generator(rand_seed)
        for _ in range(a):
            await model_adapter_https('fedml-deploy-official/Mistral-7B-LoRA-AudioWhisper')
        for _ in range(b):
            await model_adapter_https('fedml-deploy-official/FuncMaster-v0.1-Mistral-7B-Instruct-Lora')
        for _ in range(c):
            await model_adapter_https('fedml-deploy-official/dolphin-2.6-mistral-7b-dpo-laser-function-calling-lora')
        for _ in range(d):
            await model_adapter_https('fedml-deploy-official/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k')
        #for _ in range(e):
        #    await model_adapter_https()


asyncio.run(main())


