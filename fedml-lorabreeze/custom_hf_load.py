import torch

from transformers import AutoModel, pipeline

# model = AutoModel.from_pretrained("ernest/zephyr_7b_beta_bnb_int4")
pipe = pipeline("text-generation", model="ernest/zephyr_7b_beta_bnb_int4", torch_dtype=torch.bfloat16, device_map="auto")

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate."
    },
    {
        "role": "user",
        "content": "How many helicopters can a human eat in oe sitting?"
    }
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
