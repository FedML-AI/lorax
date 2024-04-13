import torch

from transformers import pipeline, AutoModelForCausalLM, BitsAndBytesConfig


def run_pipeline():
    pipe = pipeline("text-generation", model="ernest/zephyr_7b_beta_bnb_int4", device_map="auto")

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


def get_causal_lm_model():
    # Base model
    model_name_or_path = "ernest/zephyr_7b_beta_bnb_int4"

    # Load the model in int4
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 3},
        # load_in_4bit=True,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type='nf4',
        # )
    )
    model.eval()
    return model


if __name__ == "__main__":
    # run_pipeline()
    get_causal_lm_model()
