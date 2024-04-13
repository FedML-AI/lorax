import httpx
import os
import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fedml.serving import FedMLInferenceRunner
from http import HTTPStatus


class LoraxChatCompletionInferenceRunner(FedMLInferenceRunner):
    def __init__(self):
        super().__init__(None)

        max_new_tokens = os.getenv("MAX_NEW_TOKENS", "")
        try:
            max_new_tokens = int(max_new_tokens)
        except (ValueError, TypeError):
            max_new_tokens = None
        else:
            if max_new_tokens <= 0:
                max_new_tokens = None

        self.default_generation_config = dict(
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            top_p=float(os.getenv("TOP_P", 0.95)),
            max_tokens=max_new_tokens
        )

    @staticmethod
    async def stream_generator(inference_url, input_json, header=None):
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", inference_url, json=input_json, headers=header, timeout=10) as response:
                async for chunk in response.aiter_lines():
                    # we consumed a newline, need to put it back
                    yield f"{chunk}\n"

    def run(self) -> None:
        api = FastAPI()

        @api.post("/predict")
        @api.post("/completions")
        @api.post("/chat/completions")
        async def predict(request: Request):
            input_json = await request.json()
            adaptor_to_owner = {
                "qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k": "vineetsharma",
                "dolphin-2.6-mistral-7b-dpo-laser-function-calling-lora": "Yhyu13",
                "FuncMaster-v0.1-Mistral-7B-Instruct-Lora": "allyson-ai",
                "Mistral-7B-LoRA-AudioWhisper": "sshh12",
            }

            accept_type = request.headers.get("Accept", "application/json")
            assert accept_type == "application/json" or accept_type == "*/*"

            is_streaming = input_json.get("stream", False)

            lorax_base_url = "http://38.101.196.134:8080/v1"
            lorax_json = self.default_generation_config.copy()
            lorax_json.update(input_json.copy())

            fullname = f"{adaptor_to_owner[lorax_json['+ ']]}/{lorax_json['model']}"
            lorax_json["model"] = fullname

            lorax_header = {
                "Content-Type": "application/json",
                # "Authorization": f"Bearer {groq_api_key}",
            }

            if "messages" in input_json:
                lorax_inference_url = f"{lorax_base_url}/chat/completions"
            elif "prompt" in input_json:
                lorax_inference_url = f"{lorax_base_url}/completions"
            else:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail=f"Request must have either \"messages\" or \"prompt\" in the request body."
                )

            if is_streaming:
                print(f"Streaming response {lorax_inference_url} with {lorax_json} and {lorax_header}")
                return StreamingResponse(
                    self.stream_generator(lorax_inference_url, header=lorax_header, input_json=lorax_json),
                    media_type="text/event-stream",
                    headers={
                        "Content-Type": request.headers.get("Accept", "text/event-stream"),
                        "Cache-Control": "no-cache",
                    }
                )
            else:
                return httpx.post(lorax_inference_url, json=lorax_json, headers=lorax_header).json()

        @api.get("/ready")
        async def ready():
            return {"status": "Success"}

        port = 2345
        uvicorn.run(api, host="0.0.0.0", port=port)


if __name__ == "__main__":
    fedml_inference_runner = LoraxChatCompletionInferenceRunner()
    fedml_inference_runner.run()
