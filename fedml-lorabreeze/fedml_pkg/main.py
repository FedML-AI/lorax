import httpx
import os
import copy
import uvicorn
import requests

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

            accept_type = request.headers.get("Accept", "application/json")
            assert accept_type == "application/json" or accept_type == "*/*"

            is_streaming = input_json.get("stream", False)

            lorax_base_url = "http://127.0.0.1:80/v1"

            print(f"Input JSON: {input_json}")
            lorax_json = copy.deepcopy(input_json)

            lorax_json["model"] = input_json.get("adapter_id", os.getenv("LORA_BASE_MODEL_ID", None))

            if lorax_json["model"] is None:
                return {"error": "Request must have \"adapter_id\" in the request body."}

            print(f"lorax_json: {lorax_json}")

            lorax_header = {
                "Content-Type": "application/json",
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
            local_health_url = "http://127.0.0.1:80/health"
            response = None
            try:
                response = requests.get(local_health_url, timeout=3)
            except:
                pass

            if not response or response.status_code != 200:
                # Return 408 code
                raise HTTPException(
                    status_code=HTTPStatus.REQUEST_TIMEOUT.value,
                    detail=f"Local server is not ready."
                )
            return True

        port = 2345
        uvicorn.run(api, host="0.0.0.0", port=port)


if __name__ == "__main__":
    fedml_inference_runner = LoraxChatCompletionInferenceRunner()
    fedml_inference_runner.run()
