import copy
import threading
import time

import httpx
import json
import os
import uvicorn
import requests

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fedml.serving import FedMLInferenceRunner
from http import HTTPStatus
from threading import Lock


class LoraxChatCompletionInferenceRunner(FedMLInferenceRunner):

    ARE_ADAPTERS_LOADED = False

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

        self.fedml_runner_port = 2345
        self.host = "0.0.0.0"
        self.lorax_base_url = "http://127.0.0.1:80/v1"
        self.lorax_inference_url = "http://127.0.0.1:80/generate"
        self.lorax_health_url = "http://127.0.0.1:80/health"
        self.are_adapters_loaded = False

        adapters_list = os.getenv("ADAPTERS_PRELOADED", None)
        if adapters_list:
            adapters_list = [(adapter.split(":")[0], adapter.split(":")[1])
                             for adapter in adapters_list.split("|")]
            bootstrap_adapters_thread = threading.Thread(
                target=self.boostrap_adapters,
                args=[adapters_list, self.lorax_health_url, self.lorax_inference_url])
            bootstrap_adapters_thread.start()

    @classmethod
    async def stream_generator(cls, inference_url, input_json, header=None):
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", inference_url, json=input_json, headers=header, timeout=30) as response:
                async for chunk in response.aiter_lines():
                    # we consumed a newline, need to put it back
                    yield f"{chunk}\n"

    @classmethod
    def is_service_ready(cls, service_url) -> bool:
        ready = False
        try:
            response = requests.get(service_url, timeout=3)
            if response and response.status_code == 200:
                ready = True
        except Exception as e:
            pass

        return ready

    @classmethod
    def boostrap_adapters(cls, adapters_list, lorax_health_url, lorax_inference_url) -> bool:

        while not cls.is_service_ready(lorax_health_url):
            time.sleep(1)

        payload_template = lambda a_id, a_source: \
            json.dumps({
                "inputs": "[INST]Test[/INST]",
                "parameters": {
                    "adapter_id": a_id,
                    "adapter_source": a_source
                }
            })
        try:
            if adapters_list:
                print("Loading Adapters!!", flush=True)
                for adapter in adapters_list:
                    adapter_source, adapter_id = adapter[0], adapter[1]
                    payload = payload_template(adapter_id, adapter_source)
                    headers = {'Content-Type': 'application/json'}
                    response = requests.request("POST",
                                                url=lorax_inference_url,
                                                headers=headers,
                                                data=payload)
                    print("AdapterID: `{}`, AdapterSource: `{}`, LoadMessage: `{}`".format(
                        adapter_id, adapter_source, response.text), flush=True)

        except Exception as e:
            print("Error while loading adapters: ", e, flush=True)

        cls.ARE_ADAPTERS_LOADED = True

    def run(self) -> None:

        api = FastAPI()

        @api.post("/predict")
        @api.post("/completions")
        @api.post("/chat/completions")
        async def predict(request: Request):

            # Need to make sure that the service and
            # the adapters are successfully loaded to
            # start the model serving service.
            if not self.ARE_ADAPTERS_LOADED:
                return JSONResponse(
                    status_code=HTTPStatus.TOO_EARLY,
                    content={
                        "error": True,
                        "message": "Service is not ready yet."
                    })

            input_json = await request.json()

            accept_type = request.headers.get("Accept", "application/json")
            assert accept_type == "application/json" or accept_type == "*/*"

            is_streaming = input_json.get("stream", False)
            print(f"Input JSON: {input_json}", flush=True)
            lorax_json = copy.deepcopy(input_json)

            if "adapter_id" not in lorax_json:
                return {"error": "Request must have \"adapter_id\" in the request body."}
            lorax_json["model"] = lorax_json["adapter_id"]
            print(f"lorax_json: {lorax_json}", flush=True)

            lorax_header = {
                "Content-Type": "application/json",
            }

            if "messages" in input_json:
                lorax_inference_url = f"{self.lorax_base_url}/chat/completions"
            elif "prompt" in input_json:
                lorax_inference_url = f"{self.lorax_base_url}/completions"
            else:
                return JSONResponse(
                    status_code=HTTPStatus.BAD_REQUEST,
                    content={
                        "error": True,
                        "message": "Request must have either \"messages\" or \"prompt\" in the request body."
                    })

            if is_streaming:
                print(f"Streaming response {lorax_inference_url} with {lorax_json} and {lorax_header}", flush=True)
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

            # Need to make sure that the service is online
            # and the adapters are successfully loaded to
            # start the model serving service.
            if not self.is_service_ready(self.lorax_health_url) \
                    or not self.ARE_ADAPTERS_LOADED:
                return JSONResponse(
                    status_code=HTTPStatus.TOO_EARLY,
                    content={
                        "error": True,
                        "message": "Service is not ready yet!"
                    })

            return JSONResponse(content={"message": "Service is ready!"})

        uvicorn.run(api, host=self.host, port=self.fedml_runner_port)


if __name__ == "__main__":
    fedml_inference_runner = LoraxChatCompletionInferenceRunner()
    fedml_inference_runner.run()
