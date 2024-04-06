import docker
import os

import types
import yaml

from fedml.utils.logging import logger

CURRENT_DIR = os.path.dirname(__file__)


def load_endpoint(filepath: str) -> types.SimpleNamespace:
    with open(filepath, "r") as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            yaml_obj = types.SimpleNamespace(**yaml_dict)
            return yaml_obj
        except yaml.YAMLError as err:
            logger.error(err)


def spawn_lorax_container(endpoint: types.SimpleNamespace):
    inside_container_cmd = """--model-id {} --quantize {}""".format(
        endpoint.lorax_hugging_face_model_id,
        endpoint.lorax_quantize)
    docker_volume = os.path.join(CURRENT_DIR, "data") \
        if not endpoint.docker_volume else endpoint.docker_volume
    client = docker.from_env()
    exec_res = client.containers.run(
        image=endpoint.docker_image,
        command=inside_container_cmd,
        name=endpoint.docker_container_name,
        device_requests=[
            docker.types.DeviceRequest(
                device_ids=endpoint.lorax_gpu_ids,
                capabilities=[['gpu']])],
        auto_remove=True,
        detach=True,
        shm_size="1g",
        ports={"80/tcp": endpoint.lorax_port},
        volumes={docker_volume: {"bind": "/data", "mode": "rw"}})
    print(exec_res)


if __name__ == "__main__":
    endpoint_yaml_fp = os.path.join(CURRENT_DIR, 'endpoint.yaml')
    endpoint = load_endpoint(endpoint_yaml_fp)
    spawn_lorax_container(endpoint)
    print(endpoint)
