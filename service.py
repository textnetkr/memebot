import bentoml
from bentoml.io import JSON
from hydra import compose, initialize

from src.runner import SbertRunnable


with initialize(config_path="config/"):
    cfg = compose(config_name="config")

runner = bentoml.Runner(
    SbertRunnable,
    name="sbert_runner",
)
svc = bentoml.Service(cfg.MODEL.service_name, runners=[runner])

@svc.api(input=JSON(), output=JSON())
def generate(inputs):
    input_texts = inputs["inputs"]
    result = runner.generate.run(input_texts)

    return result