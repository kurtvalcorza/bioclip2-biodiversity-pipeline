
"""Run the carried capstone stages end to end with STUB models (one process per stage, as in the
notebook). BioCLIP and SigLIP are replaced by deterministic pixel-statistic encoders; the head
fit, selection, evaluation, triage, fresh-process reload and report run the carried code on CPU.
This exercises the stage chain and receipts, not model behaviour."""
import hashlib, sys, types
from pathlib import Path
import numpy as np

root = Path(sys.argv[sys.argv.index("--root") + 1])
sys.path.insert(0, str(root))
import capstone as run  # noqa: E402
from PIL import Image  # noqa: E402

def encode(image):
    grey = np.asarray(image.convert("L").resize((16, 16)), dtype=np.float64).ravel()
    colour = np.asarray(image.convert("RGB").resize((4, 4)), dtype=np.float64).ravel()
    vector = np.concatenate([grey, colour, np.zeros(768 - 256 - 48)])
    vector = vector - vector.mean()
    return vector / np.linalg.norm(vector)

def text_vector(text):
    rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
    v = rng.normal(size=768)
    return v / np.linalg.norm(v)

class StubPipe:
    logit_scale = 100.0
    def embed_images(self, images):
        return {"embeddings": [encode(im).tolist() for im in images]}
    def embed_texts(self, texts):
        return {"embeddings": [text_vector(t).tolist() for t in texts]}

run.bioclip_pipe = lambda root, record_download=True: StubPipe()

class NoGrad:
    def __enter__(self): return None
    def __exit__(self, *exc): return False

class T:
    def __init__(self, v): self.v = np.asarray(v, dtype=np.float32)
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.v

class Inputs(dict):
    def to(self, device): return self

class Processor:
    @staticmethod
    def from_pretrained(*a, **k): return Processor()
    def __call__(self, text, images, return_tensors, padding, max_length):
        return Inputs(input_ids=np.zeros((len(text), max_length)), texts=list(text), images=list(images))

class Model:
    @staticmethod
    def from_pretrained(*a, **k): return Model()
    def float(self): return self
    def cuda(self): return self
    def eval(self): return self
    def __call__(self, input_ids, texts, images):
        f = np.stack([encode(im) for im in images]); t = np.stack([text_vector("siglip " + s) for s in texts])
        return types.SimpleNamespace(logits_per_image=10 * f @ t.T)

fake_torch = types.SimpleNamespace(no_grad=NoGrad, sigmoid=lambda x: T(1 / (1 + np.exp(-x))),
                                   cuda=types.SimpleNamespace(empty_cache=lambda: None))
run.torch_runtime = lambda: fake_torch
sys.modules["transformers"] = types.SimpleNamespace(AutoModel=Model, AutoProcessor=Processor)
real_stage_model = run.stage_model
run.stage_model = lambda root, key, record_download=True: root
run.main()
