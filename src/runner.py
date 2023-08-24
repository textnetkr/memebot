import os
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import bentoml
import torch
from hydra import compose, initialize
from sentence_transformers import SentenceTransformer, util

with initialize(version_base="1.2", config_path="../config/"):
    cfg = compose(config_name="config")

device = torch.device("cuda")


class SbertRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        # data load
        def data_load(path):
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line.rstrip("\n|\r")))
            data = pd.DataFrame(data)
            return data

        # Load model from HuggingFace Hub
        self.model = SentenceTransformer(cfg.MODEL.name)

        # reference data
        self.df_ref = data_load(cfg.PATH.ref_data)

         # reference embeddings
        emb_list = os.listdir(cfg.PATH.emb_list)
        emb_list = sorted([i for i in emb_list if ".npy" in i])

        ref_embs = [np.load(cfg.PATH.emb_list + i).tolist() for i in tqdm(emb_list)]
        self.ref_emb = torch.Tensor(sum(ref_embs, []))


    @bentoml.Runnable.method(batchable=False)
    def generate(self, input_texts):        
        query_emb = self.model.encode([input_texts], convert_to_tensor=True)
        hits = util.semantic_search(query_emb, self.ref_emb, top_k=1)

        score = round(hits[0][0]["score"], 2) * 100
        df_ref = (
            self.df_ref[self.df_ref.u == self.df_ref.iloc[hits[0][0]["corpus_id"]]["u"]]
            .sample(frac=1)
            .reset_index(drop=True)
        )
        result = {}
        result['u'] = df_ref.iloc[0]['u']
        result['meme'] = df_ref.iloc[0]['meme']
        result['score'] = score

        return result