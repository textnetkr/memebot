import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import bentoml
import torch
from hydra import compose, initialize
from sentence_transformers import SentenceTransformer, util

with initialize(version_base="1.2", config_path="../config/"):
    cfg = compose(config_name="config")

device = torch.device("cuda")


def main():
    # data load
    def data_load(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.rstrip("\n|\r")))
        data = pd.DataFrame(data)
        return data

    # Load model from HuggingFace Hub
    model = SentenceTransformer(cfg.MODEL.name)

    # reference data
    df_ref = data_load(cfg.PATH.ref_data)

    # reference embeddings
    emb_list = os.listdir(cfg.PATH.emb_list)
    emb_list = sorted([i for i in emb_list if ".npy" in i])

    ref_embs = [np.load(cfg.PATH.emb_list + i).tolist() for i in tqdm(emb_list)]
    ref_emb = torch.Tensor(sum(ref_embs, []))

    input_texts = "안녕하세요."
        
    # Tokenize sentences
    query_emb = model.encode([input_texts], convert_to_tensor=True)
    hits = util.semantic_search(query_emb, ref_emb, top_k=1)

    score = round(hits[0][0]["score"], 2) * 100
    df_ref = (
        df_ref[df_ref.u == df_ref.iloc[hits[0][0]["corpus_id"]]["u"]]
        .sample(frac=1)
        .reset_index(drop=True)
    )
    result = df_ref.iloc[0]['meme']
    print(score)
    print(result)

    return result

if __name__ == "__main__":
    main()