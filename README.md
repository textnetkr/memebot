# 🤖 MemeBot

# 👉🏻 model
보유 발화문 - 밈 답변 데이터셋에서 발화문을 sbert embedding vector로 저장 후 새로운 발화문이 입력되면 보유 발화문 임베딩과 cosine similarity로 비교하여 유사도가 가장 높은 문장의 점수가 70% 이상인 발화에 대한 밈 답변 출력.
https://www.notion.so/textnet/b6ca4dec16654f81aff39816d3d63046


# 👉🏻 script
bash containerize.sh<br>
bash startup.sh


# 👉🏻 tree
```bash
.
├── src
│   └── runner.py
├── .gitignore
├── bentofile.yaml
├── containerize.sh
├── README.md
├── request.py
├── service.py
└── startup.sh
```