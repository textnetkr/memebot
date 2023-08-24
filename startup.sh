pyenv local memebot

docker run -it -d \
    -p 3000:3000 \
    --rm \
    --name textnet_memebot \
    memebot:5fiqd3ccj6ijeqqb serve