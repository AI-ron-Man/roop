docker build . -f Dockerfile.gpu -t roop-gpu
docker run -it --rm \
    --gpus all \
    -v ~/ML/media/:/roop/media/ \
    -v ~/ML/models:/roop/models \
    -v ~/ML/models/.insightface/:/root/.insightface/ \
    -v ~/ML/models/.opennsfw2/:/root/.opennsfw2/ \
    roop-gpu \
    -s ./media/source.jpeg -t ./media/target.png -o ./media/output.png