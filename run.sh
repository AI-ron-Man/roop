docker build . -f Dockerfile.cpu -t roop-cpu
docker run -it --rm \
    -v ~/ML/media/:/roop/media/ \
    -v ~/ML/models:/roop/models \
    -v ~/ML/models/.insightface/:/root/.insightface/ \
    -v ~/ML/models/.opennsfw2/:/root/.opennsfw2/ \
    roop-cpu \
    -s ./media/source.jpeg -t ./media/target.png -o ./media/output.png 