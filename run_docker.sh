docker run -it --rm --gpus '"device=0,1,2,3"' --ipc=host -v $PWD:/workspace/dockers -v $PWD/../Data/LJSpeech:/workspace/dockers/Dataset --name pablopz_glot1 pablopz_glotnet /bin/sh -c 'cd dockers; pip install -v -e .; pytest test; bash'