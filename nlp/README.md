# Fine-Tuning and Activate NLP step by step
Space to make fine-tuning, just

    cd modelbase

    git clone https://huggingface.co/google-bert/bert-base-cased

Download manualy following

    flax_model.msgpack

    model.safetensors

    pytorch_model.bin

    tf_model.h5

Or just run the script, download_model_base.py.

Run container using gpu

    docker run --gpus all -it seu_imagem

    supervisord -c /caminho/para/myapp.conf

## References
