# Identifying Fake News Using Bert
This system aims to create an API that tests the credibility of the information in a text. In other words, it will show, how reliable the information placed in it would be or not.

## System Architecture or Topology
The topology of this system is very simple. Roughly speaking, it is being divided into two parts: BackEnd and FrontEnd

## Definitions about fake news

## Building Containers

Remove container 

    docker-compose down

or 

    docker-compose down --volumes

    docker-compose down --rmi all --volumes

clean caches

    docker builder prune

## References

1. [BERT NLP building fake news system][1]
2. [What is BERT?][2]
3. [Article about BERT][3]
4. [Sample dataset to make fine-tuning][4]
5. [Bert Multilingual Huggingface][5]
6. [Bert Multilingual Case Supported languages][6]
7. [Huggingface Transformers][7]

[1]: https://qiita.com/shake54/items/66852e10a6983d6249e2
[2]: https://qiita.com/omiita/items/72998858efc19a368e50
[3]: ./article/1810.04805v2.pdf
[4]: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download
[5]: https://huggingface.co/google-bert/bert-base-multilingual-cased
[6]: https://github.com/google-research/bert/blob/master/multilingual.md
[7]: https://github.com/huggingface/transformers
