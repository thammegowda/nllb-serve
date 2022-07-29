# NLLB Serve

This project offers a web interface and REST API to Meta's No Language Left Behind (NLLB) models that can translate across 200 languages.


## Setup

```bash
git clone  https://github.com/thammegowda/nllb-serve
cd nllb-serve
pip install -e .

# either one of these should work
nllb-serve -h
python -m nllb_serve -h
```

## Start Serve

```bash
# Either one of these should work
nllb-serve
# or
python -m nllb_serve
```

This starts a service on http://localhost:6060 by default.

<img src="docs/webui-demo.png" width=600px/>



**CLI options:**

```
$ nllb-serve -h
usage: nllb-serve [-h] [-d] [-p PORT] [-ho HOST] [-b BASE] [-mi MODEL_ID] [-msl MAX_SRC_LEN]

Deploy NLLB model to a RESTful server

optional arguments:
  -h, --help            show this help message and exit
  -d, --debug           Run Flask server in debug mode (default: False)
  -p PORT, --port PORT  port to run server on (default: 6060)
  -ho HOST, --host HOST
                        Host address to bind. (default: 0.0.0.0)
  -b BASE, --base BASE  Base prefix path for all the URLs. E.g., /v1 (default: None)
  -mi MODEL_ID, --model_id MODEL_ID
                        model ID; see https://huggingface.co/models?other=nllb (default: facebook/nllb-200-distilled-600M)
  -msl MAX_SRC_LEN, --max-src-len MAX_SRC_LEN
                        max source len; longer seqs will be truncated (default: 250)
```

## REST API


* `/translate` end point accepts GET and POST requests with the following args:
  * `source` -- source text. Can be a single string or a batch (i.e., list of strings)
  * `src_lang` -- source language ID, e.g., `eng_Latn`
  * `tgt_lang` -- target language ID, e.g., `eng_Latn`

HTTP Clients may send these parameters in three ways:
1. Query parameters (GET)\
   For example:
   * http://0.0.0.0:6060/translate?source=I%20am%20testing&src_lang=eng_Latn&tgt_lang=fra_Latn
   * http://0.0.0.0:6060/translate?source=I%20am%20testing&src_lang=eng_Latn&tgt_lang=fra_Latn&source=another%20sentence

2. URL encoded form (POST)
  ```bash
   curl --data "source=Comment allez-vous?" --data "source=Bonne journée" \
   --data "src_lang=fra_Latn" --data "tgt_lang=eng_Latn" \
    http://localhost:6060/translate
  ```
3. JSON body (POST)
```bash
$ curl -H "Content-Type: application/json" -X POST \
    http://localhost:6060/translate \
   --data '{"source": ["Comment allez-vous?"], "src_lang": "fra_Latn", "tgt_lang": "kan_Knda"}'
```

## References
* https://research.facebook.com/publications/no-language-left-behind/
* https://huggingface.co/docs/transformers/main/en/model_doc/nllb
* https://ai.facebook.com/research/no-language-left-behind/
* https://github.com/facebookresearch/fairseq/tree/nllb/
