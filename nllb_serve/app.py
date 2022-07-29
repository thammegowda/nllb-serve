#!/usr/bin/env python
"""
Serves an NLLB MT model using Flask HTTP server
"""
import logging
import os
import sys
import platform
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging as log
from functools import lru_cache
import time

import flask
from flask import Flask, request, send_from_directory, Blueprint
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


from .utils import max_RSS


log.basicConfig(level=log.INFO)
DEF_MODEL_ID = "facebook/nllb-200-distilled-600M"
DEF_SRC_LNG = 'eng_Latn'
DEF_TGT_LNG = 'kan_Knda'
FLOAT_POINTS = 4
exp = None
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

bp = Blueprint('nmt', __name__, template_folder='templates', static_folder='static')


sys_info = {
    'transformer': transformers.__version__,
    'Python Version': sys.version,
    'Platform': platform.platform(),
    'Platform Version': platform.version(),
    'Processor':  platform.processor(),
    'CPU Memory Used': max_RSS()[1],
    #'GPU': '[unavailable]',
}
try:
    import torch
    #torch.set_grad_enabled(False)
    sys_info['torch']: torch.__version__
    if torch.cuda.is_available():
        sys_info['GPU'] = str(torch.cuda.get_device_properties())
        sys_info['Cuda Version'] = torch.version.cuda
    else:
        log.warning("CUDA unavailable")
except:
    pass

def render_template(*args, **kwargs):
    return flask.render_template(*args, environ=os.environ, **kwargs)


def jsonify(obj):

    if obj is None or isinstance(obj, (int, bool, str)):
        return obj
    elif isinstance(obj, float):
        return round(obj, FLOAT_POINTS)
    elif isinstance(obj, dict):
        return {key: jsonify(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [jsonify(it) for it in obj]
    #elif isinstance(ob, np.ndarray):
    #    return _jsonify(ob.tolist())
    else:
        log.warning(f"Type {type(obj)} maybe not be json serializable")
        return obj


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(bp.root_path, 'static', 'favicon'), 'favicon.ico')



def attach_translate_route(
    model_id=DEF_MODEL_ID, def_src_lang=DEF_SRC_LNG,
    def_tgt_lang=DEF_TGT_LNG, **kwargs):
    sys_info['model_id'] = model_id

    log.info(f"Loading model {model_id} ...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    log.info(f"Loading default tokenizer for {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    src_langs = tokenizer.additional_special_tokens
    tgt_langs = src_langs

    @lru_cache(maxsize=256)
    def get_tokenizer(src_lang=def_src_lang):
        log.info(f"Loading tokenizer for {model_id}; src_lang={src_lang} ...")
        #tokenizer = AutoTokenizer.from_pretrained(model_id)
        return AutoTokenizer.from_pretrained(model_id, src_lang=src_lang)

    @bp.route('/')
    def index():
        args = dict(src_langs=src_langs, tgt_langs=tgt_langs, model_id=model_id,
                    def_src_lang=def_src_lang, def_tgt_lang=def_tgt_lang)
        return render_template('index.html', **args)


    @bp.route("/translate", methods=["POST", "GET"])
    def translate():
        st = time.time()
        if request.method not in ("POST", "GET"):
            return "GET and POST are supported", 400
        if request.method == 'GET':
            args = request.args
        if request.method == 'POST':
            if request.headers.get('Content-Type') == 'application/json':
                args = request.json
            else:
                args = request.form

        if hasattr(args, 'getlist') :
            sources = args.getlist("source")
        else:
            sources = args.get("source")
            if isinstance(sources, str):
                sources = [sources]

        src_lang = args.get('src_lang') or def_src_lang
        tgt_lang = args.get('tgt_lang') or def_tgt_lang
        tokenizer = get_tokenizer(src_lang=src_lang)

        if not sources:
            return "Please submit 'source' parameter", 400
        max_length = 80
        inputs = tokenizer(sources, return_tensors="pt", padding=True,)

        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length = max_length)
        output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        res = dict(source=sources, translation=output,
                   src_lang = src_lang, tgt_lang=tgt_lang,
                   time_taken = round(time.time() - st, 3), time_units='s')

        return flask.jsonify(jsonify(res))

    @bp.route('/about')
    def about():
        sys_info['CPU Memory Used'] = max_RSS()[1]
        return render_template('about.html', sys_info=sys_info)


def parse_args():
    parser = ArgumentParser(
        prog="nllb-serve",
        description="Deploy NLLB model to a RESTful server",
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--debug", action="store_true", help="Run Flask server in debug mode")
    parser.add_argument("-p", "--port", type=int, help="port to run server on", default=6060)
    parser.add_argument("-ho", "--host", help="Host address to bind.", default='0.0.0.0')
    parser.add_argument("-b", "--base", help="Base prefix path for all the URLs. E.g., /v1")
    parser.add_argument("-mi", "--model_id", type=str, default=DEF_MODEL_ID,
                        help="model ID; see https://huggingface.co/models?other=nllb")
    parser.add_argument("-msl", "--max-src-len", type=int, default=250,
                        help="max source len; longer seqs will be truncated")
    args = vars(parser.parse_args())
    return args


# uwsgi can take CLI args too
# uwsgi --http 127.0.0.1:5000 --module nllb_serve.app:app # --pyargv "--foo=bar"
cli_args = parse_args()
attach_translate_route(**cli_args)
app.register_blueprint(bp, url_prefix=cli_args.get('base'))
if cli_args.pop('debug'):
    app.debug = True

# register a home page if needed
if cli_args.get('base'):
    @app.route('/')
    def home():
        return render_template('home.html', demo_url=cli_args.get('base'))


def main():
    log.info(f"System Info: ${sys_info}")
    # CORS(app)  # TODO: insecure
    app.run(port=cli_args["port"], host=cli_args["host"])
    # A very useful tutorial is found at:
    # https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3


if __name__ == "__main__":
    main()