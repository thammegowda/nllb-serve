from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType
import sys
from typing import List
from functools import lru_cache

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch 

from . import DEF_MODEL_ID, log
DEF_MAX_SRC_CHARS = 512 
DEF_MAX_TGT_LEN = 120

device = torch.device(torch.cuda.is_available() and 'cuda' or 'cpu')
log.info(f'torch device={device}')


class Translator:

    def __init__(self, model_id):
        self.model_id = model_id
        log.info(f"Loading model {model_id} ...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
 
    @lru_cache(maxsize=256)
    def get_tokenizer(self, src_lang):
        log.info(f"Loading tokenizer for {self.model_id}; src_lang={src_lang} ...")
        #tokenizer = AutoTokenizer.from_pretrained(model_id)
        return AutoTokenizer.from_pretrained(self.model_id, src_lang=src_lang)

    def translate_batch(self, src_sents: List[str], src_lang, tgt_lang, max_tgt_length=DEF_MAX_TGT_LEN):
        tokenizer = self.get_tokenizer(src_lang=src_lang)
        assert src_sents
        inputs = tokenizer(src_sents, return_tensors="pt", padding=True)
        inputs = {k:v.to(device) for k, v in inputs.items()}
        translated_tokens = self.model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length = max_tgt_length)
        output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return output

    def translate_all(self, src_lines, batch_size:int, max_src_chars:int=DEF_MAX_SRC_CHARS, **kwargs):
        batch = []
        for line in src_lines:
            batch.append(line.strip()[:max_src_chars])
            if len(batch) >= batch_size:
                yield from self.translate_batch(batch, **kwargs)
                batch.clear()
        if batch:
            yield from self.translate_batch(batch, **kwargs)
    
    
def main(**args):
    args = args or parse_args()
    translator = Translator(model_id=args.pop('model_id'))
    out = args.pop('out')
    out_lines = translator.translate_all(src_lines=args.pop('inp'), **args)
    for line in out_lines:
        out.write(line + '\n')

def parse_args():
    parser = ArgumentParser(
        prog="nllb-batch",
        description="Serve NLLB model via command line",
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-mi", "--model-id", type=str, default=DEF_MODEL_ID,
                        help="model ID; see https://huggingface.co/models?other=nllb")
    parser.add_argument("-sl", "--src-lang", type=str, required=True,
                        help="source language identifier; eg: eng_Latn")
    parser.add_argument("-tl", "--tgt-lang", type=str, required=True,
                        help="Target language identifier; eg: eng_Latn")
    parser.add_argument('-i', '--inp', help='Input file', type=FileType('r', encoding='utf8'), default=sys.stdin)
    parser.add_argument('-o', '--out', help='Output file', type=FileType('w', encoding='utf8'), default=sys.stdout)
    parser.add_argument("-msl", "--max-src-chars", type=int, default=DEF_MAX_SRC_CHARS,
                        help="max source chars len; longer seqs will be truncated")
    parser.add_argument('-b', '--batch-size', help='Batch size; number of sentences', type=int, default=10)
    args = vars(parser.parse_args())
    return args

if __name__ == '__main__':
    main()
