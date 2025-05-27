from sentence_splitter import split_text_into_sentences

ssplit_langs = {
    "cat_Latn": "ca",
    "ces_Latn": "cs",
    "dan_Latn": "da",
    "nld_Latn": "nl",
    "eng_Latn": "en",
    "fin_Latn": "fi",
    "fra_Latn": "fr",
    "deu_Latn": "de",
    "ell_Grek": "el",
    "hun_Latn": "hu",
    "isl_Latn": "is",
    "ita_Latn": "it",
    "Latvian": "lv",
    "lit_Latn": "lt",
    "nno_Latn": "no",
    "pol_Latn": "pl",
    "por_Latn": "pt",
    "ron_Latn": "ro",
    "rus_Cyrl": "ru",
    "slk_Latn": "sk",
    "spa_Latn": "es",
    "swe_Latn": "sv",
    "tur_Latn": "tr",
}


def sentence_splitter(src_lang, sources):
    lang = ssplit_langs.get(src_lang)
    index = [0]
    sentences = []

    for source in sources:
        sentence = split_text_into_sentences(source, language=lang)
        for sen in sentence:
            sentences.append(sen)
        index.append(len(sentences))

    return sentences, index


def ssplit_lang(src_lang):
    if src_lang in ssplit_langs:
        return True
    return False
