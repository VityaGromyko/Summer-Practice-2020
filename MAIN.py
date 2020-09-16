# %%
import pytesseract
import spacy
from spacy import displacy
from PIL import Image
from textacy import preprocessing

import json

# %%
path_to_doc_scan = 'Data/Pushkin.png'
scanned_text = pytesseract.image_to_string(Image.open(path_to_doc_scan), lang='eng')
print('Scanned text:', scanned_text, sep='\n')

# %%
normalized_text = preprocessing.normalize_hyphenated_words(
    preprocessing.normalize_quotation_marks(
        preprocessing.normalize_unicode(
            preprocessing.normalize_whitespace(
                scanned_text
            )
        )
    )
)
print('Normalized text:', normalized_text, sep='\n')

# %%
nlp = spacy.load('en_core_web_lg')
doc = nlp(normalized_text)

# %%
entities = [[ent.text, ent.label_, ent.start, ent.end] for ent in doc.ents]
print(entities)

# %%
result = {
    'text': doc.text,
    'entities': entities
}

with open('result.json', 'w') as file:
    json.dump(result, file)

# %%
displacy.serve(doc, style='ent')
