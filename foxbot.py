from pickle import dump, load
import numpy as np
import pandas as pd
from funcs import clean_text

with open("modelo.bin", "rb") as m:
    model = load(m)

with open("vectorizer.bin", "rb") as v:
    vectorizer = load(v)

while(True):
    text_in = input("Opa. Sou o Foxbot. Qual é o seu desejo?")
    counts = vectorizer.transform([text_in])
    resp = model.predict(counts)
    if (np.amax(model.predict_proba(counts)[0]) < 0.6):
        print("Desculpe, não entendi.\n")
    else:
        print (resp)