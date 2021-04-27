from pickle import dump, load
import numpy as np
import pandas as pd
from funcs import clean_text

with open("modelo.bin", "rb") as m:
    model = load(m)

with open("vectorizer.bin", "rb") as v:
    vectorizer = load(v)