import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from pickle import dump
from funcs import clean_text
from sklearn.metrics import accuracy_score

# Processing and cleaning data
data = pd.read_csv("sentencas.csv").drop(columns=["Timestamp"])
data["Sentença"] = data["Sentença"].apply(lambda x: clean_text(str(x)))

# Split test/train
X_train, X_test, y_train, y_test = train_test_split(
    data["Sentença"], data["Intenção"], test_size=0.25
)

# Naive-Bayes
vectorizer = CountVectorizer()
ft_train = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(ft_train, y_train)
tr_test = vectorizer.transform(X_test)

# Accuracy
print(f"Modelo generico: {cross_val_score(model, tr_test, y_test, cv=4)}")

# Save model and vectorizer
with open("modelo.bin", "wb+") as m:
    dump(model, m)

with open("vectorizer.bin", "wb+") as v:
    dump(vectorizer, v)