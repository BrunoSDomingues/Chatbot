from pickle import dump, load
import numpy as np
import pandas as pd
from funcs import clean_text

with open("modelo.bin", "rb") as m:
    model = load(m)

with open("vectorizer.bin", "rb") as v:
    vectorizer = load(v)

possibilities = [i for i in model.classes_]


def greetUser(p=possibilities):
    print("Opa. Sou o Foxbot. Posso responder perguntas relacionadas à: \n")
    for i in range(len(p)):
        print(f"{i+1}) {p[i]}")
    print("\nPara sair basta digitar 'sair' ou apertar Ctrl+C.\n")


def processInput(u_input, m=model, v=vectorizer):
    new_t = v.transform([u_input])
    pred = model.predict(new_t)
    if np.amax(model.predict_proba(new_t)[0]) < 0.6:
        pred = "Desculpe, nao entendi."
    return pred, new_t


def wrongPred(u_input_vectorized, p=possibilities, m=model):
    print("Selecione o que voce esperava de acordo com as opcoes abaixo: \n")

    for i in range(len(p)):
        print(f"{i+1}) {p[i]}")
    print(" ")

    while True:
        try:
            expected = int(input("Opcao escolhida: "))

            if expected not in [1, 2, 3]:
                print("Opcao invalida, por favor, digite novamente.\n")
                continue

            else:
                print(p[expected - 1] + "\n")
                model.partial_fit(u_input_vectorized, [p[expected - 1]], classes=p)
                print("Obrigado por ajudar a melhorar o FoxBot.")
                break

        except ValueError:
            print("Opcao invalida, por favor, digite novamente.\n")
            continue


greetUser()

while True:
    try:
        text_in = clean_text(str(input("O que voce gostaria de perguntar? ")))

        if text_in == "Sair" or text_in == "sair":
            print("\nAdeus.\n")
            break

        else:
            pred, transform = processInput(text_in)
            if pred != "Desculpe, nao entendi.":
                print("\nEntendi que deseja a seguinte funcao:")
                print(pred[0] + "\n")

                correct = str(input("Era isto que voce desejava? (s/n) "))

                if correct == "n":
                    wrongPred(transform)
                else:
                    continue

            else:
                print(pred + "\n")

    except KeyboardInterrupt:
        print("\nAdeus.\n")
        break