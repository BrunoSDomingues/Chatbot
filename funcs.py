def clean_text(text: str):
    text = text.lower()

    remove = [",", "-", "foxbot", "."]
    for r in remove:
        text = text.replace(r, "")
        
    text = text.replace("$", "dinheiro")

    return text
