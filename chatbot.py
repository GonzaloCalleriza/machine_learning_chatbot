import json
import nltk

nltk.download('punkt')
# Connect with database
with open("data.json") as json_data:
    data = json.load(json_data)


# Tokenize the intents
words = []
documents = []
classes = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word = nltk.word_tokenize(pattern)
        
        words.extend(word)
        documents.append((word, intent["tag"]))
        
        if intent["tag"] not in classes:
            classes.append(intent["tag"])