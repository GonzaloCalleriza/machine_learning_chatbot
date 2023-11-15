import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import random
import numpy
import tflearn

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
            
# Cleaning data
stemmer = LancasterStemmer()

# This will fraction the sentences in lower case by the root
words = [stemmer.stem(word.lower()) for word in words]

# We sort the fractions, make them a set to remove duplicates and transform it into a list
words = sorted(list(set(words)))

# Build a bag of words for our ML model
empty_output = [0] * len(classes)
training_data = []

for document in documents:
    bag_of_words = []
    pattern_words = document[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for word in words:
        bag_of_words.append(1) if word in pattern_words else bag_of_words.append(0)
    
    output_row = list(empty_output)
    output_row[classes.index(document[1])] = 1
    training_data.append([bag_of_words, output_row])
    
# Split the data for machine learning
random.shuffle(training_data)

training_numpy = numpy.array(training_data)

train_X = list(training_numpy[:, 0])

train_y = list(training_numpy[:, 1])

# Build a TensorFlow machine learning model for chat
neural_network = tflearn.input_data(shape= [None, len(train_X[0])])

neural_network = tflearn.fully_connected(neural_network, 8)

neural_network = tflearn.fully_connected(neural_network, len(train_y[0], activation="softmax"))

neural_network = tflearn.regression(neural_network)

model = tflearn.DNN(neural_network)

model.fit(train_X, train_y, n_epoch=2000, batch_size=8, show_metric=True)

# Testing the chatbot machine learning model
model.save("chatbot_dnn.tflearn")
model.load("chatbot_dnn.tflearn")

question = "Do you sell any coding course?"

def process_question(question):
    question_tokenized = nltk.word_tokenize(question)
    question_stemmed = [stemmer.stem(word.lower()) for word in question_tokenized]
    
    bag = [0] * len(words)
    
    for stem in question_stemmed:
        for index, word in enumerate(words):
            if word == stem:
                bag[index] = 1
                
    return (numpy.array(bag))

prediction = model.predict([process_question(question)])[0]