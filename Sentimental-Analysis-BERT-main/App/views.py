from django.shortcuts import render
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf

def init():
    global model, tokenizer
    model = TFBertForSequenceClassification.from_pretrained(
        "C://Users/KADIRI MOHAN KUMAR/PycharmProjects/project/bert/model/saved_model")

    tokenizer = BertTokenizer.from_pretrained(
        "C://Users/KADIRI MOHAN KUMAR/PycharmProjects/project/bert/model/tokenizer")

    return model, tokenizer

def getPredictions(query):
    tf_batch = tokenizer(query, max_length=128, padding=True, truncation=True, return_tensors='tf')
    tf_outputs = model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    return list(tf_predictions[0])



def home(request):
    if request.method == 'POST':
        query = request.POST['query']
        print(query)
        pred = getPredictions(query)
        print(pred)
        labels = ['Negative','Positive']
        neg, pos = round(float(pred[0]), 2), round(float(pred[1]), 2)
        label = labels[pred.index(max(pred))]
        return render(request, 'home.html', {'neg': neg, 'pos': pos, 'label': label, 'flag': True, 'query': query})
    init()
    return render(request, 'home.html')
