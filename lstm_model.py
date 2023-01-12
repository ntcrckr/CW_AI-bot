from tensorflow import keras
import pandas as pd
import numpy as np
from keras.utils import pad_sequences
import regex
import nltk
from sklearn import preprocessing
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.util import skipgrams


save = False
if save:
    df = pd.read_csv('./lstm_model/train.csv')
    df = df.drop(columns=['id', 'thumbs_up'])

    # Обработка текста
    df = df.astype({'review_text': 'str'})
    df['review_text'] = [i.lower() for i in df['review_text']]
    df['review_text'] = [regex.sub(r'[^\w\s]', '', i) for i in df['review_text']]
    df['review_text'].dropna(inplace=True)
    # нужно только для обучения
    df['review_text'] = [word_tokenize(i) for i in df['review_text']]

    dic = []
    for i in range(len(df)):
        dic += df['review_text'][i]

    le = preprocessing.LabelEncoder()
    le.fit(dic)

    np.save('./lstm_model/le_classes.npy', le.classes_)
else:
    le = preprocessing.LabelEncoder()
    le.classes_ = np.load('./lstm_model/le_classes.npy')

model5 = keras.models.load_model('./lstm_model/model5')
model4 = keras.models.load_model('./lstm_model/model4')
model3 = keras.models.load_model('./lstm_model/model3')
model2 = keras.models.load_model('./lstm_model/model2')
model1 = keras.models.load_model('./lstm_model/model1')


def get_predictions(text: str):
    df = pd.DataFrame({'review_text': [text]})
    df['review_text'] = [i.lower() for i in df['review_text']]
    df['review_text'] = [regex.sub(r'[^\w\s]', '', i) for i in df['review_text']]
    df['review_text'].dropna(inplace=True)
    # test_x = pd.DataFrame({'text': [le.transform(df['review_text'][0].split(' '))]})
    try:
        test_x = [le.transform(df['review_text'][0].split(' '))]
    except:
        return "Unseen label error"
    print(test_x)

    #

    max_words = 403
    X_test = pad_sequences(test_x, max_words)
    p5 = model5.predict(X_test)
    p4 = model4.predict(X_test)
    p3 = model3.predict(X_test)
    p2 = model2.predict(X_test)
    p1 = model1.predict(X_test)
    print(p5)
    print(p4)
    print(p3)
    print(p2)
    print(p1)
    if p5[0][0] >= p4[0][0] and p5[0][0] >= p3[0][0] and p5[0][0] >= p2[0][0] and p5[0][0] >= p1[0][0]:
        prediction = 5
    elif p4[0][0] >= p5[0][0] and p4[0][0] >= p3[0][0] and p4[0][0] >= p2[0][0] and p4[0][0] >= p1[0][0]:
        prediction = 4
    elif p1[0][0] >= p4[0][0] and p1[0][0] >= p3[0][0] and p1[0][0] >= p2[0][0] and p1[0][0] >= p5[0][0]:
        prediction = 1
    elif p3[0][0] >= p4[0][0] and p3[0][0] >= p5[0][0] and p3[0][0] >= p2[0][0] and p3[0][0] >= p1[0][0]:
        prediction = 3
    else:
        prediction = 2
    return prediction
