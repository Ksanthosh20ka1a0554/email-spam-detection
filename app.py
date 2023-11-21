from flask import Flask,request,render_template
import pickle
import joblib
import pandas as pd
import nltk
import string
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
#nltk.download('punkt')
#nltk.download('stopwords')

# df = pd.read_csv('spam.csv',encoding = "ISO-8859-1")

def get_importantFeatures(sent):
    sent = sent.lower()
    return [i for i in nltk.word_tokenize(sent) if i.isalnum()]

def remove_stopwords_and_punctuation(sent):
    return [i for i in sent if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation]

def perform_stemming(sent):
    ps = nltk.PorterStemmer()
    return [ps.stem(i) for i in sent]

# df['message'] = df['message'].apply(get_importantFeatures)
# df['message'] = df['message'].apply(remove_stopwords_and_punctuation)
# df['message'] = df['message'].apply(perform_stemming)

# X = df['message'].apply(lambda x: ' '.join(x))
# y = df['class']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# tfidf = TfidfVectorizer()
# X_train_tfidf = tfidf.fit_transform(X_train)

# X_test_tfidf = tfidf.transform(X_test)

# classifier = SVC(kernel='linear')
# classifier.fit(X_train_tfidf, y_train)

with open('svm_model.pkl', 'rb') as file:
    loaded_model = joblib.load(file)
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = joblib.load(file)

app = Flask(__name__)
@app.route('/')
def hello():
    return render_template('index.html')

@app.route("/review",methods=['GET','POST'])
def predict():
    if request.method=="POST":
         test_message=request.form['message']
         processed_message = perform_stemming(remove_stopwords_and_punctuation(get_importantFeatures(test_message)))
         processed_message_str = ' '.join(processed_message)
         test_message_tfidf = tfidf.transform([processed_message_str])
         prediction = loaded_model.predict(test_message_tfidf)
    return render_template('index.html',final_result=prediction[0])

if __name__ == '__main__':
     app.run(debug = True)