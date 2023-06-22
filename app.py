from flask import Flask,request,render_template
import pickle4 as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
nltk.download('punkt')
nltk.download('stopwords')

with open('svm_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

def get_importantFeatures(sent):
    sent = sent.lower()
    return [i for i in nltk.word_tokenize(sent) if i.isalnum()]

def remove_stopwords_and_punctuation(sent):
    return [i for i in sent if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation]

def perform_stemming(sent):
    ps = nltk.PorterStemmer()
    return [ps.stem(i) for i in sent]

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

# if __name__ == '__main__':
#     app.run(debug = True)