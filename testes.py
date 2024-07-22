from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import movie_reviews
import nltk

# Baixar recursos necessários do NLTK
nltk.download('movie_reviews')

# Carregar dados de reviews de filmes
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
data, labels = zip(*documents)
data = [" ".join(words) for words in data]

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Vetorização de texto
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Treinar modelo Naive Bayes
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Fazer previsões
y_pred = model.predict(X_test_vectorized)

# Avaliar precisão
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Função para prever o sentimento de um texto específico
def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]

# Receber entrada do usuário
user_input = input("Digite o texto para análise de sentimentos: ")
sentiment = predict_sentiment(user_input)

# Exibir resultado
print(f"Texto: {user_input}")
print(f"Sentimento previsto: {sentiment}")
