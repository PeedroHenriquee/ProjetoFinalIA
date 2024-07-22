import tkinter as tk
from tkinter import scrolledtext, messagebox
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import movie_reviews
import nltk
import pickle
from googletrans import Translator
from stopwords import get_stopwords

# Baixar recursos necessários do NLTK
nltk.download('movie_reviews')

# Carregar dados de reviews de filmes
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
data, labels = zip(*documents)
data = [" ".join(words) for words in data]

# Mapeamento das categorias para nomes completos
category_names = {'pos': 'Sentimento Positivo', 'neg': 'Sentimento Negativo'}

# Obter stop words em português
stop_words = get_stopwords('pt')

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Vetorização de texto com stop words em português
vectorizer = CountVectorizer(stop_words=stop_words)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Treinar modelo Naive Bayes
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Avaliar o modelo
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Salvar o modelo e o vetorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Função para prever o sentimento de um texto específico
def predict_sentiment(text):
    translator = Translator()
    translated_text = translator.translate(text, src='pt', dest='en').text
    text_vectorized = vectorizer.transform([translated_text])
    prediction = model.predict(text_vectorized)
    return category_names.get(prediction[0], "Desconhecido")

# Configuração da interface gráfica
def on_predict_button_click():
    user_input = text_entry.get("1.0", tk.END).strip()
    if user_input:
        sentiment = predict_sentiment(user_input)
        result_label.config(text=f"Sentimento previsto: {sentiment}")
    else:
        messagebox.showwarning("Entrada Inválida", "Por favor, insira um texto.")

def on_clear_button_click():
    text_entry.delete("1.0", tk.END)
    result_label.config(text="Sentimento previsto: ")

root = tk.Tk()
root.title("Análise de Sentimentos")

# Configurações da fonte
font_style = ('Helvetica', 12)

# Texto de instrução
instructions = tk.Label(root, text="Insira um texto em português e clique em 'Analisar Sentimento':", font=font_style)
instructions.pack(pady=10)

# Campo de texto para o usuário inserir o texto
text_entry = scrolledtext.ScrolledText(root, width=60, height=15, font=font_style)
text_entry.pack(pady=10)

# Botão para prever o sentimento
predict_button = tk.Button(root, text="Analisar Sentimento", command=on_predict_button_click, font=font_style, bg='#4CAF50', fg='white')
predict_button.pack(pady=5)

# Botão para limpar o texto
clear_button = tk.Button(root, text="Limpar", command=on_clear_button_click, font=font_style, bg='#f44336', fg='white')
clear_button.pack(pady=5)

# Label para exibir o resultado
result_label = tk.Label(root, text="Sentimento previsto: ", font=font_style)
result_label.pack(pady=10)

# Executar a interface
root.mainloop()
