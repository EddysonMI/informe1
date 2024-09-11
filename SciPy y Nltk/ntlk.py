from nltk.sentiment import SentimentIntensityAnalyzer

# Crear un analizador de sentimientos
sia = SentimentIntensityAnalyzer()

# Analizar sentimientos
texto = "Me encanta aprender Python. Es muy divertido y educativo."
sentimiento = sia.polarity_scores(texto)
print(f'Análisis de Sentimientos: {sentimiento}')
