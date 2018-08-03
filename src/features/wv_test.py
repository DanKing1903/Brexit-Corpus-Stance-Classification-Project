from src.features.feature_transformers import WordTokenizer

with open('data/raw/toy.csv', 'r') as f:
    data = pd.read_csv(f, header=None)[0]

wv = WordVectorizer()

result = wv.fit_transform(data)
print(result)
