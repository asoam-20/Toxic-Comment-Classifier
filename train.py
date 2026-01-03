import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Load Data
train = pd.read_csv('train.csv')
train['comment_text'] = train['comment_text'].fillna(' ')

# 2. Vectorize
print("Vectorizing text...")
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    stop_words='english',
    max_features=20000
)
# Fit on the training data
train_features = vectorizer.fit_transform(train['comment_text'])

# 3. Train Models
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
models = {}

print("Training models...")
for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(solver='sag', max_iter=1000)
    classifier.fit(train_features, train_target)
    models[class_name] = classifier
    print(f" - Trained {class_name}")

# 4. Save to .pkl files
print("Saving models to .pkl files...")
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(models, 'models.pkl')
print("Done! Files saved: 'vectorizer.pkl' and 'models.pkl'")