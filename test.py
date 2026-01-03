import joblib

# 1. Load the vectorizer and models
print("Loading models...")
vectorizer = joblib.load('vectorizer.pkl')
models = joblib.load('models.pkl')
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# 2. Define a function to predict
def predict_toxicity(text):
    # Transform the text using the loaded vectorizer
    text_features = vectorizer.transform([text])
    
    results = {}
    for class_name in class_names:
        # Predict probability (returns [prob_0, prob_1])
        prob = models[class_name].predict_proba(text_features)[0][1]
        results[class_name] = prob
    return results

# 3. Test on sample data
sample_text = "You are extremely rude and I hate you."
print(f"\nAnalyzing text: '{sample_text}'")
predictions = predict_toxicity(sample_text)

print("\nResults:")
for label, score in predictions.items():
    print(f"{label:<15}: {score:.4f}")