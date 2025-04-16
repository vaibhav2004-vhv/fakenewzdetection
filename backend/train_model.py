import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load datasets
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Add labels
df_fake["label"] = "Fake"
df_true["label"] = "True"

# Balance the dataset by undersampling
min_len = min(len(df_fake), len(df_true))
df_fake = df_fake.sample(min_len, random_state=42)
df_true = df_true.sample(min_len, random_state=42)

# Combine and shuffle
df = pd.concat([df_fake, df_true], axis=0)
df = df.sample(frac=1).reset_index(drop=True)

# Encode labels (Fake=0, True=1)
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# Text vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X = vectorizer.fit_transform(df["text"].astype("U"))  # .astype("U") ensures unicode
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100)
}

accuracies = {}

# Train and save the models
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc
    joblib.dump(model, f"{name}_model.pkl")
    print(f"✅ {name} model saved (Accuracy: {acc:.4f})")

# Save vectorizer and encoder
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("✅ Vectorizer and label encoder saved")

# Save best model name
best_model = max(accuracies, key=accuracies.get)
with open("best_model.txt", "w") as f:
    f.write(best_model)
print(f"🏆 Best model: {best_model} (Accuracy: {accuracies[best_model]:.4f})")
