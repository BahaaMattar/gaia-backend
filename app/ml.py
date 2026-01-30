import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # Critical for correct SMOTE usage

# 1. LOAD DATA
print("Step 1: Loading Data...")
df = pd.read_csv('app//Data//Diseases_and_Symptoms_dataset.csv')
X = df.drop('diseases', axis=1)
y = df['diseases']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 2. ROBUST SPLIT
# We split BEFORE any processing to prevent leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 3. DEFINE THE MODELS
# Model A: The Specialist (Linear SVC) - Good at high-dimensional separation
svc_base = LinearSVC(class_weight='balanced', max_iter=5000, dual=False, C=10)
# We must wrap SVC to get probabilities for the Voting Classifier
calibration = CalibratedClassifierCV(svc_base, method='sigmoid', cv=3)

# Model B: The Context Master (Gradient Boosting) - Good at symptom interactions
# We assume small n_estimators for speed, but high learning rate
gb_model = HistGradientBoostingClassifier(random_state=42)

# The Ensemble: Soft Voting averages the probabilities of both models
voting_model = VotingClassifier(
    estimators=[
        ('svc', calibration),
        ('gb', gb_model)
    ],
    voting='soft', # 'soft' means we average the confidence % not just the label
    n_jobs=-1
)

# 4. BUILD THE PIPELINE
# This ensures SMOTE and Feature Selection happen INSIDE the training loops
# so the validation data never leaks into the training process.
model_pipeline = ImbPipeline([
    ('selector', SelectKBest(score_func=chi2, k=230)),
    ('smote', SMOTE(random_state=42)),
    ('voter', voting_model)
])

# 5. TRAIN
print("Step 2: Training Hybrid Ensemble (This may take 2-3 mins)...")
model_pipeline.fit(X_train, y_train)

# 6. EVALUATION
print("Step 3: Evaluating...")
y_probs = model_pipeline.predict_proba(X_test)
y_pred = model_pipeline.predict(X_test)

# Top-K Accuracy Function
def get_top_k_accuracy(y_true, y_probs, k=3):
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    return np.mean(np.any(top_k_preds == y_true[:, None], axis=1))

# Extract the selected features names (a bit tricky with pipelines)
# We fit the selector manually just to get the names for saving
selector_temp = SelectKBest(score_func=chi2, k=150)
selector_temp.fit(X_train, y_train)
selected_features = X.columns[selector_temp.get_support()].tolist()

print("\n" + "="*30)
print("HYBRID MODEL PERFORMANCE")
print("="*30)
print(f"Top-1 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Top-3 Accuracy: {get_top_k_accuracy(y_test, y_probs, k=3):.4f}")
print(f"Weighted F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# 7. EXPORT
save_path = "app//ml-results//"

joblib.dump(model_pipeline, save_path + 'hybrid_model.pkl')
joblib.dump(le, save_path + 'disease_label_encoder.pkl')
joblib.dump(selected_features, save_path + 'selected_symptoms.pkl')