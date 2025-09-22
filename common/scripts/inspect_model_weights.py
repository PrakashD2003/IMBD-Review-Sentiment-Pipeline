# common/scripts/inspect_model_weights.py
import joblib
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.constants import ARTIFACT_DIR_NAME, OBJ_SAVE_DIR_NAME, VECTORIZER_OBJ_DIR, VECTORIZER_OBJ_FILE_NAME, TRAINED_MODEL_OBJ_DIR, TRAINED_MODEL_OBJ_NAME

def inspect_weights():
    """Loads the trained vectorizer and model to inspect feature weights."""
    print("--- Inspecting Model Feature Weights ---")

    # Construct paths to artifacts
    vectorizer_path = os.path.join(project_root, ARTIFACT_DIR_NAME, OBJ_SAVE_DIR_NAME, VECTORIZER_OBJ_DIR, VECTORIZER_OBJ_FILE_NAME)
    model_path = os.path.join(project_root, ARTIFACT_DIR_NAME, OBJ_SAVE_DIR_NAME, TRAINED_MODEL_OBJ_DIR, TRAINED_MODEL_OBJ_NAME)

    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        print("\nError: Model artifacts not found. Please run the training pipeline first.")
        print(f"Looked for vectorizer at: {vectorizer_path}")
        print(f"Looked for model at: {model_path}")
        return

    # Load artifacts
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    print(f"\nModel Type: {type(model).__name__}")
    print(f"Vocabulary Size: {len(vectorizer.vocabulary_)}")

    # Create a mapping from feature index to feature name
    feature_names = {v: k for k, v in vectorizer.vocabulary_.items()}

    # The words and phrases we want to inspect
    terms_to_inspect = ["good", "not good", "great", "not great"]

    print("\n--- Learned Coefficients ---")
    for term in terms_to_inspect:
        if term in vectorizer.vocabulary_:
            term_index = vectorizer.vocabulary_[term]
            # The coefficient is in model.coef_[0] for binary classification
            coefficient = model.coef_[0][term_index]
            print(f"  - Feature: '{term}' -> Weight: {coefficient:.4f}")
        else:
            print(f"  - Feature: '{term}' -> Not in model vocabulary.")

if __name__ == "__main__":
    inspect_weights()