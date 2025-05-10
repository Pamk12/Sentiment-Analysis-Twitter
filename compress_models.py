import pickle
import gzip

# Compress the model files
models = ["logistic_model.pkl", "tree_model.pkl", "vectorizer.pkl", "scaler.pkl"]

for model_file in models:
    # Load the model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Save compressed version
    with gzip.open(f"{model_file}.gz", 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Compressed {model_file}")

# Then modify app.py to load compressed models