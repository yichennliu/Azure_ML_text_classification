import json

import joblib
import numpy as np
from azureml.core.model import Model


def init():
    global model
    model_path = Model.get_model_path('dummy_classification_model')
    model = joblib.load(model_path)


def classify_text(clf, doc, labels=None):
    probas = clf.predict_proba([doc]).flatten()
    max_proba_idx = np.argmax(probas)

    if labels:
        most_proba_class = labels[max_proba_idx]
    else:
        most_proba_class = max_proba_idx

    return (most_proba_class, probas[max_proba_idx])


def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        labels = ['SPORTS', 'CRIME', 'BLACK VOICES', 'QUEER VOICES', 'MEDIA', 'SCIENCE', 'CULTURE & ARTS', 'TRAVEL', 'TECH', 'WORLD NEWS', 'PARENTING', 'WELLNESS', 'COMEDY', 'EDUCATION', 'FOOD & DRINK', 'ENVIRONMENT', 'U.S. NEWS', 'BUSINESS', 'POLITICS', 'HOME & LIVING', 'ENTERTAINMENT', 'WOMEN', 'STYLE & BEAUTY', 'WEIRD NEWS']

        # Perform prediction using the loaded scikit-learn model
        result = classify_text(model, data, labels)
        result_list = [str(element) for element in result]

        return json.dumps({"result": result_list})

    except Exception as e:
        return json.dumps({"error": str(e)})
