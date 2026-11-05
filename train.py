from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

X_TRAIN = [
    'santé maladie médicament docteur hopital',
    'maladie fièvre médicament santé',
    'je suis malade je vais à l hopital',
    'ecole université formation cours',
    'ecole formation élève professeur',
    'je vais à l université pour étudier',
    'eglise religion prière dieu',
    'foi eglise pasteur religion',
    'je vais à l eglise prier',
    'orange mangue citron fruit',
    'mangue banane fruit orange',
    'je mange un fruit orange',
]
Y_TRAIN = ['Hopital','Hopital','Hopital',
           'Education','Education','Education',
           'Foi','Foi','Foi',
           'Fruit','Fruit','Fruit']

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('model', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_TRAIN, Y_TRAIN)
joblib.dump(pipeline, "texte_model.pkl")
print("Modèle sauvegardé !")
print(pipeline.predict(['maladie de ama']))
print(pipeline.predict(['Le docteur est a l hopital']))