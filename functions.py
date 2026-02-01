import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score


def load_data():
    """Carga el dataset 20 Newsgroups (train y test)."""
    train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    return train.data, train.target, test.data, test.target


def run_experiment(max_df=0.7, ngram_range=(1, 1), clf_type="logreg"):
    """Ejecuta un experimento con MLflow registrando parámetros, métricas y modelo."""

    X_train, y_train, X_test, y_test = load_data()

    # Selección del modelo
    if clf_type == "logreg":
        clf = LogisticRegression(max_iter=1000)
    elif clf_type == "svm":
        clf = LinearSVC()
    else:
        raise ValueError(f"Clasificador no soportado: {clf_type}")

    # Pipeline TF-IDF + Modelo
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_df=max_df, ngram_range=ngram_range)),
        ("clf", clf),
    ])

    # MLflow
    mlflow.set_experiment("text_classification_20newsgroups")

    run_name = f"{clf_type}_maxdf{max_df}_ng{ngram_range}"

    with mlflow.start_run(run_name=run_name):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")

        # Log de parámetros
        mlflow.log_param("max_df", max_df)
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("clf_type", clf_type)

        # Log de métricas
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Guardar modelo
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        return acc, prec, rec
