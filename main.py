import argparse
from functions import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Ejecutar experimento de ML con MLflow")
    parser.add_argument("--max_df", type=float, default=0.7,
                        help="Valor de max_df para TfidfVectorizer")
    parser.add_argument("--clf_type", type=str, default="logreg",
                        choices=["logreg", "svm"],
                        help="Tipo de clasificador a usar")
    return parser.parse_args()


def main():
    args = parse_args()

    acc, prec, rec = run_experiment(
        max_df=args.max_df,
        ngram_range=(1, 2),
        clf_type=args.clf_type
    )

    print(f"Experimento completado:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")


if __name__ == "__main__":
    main()
