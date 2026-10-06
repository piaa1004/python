"""CLI 실행: python src/train.py (NumPy/Matplotlib 필요)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from opt_utils import (load_dataset, initialize_parameters, gradient_check,
                       run_experiments, plot_learning_curves,
                       plot_decision_boundary, save_results)


def main():
    output_dir = Path(__file__).resolve().parents[1] / "results"
    data = load_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test, stats = data
    check = gradient_check(initialize_parameters(), X_train[:5].copy(), y_train[:5])
    print("Gradient checking:", json.dumps(check, indent=2))
    assert check["passed"], "gradient checking 실패"
    networks, histories, summary = run_experiments(data)
    save_results(output_dir, histories, summary, check)
    fig = plot_learning_curves(histories)
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)
    selected = networks[summary["selected_experiment"]]
    fig = plot_decision_boundary(selected, X_test, y_test)
    fig.savefig(output_dir / "decision_boundary.png", dpi=160)
    plt.close(fig)
    print("\nFinal test:", summary["selected_experiment"],
          "loss = %.6f, accuracy = %.2f%%" % (summary["test_loss"], 100 * summary["test_accuracy"]))


if __name__ == "__main__":
    main()
