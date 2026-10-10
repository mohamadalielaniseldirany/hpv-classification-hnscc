import argparse
import importlib.util
import textwrap
from pathlib import Path

import matplotlib


def tk_backend_available():
    """Check whether Tk can really open on this machine."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return True
    except Exception as error:
        print(f"TkAgg is not available: {error}")
        return False


def choose_backend(preferred_backend):
    """Choose the backend before pyplot is imported."""
    if preferred_backend:
        matplotlib.use(preferred_backend)
        return

    if tk_backend_available():
        matplotlib.use("TkAgg")
        return

    if importlib.util.find_spec("tornado"):
        # WebAgg uses Matplotlib itself but avoids native Tk/Qt windows.
        matplotlib.use("WebAgg")
        return

    raise RuntimeError(
        "No interactive Matplotlib backend is available. "
        "TkAgg is broken, Qt is not installed, and WebAgg needs the missing 'tornado' package."
    )


def shorten_report(report_text, max_characters=2600):
    """Keep the explanation readable inside one Matplotlib figure."""
    if len(report_text) <= max_characters:
        return report_text

    return (
        report_text[:max_characters]
        + "\n\n[Report shortened inside the Matplotlib window. Full report is saved as the .txt file.]"
    )


def show_results(dataset, report_path, missingness_plot, non_numeric_plot):
    """Display the explanation and before/after plots with Matplotlib."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    report_text = Path(report_path).read_text(encoding="utf-8")
    report_text = shorten_report(report_text)

    missingness_image = mpimg.imread(missingness_plot)
    non_numeric_image = mpimg.imread(non_numeric_plot)

    fig = plt.figure(figsize=(15, 9))
    fig.canvas.manager.set_window_title(f"{dataset} preprocessing results")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.6])

    text_axis = fig.add_subplot(grid[0, :])
    text_axis.axis("off")
    text_axis.set_title("Explanation and Methods Used", loc="left", fontsize=13, pad=8)
    wrapped_report = "\n".join(
        textwrap.fill(line, width=145, replace_whitespace=False)
        for line in report_text.splitlines()
    )
    text_axis.text(
        0,
        1,
        wrapped_report,
        va="top",
        ha="left",
        family="monospace",
        fontsize=7.5,
        transform=text_axis.transAxes,
    )

    missing_axis = fig.add_subplot(grid[1, 0])
    missing_axis.imshow(missingness_image)
    missing_axis.axis("off")
    missing_axis.set_title("Missing Values Before and After", fontsize=12)

    non_numeric_axis = fig.add_subplot(grid[1, 1])
    non_numeric_axis.imshow(non_numeric_image)
    non_numeric_axis.axis("off")
    non_numeric_axis.set_title("Non-Numeric Columns Before and After", fontsize=12)

    fig.suptitle(f"{dataset} Imputation Preprocessing", fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Show Week 9 imputation results in a Matplotlib viewer."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--missingness-plot", required=True)
    parser.add_argument("--non-numeric-plot", required=True)
    parser.add_argument(
        "--backend",
        default=None,
        help="Optional Matplotlib backend, for example TkAgg. Default is WebAgg.",
    )
    args = parser.parse_args()

    try:
        choose_backend(args.backend)
        show_results(
            dataset=args.dataset,
            report_path=args.report,
            missingness_plot=args.missingness_plot,
            non_numeric_plot=args.non_numeric_plot,
        )
    except Exception as error:
        print(f"Could not open Matplotlib plot viewer: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
