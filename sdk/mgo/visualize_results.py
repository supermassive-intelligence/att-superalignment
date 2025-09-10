import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def parse_filename(filename):
    """
    Parse a filename in the form <model_name>_<database_name>_eval_results_<solve_step>.json,
    supporting underscores in the database name.
    Returns (model_name, database_name, solve_step) or None if not a match.
    """
    if not filename.endswith(".json"):
        return None
    name = filename[:-5]  # Remove '.json'
    if "_eval_results_" not in name:
        return None
    pre, step = name.rsplit("_eval_results_", 1)
    if "_" not in pre:
        return None
    model, db = pre.split("_", 1)
    try:
        solve_step = int(step)
    except ValueError:
        return None

    model = model[:7]
    if model == "None":
        model = "base"

    return model, db, solve_step


def plot_accuracy_by_database(results_dir="data/results", output_dir="plots"):
    """
    Scans result files and generates line plots (with markers) of model accuracy over solve steps.
    One plot per database, y-axis = accuracy (%), x-axis = solve step (iteration).
    Each model is indicated by a line with markers.
    Output files are named <database_name>.jpg.
    """

    # database -> model -> {solve_step: accuracy}
    data = defaultdict(lambda: defaultdict(dict))

    for filename in os.listdir(results_dir):
        parsed = parse_filename(filename)
        if not parsed:
            continue
        model_name, db_name, solve_step = parsed

        filepath = os.path.join(results_dir, filename)
        with open(filepath, "r") as f:
            try:
                results = json.load(f)
            except Exception as e:
                print(f"Could not read {filename}: {e}")
                continue

        total = len(results)
        
        if total == 0:
            continue
        num_correct = sum(1 for entry in results if "score" in entry and entry.get("score", 0) >= 1.0)
        accuracy = 100 * num_correct / total

        data[db_name][model_name][solve_step] = accuracy

    # Plot for each database
    for db_name, models in data.items():
        plt.figure()
        for model_name, step_acc in models.items():
            steps = sorted(step_acc.keys())
            accuracies = [step_acc[step] for step in steps]
            plt.plot(steps, accuracies, marker="o", label=model_name)
        plt.xlabel("Solve Step")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracy by Solve Step ({db_name})")
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 100)
        plt.tight_layout()
        output_path = f"{output_dir}/{db_name}.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(f"{output_path}")
        plt.close()


def plot_grouped_bar_highest_accuracy(results_dir="data/results", output_dir="plots"):

    # {database: {model: [accuracies at different steps]}}
    data = {}
    for filename in os.listdir(results_dir):
        parsed = parse_filename(filename)
        if not parsed:
            continue
        model, db, _ = parsed

        filepath = os.path.join(results_dir, filename)
        with open(filepath, "r") as f:
            try:
                results = json.load(f)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
        total = len(results)
        if total == 0:
            continue
        num_correct = sum(1 for entry in results if "score" in entry and entry.get("score", 0) >= 1.0)
        acc = 100 * num_correct / total
        data.setdefault(db, {}).setdefault(model, []).append(acc)

        # Compute max accuracy for each model within each database
        max_acc = {}
        for db, models in data.items():
            max_acc[db] = {
                model: max(accuracies) for model, accuracies in models.items()
            }

    # Collect list of databases and models (in sorted order)
    db_names = sorted(max_acc)
    model_names = sorted({m for db in max_acc.values() for m in db})

    # Build data array [db1_model1, db1_model2, ..., db2_model1, ...]
    bar_data = {
        model: [max_acc[db].get(model, 0) for db in db_names] for model in model_names
    }

    # Bar plotting
    x = np.arange(len(db_names))  # the group positions
    width = 0.8 / len(model_names) if model_names else 0.2  # auto width

    plt.figure(figsize=(1.5 * len(db_names) + 4, 6))
    for i, model in enumerate(model_names):
        offset = (i - (len(model_names) - 1) / 2) * width
        plt.bar(x + offset, bar_data[model], width, label=model)
    plt.xlabel("Database")
    plt.ylabel("Highest Accuracy (%)")
    plt.title("Best Accuracy by Model for Each Database")
    plt.xticks(x, db_names, rotation=20)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    output_path = f"{output_dir}/all_databases_grouped_accuracy.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(f"{output_path}")
    plt.close()


def get_max_accuracy_for_db(results_dir="data/results"):
    # {database: [accuracies]}
    db_acc = {}
    for filename in os.listdir(results_dir):
        parsed = parse_filename(filename)
        if not parsed:
            continue
        model, db, _ = parsed

        filepath = os.path.join(results_dir, filename)
        with open(filepath, "r") as f:
            try:
                results = json.load(f)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
        total = len(results)
        if total == 0:
            continue
        num_correct = sum(1 for entry in results if entry.get("score", 0) >= 1.0)
        acc = 100 * num_correct / total
        db_acc.setdefault(db, []).append(acc)

    # Compute the max accuracy for each database
    max_acc = {db: max(accs) for db, accs in db_acc.items()}
    return max_acc

# Example usage:
plot_accuracy_by_database()
plot_grouped_bar_highest_accuracy()