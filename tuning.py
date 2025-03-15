import optuna
from main import main
import csv
import os
import argparse

def save_results_to_csv(trial, trial_values, evaluation_metric, metric_name, output_file):
    """
    Save the results of an Optuna trial to a CSV file.
    
    Args:
        trial (optuna.Trial): The Optuna trial.
        trial_values (dict): Dictionary containing the hyperparameters of the trial.
        evaluation_metric (dict): Dictionary containing the evaluation metric for the trial.
        metric_name (str): The name of the evaluation metric (e.g., "F1" or "Accuracy").
        output_file (str): The path to the output CSV file.
    """
    fieldnames = ["trial_number"] + list(trial_values.keys()) + [metric_name]

    # Create the output file and write the header if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, mode="w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # Append the results of the current trial to the output file
    with open(output_file, mode="a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        row_data = {"trial_number": trial.number, **trial_values, metric_name: evaluation_metric[metric_name]}
        writer.writerow(row_data)

def objective(trial, model_type, subtask):
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
        trial (optuna.Trial): The Optuna trial.
        model_type (str): The type of model to optimize ("LSTM" or "CNN").
        subtask (str): The subtask to optimize for ("a" or "b").
        
    Returns:
        float: The value of the evaluation metric for the trial.
    """

    # Common hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    num_epochs = trial.suggest_int("num_epochs", 5, 50)
    patience = trial.suggest_int("patience", 5, 10)

    # Model-specific hyperparameters
    if model_type == "LSTM":
        hidden_dim = trial.suggest_int("hidden_dim", 10, 100)
        args = {
            "hidden_dim": hidden_dim,
        }
    elif model_type == "CNN":
        num_filters = trial.suggest_int("num_filters", 10, 200)
        num_filter_sizes = trial.suggest_int("num_filter_sizes", 1, 6)
        filter_sizes = [trial.suggest_int(f"filter_size_{i}", 2, 8) for i in range(num_filter_sizes)]
        hidden_dim = trial.suggest_int("hidden_dim", 10, 100)
        args = {
            "num_filters": num_filters,
            "filter_sizes": filter_sizes,
            'hidden_dim':hidden_dim
        }

    # Set hyperparameters as arguments for main function
    args.update({
        "model_type": model_type,
        "subtask": subtask,
        "lr": lr,
        "dropout": dropout,
        "num_epochs": num_epochs,
        "patience": patience
    })

    # Run the main function and get the evaluation metric(F1 score or accuracy)
    args_namespace = argparse.Namespace(**args)
    evaluation_metric = main(args_namespace)

    # Save the results to a CSV file
    metric_name = "F1" if subtask == "a" else "Accuracy"
    output_file = f"hyperparameter_tuning_results_{model_type}_{subtask}.csv"
    save_results_to_csv(trial, args, evaluation_metric, metric_name, output_file)

    return evaluation_metric[metric_name]

if __name__ == "__main__":
    model_types = ["LSTM", "CNN"]
    subtasks = ["a", "b"]

    for model_type in model_types:
        for subtask in subtasks:
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, model_type, subtask), n_trials=200)

            # Print the best hyperparameters
            print(f"Best trial for {model_type} model and subtask {subtask}:")
            trial = study.best_trial
            print(f"  Value: {trial.value}")
            print("  Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
