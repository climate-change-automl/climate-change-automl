import os
import subprocess

import fire
import optuna
import time


def get_metric(s, metric="node_loss"):
    """Gets metric from log file"""
    ssplit = s.split(" | ")
    mnames = ["loss", "ewth", "node_loss"]
    assert metric in mnames
    metric_dict = {}
    for term in ssplit:
        kv = term.split(" ")
        if kv[0] in mnames:
            metric_dict[kv[0]] = float(kv[1])
    return metric_dict[metric]


def process_logs(log_output, val_set="val_id", test_set="val_ood_ads"):
    valid_on = "valid on"
    log_lines = log_output.split("\n")
    log_lines_validon = list(filter(lambda s: valid_on in s, log_lines))

    log_lines_val = list(filter(lambda s: val_set in s, log_lines_validon))
    final_log_line_val = log_lines_val[-1]
    val_metric = get_metric(final_log_line_val)

    log_lines_test = list(filter(lambda s: test_set in s, log_lines_validon))
    final_log_line_test = log_lines_test[-1]
    test_metric = get_metric(final_log_line_test)

    return val_metric, test_metric


def tuning_objective(trial):

    hps = {
        "lr": f"{trial.suggest_float('lr', 1e-5, 1e-3, log=True)}",
        "warmup_steps": f"{trial.suggest_int('warmup_steps', 1, 10_000, log=True)}",
        "layers": f"{trial.suggest_int('layers', 1, 12)}",
        "num_head": f"{trial.suggest_categorical('num_head', [6, 12, 24, 32, 48])}",
        "blocks": f"{trial.suggest_int('blocks', 1, 4)}",
    }  # TODO ALSO CONVERT TO STR

    subprocess.run(
        ["bash", f"oc20_automl.sh"],
        # capture_output=True,
        # text=True,
        bufsize=1,
        env=dict(**os.environ, **hps),
    )

    # Give it time to write the log
    time.sleep(2.5)
    with open("trial.log", "r") as file:
        log_output = file.read()

    print(log_output)

    val_obj, test_obj = process_logs(log_output)
    print(val_obj, test_obj)

    return val_obj


def main(n_trials=100, delete_study=False):
    study_metadata = {"study_name": "graphormer", "storage": "sqlite:///graphormer.db'"}
    if delete_study:
        optuna.delete_study(**study_metadata)
        return

    print(f"Running {n_trials} trials...")
    try:
        study = optuna.load_study(**study_metadata)
    except KeyError:  # Study doesn't exist
        study = optuna.create_study(**study_metadata)
    study.optimize(tuning_objective, n_trials=n_trials)


if __name__ == "__main__":
    fire.Fire(main)
