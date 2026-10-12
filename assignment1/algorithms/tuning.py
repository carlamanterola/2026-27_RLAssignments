"""Shared Optuna hyperparameter tuning for tabular TD algorithms."""

import numpy as np
import optuna


def td_space(trial, n_step=False, eps_floor="eps_min"):
    """Search space shared by SARSA / n-step SARSA / Q-learning.
    eps_floor: name of the final-epsilon argument ("eps_min" or "eps_end")."""
    params = {
        "alpha": trial.suggest_float("alpha", 0.01, 0.9, log=True),
        "gamma": trial.suggest_float("gamma", 0.8, 0.999),
        eps_floor: trial.suggest_float(eps_floor, 0.001, 0.2, log=True),
        "eps_decay": trial.suggest_float("eps_decay", 0.99, 0.9999),
    }
    if n_step:
        params["n"] = trial.suggest_int("n", 1, 10)
    return params


def tune(train_fn, eval_fn, space, n_trials=40, n_seeds=2, sampler_seed=0, label=""):
    """Generic Optuna (TPE) maximisation.

    train_fn(seed=int, **params) -> Q          trains one agent
    eval_fn(Q) -> float                        score to maximise
    space(trial) -> dict                       hyperparameters for this trial
    Score of a trial = mean of eval_fn over n_seeds independent training runs.
    Returns the best hyperparameters (dict)."""

    def objective(trial):
        params = space(trial)
        scores = [eval_fn(train_fn(seed=s, **params)) for s in range(n_seeds)]
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=sampler_seed)
    )
    study.optimize(objective, n_trials=n_trials)
    print(f"[{label}] best value={study.best_value:.3f}, params={study.best_params}")
    return study.best_params