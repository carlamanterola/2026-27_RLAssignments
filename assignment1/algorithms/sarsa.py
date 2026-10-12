import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Imports setups
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tools
from env import GridworldEnv
from tuning import tune as optuna_tune, td_space

GAMMA_EVAL = 0.99  # fixed discount used to SCORE policies


def eps_greedy_fast(Q_s, eps, n_actions, rng):
    """Fast epsilon-greedy action selection with minimal allocations."""
    if rng.random() < eps:
        return rng.integers(n_actions)
    # Fast greedy selection with tie-breaking
    max_val = Q_s.max()
    best = np.flatnonzero(np.isclose(Q_s, max_val))
    return int(rng.choice(best)) if len(best) > 1 else int(best[0])


def sarsa(env, n_episodes=20000, alpha=0.1, gamma=0.99, eps=1.0,
          eps_min=0.05, eps_decay=0.999, max_steps=200, seed=0):
    """On-policy TD control."""
    rng = np.random.default_rng(seed)
    n_states, n_actions = env.n_states, env.n_actions
    Q = np.zeros((n_states, n_actions), dtype=np.float64)
    returns = np.zeros(n_episodes, dtype=np.float64)

    for ep in range(n_episodes):
        s, _ = env.reset(seed=int(rng.integers(1e6)) if ep == 0 else None)
        a = eps_greedy_fast(Q[s], eps, n_actions, rng)
        G = 0.0

        for step in range(max_steps):
            s2, r, term, trunc, _ = env.step(a)
            G += r
            
            if term or trunc:
                # Terminal state target is r (Q(terminal, .) = 0)
                Q[s, a] += alpha * (r - Q[s, a])
                break

            a2 = eps_greedy_fast(Q[s2], eps, n_actions, rng)
            Q[s, a] += alpha * (r + gamma * Q[s2, a2] - Q[s, a])
            s, a = s2, a2

        returns[ep] = G
        eps = max(eps_min, eps * eps_decay)

    return Q, returns


def evaluate(env, Q, n_episodes=1000, max_steps=200, gamma=GAMMA_EVAL):
    """Optimized greedy-policy evaluation."""
    rng = np.random.default_rng(123)
    rets = np.zeros(n_episodes, dtype=np.float64)
    goals, pits = 0, 0
    
    # Pre-calculate greedy policy to avoid inline argmax/isclose overhead
    # Pre-break ties deterministically or once statically
    max_vals = Q.max(axis=1, keepdims=True)
    is_best = np.isclose(Q, max_vals)
    
    # Precompute discount factors: gamma^0, gamma^1, ..., gamma^(max_steps-1)
    discounts = gamma ** np.arange(max_steps)

    for ep in range(n_episodes):
        s, _ = env.reset()
        G = 0.0
        for t in range(max_steps):
            best_actions = np.flatnonzero(is_best[s])
            a = int(rng.choice(best_actions)) if len(best_actions) > 1 else int(best_actions[0])
            
            s, r, term, trunc, _ = env.step(a)
            G += discounts[t] * r

            if term or trunc:
                goals += int(r > 0)
                pits += int(r < 0)
                break
        rets[ep] = G

    return rets.mean(), goals / n_episodes, pits / n_episodes


def value_iteration(env, gamma=0.99, tol=1e-10):
    """Ground truth Q* from env.P."""
    Q = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    while True:
        V = Q.max(axis=1)
        Q_new = np.zeros_like(Q)
        for s in range(env.n_states):
            for a in range(env.n_actions):
                for p, s2, r, done in env.P[s][a]:
                    Q_new[s, a] += p * (r + gamma * V[s2] * (not done))
        if np.abs(Q_new - Q).max() < tol:
            return Q_new
        Q = Q_new


def moving_average(x, w=100):
    return np.convolve(x, np.ones(w) / w, mode="valid")


N_EPISODES = 20000
N_TRIALS = 40
N_SEEDS = 2
EVAL_SEEDS = range(10, 15)

DEFAULT_PARAMS = {
    "alpha": 0.1,
    "gamma": 0.99,
    "eps_min": 0.05,
    "eps_decay": 0.999,
}


def evaluate_params(env, params, seeds=EVAL_SEEDS):
    """Trains SARSA agents and computes metrics against ground truth."""
    Q_star = value_iteration(env, gamma=params["gamma"])
    valid = [s for s in range(env.n_states)
             if s != env.wall_state and s not in env.terminal_states]

    errs, matches, v_start, greedy = [], [], [], []
    for seed in seeds:
        Q, _ = sarsa(env, n_episodes=N_EPISODES, seed=seed, **params)
        policy = Q.argmax(axis=1)
        errs.append(np.abs(Q[valid] - Q_star[valid]).max())
        matches.append(np.mean([np.isclose(Q_star[s, policy[s]], Q_star[s].max()) for s in valid]))
        v_start.append(Q[env.start_state].max())
        greedy.append(evaluate(env, Q, n_episodes=1000, gamma=GAMMA_EVAL)[0])

    return {
        "max |Q - Q*|": float(np.mean(errs)),
        "states with optimal act. (%)": 100 * float(np.mean(matches)),
        "V(start) learned": float(np.mean(v_start)),
        "V(start) true": float(Q_star[env.start_state].max()),
        "greedy avg return": float(np.mean(greedy)),
    }


def print_report(params, metrics):
    print("  params: " + ", ".join(f"{k}={v:.5g}" for k, v in params.items()))
    for k, v in metrics.items():
        print(f"  {k:<30}: {v:.4f}")


def tune(slippery):
    env = GridworldEnv(is_slippery=slippery)
    return optuna_tune(
        train_fn=lambda seed, **p: sarsa(env, n_episodes=N_EPISODES, seed=seed, **p)[0],
        eval_fn=lambda Q: evaluate(env, Q, n_episodes=500, gamma=GAMMA_EVAL)[0],
        space=lambda trial: td_space(trial, eps_floor="eps_min"),
        n_trials=N_TRIALS,
        n_seeds=N_SEEDS,
        label=f"{'stochastic' if slippery else 'deterministic'} sarsa",
    )


if __name__ == "__main__":
    for slippery in (False, True):
        mode = "stochastic" if slippery else "deterministic"
        env = GridworldEnv(is_slippery=slippery)
        tools.plot_environment(env, title=f"Environment ({mode})")

        print(f"\n{'=' * 60}\n=== {mode.upper()} ===\n{'=' * 60}")

        print("\n--- BEFORE tuning (default hyperparameters) ---")
        before = evaluate_params(env, DEFAULT_PARAMS)
        print_report(DEFAULT_PARAMS, before)

        best = tune(slippery)

        print("\n--- AFTER tuning (Optuna best hyperparameters) ---")
        after = evaluate_params(env, best)
        print_report(best, after)

        print("\n--- BEFORE vs AFTER ---")
        print(f"  {'metric':<30}{'before':>12}{'after':>12}")
        for k in before:
            print(f"  {k:<30}{before[k]:>12.4f}{after[k]:>12.4f}")

        Q, rets = sarsa(env, n_episodes=N_EPISODES, **best)
        name = "SARSA (tuned)"
        tools.plot_policy(Q.argmax(axis=1), env, title=f"{name} policy ({mode})")
        tools.plot_value_function(Q.max(axis=1), env, title=f"{name} V(s) = max_a Q ({mode})")

        plt.figure(figsize=(7, 4))
        plt.plot(moving_average(rets), label=name)
        plt.xlabel("Episode")
        plt.ylabel("Return (moving avg, w=100)")
        plt.title(f"Learning curve ({mode})")
        plt.legend()
        plt.tight_layout()
        plt.show()