import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# env.py and tools.py live in assignment1/ (parent of algorithms/); env.py itself does `import tools`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tuning.py lives next to this file

import tools
import tuning
from env import GridworldEnv

N_EPISODES = 20000          # training episodes per run (same for tuning and final eval)
N_TRIALS = 40               # Optuna trials
N_SEEDS_TUNE = 2            # seeds averaged inside each trial
N_EVAL_EPISODES_TUNE = 200  # greedy evaluation episodes inside each trial
EVAL_SEEDS = range(10, 15)  # seeds (unseen during tuning) for the final before/after report

DEFAULT_PARAMS = {
    "alpha": 0.1,
    "gamma": 0.99,
    "eps_end": 0.05,
    "eps_decay": 0.999,
}


# ------------------------------------------------------------------
# Q-learning (same implementation as before)
# ------------------------------------------------------------------
def greedy_action(Q, s, rng):
    best = np.flatnonzero(np.isclose(Q[s], Q[s].max()))
    return int(rng.choice(best))


def epsilon_greedy(Q, s, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))  # explorar
    return greedy_action(Q, s, rng)           # explotar


def q_learning(
    env,
    n_episodes=20000,
    alpha=0.1,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.999,
    max_steps=200,
    seed=0,
):
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    Q = np.zeros((env.n_states, env.n_actions))
    returns, lengths = [], []
    epsilon = eps_start

    for _ in range(n_episodes):
        s, _ = env.reset()
        total, steps = 0.0, 0

        for _ in range(max_steps):
            a = epsilon_greedy(Q, s, epsilon, rng)  # behaviour policy: eps-greedy
            s2, r, terminated, truncated, _ = env.step(a)
            target = r + gamma * np.max(Q[s2]) * (not terminated)
            Q[s, a] += alpha * (target - Q[s, a])

            total += r
            steps += 1
            s = s2
            if terminated:
                break

        returns.append(total)
        lengths.append(steps)
        epsilon = max(eps_end, epsilon * eps_decay)

    return Q, np.array(returns), np.array(lengths)


def value_iteration(env, gamma=0.99, tol=1e-10):
    Q = np.zeros((env.n_states, env.n_actions))
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


def evaluate_greedy(env, Q, n_episodes=1000, max_steps=200):
    """Average return of the purely greedy policy (epsilon = 0)."""
    rng = np.random.default_rng(123)
    total = 0.0
    for _ in range(n_episodes):
        s, _ = env.reset()
        for _ in range(max_steps):
            s, r, terminated, _, _ = env.step(greedy_action(Q, s, rng))
            total += r
            if terminated:
                break
    return total / n_episodes


# ------------------------------------------------------------------
# Optuna tuning
# ------------------------------------------------------------------
def make_train_fn(env):
    def train_fn(seed, **params):
        Q, _, _ = q_learning(env, n_episodes=N_EPISODES, seed=seed, **params)
        return Q
    return train_fn


def make_eval_fn(env):
    def eval_fn(Q):
        return evaluate_greedy(env, Q, n_episodes=N_EVAL_EPISODES_TUNE)
    return eval_fn


def evaluate_params(env, params, seeds=EVAL_SEEDS):
    """Trains one agent per seed and averages the metrics."""
    Q_star = value_iteration(env, gamma=params["gamma"])
    valid = [s for s in range(env.n_states)
             if s != env.wall_state and s not in env.terminal_states]

    errs, matches, v_start, greedy = [], [], [], []
    for seed in seeds:
        Q, _, _ = q_learning(env, n_episodes=N_EPISODES, seed=seed, **params)
        policy = Q.argmax(axis=1)
        errs.append(np.abs(Q[valid] - Q_star[valid]).max())
        matches.append(np.mean([np.isclose(Q_star[s, policy[s]], Q_star[s].max()) for s in valid]))
        v_start.append(Q[env.start_state].max())
        greedy.append(evaluate_greedy(env, Q))

    return {
        "max |Q - Q*|": np.mean(errs),
        "states with optimal act. (%)": 100 * np.mean(matches),
        "V(start) learned": np.mean(v_start),
        "V(start) true": Q_star[env.start_state].max(),
        "greedy avg return": np.mean(greedy),
    }


def print_report(params, metrics):
    print("  params: " + ", ".join(f"{k}={v:.5g}" for k, v in params.items()))
    for k, v in metrics.items():
        print(f"  {k:<30}: {v:.4f}")


def run(is_slippery):
    name = "stochastic" if is_slippery else "deterministic"
    env = GridworldEnv(is_slippery=is_slippery)

    print(f"\n{'=' * 60}\n=== {name.upper()} ===\n{'=' * 60}")

    # ---------- BEFORE tuning ----------
    print("\n--- BEFORE tuning (default hyperparameters) ---")
    before = evaluate_params(env, DEFAULT_PARAMS)
    print_report(DEFAULT_PARAMS, before)

    # ---------- Optuna tuning ----------
    best_params = tuning.tune(
        train_fn=make_train_fn(env),
        eval_fn=make_eval_fn(env),
        space=lambda trial: tuning.td_space(trial, n_step=False, eps_floor="eps_end"),
        n_trials=N_TRIALS,
        n_seeds=N_SEEDS_TUNE,
        label=f"Q-learning {name}",
    )

    # ---------- AFTER tuning ----------
    print("\n--- AFTER tuning (Optuna best hyperparameters) ---")
    after = evaluate_params(env, best_params)
    print_report(best_params, after)

    # ---------- Comparison ----------
    print("\n--- BEFORE vs AFTER ---")
    print(f"  {'metric':<30}{'before':>12}{'after':>12}")
    for k in before:
        print(f"  {k:<30}{before[k]:>12.4f}{after[k]:>12.4f}")

    # ---------- Plots with the tuned agent ----------
    Q, returns, lengths = q_learning(env, n_episodes=N_EPISODES, **best_params)
    tools.plot_policy(Q.argmax(axis=1), env, title=f"Q-learning policy ({name}, tuned)")
    tools.plot_value_function(Q.max(axis=1), env, title=f"Q-learning V(s) ({name}, tuned)")

    return best_params, returns, lengths


def moving_average(x, w=100):
    return np.convolve(x, np.ones(w) / w, mode="valid")


if __name__ == "__main__":
    best, results = {}, []
    for slippery in (False, True):
        name = "stochastic" if slippery else "deterministic"
        best_params, returns, lengths = run(slippery)
        best[name] = best_params
        results.append((returns, lengths, name))

    print("\n=== BEST HYPERPARAMETERS ===")
    for name, p in best.items():
        print(f"{name}: {p}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for returns, lengths, name in results:
        axes[0].plot(moving_average(returns), label=name)
        axes[1].plot(moving_average(lengths), label=name)
    axes[0].set(title="Return per episode (moving avg, w=100)", xlabel="episode")
    axes[1].set(title="Episode length (moving avg, w=100)", xlabel="episode")
    axes[0].legend()
    plt.tight_layout()
    plt.show()