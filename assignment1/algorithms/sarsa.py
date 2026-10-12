import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# env.py and tools.py live in assignment1/ (parent of algorithms/); env.py itself does `import tools`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tuning.py lives next to this file

import tools
from env import GridworldEnv
from tuning import tune as optuna_tune, td_space

GAMMA_EVAL = 0.99  # fixed discount used to SCORE policies (shorter paths score higher)


def eps_greedy(Q, s, eps):
    """epsilon-greedy action from Q (random tie-breaking among greedy actions)."""
    if np.random.rand() < eps:
        return np.random.randint(Q.shape[1])
    best = np.flatnonzero(np.isclose(Q[s], Q[s].max()))
    return int(np.random.choice(best))


def sarsa(env, n_episodes=5000, alpha=0.1, gamma=0.99, eps=1.0,
          eps_min=0.05, eps_decay=0.999, max_steps=200, seed=0):
    """On-policy TD control: q(s,a) <- q(s,a) + alpha [r + gamma q(s',a') - q(s,a)]."""
    Q = np.zeros((env.n_states, env.n_actions))
    returns = np.zeros(n_episodes)
    env.reset(seed=seed)

    for ep in range(n_episodes):
        s, _ = env.reset()
        a = eps_greedy(Q, s, eps)
        G = 0.0
        for _ in range(max_steps):
            s2, r, term, trunc, _ = env.step(a)
            G += r
            if term:
                target = r  # q(terminal, .) = 0
                Q[s, a] += alpha * (target - Q[s, a])
                break
            a2 = eps_greedy(Q, s2, eps)
            target = r + gamma * Q[s2, a2]
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s2, a2
        returns[ep] = G
        eps = max(eps_min, eps * eps_decay)
    return Q, returns


def greedy_action(Q, s, rng):
    """argmax_a Q[s, a] with random tie-breaking (same as q-learning.py)."""
    best = np.flatnonzero(np.isclose(Q[s], Q[s].max()))
    return int(rng.choice(best))


def evaluate(env, Q, n_episodes=1000, max_steps=200, gamma=GAMMA_EVAL):
    """Greedy-policy evaluation: (mean DISCOUNTED return, goal rate, pit rate).
    Discounting makes the score sensitive to path length, so it no longer
    saturates at 1.0 for every policy that eventually reaches the goal."""
    rng = np.random.default_rng(123)
    rets, goals, pits = [], 0, 0
    for _ in range(n_episodes):
        s, _ = env.reset()
        G = 0.0
        for t in range(max_steps):
            s, r, term, trunc, _ = env.step(greedy_action(Q, s, rng))
            G += (gamma ** t) * r
            if term:
                goals += r > 0
                pits += r < 0
                break
        rets.append(G)
    return np.mean(rets), goals / n_episodes, pits / n_episodes


def value_iteration(env, gamma=0.99, tol=1e-10):
    """Ground truth Q* from env.P (used to verify the learned Q)."""
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


def report(env, Q, title, gamma):
    """Console summary. Adds max|V-V*| (less dominated by off-path actions than
    max|Q-Q*|) and the discounted greedy return next to the optimal V*(start)."""
    Q_star = value_iteration(env, gamma=gamma)
    valid = [s for s in range(env.n_states)
             if s != env.wall_state and s not in env.terminal_states]
    policy = Q.argmax(axis=1)
    match = np.mean([np.isclose(Q_star[s, policy[s]], Q_star[s].max()) for s in valid])
    v_err = np.abs(Q.max(axis=1)[valid] - Q_star.max(axis=1)[valid]).max()
    regret = max(Q_star[s].max() - Q_star[s, policy[s]] for s in valid)
    ret, goal_rate, pit_rate = evaluate(env, Q, n_episodes=1000, gamma=gamma)

    print(f"\n=== {title.upper()} ===")
    print(f"max |Q - Q*|            : {np.abs(Q[valid] - Q_star[valid]).max():.4f}")
    print(f"max |V - V*|            : {v_err:.4f}")
    print(f"states with optimal act.: {match * 100:.0f}%  (max action gap {regret:.4f})")
    print(f"V(start) learned / true : {Q[env.start_state].max():.3f} / {Q_star[env.start_state].max():.3f}")
    print(f"greedy discounted return: {ret:.3f}  (optimal V*(start) = {Q_star[env.start_state].max():.3f})")
    print(f"goal / pit rate         : {goal_rate:.3f} / {pit_rate:.3f}")


def moving_average(x, w=100):
    return np.convolve(x, np.ones(w) / w, mode="valid")


N_EPISODES = 5000
N_TRIALS = 40
N_SEEDS = 2


def tune(slippery):
    """Optuna search via the shared tuning module.
    Score = DISCOUNTED greedy return with a fixed GAMMA_EVAL, so trials are
    ranked by path efficiency instead of tying at 1.0."""
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

        best = tune(slippery)
        Q, rets = sarsa(env, n_episodes=N_EPISODES, **best)
        name = "SARSA (tuned)"

        report(env, Q, f"{mode} - {name}", best["gamma"])
        tools.plot_policy(Q.argmax(axis=1), env, title=f"{name} policy ({mode})")
        tools.plot_value_function(Q.max(axis=1), env, title=f"{name} V(s) = max_a Q ({mode})")

        # learning curve drawn after the tools' plt.show() calls (they open/close their own figures)
        plt.figure(figsize=(7, 4))
        plt.plot(moving_average(rets), label=name)
        plt.xlabel("Episode")
        plt.ylabel("Return (moving avg, w=100)")
        plt.title(f"Learning curve ({mode})")
        plt.legend()
        plt.tight_layout()
        plt.show()