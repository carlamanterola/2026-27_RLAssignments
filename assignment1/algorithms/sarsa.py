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


def n_step_sarsa(env, n=3, n_episodes=5000, alpha=0.1, gamma=0.99, eps=1.0,
                 eps_min=0.05, eps_decay=0.999, max_steps=200, seed=0):
    """On-policy n-step TD control.
    G_t^(n) = R_{t+1} + ... + gamma^{n-1} R_{t+n} + gamma^n q(S_{t+n}, A_{t+n})
    (bootstrapping term dropped if the episode ends before t+n)."""
    Q = np.zeros((env.n_states, env.n_actions))
    returns = np.zeros(n_episodes)
    env.reset(seed=seed)

    for ep in range(n_episodes):
        s0, _ = env.reset()
        states = [s0]
        actions = [eps_greedy(Q, s0, eps)]
        rewards = [0.0]  # dummy R_0 so that rewards[i] = R_i
        T = 10**9
        t = 0
        while True:
            if t < T:
                s2, r, term, trunc, _ = env.step(actions[t])
                rewards.append(r)
                states.append(s2)
                if term or t + 1 >= max_steps:
                    T = t + 1
                else:
                    actions.append(eps_greedy(Q, s2, eps))
            tau = t - n + 1
            if tau >= 0:
                G = sum(gamma ** (i - tau - 1) * rewards[i]
                        for i in range(tau + 1, min(tau + n, T) + 1))
                if tau + n < T:
                    G += gamma ** n * Q[states[tau + n], actions[tau + n]]
                Q[states[tau], actions[tau]] += alpha * (G - Q[states[tau], actions[tau]])
            if tau == T - 1:
                break
            t += 1
        returns[ep] = sum(rewards)
        eps = max(eps_min, eps * eps_decay)
    return Q, returns


def evaluate(env, Q, n_episodes=1000, max_steps=100):
    """Greedy-policy evaluation: (mean return, goal rate, pit rate)."""
    rets, goals, pits = [], 0, 0
    for _ in range(n_episodes):
        s, _ = env.reset()
        G = 0.0
        for _ in range(max_steps):
            s, r, term, trunc, _ = env.step(int(np.argmax(Q[s])))
            G += r
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
    """Console summary (same format as q-learning.py)."""
    Q_star = value_iteration(env, gamma=gamma)
    valid = [s for s in range(env.n_states)
             if s != env.wall_state and s not in env.terminal_states]
    policy = Q.argmax(axis=1)
    match = np.mean([np.isclose(Q_star[s, policy[s]], Q_star[s].max()) for s in valid])

    print(f"\n=== {title.upper()} ===")
    print(f"max |Q - Q*|            : {np.abs(Q[valid] - Q_star[valid]).max():.4f}")
    print(f"states with optimal act.: {match * 100:.0f}%")
    print(f"V(start) learned / true : {Q[env.start_state].max():.3f} / {Q_star[env.start_state].max():.3f}")
    print(f"greedy avg return       : {evaluate(env, Q, n_episodes=1000, max_steps=200)[0]:.3f}")


def moving_average(x, w=100):
    return np.convolve(x, np.ones(w) / w, mode="valid")


N_EPISODES = 5000
N_TRIALS = 40
N_SEEDS = 2


def tune(algo, slippery):
    """Optuna search via the shared tuning module. algo in {"sarsa", "n_step"}."""
    env = GridworldEnv(is_slippery=slippery)
    fn = sarsa if algo == "sarsa" else n_step_sarsa
    return optuna_tune(
        train_fn=lambda seed, **p: fn(env, n_episodes=N_EPISODES, seed=seed, **p)[0],
        eval_fn=lambda Q: evaluate(env, Q, n_episodes=500)[0],
        space=lambda trial: td_space(trial, n_step=(algo == "n_step"), eps_floor="eps_min"),
        n_trials=N_TRIALS,
        n_seeds=N_SEEDS,
        label=f"{'stochastic' if slippery else 'deterministic'} {algo}",
    )


if __name__ == "__main__":
    for slippery in (False, True):
        mode = "stochastic" if slippery else "deterministic"
        env = GridworldEnv(is_slippery=slippery)
        tools.plot_environment(env, title=f"Environment ({mode})")

        best_s = tune("sarsa", slippery)
        best_n = tune("n_step", slippery)
        results = {
            "SARSA (tuned)": (*sarsa(env, n_episodes=N_EPISODES, **best_s), best_s["gamma"]),
            f"{best_n['n']}-step SARSA (tuned)": (*n_step_sarsa(env, n_episodes=N_EPISODES, **best_n), best_n["gamma"]),
        }

        plt.figure(figsize=(7, 4))
        for name, (Q, rets, gamma) in results.items():
            report(env, Q, f"{mode} - {name}", gamma)
            plt.plot(moving_average(rets), label=name)
            tools.plot_policy(Q.argmax(axis=1), env, title=f"{name} policy ({mode})")
            tools.plot_value_function(Q.max(axis=1), env, title=f"{name} V(s) = max_a Q ({mode})")
        plt.xlabel("Episode")
        plt.ylabel("Return (moving avg, w=100)")
        plt.title(f"Learning curves ({mode})")
        plt.legend()
        plt.tight_layout()
        plt.show()