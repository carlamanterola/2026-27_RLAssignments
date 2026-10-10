import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# env.py and tools.py live in assignment1/ (parent of algorithms/); env.py itself does `import tools`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools
from env import GridworldEnv


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


def moving_average(x, w=100):
    return np.convolve(x, np.ones(w) / w, mode="valid")


if __name__ == "__main__":
    N_STEP = 3
    for slippery in (False, True):
        mode = "stochastic" if slippery else "deterministic"
        env = GridworldEnv(is_slippery=slippery)
        tools.plot_environment(env, title=f"Environment ({mode})")

        results = {}
        results["SARSA"] = sarsa(env)
        results[f"{N_STEP}-step SARSA"] = n_step_sarsa(env, n=N_STEP)

        plt.figure(figsize=(7, 4))
        for name, (Q, rets) in results.items():
            plt.plot(moving_average(rets), label=name)
            tools.plot_policy(Q.argmax(axis=1), env, title=f"{name} policy ({mode})")
            tools.plot_value_function(Q.max(axis=1), env, title=f"{name} V(s) = max_a Q ({mode})")
            m, g, p = evaluate(env, Q)
            print(f"[{mode}] {name}: mean return={m:.3f}, goal rate={g:.3f}, pit rate={p:.3f}")
        plt.xlabel("Episode")
        plt.ylabel("Return (moving avg, w=100)")
        plt.title(f"Learning curves ({mode})")
        plt.legend()
        plt.tight_layout()
        plt.show()