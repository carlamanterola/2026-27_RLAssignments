import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib.pyplot as plt
import numpy as np
import tools
from env import GridworldEnv


def greedy_action(Q, s, rng):
    best = np.flatnonzero(np.isclose(Q[s], Q[s].max()))
    return int(rng.choice(best))


def epsilon_greedy(Q, s, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))  #explorar
    return greedy_action(Q, s, rng)           #explotar


# -----------
# Q-learning
# -----------
def q_learning(
    env,
    n_episodes=20000,
    alpha=0.1,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.9997,  
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


# ----------------------------------------------------
# Value iteration on env.P -> To verify our Q-learning
# ----------------------------------------------------
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


def moving_average(x, w=100):
    return np.convolve(x, np.ones(w) / w, mode="valid")


# ---------------------------------
# Main: deterministic + stochastic
# ---------------------------------
def run(is_slippery, gamma=0.99):
    name = "stochastic" if is_slippery else "deterministic" #If is_slippery is True, name is "stochastic", otherwise it's "deterministic"
    env = GridworldEnv(is_slippery=is_slippery)

    Q, returns, lengths = q_learning(env, gamma=gamma)
    Q_star = value_iteration(env, gamma=gamma)

    valid = [s for s in range(env.n_states)
             if s != env.wall_state and s not in env.terminal_states]
    policy = Q.argmax(axis=1)
    policy_star = Q_star.argmax(axis=1)
    match = np.mean([np.isclose(Q_star[s, policy[s]], Q_star[s].max()) for s in valid])

    print(f"\n=== {name.upper()} ===")
    print(f"max |Q - Q*|            : {np.abs(Q[valid] - Q_star[valid]).max():.4f}")
    print(f"states with optimal act.: {match * 100:.0f}%")
    print(f"V(start) learned / true : {Q[env.start_state].max():.3f} / {Q_star[env.start_state].max():.3f}")
    print(f"greedy avg return       : {evaluate_greedy(env, Q):.3f}")

    tools.plot_policy(policy, env, title=f"Q-learning policy ({name})")
    tools.plot_value_function(Q.max(axis=1), env, title=f"Q-learning V(s) ({name})")
    return returns, lengths, name


if __name__ == "__main__":
    results = [run(False), run(True)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for returns, lengths, name in results:
        axes[0].plot(moving_average(returns), label=name)
        axes[1].plot(moving_average(lengths), label=name)
    axes[0].set(title="Return per episode (moving avg, w=100)", xlabel="episode")
    axes[1].set(title="Episode length (moving avg, w=100)", xlabel="episode")
    axes[0].legend()
    plt.tight_layout()
    plt.show()