import matplotlib.pyplot as plt
import numpy as np

ACTION_ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}


def extract_env_info(env):
    """Automatically extracts shape, walls, holes, terminals, and rewards from any gridworld environment (Custom or Gymnasium FrozenLake)."""
    unwrapped = getattr(env, "unwrapped", env)

    # 1. Extract Shape
    if hasattr(unwrapped, "desc"):  # Gymnasium FrozenLake
        n_rows, n_cols = unwrapped.desc.shape
    elif hasattr(unwrapped, "n_rows") and hasattr(unwrapped, "n_cols"):
        n_rows, n_cols = unwrapped.n_rows, unwrapped.n_cols
    else:
        n_states = unwrapped.observation_space.n
        if n_states == 12:
            n_rows, n_cols = 3, 4
        else:
            side = int(np.sqrt(n_states))
            n_rows, n_cols = side, side

    shape = (n_rows, n_cols)
    n_states = n_rows * n_cols

    walls = set()
    holes = set()
    terminals = set()
    rewards = np.zeros(n_states)

    # 2. Extract from explicit attributes (Custom GridworldEnv)
    if hasattr(unwrapped, "wall_state"):
        w = unwrapped.wall_state
        walls.update([w] if isinstance(w, (int, np.integer)) else w)

    if hasattr(unwrapped, "terminal_states"):
        term_dict = unwrapped.terminal_states
        if isinstance(term_dict, dict):
            for s, r in term_dict.items():
                rewards[s] = r
                if r < 0:
                    holes.add(s)
                else:
                    terminals.add(s)
        elif isinstance(term_dict, (list, tuple, set)):
            terminals.update(term_dict)

    # 3. Extract from character map (FrozenLake)
    if hasattr(unwrapped, "desc"):
        desc = unwrapped.desc
        for r in range(n_rows):
            for c in range(n_cols):
                s = r * n_cols + c
                char = desc[r, c]
                if isinstance(char, bytes):
                    char = char.decode("utf-8")

                if char == "H":
                    holes.add(s)
                    rewards[s] = -1.0
                elif char == "G":
                    terminals.add(s)
                    rewards[s] = +1.0
                elif char == "W":
                    walls.add(s)

    # 4. Fallback inspection via transition dynamics P
    if hasattr(unwrapped, "P"):
        for s in range(n_states):
            if s in walls or s in holes or s in terminals or s not in unwrapped.P:
                continue

            # Check if s is terminal in P dynamics
            dones = [t[3] for a in unwrapped.P[s] for t in unwrapped.P[s][a]]
            if dones and all(dones):
                max_r = max([t[2] for a in unwrapped.P[s] for t in unwrapped.P[s][a]])
                rewards[s] = max_r
                if max_r < 0:
                    holes.add(s)
                else:
                    terminals.add(s)

    return {
        "shape": shape,
        "walls": walls,
        "holes": holes,
        "terminals": terminals,
        "rewards": rewards,
    }


# =====================================================================
# TOOL 1: ENVIRONMENT VISUALIZER
# =====================================================================
def plot_environment(env, title="Environment Grid"):
    """Draws the environment grid auto-extracting parameters from env."""
    info = extract_env_info(env)
    shape, rewards, walls, holes = (
        info["shape"],
        info["rewards"],
        info["walls"],
        info["holes"],
    )
    n_rows, n_cols = shape

    fig, ax = plt.subplots(figsize=(n_cols * 1.5, n_rows * 1.5))
    ax.imshow(np.zeros(shape), cmap="Pastel1", vmin=0, vmax=1)

    ax.set_xticks(np.arange(-0.5, n_cols, 1))
    ax.set_yticks(np.arange(-0.5, n_rows, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color="black", linestyle="-", linewidth=2)

    for s in range(n_rows * n_cols):
        r, c = divmod(s, n_cols)

        if s in walls:
            ax.add_patch(
                plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, color="gray", hatch="//"
                )
            )
            ax.text(
                c, r, "WALL", ha="center", va="center", color="white", weight="bold"
            )
        elif s in holes:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="salmon"))
            ax.text(
                c,
                r,
                f"HOLE\n({rewards[s]:.0f})",
                ha="center",
                va="center",
                color="white",
                weight="bold",
            )
        else:
            r_val = rewards[s]
            if r_val > 0:
                ax.add_patch(
                    plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="lightgreen")
                )
                ax.text(
                    c,
                    r,
                    f"+{r_val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    weight="bold",
                )
            else:
                ax.text(
                    c,
                    r,
                    f"{r_val:.2f}",
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=10,
                )

    plt.title(title, fontsize=14, pad=10)
    plt.tight_layout()
    plt.show()


# =====================================================================
# TOOL 2: POLICY VISUALIZER
# =====================================================================
def plot_policy(
    policy,
    env,
    action_arrows=ACTION_ARROWS,
    title="Policy Visualization π(s)",
):
    """Draws policy arrows in each cell using auto-extracted env metadata."""
    info = extract_env_info(env)
    shape, walls, holes, terminals = (
        info["shape"],
        info["walls"],
        info["holes"],
        info["terminals"],
    )
    n_rows, n_cols = shape

    fig, ax = plt.subplots(figsize=(n_cols * 1.5, n_rows * 1.5))
    ax.imshow(np.zeros(shape), cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(np.arange(-0.5, n_cols, 1))
    ax.set_yticks(np.arange(-0.5, n_rows, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color="black", linestyle="-", linewidth=2)

    # Convert 1D deterministic policy array to a 2D probability matrix
    if policy.ndim == 1:
        n_actions = len(action_arrows)
        policy_prob = np.zeros((len(policy), n_actions))
        for s, a in enumerate(policy):
            policy_prob[s, int(a)] = 1.0
        policy = policy_prob

    for s in range(n_rows * n_cols):
        r, c = divmod(s, n_cols)

        if s in walls:
            ax.add_patch(
                plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, color="gray", hatch="//"
                )
            )
            ax.text(
                c, r, "WALL", ha="center", va="center", color="white", weight="bold"
            )
        elif s in holes:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="salmon"))
            ax.text(
                c, r, "HOLE", ha="center", va="center", color="white", weight="bold"
            )
        elif s in terminals:
            ax.add_patch(
                plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="lightgray")
            )
            ax.text(
                c, r, "GOAL", ha="center", va="center", fontsize=10, weight="bold"
            )
        else:
            max_prob = np.max(policy[s])
            if max_prob > 0:
                best_actions = np.where(np.isclose(policy[s], max_prob))[0]
                arrow_str = " ".join([action_arrows[a] for a in best_actions])
                ax.text(
                    c,
                    r,
                    arrow_str,
                    ha="center",
                    va="center",
                    fontsize=18,
                    weight="bold",
                )

    plt.title(title, fontsize=14, pad=10)
    plt.tight_layout()
    plt.show()


# =====================================================================
# TOOL 3: VALUE STATE VISUALIZER
# =====================================================================
def plot_value_function(V, env, title="State-Value Function V(s)"):
    """Draws a heatmap overlay of state values V(s) auto-extracting metadata."""
    info = extract_env_info(env)
    shape, walls, holes, terminals = (
        info["shape"],
        info["walls"],
        info["holes"],
        info["terminals"],
    )
    n_rows, n_cols = shape

    fig, ax = plt.subplots(figsize=(n_cols * 1.6, n_rows * 1.5))
    V_grid = V.reshape(shape).copy()

    mask_indices = list(walls) + list(holes) + list(terminals)
    unmasked_vals = [V[i] for i in range(len(V)) if i not in mask_indices]
    vmin = min(unmasked_vals) if unmasked_vals else -1.0
    vmax = max(unmasked_vals) if unmasked_vals else 1.0

    im = ax.imshow(V_grid, cmap="YlGnBu", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(-0.5, n_cols, 1))
    ax.set_yticks(np.arange(-0.5, n_rows, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color="black", linestyle="-", linewidth=2)

    for s in range(n_rows * n_cols):
        r, c = divmod(s, n_cols)

        if s in walls:
            ax.add_patch(
                plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, color="gray", hatch="//"
                )
            )
            ax.text(
                c, r, "WALL", ha="center", va="center", color="white", weight="bold"
            )
        elif s in holes:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="salmon"))
            ax.text(
                c,
                r,
                f"V={V[s]:.2f}",
                ha="center",
                va="center",
                color="white",
                weight="bold",
            )
        elif s in terminals:
            ax.add_patch(
                plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="lightgray")
            )
            ax.text(
                c,
                r,
                f"V={V[s]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
            )
        else:
            ax.text(
                c,
                r,
                f"{V[s]:.3f}",
                ha="center",
                va="center",
                fontsize=11,
                weight="bold",
            )

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.title(title, fontsize=14, pad=10)
    plt.tight_layout()
    plt.show()


