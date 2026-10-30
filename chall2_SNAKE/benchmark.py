import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

import snake_env  # noqa: F401 - Registers Snake environments via gymnasium.register

METRICS = {
    "reward": {"label": "retorno", "unit": "puntos"},
    "apples": {"label": "manzanas recogidas", "unit": "manzanas"},
}


def resolve_model_path(model_path: str) -> Path:
    """Resolve model path allowing both with and without .zip extension."""
    path = Path(model_path)
    if path.exists():
        return path
    zip_path = path.with_suffix(".zip")
    if zip_path.exists():
        return zip_path
    raise FileNotFoundError(f"No se encontro el modelo en '{model_path}' ni en '{zip_path}'.")


def run_episode(model: PPO, env: gym.Env, metric: str, deterministic: bool = True) -> float:
    """Run one episode and return the requested metric (reward return or apples eaten)."""
    obs, info = env.reset()
    done = False
    episode_return = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        done = terminated or truncated
    return episode_return if metric == "reward" else float(info["apples"])


def evaluate_policy_metric(model: PPO, env: gym.Env, metric: str, num_episodes: int = 100, deterministic: bool = True) -> np.ndarray:
    """Evaluate policy over num_episodes and return the chosen metric per episode."""
    values = []
    for episode in range(1, num_episodes + 1):
        value = run_episode(model, env, metric=metric, deterministic=deterministic)
        values.append(value)
        print(f"Episodio {episode:03d} | {METRICS[metric]['label']} = {value:.2f}")
    return np.array(values, dtype=np.float64)


def plot_metric(values: np.ndarray, metric: str, output_path: Path, show_plot: bool = False) -> None:
    """Plot per-episode metric with mean line and highlight max/min in red."""
    episodes = np.arange(1, len(values) + 1)
    max_idx, min_idx = int(np.argmax(values)), int(np.argmin(values))
    colors = np.array(["tab:blue"] * len(values), dtype=object)
    colors[max_idx] = colors[min_idx] = "red"
    mean_value = float(np.mean(values))
    label = METRICS[metric]["label"]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(episodes, values, color=colors, edgecolor="black", linewidth=0.2)
    ax.axhline(mean_value, color="black", linestyle="--", linewidth=2, label=f"{label.capitalize()} medio = {mean_value:.2f}")
    ax.set_title(f"Benchmark de {label} por episodio ({len(values)} episodios)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel(label.capitalize())
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right")
    ax.text(episodes[max_idx], values[max_idx], f"MAX {values[max_idx]:.2f}", color="red", fontsize=9, ha="center", va="bottom")
    ax.text(episodes[min_idx], values[min_idx], f"MIN {values[min_idx]:.2f}", color="red", fontsize=9, ha="center", va="top")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Grafica guardada en: {output_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark de politicas entrenadas en Snake.")
    parser.add_argument("--model-path", type=str, default="models/PPO/100000", help="Ruta al modelo entrenado de SB3 (con o sin .zip).")
    parser.add_argument("--env-id", type=str, default="Snake-v0", help="ID del entorno registrado en gymnasium.")
    parser.add_argument("--episodes", type=int, default=100, help="Numero de episodios de evaluacion.")
    parser.add_argument("--metric", choices=list(METRICS), default="reward", help="Metrica a evaluar: 'reward' (retorno acumulado) o 'apples' (manzanas recogidas).")
    parser.add_argument("--stochastic", action="store_true", help="Si se activa, usa politica no determinista (por defecto es determinista).")
    parser.add_argument("--plot-path", type=str, default="logs/benchmark_returns.png", help="Ruta donde guardar la grafica de barras.")
    parser.add_argument("--show-plot", action="store_true", help="Mostrar la grafica en pantalla al finalizar.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    deterministic = not args.stochastic

    model = PPO.load(resolve_model_path(args.model_path))
    env = gym.make(args.env_id, render_mode="rgb_array")

    try:
        values = evaluate_policy_metric(model, env, metric=args.metric, num_episodes=args.episodes, deterministic=deterministic)
    finally:
        env.close()

    label = METRICS[args.metric]["label"]
    print("\nResumen benchmark")
    print(f"{label.capitalize()} medio: {np.mean(values):.2f}")
    print(f"{label.capitalize()} maximo: {np.max(values):.2f}")
    print(f"{label.capitalize()} minimo: {np.min(values):.2f}")

    plot_metric(values, metric=args.metric, output_path=Path(args.plot_path), show_plot=args.show_plot)


if __name__ == "__main__":
    main()

