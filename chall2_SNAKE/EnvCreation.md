# Creacion de variaciones del entorno con `register`

Este documento explica como funciona el registro de entornos de Gymnasium (`gymnasium.envs.registration.register`) y como usarlo para crear variantes de `SnakeEnv`.

## Que hace `register()`

`register()` no crea el entorno; solo guarda una "receta" en un registro global. Cuando luego llamas a `gym.make("Snake-v0", ...)`, Gymnasium busca esa receta y construye el entorno.

Sus parametros se dividen en dos grupos.

### 1) Parametros propios de `register()` (metadatos del registro)

Estos configuran wrappers y comportamiento del registro. **No** se pasan al `__init__` del entorno:

- `id`: nombre unico (`"Snake-v0"`).
- `entry_point`: ruta a la clase (`"snake_env.env:SnakeEnv"`).
- `max_episode_steps`: Gymnasium envuelve automaticamente el entorno con un wrapper `TimeLimit`. Ese wrapper cuenta los pasos y, al superar el limite, pone `truncated=True` desde fuera, sin que `SnakeEnv` sepa nada de esto. Por eso no existe `max_episode_steps` como atributo en `env.py` — no hace falta, lo gestiona el wrapper.
- `reward_threshold`: umbral de recompensa para considerar el entorno "resuelto" (opcional).
- `nondeterministic`, `order_enforce`, `autoreset`, `disable_env_checker`, `apply_api_compatibility`: opcionales, configuran otros wrappers/validaciones.

### 2) `kwargs` (dict) — esto si va al `__init__` del entorno

```python
register(
    id="Snake-v0-step5",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=500,
    kwargs={"action_size": 2},
)
```

Aqui `kwargs={"action_size": 2}` se pasa tal cual como argumento con nombre a `SnakeEnv.__init__(self, render_mode=None, action_size=1, render_fps=10)`. Es decir, es equivalente a llamar `SnakeEnv(action_size=2)`.

Cualquier parametro que exista en `__init__` de tu entorno (`render_mode`, `action_size`, `render_fps`) puede ir en ese dict `kwargs` del `register`.

## Sobre `render_mode` en concreto

`render_mode` si es un argumento del constructor de `SnakeEnv`, pero normalmente **no** se fija en `register()` (aunque podrias, via `kwargs={"render_mode": "human"}`). Lo habitual es dejarlo fuera del registro y pasarlo en el momento de crear el entorno:

```python
env = gym.make("Snake-v0", render_mode="human", render_fps=10)
```

`gym.make()` toma los `kwargs` extra que le pasas (`render_mode="human"`, `render_fps=10`) y los combina con los `kwargs` definidos en el `register()`, pasando todo junto al constructor. Si un mismo parametro aparece en ambos sitios, el que pasas en `gym.make()` tiene prioridad sobre el del `register`.

## Resumen practico

| Quieres controlar... | Donde lo pones |
|---|---|
| Limite de pasos por episodio | `max_episode_steps=` en `register()` (gestionado por wrapper `TimeLimit`, no toca tu `__init__`) |
| Parametro propio del constructor (`action_size`, `render_fps` fijo por variante) | `kwargs={...}` dentro de `register()` |
| `render_mode` (normalmente cambia segun si entrenas o visualizas) | Como argumento en `gym.make(id, render_mode=...)`, no en el `register` |
| Umbral de recompensa "resuelto" | `reward_threshold=` en `register()` |

## Ejemplo completo

```python
from gymnasium.envs.registration import register

register(
    id="Snake-v3",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=700,
    kwargs={"action_size": 2},
)
```

Luego se crea con:

```python
import gymnasium as gym
import snake_env  # importante para ejecutar los register

env = gym.make("Snake-v3", render_mode="rgb_array")
```

Otro ejemplo, fijando `render_fps` por defecto en la variante:

```python
register(
    id="Snake-v4",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=300,
    kwargs={"action_size": 1, "render_fps": 30},
)
```

Y `render_mode` se sigue decidiendo en `gym.make(...)` segun el uso (entrenamiento vs. visualizacion).

## Ideas de variaciones pedagogicas

- Cambiar `max_episode_steps`.
- Variar `action_size` para cambiar velocidad efectiva.
- Modificar la recompensa en `step` para incentivar estrategias distintas.
- Cambiar `STACKED_ACTIONS` para dar mas o menos memoria temporal.
