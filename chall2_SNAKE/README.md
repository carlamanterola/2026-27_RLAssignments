# Challenge 2: Snake (RL)

Este challenge esta pensado para que el alumnado pueda **modificar el entorno** y, a la vez, mantener un **protocolo de evaluacion comun** con `benchmark.py`.

## Estructura relevante

- `snake_env/env.py`: implementacion del entorno `SnakeEnv`.
- `snake_env/__init__.py`: registro de variantes del entorno con `gymnasium.register`.
- `EnvCreation.md`: guia detallada sobre `register` y como crear variantes del entorno.
- `training.py`: ejemplo base de entrenamiento con PPO.
- `benchmark.py`: evaluacion comun para comparar agentes.

## API del entorno

La clase principal es `SnakeEnv(gym.Env)` en `snake_env/env.py`.

### Metodos mas importantes

- `__init__(render_mode=None, action_size=1, render_fps=10)`
  - Define espacios (`action_space`, `observation_space`) y parametros de dinamica.
- `reset(seed=None, options=None)`
  - Reinicia tablero, serpiente, manzana y devuelve estado inicial.
- `step(action)`
  - Aplica accion, calcula recompensa, avanza el episodio y devuelve:
  - `(observation, reward, terminated, truncated, info)`.
- `render(mode=None)`
  - Renderizado visual (`human`) o imagen (`rgb_array`).
- `_take_action(action)`
  - Logica de movimiento de la cabeza con bloqueo de giros opuestos directos.
- `_update_ui()`
  - Dibujo de serpiente/manzana con OpenCV.
- `close()`
  - Cierra ventanas de OpenCV.

## Estado (observacion)

La observacion es un vector de longitud `5 + STACKED_ACTIONS`.

Componentes en orden:

1. `head_x`: coordenada x de la cabeza.
2. `head_y`: coordenada y de la cabeza.
3. `apple_delta_x`: diferencia x entre manzana y cabeza.
4. `apple_delta_y`: diferencia y entre manzana y cabeza.
5. `snake_length`: longitud de la serpiente.
6. `prev_actions`: ultimas acciones (`STACKED_ACTIONS`, por defecto 5).

El espacio es:

- `observation_space = Box(low=-tableSizeObs, high=tableSizeObs, shape=(5 + STACKED_ACTIONS,), dtype=float64)`

## Acciones posibles

Espacio de acciones discreto: `Discrete(4)`.

Mapeo:

- `0`: izquierda
- `1`: derecha
- `2`: abajo
- `3`: arriba

Nota: en `_take_action`, los giros opuestos inmediatos se bloquean para evitar invertir direccion en un solo paso.

## Recompensa y fin de episodio

Resumen de la logica actual en `step`:

- Comer manzana: `+10`.
- Morir por colision (pared o cuerpo): `-10`.
- Acercarse a la manzana: `+1`.
- Alejarse de la manzana: `-1`.
- Penalizacion por paso: `-0.1`.

Finalizacion:

- `terminated=True` por colision.
- `truncated` depende de `max_episode_steps` del wrapper de Gym (segun la version registrada).

## Variaciones del entorno con register

En `snake_env/__init__.py` se usan varios `register(...)` de Gymnasium para definir IDs diferentes. Ver [EnvCreation.md](EnvCreation.md) para una explicacion detallada de como funciona `register`, que va como parametro propio y que va en `kwargs`, y ejemplos completos de nuevas variantes.

## Benchmark comun (100 episodios, sin render)

`benchmark.py` ejecuta evaluacion estandarizada:

- 100 episodios por defecto,
- sin renderizado visual del juego,
- guarda grafica de barras con retorno por episodio,
- dibuja linea horizontal con retorno medio,
- resalta en rojo episodios de retorno maximo y minimo.

Uso recomendado:

```bash
python benchmark.py --model-path models/PPO/100000 --episodes 100
```

Con ventana de grafica:

```bash
python benchmark.py --model-path models/PPO/100000 --episodes 100 --show-plot
```
