# Reinforcement Learning

Materiales completos para aprender Reinforcement Learning con proyectos visuales.

## Estructura

```
Reinforcement Learning/
├── CLAUDE.md                              # Esta documentación
├── requirements.txt                       # Dependencias
├── reinforcement_learning_clase.ipynb     # Notebook principal con teoría
│
├── core/                                  # Implementaciones base
│   ├── __init__.py
│   ├── agentes.py                         # Q-Learning, SARSA, DQN
│   └── utils.py                           # Gráficos, métricas
│
├── ejemplos/                              # Ejemplos básicos
│   ├── ejemplo_cartpole_dqn.py            # DQN en CartPole
│   └── ejemplo_qlearning_taxi.py          # Q-Learning en Taxi
│
├── modelos/                               # Modelos guardados
│   ├── cartpole_dqn.pth
│   ├── nibbler_best.pth
│   ├── car_agent_*.pth
│   └── *_training.png                     # Gráficas de entrenamiento
│
└── proyectos/                             # Proyectos visuales
    ├── lunarlander/                       # LunarLander con SB3
    ├── minigrid/                          # Laberintos 2D
    ├── flappybird/                        # Flappy Bird
    ├── highway/                           # Conducción autónoma
    ├── pybullet/                          # Robótica 3D
    ├── nibbler/                           # Snake con DQN
    └── racing/                            # Carreras multi-agente
```

## Instalación

```bash
cd "Reinforcement Learning"

# Dependencias base
pip install gymnasium stable-baselines3 torch numpy matplotlib pygame

# Por proyecto (opcional)
pip install gymnasium[box2d]        # LunarLander
pip install minigrid                # Laberintos
pip install flappy-bird-gymnasium   # Flappy Bird
pip install highway-env             # Conducción
pip install pybullet                # Robótica 3D
```

## Uso Rápido

### Ejemplos básicos
```bash
cd ejemplos
python ejemplo_cartpole_dqn.py      # DQN en CartPole
python ejemplo_qlearning_taxi.py    # Q-Learning en Taxi
```

### Proyectos
```bash
# LunarLander (el más visual)
cd proyectos/lunarlander
python lunarlander_sb3.py           # Entrenar
python lunarlander_sb3.py --demo    # Ver agente

# Nibbler (Snake)
cd proyectos/nibbler
python nibbler_game.py              # Jugar
python nibbler_game.py --train      # Entrenar

# Racing (multi-agente)
cd proyectos/racing
python racing_game.py               # Ver coches
python racing_game.py --train       # Entrenar
```

## Proyectos Visuales

| Proyecto | Descripción | Comando rápido |
|----------|-------------|----------------|
| **LunarLander** | Aterrizar nave espacial | `python lunarlander_sb3.py` |
| **MiniGrid** | Navegar laberintos | `python minigrid_navegacion.py` |
| **Flappy Bird** | Pasar tubos | `python flappybird_dqn.py` |
| **Highway** | Conducción autónoma | `python highway_conduccion.py` |
| **PyBullet** | Robots 3D caminando | `python pybullet_robotica.py` |
| **Nibbler** | Snake con DQN | `python nibbler_game.py` |
| **Racing** | Carreras multi-agente | `python racing_game.py` |

## Código Mínimo (5 líneas)

```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)
```

## Algoritmos Disponibles

### En `core/agentes.py`:
- **Q-Learning**: Estados discretos, tabular
- **SARSA**: On-policy, más conservador
- **DQN**: Deep Q-Network para estados continuos

### En Stable-Baselines3:
- **PPO**: General, robusto (recomendado)
- **DQN**: Acciones discretas
- **SAC/TD3**: Acciones continuas, robótica

## Hiperparámetros Recomendados

| Entorno | Algoritmo | LR | Gamma | Timesteps |
|---------|-----------|-----|-------|-----------|
| CartPole | DQN | 0.001 | 0.99 | 30K |
| LunarLander | PPO | 0.0003 | 0.99 | 100K |
| Highway | DQN | 0.0005 | 0.8 | 100K |
| Robótica | SAC | 0.0003 | 0.99 | 500K+ |

## Tips Windows

- **DataLoader**: Usar `num_workers=0` siempre
- **Box2D**: Instalar `swig` primero si falla
- **PyBullet**: Actualizar drivers GPU si hay errores OpenGL

## Recursos

- **Libro**: Sutton & Barto - "Reinforcement Learning: An Introduction"
- **Curso**: Hugging Face Deep RL Course
- **Docs**: stable-baselines3.readthedocs.io
