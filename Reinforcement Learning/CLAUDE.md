# Reinforcement Learning

Materiales completos para aprender Reinforcement Learning, organizados por nivel de complejidad.

## Estructura del Directorio

```
Reinforcement Learning/
│
├── 01_teoria/                                    # NIVEL 1: Fundamentos teóricos
│   ├── reinforcement_learning_clase.ipynb        # Notebook principal con toda la teoría
│   └── recorrido_proyectos_rl.ipynb              # Guía que relaciona teoría con proyectos
│
├── 02_fundamentos/                               # NIVEL 2: Implementaciones base
│   ├── core/                                     # Módulo con algoritmos
│   │   ├── __init__.py
│   │   ├── agentes.py                            # Q-Learning, SARSA, DQN
│   │   └── utils.py                              # Gráficos, métricas
│   └── ejemplos/                                 # Tutoriales básicos
│       ├── ejemplo_cartpole_dqn.py               # DQN en CartPole
│       └── ejemplo_qlearning_taxi.py             # Q-Learning en Taxi
│
├── 03_proyectos_dqn/                             # NIVEL 3: Proyectos con DQN
│   ├── nibbler/                                  # Snake con DQN + Pygame
│   │   ├── nibbler_game.py
│   │   └── nibbler_rl.ipynb
│   ├── flappybird/                               # Flappy Bird
│   │   └── flappybird_dqn.py
│   └── racing/                                   # Carreras multi-agente
│       └── racing_game.py
│
├── 04_proyectos_avanzados/                       # NIVEL 4: Algoritmos avanzados (PPO, SAC)
│   ├── lunarlander/                              # LunarLander con Stable-Baselines3
│   │   └── lunarlander_sb3.py
│   ├── highway/                                  # Conducción autónoma
│   │   └── highway_conduccion.py
│   ├── minigrid/                                 # Laberintos 2D
│   │   └── minigrid_navegacion.py
│   └── pybullet/                                 # Robótica 3D
│       └── pybullet_robotica.py
│
├── modelos/                                      # Modelos entrenados (.pth)
│   ├── cartpole_dqn.pth
│   ├── nibbler_best.pth
│   └── car_agent_*.pth
│
├── requirements.txt                              # Dependencias
└── CLAUDE.md                                     # Esta documentación
```

## Mapa de Aprendizaje

```
TEORÍA                           PRÁCTICA
──────                           ────────

Conceptos Básicos ──────────────► 02_fundamentos/core/agentes.py
  • Agente, Entorno, Estado         (QLearningAgent, SARSAAgent)
  • Política, Recompensa
  • ε-greedy

Q-Learning Tabular ─────────────► 02_fundamentos/ejemplos/ejemplo_qlearning_taxi.py
  • Tabla Q                         (Taxi-v3: 500 estados discretos)
  • Ecuación de Bellman

DQN ────────────────────────────► 02_fundamentos/ejemplos/ejemplo_cartpole_dqn.py
  • Red neuronal para Q             03_proyectos_dqn/nibbler/
  • Experience Replay               03_proyectos_dqn/flappybird/
  • Target Network                  03_proyectos_dqn/racing/

Algoritmos Avanzados ───────────► 04_proyectos_avanzados/
  • PPO (Policy Gradient)           lunarlander/ (PPO, A2C, DQN)
  • SAC/TD3 (Continuo)              highway/ (DQN, PPO)
                                    pybullet/ (SAC, TD3)
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

### Nivel 1: Teoría
```bash
# Abrir notebooks de teoría
jupyter notebook 01_teoria/reinforcement_learning_clase.ipynb
jupyter notebook 01_teoria/recorrido_proyectos_rl.ipynb
```

### Nivel 2: Fundamentos
```bash
cd 02_fundamentos/ejemplos
python ejemplo_qlearning_taxi.py      # Q-Learning tabular
python ejemplo_cartpole_dqn.py        # DQN básico
```

### Nivel 3: Proyectos DQN
```bash
# Nibbler (Snake)
cd 03_proyectos_dqn/nibbler
python nibbler_game.py                # Jugar manualmente
python nibbler_game.py --train        # Entrenar agente

# Racing (multi-agente)
cd 03_proyectos_dqn/racing
python racing_game.py                 # Ver coches
python racing_game.py --train         # Entrenar

# Flappy Bird
cd 03_proyectos_dqn/flappybird
python flappybird_dqn.py
```

### Nivel 4: Proyectos Avanzados
```bash
# LunarLander (requiere gymnasium[box2d])
cd 04_proyectos_avanzados/lunarlander
python lunarlander_sb3.py             # Entrenar
python lunarlander_sb3.py --demo      # Ver agente

# Highway (requiere highway-env)
cd 04_proyectos_avanzados/highway
python highway_conduccion.py

# MiniGrid (requiere minigrid)
cd 04_proyectos_avanzados/minigrid
python minigrid_navegacion.py

# PyBullet (requiere pybullet)
cd 04_proyectos_avanzados/pybullet
python pybullet_robotica.py
```

## Resumen de Proyectos

| Nivel | Proyecto | Algoritmo | Descripción |
|-------|----------|-----------|-------------|
| 2 | Taxi | Q-Learning | Estados discretos, tabla Q |
| 2 | CartPole | DQN | Estados continuos, red neuronal |
| 3 | Nibbler | DQN | Snake con Pygame, DQN desde cero |
| 3 | Flappy Bird | DQN/PPO | Timing crítico |
| 3 | Racing | DQN | Multi-agente, sensores |
| 4 | LunarLander | PPO/DQN/A2C | Stable-Baselines3 |
| 4 | Highway | DQN/PPO | Conducción autónoma |
| 4 | MiniGrid | PPO | Observación parcial, wrappers |
| 4 | PyBullet | SAC/TD3 | Robótica 3D, acciones continuas |

## Algoritmos por Ubicación

### En `02_fundamentos/core/agentes.py`:
- **Q-Learning**: Estados discretos, tabular
- **SARSA**: On-policy, más conservador
- **DQN**: Deep Q-Network para estados continuos

### En proyectos con Stable-Baselines3:
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
