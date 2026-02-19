# Reinforcement Learning

Materiales completos para aprender Reinforcement Learning, organizados por nivel de complejidad.

## Estructura del Directorio

```
Reinforcement Learning/
│
├── 01_teoria/                                    # NIVEL 1: Fundamentos teóricos
│   ├── RL_00_proyectos.ipynb                     # Guía que relaciona teoría con proyectos
│   ├── RL_01_fundamentos.ipynb                   # Parte 1: Fundamentos, SARSA, Q-Learning
│   ├── RL_02_dqn.ipynb                           # Parte 2: DQN, Gymnasium, Práctica
│   └── RL_03_sb3.ipynb                           # Parte 3: Stable-Baselines3, PPO, SAC
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
│   │   ├── nibbler_game.py                       # Variantes: A=estándar, B=reward shaping, C=curiosidad
│   │   └── nibbler_rl.ipynb
│   ├── flappybird/                               # Flappy Bird
│   │   ├── flappybird_dqn.py                     # Variantes: A=DQN+simple, B=PPO+simple, C=DQN+RGB, D=comparativa
│   │   └── flappybird_rl.ipynb
│   └── racing/                                   # Carreras multi-agente
│       ├── racing_game.py                        # Variantes: A=independ., B=red compartida, C=maestro-alumno, D=competitivo
│       └── racing_rl.ipynb
│
├── 04_proyectos_avanzados/                       # NIVEL 4: Algoritmos avanzados (PPO, SAC)
│   ├── lunarlander/                              # LunarLander con Stable-Baselines3
│   │   ├── lunarlander_sb3.py                    # Variantes: A=PPO, B=DQN, C=A2C (discreto), D=SAC/TD3 (continuo)
│   │   └── lunarlander_rl.ipynb
│   ├── highway/                                  # Conducción autónoma
│   │   ├── highway_conduccion.py                 # Variantes: A=un entorno, B=transfer learning, C=curriculum
│   │   └── highway_rl.ipynb
│   ├── minigrid/                                 # Laberintos 2D
│   │   ├── minigrid_navegacion.py                # Variantes: A=MLP+flat, B=CNN, C=curriculum
│   │   └── minigrid_rl.ipynb
│   └── pybullet/                                 # Robótica 3D
│       ├── pybullet_robotica.py                  # Variantes: A=PPO, B=SAC, C=TD3, D=matriz Algo×Robot
│       └── pybullet_rl.ipynb
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
# Abrir notebooks de teoría (en orden)
jupyter notebook 01_teoria/RL_01_fundamentos.ipynb   # Parte 1: Fundamentos, SARSA, Q-Learning
jupyter notebook 01_teoria/RL_02_dqn.ipynb           # Parte 2: DQN, Gymnasium, Práctica
jupyter notebook 01_teoria/RL_03_sb3.ipynb           # Parte 3: Stable-Baselines3, PPO, SAC
jupyter notebook 01_teoria/RL_00_proyectos.ipynb     # Mapa teoría → proyectos
```

### Nivel 2: Fundamentos
```bash
cd 02_fundamentos/ejemplos
python ejemplo_qlearning_taxi.py      # Q-Learning tabular
python ejemplo_cartpole_dqn.py        # DQN básico
```

### Nivel 3: Proyectos DQN
```bash
# Nibbler (Snake) — Variantes de función de reward
cd 03_proyectos_dqn/nibbler
python nibbler_game.py                           # Jugar manualmente
python nibbler_game.py --train                   # A: DQN estándar
python nibbler_game.py --train --variant shaped  # B: Reward shaping (penaliza bucles)
python nibbler_game.py --train --variant curiosity  # C: Curiosidad intrínseca (ICM)

# Racing (multi-agente) — Variantes de arquitectura multi-agente
cd 03_proyectos_dqn/racing
python racing_game.py                                      # Ver coches aleatorios
python racing_game.py --train                              # A: 4 redes independientes
python racing_game.py --train --variant shared             # B: 1 red compartida
python racing_game.py --train --variant master_student     # C: Maestro-alumno
python racing_game.py --train --variant competitive        # D: Recompensa por ranking

# Flappy Bird — Variantes de algoritmo y observación
cd 03_proyectos_dqn/flappybird
python flappybird_dqn.py --algorithm DQN --simple          # A: DQN + obs. simple
python flappybird_dqn.py --algorithm PPO --simple          # B: PPO + obs. simple
python flappybird_dqn.py --algorithm DQN                   # C: DQN + imagen RGB
python flappybird_dqn.py --compare-algorithms              # D: Comparar DQN vs PPO
python flappybird_dqn.py --compare-all                     # Todas las combinaciones
```

### Nivel 4: Proyectos Avanzados
```bash
# LunarLander (requiere gymnasium[box2d]) — Variantes de algoritmo y espacio de acción
cd 04_proyectos_avanzados/lunarlander
python lunarlander_sb3.py                                  # A: PPO (discreto)
python lunarlander_sb3.py --algorithm DQN                  # B: DQN (discreto)
python lunarlander_sb3.py --algorithm A2C                  # C: A2C (discreto)
python lunarlander_sb3.py --continuous                     # D: SAC en continuo
python lunarlander_sb3.py --continuous --algorithm TD3     # D: TD3 en continuo
python lunarlander_sb3.py --compare                        # Comparar A+B+C
python lunarlander_sb3.py --compare-cont                   # Discreto vs continuo

# Highway (requiere highway-env) — Variantes de transferencia
cd 04_proyectos_avanzados/highway
python highway_conduccion.py --env highway                 # A: un entorno
python highway_conduccion.py --transfer                    # B: highway→intersection
python highway_conduccion.py --transfer --source merge --target roundabout
python highway_conduccion.py --curriculum                  # C: 5 entornos progresivos

# MiniGrid (requiere minigrid) — Variantes de arquitectura de red
cd 04_proyectos_avanzados/minigrid
python minigrid_navegacion.py --variant flat               # A: MLP + obs. aplanada
python minigrid_navegacion.py --variant cnn                # B: CNN personalizada
python minigrid_navegacion.py --curriculum                 # C: Empty→DoorKey→Lava

# PyBullet (requiere pybullet) — Variantes de algoritmo y robot
cd 04_proyectos_avanzados/pybullet
python pybullet_robotica.py --algorithm PPO --env ant      # A: PPO
python pybullet_robotica.py --algorithm SAC --env ant      # B: SAC
python pybullet_robotica.py --algorithm TD3 --env ant      # C: TD3
python pybullet_robotica.py --compare                      # Comparar PPO/SAC/TD3
python pybullet_robotica.py --compare-matrix               # D: Matriz Algo×Robot
```

## Resumen de Proyectos y Variantes

| Nivel | Proyecto | Variante | Concepto demostrado |
|-------|----------|----------|---------------------|
| 3 | Nibbler | A: DQN estándar | Baseline, reward básico |
| 3 | Nibbler | B: Reward shaping | Diseño de función de reward |
| 3 | Nibbler | C: Curiosidad intrínseca | Exploración motivada internamente |
| 3 | Flappy Bird | A: DQN + simple | Off-policy, observación vectorial |
| 3 | Flappy Bird | B: PPO + simple | On-policy, misma observación |
| 3 | Flappy Bird | C: DQN + RGB | Off-policy, observación visual |
| 3 | Flappy Bird | D: Comparativa | Impacto de algoritmo vs observación |
| 3 | Racing | A: Independientes | Multi-agente sin compartir |
| 3 | Racing | B: Red compartida | Centralized Training (CTDE) |
| 3 | Racing | C: Maestro-alumno | Transfer learning entre agentes |
| 3 | Racing | D: Competitivo | Recompensa relativa por ranking |
| 4 | LunarLander | A: PPO | On-policy, política estocástica |
| 4 | LunarLander | B: DQN | Off-policy, replay buffer |
| 4 | LunarLander | C: A2C | Actor-Critic síncrono |
| 4 | LunarLander | D: SAC/TD3 continuo | Espacio de acción continuo |
| 4 | Highway | A: Entorno único | Baseline por entorno |
| 4 | Highway | B: Transfer learning | Fine-tuning entre entornos |
| 4 | Highway | C: Curriculum | Progresión de dificultad |
| 4 | MiniGrid | A: MLP + flat | Sin estructura espacial |
| 4 | MiniGrid | B: CNN | Convoluciones, estructura espacial |
| 4 | MiniGrid | C: CNN + curriculum | CNN + progresión de entornos |
| 4 | PyBullet | A/B/C: PPO/SAC/TD3 | Algoritmos de control continuo |
| 4 | PyBullet | D: Matriz Algo×Robot | Comparativa cruzada |

## Variantes de Entrenamiento — Guía Rápida

Cada proyecto implementa múltiples variantes seleccionables por argumento CLI.
Los notebooks `.ipynb` de cada proyecto documentan cada variante con explicación conceptual y código.

### Racing (`--variant`)
| Flag | Concepto clave |
|------|---------------|
| `independent` | Cada agente aprende solo (baseline) |
| `shared` | 1 red compartida, buffer común → 4× datos |
| `master_student` | Maestro entrena, alumnos copian pesos |
| `competitive` | Bonus/penalización por posición en carrera |

### Nibbler (`--variant`)
| Flag | Concepto clave |
|------|---------------|
| `standard` | Reward: +10 comer, -10 morir (baseline) |
| `shaped` | Penaliza bucles, premia exploración de celdas nuevas |
| `curiosity` | ICM: bonus por error de predicción de transición |

### Flappy Bird
| Flag | Concepto clave |
|------|---------------|
| `--algorithm DQN --simple` | Off-policy, vector 12D |
| `--algorithm PPO --simple` | On-policy, vector 12D |
| `--algorithm DQN` | Off-policy, imagen RGB (CNN) |
| `--compare-algorithms` | DQN vs PPO con misma observación |

### LunarLander
| Flag | Concepto clave |
|------|---------------|
| `--algorithm PPO/DQN/A2C` | Discreto: 4 acciones on/off |
| `--continuous` | Continuo: potencia real [−1, 1]², requiere SAC/TD3 |
| `--compare-cont` | Discreto vs continuo |

### Highway (`--transfer` / `--curriculum`)
| Flag | Concepto clave |
|------|---------------|
| `--env X` | Un solo entorno (baseline) |
| `--transfer` | Fase 1: highway → Fase 2: fine-tune en intersection |
| `--curriculum` | highway-fast → highway → merge → roundabout → intersection |

### MiniGrid (`--variant` / `--curriculum`)
| Flag | Concepto clave |
|------|---------------|
| `--variant flat` | MLP con obs. aplanada (sin estructura espacial) |
| `--variant cnn` | CNN que respeta la imagen 7×7×3 |
| `--curriculum` | Empty → FourRooms → DoorKey → LavaCrossing |

### PyBullet (`--compare-matrix`)
| Flag | Concepto clave |
|------|---------------|
| `--algorithm PPO/SAC/TD3` | Un algoritmo en un robot |
| `--compare` | PPO vs SAC vs TD3 en el mismo robot |
| `--compare-matrix` | Matriz 3 algoritmos × 4 robots |

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
