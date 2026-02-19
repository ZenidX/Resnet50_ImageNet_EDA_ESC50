"""
Highway-Env - Conducción Autónoma con RL
========================================
Simulación 2D de conducción: autopista, parking, intersecciones.

VARIANTES:
  A — Por entorno  (--env X):        Entrenar en un entorno específico con DQN/PPO
  B — Transfer     (--transfer):     Entrenar en highway, fine-tune en intersección
  C — Curriculum   (--curriculum):   highway-fast → highway → merge → roundabout → intersection

Instalación:
    pip install highway-env stable-baselines3

Uso:
    python highway_conduccion.py                      # Entrenar autopista (var. A)
    python highway_conduccion.py --env parking        # Entrenar parking
    python highway_conduccion.py --transfer           # Var. B: transfer highway→intersection
    python highway_conduccion.py --transfer --source merge --target roundabout
    python highway_conduccion.py --curriculum         # Var. C: curriculum progresivo
    python highway_conduccion.py --demo               # Ver agente
    python highway_conduccion.py --manual             # Conducir manualmente
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import highway_env

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# =============================================================================
# ENTORNOS DISPONIBLES
# =============================================================================

ENTORNOS = {
    "highway": {
        "id": "highway-v0",
        "desc": "Autopista: adelantar coches, mantener velocidad",
        "obs_type": "Kinematics",
    },
    "highway-fast": {
        "id": "highway-fast-v0",
        "desc": "Autopista simplificada (más rápida)",
        "obs_type": "Kinematics",
    },
    "merge": {
        "id": "merge-v0",
        "desc": "Incorporarse a autopista",
        "obs_type": "Kinematics",
    },
    "roundabout": {
        "id": "roundabout-v0",
        "desc": "Rotonda",
        "obs_type": "Kinematics",
    },
    "parking": {
        "id": "parking-v0",
        "desc": "Aparcar en plaza",
        "obs_type": "KinematicsGoal",
    },
    "intersection": {
        "id": "intersection-v0",
        "desc": "Cruzar intersección",
        "obs_type": "Kinematics",
    },
    "racetrack": {
        "id": "racetrack-v0",
        "desc": "Circuito de carreras",
        "obs_type": "Kinematics",
    },
}


# =============================================================================
# CALLBACK
# =============================================================================

class HighwayCallback(BaseCallback):
    """Callback para registrar progreso en highway-env."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.crashes = []
        self.goals_reached = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                # Algunas métricas específicas
                self.crashes.append(info.get('crashed', False))
                self.goals_reached.append(info.get('is_success', False))
        return True


# =============================================================================
# CREAR ENTORNO
# =============================================================================

def crear_entorno(env_name="highway", render=False, config_override=None):
    """
    Crea un entorno de highway-env con configuración.

    Args:
        env_name: Nombre del entorno
        render: Si renderizar
        config_override: Dict con configuración extra

    Returns:
        Entorno configurado
    """
    env_info = ENTORNOS.get(env_name, ENTORNOS["highway"])
    env_id = env_info["id"]

    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)

    # Configuración base
    config = {
        "observation": {
            "type": env_info["obs_type"],
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": True,
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "duration": 40,  # Pasos por episodio
        "collision_reward": -1,
        "high_speed_reward": 0.4,
        "right_lane_reward": 0.1,
    }

    # Configuración específica por entorno
    if env_name == "parking":
        config = {
            "observation": {
                "type": "KinematicsGoal",
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False,
            },
            "action": {
                "type": "ContinuousAction",
            },
        }

    if config_override:
        config.update(config_override)

    env.unwrapped.configure(config)

    return env


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def entrenar_highway(env_name="highway", timesteps=100000, algorithm="DQN"):
    """
    Entrena un agente en highway-env.

    Args:
        env_name: Nombre del entorno
        timesteps: Pasos de entrenamiento
        algorithm: "DQN" o "PPO"

    Returns:
        Modelo y callback
    """
    print(f"\n{'='*60}")
    print(f"  Entrenando {algorithm} en {env_name}")
    env_info = ENTORNOS.get(env_name, ENTORNOS["highway"])
    print(f"  Descripción: {env_info['desc']}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"{'='*60}\n")

    env = crear_entorno(env_name, render=False)

    # Determinar tipo de política
    if env_name == "parking":
        # Parking usa acciones continuas, solo PPO/SAC
        algorithm = "PPO"
        print("Nota: Parking requiere acciones continuas, usando PPO")

    if algorithm == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=5e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.8,  # Horizonte corto para conducción
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            target_update_interval=50,
            policy_kwargs={"net_arch": [256, 256]},
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=5e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.8,
            gae_lambda=0.95,
            policy_kwargs={"net_arch": [256, 256]},
        )

    callback = HighwayCallback()

    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        progress_bar=True
    )

    # Guardar
    save_path = f"highway_{env_name}_{algorithm.lower()}"
    model.save(save_path)
    print(f"\nModelo guardado: {save_path}.zip")

    env.close()
    return model, callback


def demo_highway(env_name="highway", model_path=None, n_episodios=5):
    """
    Muestra el agente conduciendo.
    """
    if model_path is None:
        model_path = f"highway_{env_name}_dqn"

    print(f"\nCargando modelo: {model_path}")

    # Intentar cargar como DQN, si falla como PPO
    try:
        model = DQN.load(model_path)
    except:
        model = PPO.load(model_path)

    env = crear_entorno(env_name, render=True)

    print(f"\nMostrando {n_episodios} episodios...")
    print("\nAcciones disponibles:")
    print("  0: LANE_LEFT")
    print("  1: IDLE")
    print("  2: LANE_RIGHT")
    print("  3: FASTER")
    print("  4: SLOWER\n")

    stats = {"crashes": 0, "success": 0, "rewards": []}

    for ep in range(n_episodios):
        obs, info = env.reset()
        total_reward = 0
        done = False
        crashed = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            crashed = info.get('crashed', False)

        stats["rewards"].append(total_reward)
        if crashed:
            stats["crashes"] += 1
            resultado = "CHOQUE"
        elif info.get('is_success', False):
            stats["success"] += 1
            resultado = "EXITO"
        else:
            resultado = "TIEMPO"

        print(f"Episodio {ep+1}: {total_reward:.2f} puntos - {resultado}")

    print(f"\nResumen:")
    print(f"  Recompensa media: {np.mean(stats['rewards']):.2f}")
    print(f"  Choques: {stats['crashes']}/{n_episodios}")
    print(f"  Éxitos: {stats['success']}/{n_episodios}")

    env.close()


def conducir_manual(env_name="highway"):
    """
    Permite conducir manualmente.
    """
    import pygame

    print(f"\n{'='*60}")
    print(f"  Conducción Manual - {env_name}")
    print(f"{'='*60}")
    print("\nControles:")
    print("  Flechas Arriba/Abajo: Acelerar/Frenar")
    print("  Flechas Izq/Der: Cambiar carril")
    print("  R: Reiniciar")
    print("  ESC: Salir\n")

    env = crear_entorno(env_name, render=True)
    obs, info = env.reset()

    total_reward = 0
    running = True
    clock = pygame.time.Clock()

    # Mapeo de teclas a acciones
    # 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
    action = 1  # IDLE por defecto

    while running:
        keys = pygame.key.get_pressed()

        # Determinar acción basada en teclas presionadas
        if keys[pygame.K_LEFT]:
            action = 0  # LANE_LEFT
        elif keys[pygame.K_RIGHT]:
            action = 2  # LANE_RIGHT
        elif keys[pygame.K_UP]:
            action = 3  # FASTER
        elif keys[pygame.K_DOWN]:
            action = 4  # SLOWER
        else:
            action = 1  # IDLE

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("Reiniciado")
                elif event.key == pygame.K_ESCAPE:
                    running = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            crashed = info.get('crashed', False)
            resultado = "CHOQUE!" if crashed else "Fin del episodio"
            print(f"{resultado} - Recompensa: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(15)  # 15 FPS

    env.close()


def plot_training(callback, save_path="highway_training.png"):
    """Grafica el progreso."""
    if not callback.episode_rewards:
        print("No hay datos")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    rewards = callback.episode_rewards
    window = min(50, len(rewards) // 4) if len(rewards) > 4 else 1

    # Recompensas
    axes[0].plot(rewards, alpha=0.3, color='blue')
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), smoothed, 'r-', linewidth=2)
    axes[0].set_xlabel('Episodio')
    axes[0].set_ylabel('Recompensa')
    axes[0].set_title('Recompensa por Episodio')
    axes[0].grid(True, alpha=0.3)

    # Longitud
    lengths = callback.episode_lengths
    axes[1].plot(lengths, alpha=0.3, color='green')
    if window > 1:
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(lengths)), smoothed, 'r-', linewidth=2)
    axes[1].set_xlabel('Episodio')
    axes[1].set_ylabel('Pasos')
    axes[1].set_title('Duración del Episodio')
    axes[1].grid(True, alpha=0.3)

    # Tasa de choques
    if callback.crashes:
        crashes = [1 if c else 0 for c in callback.crashes]
        crash_rate = 1 - np.cumsum(crashes) / (np.arange(len(crashes)) + 1)
        axes[2].plot(crash_rate, 'b-', linewidth=2)
        axes[2].set_xlabel('Episodio')
        axes[2].set_ylabel('Tasa de Supervivencia')
        axes[2].set_title('Supervivencia Acumulada')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Gráfica guardada: {save_path}")


def entrenar_todos(timesteps=50000):
    """
    Entrena en todos los entornos disponibles.
    """
    print("\n" + "="*60)
    print("  Entrenando en TODOS los entornos")
    print("="*60)

    resultados = {}

    for env_name in ["highway", "merge", "intersection", "roundabout"]:
        print(f"\n>>> {env_name.upper()} <<<")
        try:
            model, callback = entrenar_highway(
                env_name=env_name,
                timesteps=timesteps,
                algorithm="DQN"
            )
            resultados[env_name] = {
                "mean_reward": np.mean(callback.episode_rewards[-50:]),
                "crash_rate": sum(callback.crashes) / len(callback.crashes) if callback.crashes else 0
            }
        except Exception as e:
            print(f"Error en {env_name}: {e}")
            resultados[env_name] = {"error": str(e)}

    # Resumen
    print("\n" + "="*60)
    print("  RESUMEN")
    print("="*60)
    for env_name, data in resultados.items():
        if "error" in data:
            print(f"{env_name}: ERROR - {data['error']}")
        else:
            print(f"{env_name}: reward={data['mean_reward']:.2f}, crashes={data['crash_rate']*100:.1f}%")

    return resultados


def transfer_learning(source_env="highway", target_env="intersection",
                      source_timesteps=80000, target_timesteps=40000,
                      algorithm="DQN"):
    """
    Variante B: Transfer Learning entre entornos de conducción.

    Entrena primero en un entorno fuente (más sencillo) y luego
    transfiere los pesos al entorno objetivo (más complejo).

    La idea: las habilidades básicas aprendidas en autopista
    (mantener velocidad, no chocar) son útiles al cruzar
    una intersección, aunque el escenario sea diferente.

    Sin transfer: el agente empieza desde cero en el entorno difícil.
    Con transfer: el agente ya sabe conducir, solo aprende lo nuevo.

    Ejecutar:
        python highway_conduccion.py --transfer
        python highway_conduccion.py --transfer --source highway --target intersection
    """
    print(f"\n{'='*60}")
    print(f"  Variante B: Transfer Learning")
    print(f"  Fuente: {source_env} ({source_timesteps:,} steps)")
    print(f"  Objetivo: {target_env} ({target_timesteps:,} steps)")
    print(f"  Algoritmo: {algorithm}")
    print(f"{'='*60}\n")

    # --- FASE 1: Entrenar en entorno fuente ---
    print(f"--- FASE 1: Entrenando en {source_env} ---")
    model_source, cb_source = entrenar_highway(
        env_name=source_env,
        timesteps=source_timesteps,
        algorithm=algorithm
    )
    source_mean = np.mean(cb_source.episode_rewards[-50:]) if len(cb_source.episode_rewards) >= 50 else np.mean(cb_source.episode_rewards) if cb_source.episode_rewards else 0
    print(f"\nFase 1 completada. Recompensa media final: {source_mean:.2f}")

    # --- FASE 2: Fine-tuning en entorno objetivo ---
    print(f"\n--- FASE 2: Fine-tuning en {target_env} ---")
    print("Los pesos del modelo fuente se transfieren al entorno objetivo.")
    print("El agente empieza con conocimiento de conducción básica.\n")

    env_target = crear_entorno(target_env, render=False)

    # Reutilizar el modelo entrenado, solo cambiar el entorno
    model_source.set_env(env_target)

    # Reducir learning rate para fine-tuning (ajuste fino)
    if algorithm == "DQN":
        model_source.learning_rate = 1e-4  # Más pequeño que en entrenamiento base
    else:
        model_source.learning_rate = 1e-4

    callback_target = HighwayCallback()
    model_source.learn(
        total_timesteps=target_timesteps,
        callback=callback_target,
        progress_bar=True,
        reset_num_timesteps=False  # No reiniciar contador
    )

    save_path = f"highway_transfer_{source_env}_to_{target_env}"
    model_source.save(save_path)
    print(f"\nModelo con transfer guardado: {save_path}.zip")

    # Comparar sin transfer vs con transfer
    print("\n--- Comparando resultados ---")

    # Entrenar desde cero en el target para comparar
    print(f"Entrenando desde cero en {target_env} para comparar...")
    model_scratch, cb_scratch = entrenar_highway(
        env_name=target_env,
        timesteps=target_timesteps,
        algorithm=algorithm
    )

    # Graficar comparación
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Curva de aprendizaje en entorno objetivo
    ax = axes[0]
    if cb_scratch.episode_rewards:
        window = min(20, len(cb_scratch.episode_rewards) // 4) if len(cb_scratch.episode_rewards) > 4 else 1
        if window > 1:
            smoothed = np.convolve(cb_scratch.episode_rewards, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label="Desde cero", color="red")
    if callback_target.episode_rewards:
        window = min(20, len(callback_target.episode_rewards) // 4) if len(callback_target.episode_rewards) > 4 else 1
        if window > 1:
            smoothed = np.convolve(callback_target.episode_rewards, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label="Con transfer", color="blue")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa")
    ax.set_title(f"Transfer Learning: {source_env} → {target_env}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Recompensa fuente durante entrenamiento inicial
    ax2 = axes[1]
    if cb_source.episode_rewards:
        window = min(20, len(cb_source.episode_rewards) // 4) if len(cb_source.episode_rewards) > 4 else 1
        if window > 1:
            smoothed = np.convolve(cb_source.episode_rewards, np.ones(window)/window, mode='valid')
            ax2.plot(smoothed, label=f"Fase 1: {source_env}", color="green")
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("Recompensa")
    ax2.set_title("Fase 1: Entrenamiento en entorno fuente")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("highway_transfer_learning.png", dpi=150)
    plt.show()
    print("Gráfica guardada: highway_transfer_learning.png")

    env_target.close()
    return model_source, cb_source, callback_target


def curriculum_learning(timesteps_per_env=50000, algorithm="DQN"):
    """
    Variante C: Curriculum Learning progresivo.

    El agente entrena en entornos progresivamente más difíciles.
    Cuando domina uno, se introduce el siguiente.

    Orden de dificultad:
      1. highway-fast  — Autopista simplificada (más sencilla)
      2. highway       — Autopista completa
      3. merge         — Incorporarse a autopista (fusión de carriles)
      4. roundabout    — Rotonda (tomar decisiones de giro)
      5. intersection  — Cruzar intersecciones (el más complejo)

    Idea pedagógica: como aprender a conducir. Primero carretera vacía,
    luego tráfico, luego maniobras complejas.

    Ejecutar: python highway_conduccion.py --curriculum
    """
    curriculum = [
        ("highway-fast",  "Autopista simplificada - flujo básico"),
        ("highway",       "Autopista completa - adelantamientos"),
        ("merge",         "Incorporación - fusión de carriles"),
        ("roundabout",    "Rotonda - decisiones de giro"),
        ("intersection",  "Intersección - el más complejo"),
    ]

    print(f"\n{'='*60}")
    print(f"  Variante C: Curriculum Learning")
    print(f"  {len(curriculum)} entornos progresivos")
    print(f"  {timesteps_per_env:,} steps por entorno")
    print(f"  Algoritmo: {algorithm}")
    print(f"{'='*60}")

    for i, (env_name, desc) in enumerate(curriculum):
        print(f"\n  Nivel {i+1}: {env_name} — {desc}")
    print()

    model = None
    historial = {}

    for nivel, (env_name, desc) in enumerate(curriculum):
        print(f"\n--- NIVEL {nivel+1}: {env_name} ---")
        print(f"Descripción: {desc}")

        env = crear_entorno(env_name, render=False)

        if model is None:
            # Primer nivel: crear modelo desde cero
            if algorithm == "DQN":
                from stable_baselines3 import DQN as _DQN
                model = _DQN(
                    "MlpPolicy", env, verbose=1,
                    learning_rate=5e-4, buffer_size=50000,
                    learning_starts=1000, batch_size=32,
                    gamma=0.8, exploration_fraction=0.2,
                    exploration_final_eps=0.05,
                    target_update_interval=50,
                    policy_kwargs={"net_arch": [256, 256]},
                )
            else:
                from stable_baselines3 import PPO as _PPO
                model = _PPO(
                    "MlpPolicy", env, verbose=1,
                    learning_rate=5e-4, n_steps=256,
                    batch_size=64, n_epochs=10, gamma=0.8,
                    policy_kwargs={"net_arch": [256, 256]},
                )
        else:
            # Niveles siguientes: transferir al nuevo entorno
            print(f"Transfiriendo pesos del nivel anterior a {env_name}...")
            model.set_env(env)
            model.learning_rate = model.learning_rate * 0.8  # Reducir LR progresivamente

        callback = HighwayCallback()
        model.learn(
            total_timesteps=timesteps_per_env,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=(nivel == 0)  # Solo resetear en el primero
        )

        historial[env_name] = {
            "rewards": callback.episode_rewards,
            "crashes": callback.crashes,
            "mean": np.mean(callback.episode_rewards[-30:]) if len(callback.episode_rewards) >= 30 else np.mean(callback.episode_rewards) if callback.episode_rewards else 0
        }

        save_path = f"highway_curriculum_nivel{nivel+1}_{env_name}"
        model.save(save_path)
        print(f"Nivel {nivel+1} completado. Recompensa media: {historial[env_name]['mean']:.2f}")
        print(f"Modelo guardado: {save_path}.zip")

        env.close()

    # Graficar progreso en todos los niveles
    fig, axes = plt.subplots(1, len(curriculum), figsize=(4*len(curriculum), 4))
    if len(curriculum) == 1:
        axes = [axes]

    for idx, (env_name, desc) in enumerate(curriculum):
        ax = axes[idx]
        data = historial.get(env_name, {})
        rewards = data.get("rewards", [])
        if rewards:
            ax.plot(rewards, alpha=0.3, color='blue')
            window = min(20, len(rewards) // 4) if len(rewards) > 4 else 1
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(rewards)), smoothed, 'r-', linewidth=2)
        ax.set_title(f"Nivel {idx+1}: {env_name}")
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Recompensa")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Curriculum Learning — Highway ({algorithm})")
    plt.tight_layout()
    plt.savefig("highway_curriculum.png", dpi=150)
    plt.show()
    print("\nGráfica guardada: highway_curriculum.png")

    # Resumen
    print("\n" + "="*60)
    print("  RESUMEN CURRICULUM")
    print("="*60)
    for nivel, (env_name, _) in enumerate(curriculum):
        data = historial.get(env_name, {})
        mean = data.get("mean", 0)
        print(f"  Nivel {nivel+1} ({env_name}): recompensa media = {mean:.2f}")

    return model, historial


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Highway-Env - Conducción con RL")
    parser.add_argument("--demo", action="store_true", help="Ver agente")
    parser.add_argument("--manual", action="store_true", help="Conducir")
    parser.add_argument("--all", action="store_true", help="Entrenar todos")
    parser.add_argument("--transfer", action="store_true",
                        help="Variante B: Transfer learning entre entornos")
    parser.add_argument("--curriculum", action="store_true",
                        help="Variante C: Curriculum learning progresivo")
    parser.add_argument("--source", type=str, default="highway",
                        choices=list(ENTORNOS.keys()),
                        help="Entorno fuente para transfer learning")
    parser.add_argument("--target", type=str, default="intersection",
                        choices=list(ENTORNOS.keys()),
                        help="Entorno objetivo para transfer learning")
    parser.add_argument("--env", type=str, default="highway",
                        choices=list(ENTORNOS.keys()), help="Entorno")
    parser.add_argument("--algorithm", type=str, default="DQN",
                        choices=["DQN", "PPO"], help="Algoritmo")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Timesteps")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodios demo")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("\nEntornos disponibles:")
    for name, info in ENTORNOS.items():
        print(f"  {name}: {info['desc']}")

    if args.manual:
        conducir_manual(args.env)
    elif args.demo:
        demo_highway(args.env, n_episodios=args.episodes)
    elif args.all:
        entrenar_todos(args.timesteps)
    elif args.transfer:
        transfer_learning(
            source_env=args.source,
            target_env=args.target,
            source_timesteps=args.timesteps,
            target_timesteps=args.timesteps // 2,
            algorithm=args.algorithm
        )
    elif args.curriculum:
        curriculum_learning(
            timesteps_per_env=args.timesteps // 5,
            algorithm=args.algorithm
        )
    else:
        model, callback = entrenar_highway(
            env_name=args.env,
            timesteps=args.timesteps,
            algorithm=args.algorithm
        )
        plot_training(callback)
        demo_highway(args.env, n_episodios=3)
