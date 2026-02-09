"""
Ejemplo Completo: Q-Learning en Taxi-v3
=======================================
Este script muestra cómo entrenar un agente Q-Learning tabular
para resolver el entorno Taxi de Gymnasium.

Uso:
    python ejemplo_qlearning_taxi.py

El entorno Taxi tiene estados discretos, ideal para Q-Learning tabular.
El taxi debe recoger pasajeros y dejarlos en su destino.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from core.agentes import QLearningAgent
from core.utils import plot_learning_curve, MetricsTracker

# Configuración
ENV_NAME = "Taxi-v3"
N_EPISODIOS = 2000
MAX_PASOS = 200

# Hiperparámetros
ALPHA = 0.1        # Tasa de aprendizaje
GAMMA = 0.99       # Factor de descuento
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999


def main():
    print("=" * 60)
    print("ENTRENAMIENTO Q-LEARNING - Taxi-v3")
    print("=" * 60)

    # Crear entorno
    env = gym.make(ENV_NAME)

    print(f"\nEntorno: {ENV_NAME}")
    print(f"Espacio de observación: {env.observation_space}")
    print(f"  - {env.observation_space.n} estados discretos")
    print(f"Espacio de acciones: {env.action_space}")
    print(f"  - {env.action_space.n} acciones: sur, norte, este, oeste, recoger, dejar")

    # Crear agente
    agente = QLearningAgent(
        n_acciones=env.action_space.n,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY
    )

    print(f"\nAgente Q-Learning creado:")
    print(f"  - Alpha: {ALPHA}")
    print(f"  - Gamma: {GAMMA}")
    print(f"  - Epsilon decay: {EPSILON_DECAY}")

    # Entrenar
    print(f"\n{'=' * 60}")
    print("ENTRENANDO...")
    print("=" * 60)

    recompensas = []
    pasos = []

    for ep in range(N_EPISODIOS):
        estado, _ = env.reset()
        recompensa_total = 0

        for paso in range(MAX_PASOS):
            accion = agente.seleccionar_accion(estado)
            siguiente, recompensa, terminado, truncado, _ = env.step(accion)

            agente.aprender(estado, accion, recompensa, siguiente, terminado or truncado)

            recompensa_total += recompensa
            estado = siguiente

            if terminado or truncado:
                break

        agente.decay_epsilon()
        recompensas.append(recompensa_total)
        pasos.append(paso + 1)

        if (ep + 1) % 200 == 0:
            promedio = np.mean(recompensas[-200:])
            print(f"Episodio {ep + 1:4d} | Recompensa: {recompensa_total:7.1f} | "
                  f"Promedio: {promedio:7.2f} | ε: {agente.epsilon:.4f}")

    # Resultados
    print(f"\n{'=' * 60}")
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("=" * 60)

    print(f"Recompensa promedio (últimos 100 ep): {np.mean(recompensas[-100:]):.2f}")
    print(f"Estados visitados: {len(agente.Q)}")
    print(f"Epsilon final: {agente.epsilon:.4f}")

    # Evaluación
    print(f"\n{'=' * 60}")
    print("EVALUACIÓN (10 episodios, sin exploración)")
    print("=" * 60)

    agente.epsilon = 0  # Desactivar exploración
    eval_recompensas = []
    eval_pasos = []

    for _ in range(10):
        estado, _ = env.reset()
        recompensa_total = 0

        for paso in range(MAX_PASOS):
            accion = agente.seleccionar_accion(estado)
            estado, recompensa, terminado, truncado, _ = env.step(accion)
            recompensa_total += recompensa

            if terminado or truncado:
                break

        eval_recompensas.append(recompensa_total)
        eval_pasos.append(paso + 1)

    print(f"Recompensa: {np.mean(eval_recompensas):.1f} ± {np.std(eval_recompensas):.1f}")
    print(f"Pasos promedio: {np.mean(eval_pasos):.1f}")

    # Demo visual
    print(f"\n{'=' * 60}")
    print("DEMO: Un episodio completo")
    print("=" * 60)

    env_render = gym.make(ENV_NAME, render_mode="ansi")
    estado, _ = env_render.reset()

    print("\nEstado inicial:")
    print(env_render.render())

    recompensa_total = 0
    for paso in range(20):
        accion = agente.seleccionar_accion(estado)
        acciones = ['Sur', 'Norte', 'Este', 'Oeste', 'Recoger', 'Dejar']

        estado, recompensa, terminado, truncado, _ = env_render.step(accion)
        recompensa_total += recompensa

        print(f"\nPaso {paso + 1}: {acciones[accion]} | Recompensa: {recompensa}")
        print(env_render.render())

        if terminado:
            print(f"\n🎉 ¡Completado! Recompensa total: {recompensa_total}")
            break

    env_render.close()

    # Visualizar curvas
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Curva de recompensas
    window = 50
    recompensas_suav = np.convolve(recompensas, np.ones(window)/window, mode='valid')

    axes[0].plot(recompensas, alpha=0.3, label='Por episodio')
    axes[0].plot(range(window-1, len(recompensas)), recompensas_suav,
                 'r-', linewidth=2, label=f'Media móvil ({window} ep)')
    axes[0].axhline(y=8, color='g', linestyle='--', label='Objetivo (~8)')
    axes[0].set_xlabel('Episodio')
    axes[0].set_ylabel('Recompensa')
    axes[0].set_title('Curva de Aprendizaje - Q-Learning Taxi')
    axes[0].legend()

    # Curva de pasos
    pasos_suav = np.convolve(pasos, np.ones(window)/window, mode='valid')
    axes[1].plot(pasos, alpha=0.3, color='green')
    axes[1].plot(range(window-1, len(pasos)), pasos_suav,
                 'darkgreen', linewidth=2)
    axes[1].set_xlabel('Episodio')
    axes[1].set_ylabel('Pasos hasta completar')
    axes[1].set_title('Eficiencia del Agente')

    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent.parent / "modelos" / "taxi_training.png"), dpi=150)
    plt.show()

    print("\n✅ Gráficos guardados en: taxi_training.png")

    env.close()


if __name__ == "__main__":
    main()
