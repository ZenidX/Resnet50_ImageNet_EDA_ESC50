"""
Implementaciones de Agentes de Reinforcement Learning
=====================================================
Incluye: Q-Learning, DQN, y utilidades comunes.
"""

import numpy as np
import random
from collections import defaultdict, deque

# Importar PyTorch (opcional, para DQN)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch no disponible. DQN no funcionará.")


# =============================================================================
# REPLAY BUFFER (Para DQN)
# =============================================================================

class ReplayBuffer:
    """
    Buffer circular para almacenar experiencias (Experience Replay).

    Almacena transiciones (s, a, r, s', done) y permite samplear
    batches aleatorios para entrenar redes neuronales.
    """

    def __init__(self, capacidad=10000):
        """
        Args:
            capacidad: Número máximo de transiciones a almacenar
        """
        self.buffer = deque(maxlen=capacidad)

    def agregar(self, estado, accion, recompensa, siguiente_estado, terminado):
        """Añade una transición al buffer."""
        self.buffer.append((estado, accion, recompensa, siguiente_estado, terminado))

    def sample(self, batch_size):
        """
        Samplea un batch aleatorio de transiciones.

        Returns:
            Tupla de arrays numpy: (estados, acciones, recompensas, siguientes, terminados)
        """
        batch = random.sample(self.buffer, batch_size)
        estados, acciones, recompensas, siguientes, terminados = zip(*batch)

        return (np.array(estados),
                np.array(acciones),
                np.array(recompensas, dtype=np.float32),
                np.array(siguientes),
                np.array(terminados, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# Q-LEARNING (Tabular)
# =============================================================================

class QLearningAgent:
    """
    Agente Q-Learning con tabla Q.

    Ideal para entornos con estados discretos o discretizables.
    Aprende la función Q(s, a) directamente mediante la ecuación de Bellman.

    Attributes:
        Q: Diccionario que mapea estados a arrays de Q-valores
        alpha: Tasa de aprendizaje
        gamma: Factor de descuento
        epsilon: Probabilidad de exploración
    """

    def __init__(self, n_acciones, alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        """
        Args:
            n_acciones: Número de acciones posibles
            alpha: Tasa de aprendizaje (0 a 1)
            gamma: Factor de descuento (0 a 1)
            epsilon: Probabilidad inicial de exploración
            epsilon_min: Valor mínimo de epsilon
            epsilon_decay: Factor de decay por episodio
        """
        self.n_acciones = n_acciones
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Tabla Q: {estado: [Q(s,a0), Q(s,a1), ...]}
        self.Q = defaultdict(lambda: np.zeros(n_acciones))

    def seleccionar_accion(self, estado):
        """
        Selecciona una acción usando política ε-greedy.

        Args:
            estado: Estado actual (debe ser hasheable, ej: tupla)

        Returns:
            Índice de la acción seleccionada
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_acciones - 1)  # Explorar
        else:
            return np.argmax(self.Q[estado])  # Explotar

    def aprender(self, estado, accion, recompensa, siguiente_estado, terminado):
        """
        Actualiza la tabla Q usando la ecuación de Bellman.

        Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida
            siguiente_estado: Estado resultante
            terminado: Si el episodio terminó

        Returns:
            Error TD (diferencia temporal)
        """
        # Calcular target
        if terminado:
            target = recompensa
        else:
            target = recompensa + self.gamma * np.max(self.Q[siguiente_estado])

        # Error TD
        error_td = target - self.Q[estado][accion]

        # Actualizar Q
        self.Q[estado][accion] += self.alpha * error_td

        return error_td

    def decay_epsilon(self):
        """Reduce epsilon según el factor de decay."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        """
        Extrae la política greedy de la tabla Q.

        Returns:
            Dict {estado: mejor_accion}
        """
        return {estado: np.argmax(q_vals) for estado, q_vals in self.Q.items()}

    def get_value_function(self):
        """
        Extrae la función de valor V(s) = max_a Q(s,a).

        Returns:
            Dict {estado: valor}
        """
        return {estado: np.max(q_vals) for estado, q_vals in self.Q.items()}


# =============================================================================
# SARSA
# =============================================================================

class SARSAAgent:
    """
    Agente SARSA (State-Action-Reward-State-Action).

    Similar a Q-Learning pero es on-policy: usa la acción real
    del siguiente paso en lugar del máximo.

    Más conservador que Q-Learning en entornos con penalizaciones.
    """

    def __init__(self, n_acciones, alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.n_acciones = n_acciones
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = defaultdict(lambda: np.zeros(n_acciones))

    def seleccionar_accion(self, estado):
        """Política ε-greedy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_acciones - 1)
        else:
            return np.argmax(self.Q[estado])

    def aprender(self, estado, accion, recompensa, siguiente_estado, siguiente_accion, terminado):
        """
        Actualización SARSA.

        Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]

        Nota: Requiere la siguiente acción (a'), a diferencia de Q-Learning.
        """
        if terminado:
            target = recompensa
        else:
            target = recompensa + self.gamma * self.Q[siguiente_estado][siguiente_accion]

        error_td = target - self.Q[estado][accion]
        self.Q[estado][accion] += self.alpha * error_td

        return error_td

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# =============================================================================
# DQN (Deep Q-Network)
# =============================================================================

if TORCH_AVAILABLE:

    class DQNetwork(nn.Module):
        """
        Red neuronal para aproximar Q(s, a).

        Arquitectura: MLP con 2 capas ocultas.
        Input: Estado (vector)
        Output: Q-valor para cada acción
        """

        def __init__(self, input_size, n_acciones, hidden_sizes=[128, 128]):
            """
            Args:
                input_size: Dimensión del estado
                n_acciones: Número de acciones
                hidden_sizes: Lista con tamaños de capas ocultas
            """
            super(DQNetwork, self).__init__()

            layers = []
            prev_size = input_size

            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, n_acciones))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)


    class DQNAgent:
        """
        Agente Deep Q-Network.

        Usa una red neuronal para aproximar Q(s, a) y técnicas
        de estabilización: Experience Replay y Target Network.

        Ideal para estados continuos de alta dimensionalidad.
        """

        def __init__(self, input_size, n_acciones, lr=0.001, gamma=0.99,
                     epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                     buffer_size=10000, batch_size=64, target_update=10,
                     hidden_sizes=[128, 128], device=None):
            """
            Args:
                input_size: Dimensión del estado
                n_acciones: Número de acciones
                lr: Learning rate
                gamma: Factor de descuento
                epsilon: Exploración inicial
                epsilon_min: Exploración mínima
                epsilon_decay: Decay de epsilon
                buffer_size: Tamaño del replay buffer
                batch_size: Tamaño del batch para entrenamiento
                target_update: Episodios entre actualizaciones del target
                hidden_sizes: Arquitectura de la red
                device: 'cpu' o 'cuda'
            """
            self.n_acciones = n_acciones
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.batch_size = batch_size
            self.target_update = target_update

            # Device
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            # Redes
            self.q_network = DQNetwork(input_size, n_acciones, hidden_sizes).to(self.device)
            self.target_network = DQNetwork(input_size, n_acciones, hidden_sizes).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()

            # Optimizador
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
            self.loss_fn = nn.MSELoss()

            # Replay Buffer
            self.memory = ReplayBuffer(buffer_size)

            # Contador de episodios
            self.episode_count = 0

        def seleccionar_accion(self, estado):
            """
            Selecciona acción con ε-greedy.

            Args:
                estado: Array numpy con el estado

            Returns:
                Índice de la acción
            """
            if random.random() < self.epsilon:
                return random.randint(0, self.n_acciones - 1)

            with torch.no_grad():
                estado_tensor = torch.FloatTensor(estado).unsqueeze(0).to(self.device)
                q_valores = self.q_network(estado_tensor)
                return q_valores.argmax().item()

        def recordar(self, estado, accion, recompensa, siguiente_estado, terminado):
            """Almacena una transición en el buffer."""
            self.memory.agregar(estado, accion, recompensa, siguiente_estado, terminado)

        def entrenar_paso(self):
            """
            Realiza un paso de entrenamiento.

            Returns:
                Pérdida del paso (o 0 si no hay suficientes muestras)
            """
            if len(self.memory) < self.batch_size:
                return 0

            # Samplear batch
            estados, acciones, recompensas, siguientes, terminados = \
                self.memory.sample(self.batch_size)

            # Convertir a tensores
            estados_t = torch.FloatTensor(estados).to(self.device)
            acciones_t = torch.LongTensor(acciones).to(self.device)
            recompensas_t = torch.FloatTensor(recompensas).to(self.device)
            siguientes_t = torch.FloatTensor(siguientes).to(self.device)
            terminados_t = torch.FloatTensor(terminados).to(self.device)

            # Q-valores actuales para las acciones tomadas
            q_actual = self.q_network(estados_t).gather(1, acciones_t.unsqueeze(1)).squeeze()

            # Q-valores target (usando target network)
            with torch.no_grad():
                q_siguiente_max = self.target_network(siguientes_t).max(1)[0]
                q_target = recompensas_t + (1 - terminados_t) * self.gamma * q_siguiente_max

            # Calcular pérdida y actualizar
            loss = self.loss_fn(q_actual, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()

            return loss.item()

        def fin_episodio(self):
            """Llamar al final de cada episodio."""
            self.episode_count += 1

            # Actualizar target network periódicamente
            if self.episode_count % self.target_update == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        def guardar(self, filepath):
            """Guarda el modelo entrenado."""
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'episode_count': self.episode_count
            }, filepath)
            print(f"Modelo guardado en: {filepath}")

        def cargar(self, filepath):
            """Carga un modelo guardado."""
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.episode_count = checkpoint['episode_count']
            print(f"Modelo cargado desde: {filepath}")


# =============================================================================
# FUNCIONES DE ENTRENAMIENTO
# =============================================================================

def entrenar_qlearning(env, agente, n_episodios=500, max_pasos=1000, verbose=True):
    """
    Entrena un agente Q-Learning.

    Args:
        env: Entorno con interfaz step()/reset()
        agente: Instancia de QLearningAgent
        n_episodios: Número de episodios de entrenamiento
        max_pasos: Máximo de pasos por episodio
        verbose: Si mostrar progreso

    Returns:
        Tupla (recompensas, pasos) con historiales
    """
    recompensas = []
    pasos = []

    for ep in range(n_episodios):
        estado = env.reset()
        if isinstance(estado, tuple):  # Gymnasium devuelve (obs, info)
            estado = estado[0]

        recompensa_total = 0

        for paso in range(max_pasos):
            accion = agente.seleccionar_accion(estado)

            resultado = env.step(accion)
            if len(resultado) == 5:  # Gymnasium
                siguiente, recompensa, terminado, truncado, _ = resultado
                done = terminado or truncado
            else:  # Entorno simple
                siguiente, recompensa, done = resultado

            agente.aprender(estado, accion, recompensa, siguiente, done)

            recompensa_total += recompensa
            estado = siguiente

            if done:
                break

        agente.decay_epsilon()
        recompensas.append(recompensa_total)
        pasos.append(paso + 1)

        if verbose and (ep + 1) % 100 == 0:
            promedio = np.mean(recompensas[-100:])
            print(f"Episodio {ep + 1:4d} | Recompensa: {recompensa_total:7.1f} | "
                  f"Promedio: {promedio:7.2f} | ε: {agente.epsilon:.3f}")

    return recompensas, pasos


def entrenar_dqn(env, agente, n_episodios=500, max_pasos=1000, verbose=True):
    """
    Entrena un agente DQN.

    Args:
        env: Entorno de Gymnasium
        agente: Instancia de DQNAgent
        n_episodios: Número de episodios
        max_pasos: Máximo de pasos por episodio
        verbose: Si mostrar progreso

    Returns:
        Tupla (recompensas, perdidas) con historiales
    """
    recompensas = []
    perdidas = []

    for ep in range(n_episodios):
        estado, _ = env.reset()
        recompensa_total = 0
        perdida_total = 0
        n_updates = 0

        for paso in range(max_pasos):
            accion = agente.seleccionar_accion(estado)
            siguiente, recompensa, terminado, truncado, _ = env.step(accion)
            done = terminado or truncado

            agente.recordar(estado, accion, recompensa, siguiente, done)
            loss = agente.entrenar_paso()

            recompensa_total += recompensa
            if loss > 0:
                perdida_total += loss
                n_updates += 1

            estado = siguiente

            if done:
                break

        agente.fin_episodio()
        recompensas.append(recompensa_total)
        perdidas.append(perdida_total / max(n_updates, 1))

        if verbose and (ep + 1) % 50 == 0:
            promedio = np.mean(recompensas[-50:])
            print(f"Episodio {ep + 1:4d} | Recompensa: {recompensa_total:7.1f} | "
                  f"Promedio: {promedio:7.2f} | ε: {agente.epsilon:.3f}")

    return recompensas, perdidas


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Test de Agentes de RL ===\n")

    # Test Q-Learning
    print("Test QLearningAgent:")
    agent = QLearningAgent(n_acciones=4)
    estado = (0, 0)
    accion = agent.seleccionar_accion(estado)
    error = agent.aprender(estado, accion, -1, (0, 1), False)
    print(f"  Acción seleccionada: {accion}")
    print(f"  Error TD: {error:.4f}")
    print(f"  Q[{estado}]: {agent.Q[estado]}")

    # Test DQN
    if TORCH_AVAILABLE:
        print("\nTest DQNAgent:")
        dqn = DQNAgent(input_size=4, n_acciones=2, hidden_sizes=[64, 64])
        estado = np.random.randn(4)
        accion = dqn.seleccionar_accion(estado)
        print(f"  Estado: {estado}")
        print(f"  Acción seleccionada: {accion}")
        print(f"  Device: {dqn.device}")

    print("\n✅ Todos los agentes funcionan correctamente")
