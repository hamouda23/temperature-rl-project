import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class RoomTemperatureEnv(gym.Env):
    """
    Environnement de contrôle de température d'une chambre.
    
    Objectif: Maintenir la température entre 20°C et 22°C
    Actions: 0=Refroidir, 1=Ne rien faire, 2=Chauffer
    """
    
    def __init__(self):
        super(RoomTemperatureEnv, self).__init__()
        
        # Température cible (20-22°C)
        self.target_temp_min = 20.0
        self.target_temp_max = 22.0
        
        # Espaces d'action et d'observation
        self.action_space = spaces.Discrete(3)  # 0: refroidir, 1: rien, 2: chauffer
        
        # Observation: [température actuelle, température extérieure, heure du jour]
        self.observation_space = spaces.Box(
            low=np.array([10.0, -10.0, 0.0]),
            high=np.array([35.0, 40.0, 24.0]),
            dtype=np.float32
        )
        
        # Paramètres physiques
        self.heating_power = 2.0  # °C par step
        self.cooling_power = 1.5  # °C par step
        self.heat_loss_coef = 0.1  # Coefficient de perte de chaleur
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # État initial aléatoire
        self.current_temp = np.random.uniform(15, 28)
        self.outdoor_temp = np.random.uniform(-5, 35)
        self.time_of_day = np.random.uniform(0, 24)
        self.step_count = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.array([
            self.current_temp,
            self.outdoor_temp,
            self.time_of_day
        ], dtype=np.float32)
    
    def step(self, action):
        # Effet de l'action
        if action == 0:  # Refroidir
            self.current_temp -= self.cooling_power
            energy_cost = 0.05
        elif action == 2:  # Chauffer
            self.current_temp += self.heating_power
            energy_cost = 0.08
        else:  # Ne rien faire
            energy_cost = 0.0
        
        # Pertes thermiques naturelles (vers la température extérieure)
        temp_diff = self.current_temp - self.outdoor_temp
        self.current_temp -= self.heat_loss_coef * temp_diff
        
        # Variation de température extérieure et temps
        self.outdoor_temp += np.random.uniform(-0.5, 0.5)
        self.outdoor_temp = np.clip(self.outdoor_temp, -10, 40)
        self.time_of_day = (self.time_of_day + 0.25) % 24
        
        # Calcul de la récompense
        if self.target_temp_min <= self.current_temp <= self.target_temp_max:
            comfort_reward = 1.0
        else:
            # Pénalité proportionnelle à la distance de la zone de confort
            distance = min(
                abs(self.current_temp - self.target_temp_min),
                abs(self.current_temp - self.target_temp_max)
            )
            comfort_reward = -distance * 0.5
        
        reward = comfort_reward - energy_cost
        
        self.step_count += 1
        terminated = self.step_count >= 200
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}


class QLearningAgent:
    """Agent Q-Learning simple pour le contrôle de température"""
    
    def __init__(self, env, learning_rate=0.1, discount=0.95, epsilon=1.0):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Table Q discrétisée
        self.q_table = {}
    
    def discretize_state(self, state):
        """Convertit l'état continu en état discret"""
        temp = round(state[0] / 2) * 2  # Arrondir à 2°C près
        outdoor = round(state[1] / 5) * 5  # Arrondir à 5°C près
        time = round(state[2] / 4) * 4  # Arrondir à 4h près
        return (temp, outdoor, time)
    
    def get_q_value(self, state, action):
        """Récupère la valeur Q d'un état-action"""
        return self.q_table.get((state, action), 0.0)
    
    def choose_action(self, state, training=True):
        """Choisit une action avec epsilon-greedy"""
        if training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # Choisir la meilleure action
        q_values = [self.get_q_value(state, a) for a in range(3)]
        return np.argmax(q_values)
    
    def update_q_table(self, state, action, reward, next_state):
        """Met à jour la table Q"""
        current_q = self.get_q_value(state, action)
        max_next_q = max([self.get_q_value(next_state, a) for a in range(3)])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
    
    def train(self, episodes=500):
        """Entraîne l'agent"""
        rewards_history = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            total_reward = 0
            
            for _ in range(200):
                action = self.choose_action(state, training=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state_discrete = self.discretize_state(next_state)
                
                self.update_q_table(state, action, reward, next_state_discrete)
                
                state = next_state_discrete
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards_history.append(total_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Épisode {episode + 1}/{episodes} - Récompense moyenne: {avg_reward:.2f} - Epsilon: {self.epsilon:.3f}")
        
        return rewards_history


# Entraînement de l'agent
if __name__ == "__main__":
    print("=== Entraînement de l'agent RL pour contrôle de température ===\n")
    
    env = RoomTemperatureEnv()
    agent = QLearningAgent(env)
    
    # Entraînement
    rewards = agent.train(episodes=500)
    
    # Visualisation des résultats
    plt.figure(figsize=(12, 4))
    
    # Graphique 1: Récompenses
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3)
    plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'))
    plt.xlabel('Épisode')
    plt.ylabel('Récompense totale')
    plt.title('Évolution de l\'apprentissage')
    plt.grid(True)
    
    # Graphique 2: Test de l'agent entraîné
    plt.subplot(1, 2, 2)
    state, _ = env.reset()
    temps = []
    actions = []
    
    for step in range(200):
        state_discrete = agent.discretize_state(state)
        action = agent.choose_action(state_discrete, training=False)
        state, reward, terminated, truncated, _ = env.step(action)
        temps.append(state[0])
        actions.append(action)
        
        if terminated or truncated:
            break
    
    plt.plot(temps, label='Température')
    plt.axhline(y=20, color='g', linestyle='--', label='Zone confort min')
    plt.axhline(y=22, color='g', linestyle='--', label='Zone confort max')
    plt.xlabel('Temps (steps)')
    plt.ylabel('Température (°C)')
    plt.title('Performance de l\'agent entraîné')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== Entraînement terminé ===")
    print(f"Taille de la table Q: {len(agent.q_table)}")
    print(f"Température finale: {temps[-1]:.2f}°C")