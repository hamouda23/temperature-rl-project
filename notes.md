# MES NOTES Q-LEARNING

## Ce que j'ai compris :

### Q-Table
- Dictionnaire Python : {(état, action): valeur}
- Commence vide, se remplit pendant l'entraînement
- Exemple : (18, 2): 5.8 = "Chauffer à 18°C vaut 5.8"

### Les 5 fonctions clés :
1. reset_chambre() - Nouvelle température aléatoire
2. faire_action() - Physique + récompense
3. discretiser() - 21.3 → 21
4. choisir_action() - Epsilon-greedy
5. apprendre() - Q = Q + α(r + γmax(Q') - Q)

### Epsilon :
- Début : 1.0 (explore tout)
- Fin : 0.01 (exploite ce qu'il sait)

### Alpha
- La vitesse à laquelle l'agent met à jour ses connaissances
- ALPHA = 0.1 (10%) Apprentissage ÉQUILIBRÉ
- ALPHA = 0.01 (1%) Apprentissage STABLE mais LENT

### Résultats :
- Agent atteint 85-95% du temps dans zone confort
- Après 100 épisodes seulement !

## Amélioration :
- Augmenter episodes 200
- Améliorer les récompose ex : mettre une zone tampo 
  if 20 <= chambre['temp'] <= 22:
      reward = 10
 elif 19 <= chambre['temp'] <= 23:  # Zone tampon
      reward = 5
 else:
      reward = -1

### Bibliothèques Python pour RL :
- Gymnasium 

- RLlib (de Ray): Framework scalable pour RL multi-agents et deep RL. Idéal pour des simulations de chauffage complexes (ex. : district heating). Exemple : Intégrez un env Gym pour optimiser l'énergie sur plusieurs pièces. Installation : pip install ray[rllib].
Projet exemple : RLHeatingController sur GitHub.

- Stable Baselines3: Implémentations fiables d'algos RL (PPO, DDPG, SAC) sur Gymnasium. Parfait pour affiner un contrôleur PID-like en RL pour température (ex. : TCLab hardware). Facile à tuner pour minimiser overshoots. Installation : pip install stable-baselines3.
Utile pour : Simulations rapides de régulation thermique.

- pyqlearning: Bibliothèque légère pour Q-Learning et Deep Q-Network, sans dépendances lourdes. Directement applicable à des envs température discrets (comme votre code). Supporte multi-agents pour systèmes HVAC. Installation : pip install pyqlearning.
Avantage : Minimaliste, proche de votre approche simple.

- BuildingGym: Toolbox open-source pour RL en gestion énergétique de bâtiments (inclut envs pour température/chauffage). Compatible avec RLlib ou Stable Baselines3 ; simule variations climatiques. Installation : Via GitHub.
Exemple : Optimisation de consommation en temps réel. Détails sur arXiv.