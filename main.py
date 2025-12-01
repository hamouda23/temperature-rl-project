# -*- coding: utf-8 -*-
"""

Q-LEARNING VERSION ULTRA-SIMPLE SANS GYMNASIUM ET SANS DEPENDANCES
===============================================================================
Contrôle de température - Code minimal pour apprendre Le Q-Learning
-------------------------------------------------------------------------------
Auteur: Samir HAMOUDA Raouf Ochi
Date: Novembre 2025
"""

import random
import matplotlib.pyplot as plt
# ═══════════════════════════════════════════════════════════════
# PARTIE 1 : LA CHAMBRE (simulation simple)
# ═══════════════════════════════════════════════════════════════

def reset_chambre():
    """Commence un nouvel épisode"""
    return {
        'temp': random.randint(15, 28),  # Température initiale aléatoire
        'step': 0
    }


def faire_action(chambre, action):
    """
    Exécute une action et retourne ce qui se passe.
    
    Actions:
        0 = Refroidir
        1 = Rien faire
        2 = Chauffer
    """
    
    # Effet de l'action
    if action == 0:  # Refroidir
        chambre['temp'] -= 2
    elif action == 2:  # Chauffer
        chambre['temp'] += 2
    # Sinon (action == 1) : rien faire
    
    # Un peu de variation naturelle
    chambre['temp'] += random.uniform(-0.5, 0.5)
    
    # Récompense
    if 20 <= chambre['temp'] <= 22:
        reward = 10  # BIEN ! Dans la zone confort
    else:
        reward = -1  # MAL ! Hors zone
    
    # Avancer le temps
    chambre['step'] += 1
    done = chambre['step'] >= 50  # Épisode de 50 steps
    
    return reward, done


# ═══════════════════════════════════════════════════════════════
# PARTIE 2 : Q-LEARNING (le cerveau)
# ═══════════════════════════════════════════════════════════════

def discretiser(temperature):
    """Arrondit la température pour la Q-table"""
    return round(temperature)  # 21.3 → 21


def choisir_action(q_table, temp, epsilon):
    """
    Choisit une action (exploration vs exploitation).
    
    epsilon = probabilité d'explorer (essayer au hasard)
    """
    
    # EXPLORATION : Action aléatoire
    if random.random() < epsilon:
        return random.randint(0, 2)
    
    # EXPLOITATION : Meilleure action connue
    # Regarde Q(temp, 0), Q(temp, 1), Q(temp, 2)
    q0 = q_table.get((temp, 0), 0)
    q1 = q_table.get((temp, 1), 0)
    q2 = q_table.get((temp, 2), 0)
    
    # Trouve l'action avec la plus grande valeur Q
    if q0 >= q1 and q0 >= q2:
        return 0
    elif q1 >= q2:
        return 1
    else:
        return 2


def apprendre(q_table, temp, action, reward, nouvelle_temp, alpha, gamma):
    """
    Met à jour la Q-table (APPRENTISSAGE).
    
    alpha = vitesse d'apprentissage (0.1 = 10% de mise à jour)
    gamma = importance du futur (0.9 = regarde 90% loin)
    """
    
    # Valeur Q actuelle
    q_actuel = q_table.get((temp, action), 0)
    
    # Meilleure valeur Q au prochain état
    q_next_0 = q_table.get((nouvelle_temp, 0), 0)
    q_next_1 = q_table.get((nouvelle_temp, 1), 0)
    q_next_2 = q_table.get((nouvelle_temp, 2), 0)
    max_q_next = max(q_next_0, q_next_1, q_next_2)
    
    # FORMULE Q-LEARNING
    nouveau_q = q_actuel + alpha * (reward + gamma * max_q_next - q_actuel)
    
    # Mettre à jour
    q_table[(temp, action)] = nouveau_q


# ═══════════════════════════════════════════════════════════════
# PARTIE 3 : ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════

def entrainer(episodes=200):
    q_table = {}
    alpha = 0.01
    gamma = 0.9
    epsilon = 1.0
    rewards_history = []  # <-- stocke la récompense totale de chaque épisode
    
    for episode in range(episodes):
        chambre = reset_chambre()
        temp = discretiser(chambre['temp'])
        reward_total = 0
        
        for step in range(50):
            action = choisir_action(q_table, temp, epsilon)
            reward, done = faire_action(chambre, action)
            nouvelle_temp = discretiser(chambre['temp'])
            apprendre(q_table, temp, action, reward, nouvelle_temp, alpha, gamma)
            temp = nouvelle_temp
            reward_total += reward
            if done:
                break
        
        rewards_history.append(reward_total)  # <-- ajoute la récompense totale
        epsilon = max(0.01, epsilon * 0.995)
        
        if (episode + 1) % 20 == 0:
            print(f"Épisode {episode+1:3d} | Reward: {reward_total:6.1f} | Epsilon: {epsilon:.3f} | Q-table: {len(q_table)}")
    
    # ── Créer et enregistrer le plot ──
    plt.figure(figsize=(8,4))
    plt.plot(rewards_history, label="Reward total par épisode")
    plt.xlabel("Épisode")
    plt.ylabel("Reward total")
    plt.title("Évolution des récompenses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rewards_plot1.png")  # <-- sauvegarde en PNG
    plt.close()

    print("\n✅ Entraînement terminé ! Le plot a été enregistré sous 'rewards_plot1.png'\n")
    return q_table



# ═══════════════════════════════════════════════════════════════
# PARTIE 4 : TEST
# ═══════════════════════════════════════════════════════════════

def tester(q_table):
    chambre = reset_chambre()
    temp_initiale = chambre['temp']
    temperatures = []
    actions_taken = []

    for step in range(50):
        temp = discretiser(chambre['temp'])
        action = choisir_action(q_table, temp, epsilon=0)
        reward, done = faire_action(chambre, action)
        temperatures.append(chambre['temp'])
        actions_taken.append(action)
        if done:
            break
    
    # ── Plot de la température ──
    plt.figure(figsize=(8,4))
    plt.plot(temperatures, marker='o', label="Température")
    plt.axhline(20, color='green', linestyle='--', label="Zone confort min")
    plt.axhline(22, color='green', linestyle='--', label="Zone confort max")
    plt.xlabel("Step")
    plt.ylabel("Température (°C)")
    plt.title("Évolution de la température pendant le test")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("temperature_test1.png")
    plt.close()

    print("\n✅ Test terminé ! Le plot a été enregistré sous 'temperature_test1.png'\n")


# ═══════════════════════════════════════════════════════════════
# PROGRAMME PRINCIPAL
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("\n" + "="*50)
    print("  Q-LEARNING ULTRA-SIMPLE")
    print("  Contrôle de Température")
    print("="*50 + "\n")
    
    # 1. ENTRAÎNER
    q_table = entrainer(episodes=200)
    
    # 2. TESTER
    tester(q_table)
    
    print("\n" + "="*50)
    print("  TERMINÉ !")
    print("="*50 + "\n")