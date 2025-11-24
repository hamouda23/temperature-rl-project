# -*- coding: utf-8 -*-
"""

Q-LEARNING VERSION ULTRA-SIMPLE SANS GYMNASIUM ET SANS DEPENDANCES
===============================================================
Contrôle de température - Code minimal pour apprendre
Auteur: Samir HAMOUDA Raouf Ochi
Date: Novembre 2025
"""

import random

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
    """Entraîne l'agent"""
    
    print("="*50)
    print("  ENTRAÎNEMENT Q-LEARNING")
    print("="*50)
    
    # Initialisation
    q_table = {}  # Mémoire vide
    alpha = 0.1   # Vitesse apprentissage
    gamma = 0.9   # Importance futur
    epsilon = 1.0 # Exploration initiale (100%)
    
    # Entraînement
    for episode in range(episodes):
        
        # Nouvel épisode
        chambre = reset_chambre()
        temp = discretiser(chambre['temp'])
        reward_total = 0
        
        # Jouer l'épisode
        for step in range(50):
            
            # 1. Choisir action
            action = choisir_action(q_table, temp, epsilon)
            
            # 2. Exécuter
            reward, done = faire_action(chambre, action)
            nouvelle_temp = discretiser(chambre['temp'])
            
            # 3. Apprendre
            apprendre(q_table, temp, action, reward, nouvelle_temp, alpha, gamma)
            
            # 4. Avancer
            temp = nouvelle_temp
            reward_total += reward
            
            if done:
                break
        
        # Diminuer exploration
        epsilon = max(0.01, epsilon * 0.995)
        
        # Afficher progression tous les 20 épisodes
        if (episode + 1) % 20 == 0:
            print(f"Épisode {episode+1:3d} | Reward: {reward_total:6.1f} | Epsilon: {epsilon:.3f} | Q-table: {len(q_table)}")
    
    print("\n✅ Entraînement terminé !\n")
    return q_table


# ═══════════════════════════════════════════════════════════════
# PARTIE 4 : TEST
# ═══════════════════════════════════════════════════════════════

def tester(q_table):
    """Teste l'agent entraîné"""
    
    print("="*50)
    print("  TEST DE L'AGENT")
    print("="*50)
    
    chambre = reset_chambre()
    temp_initiale = chambre['temp']
    print(f"\n🌡️  Température initiale: {temp_initiale:.1f}°C")
    print(f"🎯 Objectif: Maintenir 20-22°C\n")
    print("Step | Temp  | Action      | Reward")
    print("-----+-------+-------------+--------")
    
    temps_confort = 0
    
    for step in range(50):
        temp = discretiser(chambre['temp'])
        
        # Pas d'exploration (epsilon=0)
        action = choisir_action(q_table, temp, epsilon=0)
        
        # Exécuter
        reward, done = faire_action(chambre, action)
        
        # Noms d'actions
        actions = ["Refroidir", "Rien", "Chauffer"]
        
        # Afficher
        print(f"{step:4d} | {chambre['temp']:5.1f} | {actions[action]:11s} | {reward:6.1f}")
        
        # Compter temps dans zone confort
        if 20 <= chambre['temp'] <= 22:
            temps_confort += 1
        
        if done:
            break
    
    # Résultats
    print("\n" + "="*50)
    print("  RÉSULTATS")
    print("="*50)
    print(f"Température finale: {chambre['temp']:.1f}°C")
    print(f"Temps dans zone confort: {temps_confort}/50 ({temps_confort*2}%)")
    print(f"États appris: {len(q_table)}")
    
    # Montrer quelques valeurs Q
    print("\n📊 Exemples de Q-values apprises:")
    for temp_val in [18, 20, 21, 22, 24]:
        if any((temp_val, a) in q_table for a in range(3)):
            print(f"\n   Température {temp_val}°C:")
            for action in range(3):
                q = q_table.get((temp_val, action), 0)
                actions = ["Refroidir", "Rien", "Chauffer"]
                print(f"      {actions[action]:10s}: {q:7.2f}")


# ═══════════════════════════════════════════════════════════════
# PROGRAMME PRINCIPAL
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("\n" + "="*50)
    print("  Q-LEARNING ULTRA-SIMPLE")
    print("  Contrôle de Température")
    print("="*50 + "\n")
    
    # 1. ENTRAÎNER
    q_table = entrainer(episodes=100)
    
    # 2. TESTER
    tester(q_table)
    
    print("\n" + "="*50)
    print("  TERMINÉ !")
    print("="*50 + "\n")