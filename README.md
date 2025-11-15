# temperature-rl-project
🌡️ Contrôle intelligent de température par apprentissage par renforcement (Q-Learning). Agent autonome qui maintient une pièce entre 20-22°C en optimisant confort et consommation énergétique.
# 🌡️ Temperature Control with Q-Learning

Système de contrôle intelligent de température utilisant l'apprentissage par renforcement (Reinforcement Learning). Un agent Q-Learning apprend automatiquement à maintenir une pièce dans une zone de confort (20-22°C) tout en minimisant la consommation énergétique.

## 🎯 Objectif

Développer un agent intelligent capable de :
- ✅ Maintenir la température dans une zone de confort (20-22°C)
- ✅ Minimiser la consommation énergétique
- ✅ S'adapter aux variations de température extérieure
- ✅ Apprendre sans programmation explicite des règles

## 🧠 Approche

**Algorithme** : Q-Learning (Reinforcement Learning)
- **État** : Température intérieure et extérieure
- **Actions** : Refroidir, Ne rien faire, Chauffer
- **Récompense** : +10 dans zone confort, -1 hors zone, pénalité énergétique

## 🚀 Quick Start

### Installation
```bash
# Version simple (Python pur)
python simple_qlearning.py

# Version avec Gymnasium (graphiques)
pip install gymnasium numpy matplotlib
python main.py
```

### Résultats attendus
- ✅ Agent apprend en ~100 épisodes
- ✅ Maintient la température 80-90% du temps
- ✅ Table Q avec ~150-200 états appris

## 📊 Résultats

L'agent apprend à :
1. **Chauffer** quand température < 20°C
2. **Refroidir** quand température > 22°C
3. **Ne rien faire** quand dans la zone de confort
4. **Anticiper** les variations thermiques

## 🛠️ Technologies

- **Python** 3.8+
- **Q-Learning** (Tabular RL)
- **Gymnasium** (environnement RL standard)
- **NumPy** (calculs)
- **Matplotlib** (visualisation)

## 📁 Structure du projet
```
temperature-rl/
├── simple_qlearning.py    # Version minimale (débutant)
├── main.py                # Version complète avec Gymnasium
├── README.md              # Documentation
└── results/               # Graphiques et résultats
```

## 🔄 Perspectives

- [ ] Intégration Node-RED/MQTT pour déploiement IoT
- [ ] Multi-pièces avec coordination
- [ ] Algorithmes avancés (SARSA, DQN)
- [ ] Prise en compte tarification électrique variable
- [ ] Prédiction météo intégrée

## 📚 Références

- **Gymnasium** : https://gymnasium.farama.org/
- **Q-Learning** : Sutton & Barto - Reinforcement Learning: An Introduction
- **Projets similaires** : 
  - [VectorInstitute/HV-Ai-C](https://github.com/VectorInstitute/HV-Ai-C)
  - [the5avage/Q-Learning](https://github.com/the5avage/Q-Learning)

## 👤 Auteur

[Ton Nom] - Projet d'apprentissage par renforcement

## 📄 Licence

MIT License - Libre d'utilisation et modification

## 🌟 Remerciements

Merci à la communauté Gymnasium et aux projets open-source qui ont inspiré ce travail.
```

---

## 🏷️ **Topics à ajouter sur GitHub**

Clique sur "Add topics" et ajoute :
```
reinforcement-learning
q-learning
temperature-control
hvac
iot
smart-home
gymnasium
openai-gym
machine-learning
python
automation
energy-optimization
domotics
