
# Snake Game avec Deep Q-Learning (DQN)

Ce projet implémente un jeu Snake en utilisant la bibliothèque Pygame, avec un agent d'apprentissage par renforcement basé sur un Deep Q-Network (DQN). L'agent apprend à jouer au Snake en maximisant une fonction de récompense à travers une exploration et une exploitation des états du jeu.

## Fonctionnalités

- **Jeu Snake** : Un jeu Snake classique où le serpent mange de la nourriture pour grandir tout en évitant de se heurter aux murs et à lui-même.
- **Agent d'apprentissage (DQN)** : Un agent qui apprend à jouer au jeu Snake en utilisant l'algorithme Q-learning avec une stratégie epsilon-greedy.
- **Exploration et exploitation** : L'agent choisit ses actions en fonction de l'état du jeu, en équilibrant exploration et exploitation.
- **Sauvegarde et chargement du modèle** : Le modèle d'agent (Q-table) peut être sauvegardé et chargé pour continuer l'entraînement.

## Prérequis

Pour faire fonctionner ce projet, vous devez avoir Python et les bibliothèques suivantes installées :

- `pygame` : pour l'interface du jeu.
- `numpy` : pour la gestion des tableaux et des calculs numériques.
- `random` : pour les actions aléatoires de l'agent.
- `json` : pour la sauvegarde et le chargement du modèle de l'agent.

Vous pouvez installer les dépendances nécessaires avec `pip` :

```bash
pip install pygame numpy
```

## Structure du Projet

- `snake_game.py` : Contient la classe `SnakeGame` qui gère le jeu et l'interface.
- `dqn_agent.py` : Contient la classe `DQNAgent` qui implémente l'algorithme Deep Q-Network pour l'apprentissage de l'agent.
- `train.py` : Le script principal qui lance l'entraînement de l'agent sur plusieurs épisodes et sauvegarde périodiquement le modèle.
- `snake_model.json` : Le fichier JSON où les poids du modèle (Q-table) sont sauvegardés.

## Comment Exécuter le Jeu

Pour entraîner l'agent, exécutez le fichier `train.py` :

```bash
python train.py
```

Vous pouvez définir l'option `continue_training=True` pour reprendre l'entraînement à partir du modèle sauvegardé si vous avez déjà un fichier `snake_model.json`.

### Exemple de commande pour reprendre l'entraînement :

```bash
python train.py --continue_training
```

## Fonctionnement du Code

1. **SnakeGame Class** :
   - Cette classe gère l'état du jeu, la gestion de la nourriture, la détection des dangers (collisions), et le rendu visuel à l'aide de Pygame.
   
2. **DQNAgent Class** :
   - Cette classe implémente l'algorithme Deep Q-Learning pour permettre à l'agent d'apprendre à jouer au Snake. Elle gère le choix des actions via une stratégie epsilon-greedy et le stockage des expériences pour l'apprentissage par replay.

3. **Entraînement** :
   - L'agent apprend en jouant des épisodes du jeu, en choisissant des actions et en recevant des récompenses. L'algorithme met à jour les valeurs Q en fonction de ces expériences.
   - Le modèle est sauvegardé périodiquement, chaque 100 épisodes.

## Sauvegarde et Chargement du Modèle

Le modèle d'agent (Q-table) peut être sauvegardé en utilisant la fonction `save_model()` et chargé avec la fonction `load_model()`.

- Sauvegarde du modèle :
  ```python
  agent.save_model()
  ```

- Chargement du modèle :
  ```python
  agent.load_model()
  ```

Le fichier `snake_model.json` contiendra les informations du modèle.

## Comment Améliorer l'Agent

- **Récompenses et Pénalités** : Vous pouvez ajuster les récompenses pour guider l'agent dans son apprentissage, comme donner plus de récompenses pour éviter les collisions et manger la nourriture.
- **DQN avec Réseaux de Neurones** : Remplacer la table Q par un réseau de neurones pour gérer des espaces d'états plus larges.
- **Actions Supplémentaires** : Ajouter des actions plus complexes comme des mouvements diagonaux ou des changements de vitesse.

## À Propos

Ce projet est un exercice d'apprentissage par renforcement appliqué à un jeu classique. Il permet de comprendre comment un agent peut apprendre à jouer à un jeu à partir d'une série d'expériences, tout en s'appuyant sur des techniques d'intelligence artificielle comme les Q-Learning et les réseaux neuronaux.
