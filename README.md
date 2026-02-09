# Deep Learning Intensive Projects - 10-Day Challenge

This repository contains the code and results of two Deep Learning projects completed within a 10-day timeframe as part of a Deep Learning course. The goal was to implement, train, and optimize Neural Networks using PyTorch from scratch.

## Project Overview

The repository is divided into two main parts, corresponding to the two distinct tasks assigned.

### Project 1: [Gymnastic exam pass/fail prediction]
* Objective: [Predict the binary result of a gymnastic entrance exam] (https://www.kaggle.com/competitions/gymnastic-exam/overview)

### Project 2: [20-minute shooting drill] (https://www.kaggle.com/competitions/10-minute-shooting-drill) 
* Objective: [Predict number of crossbars hit by the player during a 20-minute shooting drill]


## Key Features & Methodology

Across both projects, the following methodologies were applied:

* Framework: PyTorch for model architecture and automatic differentiation.
* Data Processing: 
    * Custom PyTorch import `Dataset` and `DataLoader` classes.
    * Preprocessing using Scikit-Learn (StandardScaler).
* Optimization:
    * Hyperparameter tuning using Optuna (visualizations available in the results).
    * K-Fold Cross-Validation to ensure model robustness.
    * Adaptive Learning Rate (ReduceLROnPlateau) to converge efficiently.

## Installation

This project uses uv for dependency management.

1. Clone the repository:
```bash
git clone [URL_OF_YOUR_REPO](https://github.com/TOUJAN-Oceam/Kaggle_DL_3A_CentraleSupelec)
cd [NAME_OF_YOUR_FOLDER](https://github.com/TOUJAN-Oceam/Kaggle_DL_3A_CentraleSupelec)
```
2. To sync the .toml : 
```bash
  uv sync
```
then run both scripts :
```bash  
  uv run Shooting_drill_scripts/Shooting_drill_prediction_TOUJAN_Oceam.py
  uv run gymnastic_scripts/gymnastic_prediction_TOUJAN_Oceam.py
```

## Results & Analysis : 
The repository includes visual analysis of the training process:

  * `Optuna` Plots: Visualizations of the hyperparameter search space and objective value history are available in the graphs/ or optuna_results/ folder.

  * Loss Curves: Training and validation loss evolution to monitor overfitting.
  
  * Performance: Metric evaluated:
  * The shooting exercise will correspond to the absolute error between your prediction and the actual number of crossbars hit by the player, 41 participants, public score: 1.17240 (top 2 in leaderboard), private score: 1.14519 (top 4 in leaderboard).
  * The gymnastics test will correspond to the accuracy score, 41 participants, public score: 0.90208 (top 3 in leaderboard), private score: 0.87142 (top 4 in leaderboard).


[FRENCH VERSION]

# Projets intensifs d'apprentissage profond - Défi de 10 jours

Ce référentiel contient le code et les résultats de deux projets d'apprentissage profond réalisés en 10 jours dans le cadre d'un cours sur l'apprentissage profond. L'objectif était de mettre en œuvre, d'entraîner et d'optimiser des réseaux neuronaux à partir de zéro à l'aide de PyTorch.

## Aperçu du projet

Le référentiel est divisé en deux parties principales, correspondant aux deux tâches distinctes assignées.

### Projet 1 : [Prédiction de la réussite/l'échec à un examen de gymnastique]
* Objectif : [Prédire le résultat binaire d'un examen d'entrée en gymnastique] (https://www.kaggle.com/competitions/gymnastic-exam/overview)

### Projet 2 : [Exercice de tir de 20 minutes] (https://www.kaggle.com/competitions/10-minute-shooting-drill) 
* Objectif : [Prédire le nombre de barres transversales touchées par le joueur pendant un exercice de tir de 20 minutes]
## Principales caractéristiques et méthodologie

Les méthodologies suivantes ont été appliquées aux deux projets :

* Cadre : PyTorch pour l'architecture du modèle et la différenciation automatique.
* Traitement des données :
* Classes personnalisées PyTorch import `Dataset` et `DataLoader`.
* Prétraitement à l'aide de Scikit-Learn (StandardScaler).
* Optimisation :
* Réglage des hyperparamètres à l'aide d'Optuna (visualisations disponibles dans les résultats).
* Validation croisée K-Fold pour garantir la robustesse du modèle.
* Taux d'apprentissage adaptatif (ReduceLROnPlateau) pour une convergence efficace.

## Installation

Ce projet utilise uv pour la gestion des dépendances.

1. Clonez le référentiel :
```bash
git clone [URL_DE_VOTRE_RÉFÉRENTIEL](https://github.com/TOUJAN-Oceam/Kaggle_DL_3A_CentraleSupelec)
cd [NOM_DE_VOTRE_DOSSIER](https://github.com/TOUJAN-Oceam/Kaggle_DL_3A_CentraleSupelec)
```
2. Pour synchroniser le fichier .toml : 
```bash
  uv sync
```
puis exécutez les deux scripts :
```bash  
  uv run Shooting_drill_scripts/Shooting_drill_prediction_TOUJAN_Oceam.py
  uv run gymnastic_scripts/gymnastic_prediction_TOUJAN_Oceam.py
```

## Résultats et analyse : 
Le référentiel comprend une analyse visuelle du processus d'entraînement :

  * Graphiques `Optuna` : des visualisations de l'espace de recherche des hyperparamètres et de l'historique des valeurs objectives sont disponibles dans le dossier graphs/ ou optuna_results/.

  * Courbes de perte : évolution des pertes d'entraînement et de validation pour surveiller le surajustement.

  * Performance : Métrique évaluée :
  * L'exercice de tir correspondra à l'erreur absolue entre votre prédiction et le nombre réel de barres transversales touchées par le joueur, 41 participants, score public : 1,17240 (top 2 du leaderboard), score privé : 1,14519 (top 4 du leaderboard).
  * L'examen de gymnastique correspondra au score de précision, 41 participants, score public : 0,90208 (top 3 du leaderboard), score privé : 0,87142 (top 4 du leaderboard).

