# Mistral Assistant - Système de Fine-tuning et d'Interface

Un système complet pour fine-tuner le modèle Mistral 7B et l'utiliser via une interface interactive. Ce projet permet d'adapter le modèle à n'importe quel domaine spécifique et inclut un exemple d'application pour l'e-commerce.

## Aperçu du Projet

Ce projet comprend deux composants principaux :
1. Un script de fine-tuning permettant d'adapter le modèle Mistral à n'importe quel domaine ou cas d'usage
2. Une interface de chat qui utilise le modèle personnalisé avec un système de raisonnement avancé

L'exemple fourni montre une application pour l'e-commerce, mais la méthodologie peut être adaptée à divers secteurs comme le support technique, la santé, l'éducation, etc.

## Prérequis

- Python 3.8+
- PyTorch
- CUDA compatible GPU (fortement recommandé)
- 16+ GB de RAM
- 10+ GB d'espace disque pour le modèle

## Dépendances Principales

```
torch
transformers
peft
colorama
numpy
pandas
datasets
```

## Installation

1. Clonez ce dépôt
   ```bash
   git clone [URL_DU_REPO]
   cd [NOM_DU_REPO]
   ```

2. Installez les dépendances
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

### 1. Fine-tuning du Modèle

Le script `fine_tune.py` permet d'adapter le modèle Mistral à votre cas d'usage spécifique.

```bash
python fine_tune.py
```

#### Format des Données d'Entraînement

Votre CSV d'entraînement doit contenir au minimum deux colonnes :
- `instruction` : requête/instruction pour le modèle
- `response` : réponse désirée

#### Paramètres Ajustables

- `MODEL_NAME` : modèle de base (par défaut "mistralai/Mistral-7B-Instruct-v0.3")
- `OUTPUT_DIR` : dossier de sauvegarde du modèle fine-tuné
- `MICRO_BATCH_SIZE` : taille de batch (ajuster selon votre mémoire GPU)
- `EPOCHS` : nombre d'époques d'entraînement
- `LEARNING_RATE` : taux d'apprentissage
- `LORA_R` et autres paramètres LoRA pour l'efficacité du fine-tuning

### 2. Interface de Chat

Après le fine-tuning, vous pouvez interagir avec votre modèle via l'interface de chat.

```bash
python chat_interface.py
```

#### Personnalisation de l'Interface

Pour adapter l'interface à votre domaine spécifique, modifiez :
- Le dictionnaire `CATEGORIES` pour refléter les catégories pertinentes à votre cas d'usage
- Les `RESPONSE_TEMPLATES` pour des réponses préconfigurées
- Le système de raisonnement dans la classe `AdvancedReasoningSystem`

#### Commandes de l'Interface

- `aide` : affiche les commandes disponibles
- `quitter` : termine la conversation
- `effacer` : réinitialise l'historique de conversation
- `debug` : active/désactive l'affichage du raisonnement interne

## Architecture Technique

### Fine-tuning avec QLoRA

Le système utilise QLoRA (Quantized Low-Rank Adaptation) pour un fine-tuning efficace avec une empreinte mémoire réduite :
- Quantification 4-bit du modèle de base
- Adaptation des paramètres via matrices de faible rang
- Optimisation pour l'entraînement sur GPU avec mémoire limitée

### Système de Raisonnement

L'interface inclut un système de raisonnement modulable :
1. **Catégorisation** des requêtes utilisateur
2. **Analyse contextuelle** de la conversation
3. **Extraction d'entités** pertinentes
4. **Détermination de l'approche** de réponse
5. **Vérification de cohérence**

Ce système peut être adapté à votre domaine spécifique en modifiant les catégories et les règles de raisonnement.

## Adaptation à Différents Domaines

Le système peut être adapté à divers domaines en :
1. Préparant un dataset d'entraînement spécifique à votre domaine
2. Ajustant les catégories et mots-clés dans `CATEGORIES`
3. Modifiant les vecteurs thématiques dans `THEME_VECTORS`
4. Adaptant les règles de raisonnement selon les besoins de votre domaine

## Dépannage

- **Erreur CUDA out of memory** : Réduisez `MICRO_BATCH_SIZE` ou utilisez un appareil avec plus de VRAM
- **Problèmes de lecture CSV** : Le script inclut des fonctions de réparation pour les fichiers CSV problématiques
- **Performance lente** : Assurez-vous d'utiliser un GPU compatible CUDA

## Structure du Projet

```
.
├── fine_tune.py           # Script de fine-tuning générique
├── chat_interface.py      # Interface de chat avec système de raisonnement
├── requirements.txt       # Dépendances du projet
├── mistral-finetuned/     # Dossier pour le modèle fine-tuné
└── README.md              # Documentation
```

---

*Ce projet utilise le modèle Mistral-7B-Instruct-v0.3 comme base, en permettant son adaptation à divers domaines et cas d'usage.*
