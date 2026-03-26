# Turtlebot MiniGrid (2D, obstacles statiques) avec CleanRL

Ce mini-framework ajoute un environnement Gymnasium de type *minigrid* pour un robot style Turtlebot et permet de lancer facilement les algorithmes CleanRL dessus.

## Fichiers ajoutés

- `cleanrl/turtlebot_minigrid_env.py` : environnement 2D (grille, orientation, obstacles statiques)
- `cleanrl/run_cleanrl_turtlebot.py` : lanceur générique pour exécuter un script CleanRL sur cet env
- `cleanrl/eval_turtlebot_dqn.py` : évaluation d'un modèle DQN sauvegardé

## Espace d'état et actions

- **Actions discrètes** (`Discrete(3)`) :
  - `0` tourner à gauche
  - `1` tourner à droite
  - `2` avancer
- **Observation** (`Box`) :
  - grille aplatie (obstacles, robot, objectif)
  - état compact 
    - position robot `(x, y)` normalisée
    - position objectif `(x, y)` normalisée
    - orientation one-hot (N/E/S/O)

## Récompenses

- pénalité de pas (`-0.01`)
- pénalité collision (`-0.10` en plus)
- bonus de progression vers l'objectif (shaping Manhattan)
- récompense d'objectif (`+1.0`) et fin d'épisode

## Lancer DQN (recommandé)

Depuis la racine du dépôt `cleanrl` :

```bash
python cleanrl/run_cleanrl_turtlebot.py \
  --algo dqn_cbr.py \
  --total-timesteps 200000 \
  --learning-starts 1000 \
  --buffer-size 50000 \
  --batch-size 128 \
  --save-model
```

> `--env-id` est injecté automatiquement (`TurtlebotMiniGrid-StaticObstacles-v0`) si tu ne le fournis pas.

## Évaluer un modèle DQN

```bash
python cleanrl/eval_turtlebot_dqn.py \
  --model-path runs/<run_name>/dqn.cleanrl_model \
  --eval-episodes 20 \
  --capture-video
```

## Lancer un autre algo CleanRL

Exemple PPO (si le script supporte `--env-id` et action discrète) :

```bash
python cleanrl/run_cleanrl_turtlebot.py --algo ppo.py --total-timesteps 200000
```

## Notes

- L'environnement est enregistré à l'import sous l'id : `TurtlebotMiniGrid-StaticObstacles-v0`.
- Le rendu vidéo fonctionne avec `render_mode="rgb_array"` (utile pour `--capture-video`).
- Le cas d'usage principal ici est DQN.
