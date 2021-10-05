# Cours sur le Deep Reinforcement Learning

Vous trouverez dans ce dépôt tous les fichiers nécessaires pour suivre le cours.

Les *notebooks* ne s'affichent pas correctement dans l'apperçu Github. Ouvrez les avec votre installation locale de Jupyter, ou bien utilisez les liens Google Colab qui les accompagnent.

Il est fortement recommandé de cloner le dépôt localement sur votre machine, et de créer une branche pour vos notes et modifications. Ainsi, vous pourrez *merger* vos notes personnelles avec les éventuelles révisions des notebooks, et vos modifications apportées au code avec les ajouts que j'y ferais tout au long du semestre.

Vous devrez aussi installer les modules Python nécessaires pour suivre les exercices pratiques. Une simple commande `pip` suffit. Vous pouvez bien sur utiliser votre système d'environnement virtuel préféré pour isoler l'installation.

```
git clone https://github.com/pcouy/ESGI-M2-IABD
pip install -e ESGI-M2-IABD/code
```

L'argument `-e` passé à `pip` vous permet d'apporter des modifications au contenu du dossier `code` et de les tester sans avoir à réinstaller le module.

Si vous ouvrez les notebooks dans Google Colab, le clonage du dépôt et l'installation du module se font automatiquement dans la première cellule (qui ne fait rien si le notebook est ouvert en dehors de colab).
