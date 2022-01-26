# Précisions sur le projet

## Pistes à  explorer

Dans le cadre de ce projet, vous devez d'une part proposer une implémentation d'un algorithme d'apprentissage profond par renforcement, et d'autre part réaliser des expériences mettant en évidence certaines propriétés de vos modèles.

### Implémentation d'une amélioration des algorithmes fournis

Si vous hésitez sur la pertinence ou la difficulté d'un algorithme que vous envisagez d'implémenter pour votre projet, contactez moi pour en discuter.

#### Exploration

Outre les améliorations intégrées dans *Rainbow*, un axe de recherche intéressant et accessible concerne le problème de l'exploration. En effet, vous avez jusqu'à présent utilisé uniquement une politique *epsilon-greedy* avec diminution de epsilon au cours de l'entrainement. Bien que produisant des résultats satisfaisants, il s'agit d'un méthode très naïve et rudimentaire pour explorer l'espace des politiques.

En effet, le début de l'entrainement est constitué uniquement de *trajectoires* presque totalement aléatoires (epsilon est grand), et les chances de tenter une politique radicalement différente de la politique *greedy* deviennent nulles en fin d'entrainement (epsilon devient petit). Pour illustrer ces problèmes sur le jeu *snake* :

* Vos agents découvrent assez vite comment marquer leurs premiers points en début d'entrainement (cf épisodes de test, où epsilon est fixé à 0). Cependant, tant que epsilon reste élevé, vos épisodes d'entrainement conduisent tous à une fin de partie rapide car l'agent n'utilise pas encore ce qu'il a appris. Ainsi, il continue à "explorer" par des actions aléatoires les états de début de partie (qu'il semble déjà bien connaitre d'après les épisodes de test), alors qu'il y aurait plus de nouvelles connaissances à extraire en exploitant ce qu'on connait déjà sur le début de partie pour atteindre des états plus avancés.
* Lorsque epsilon devient très faible, vos agents eploitent presque tout le temps ce qu'ils pensent être la meilleure action. Comme l'agent n'expérimente plus les autres actions, il n'y a plus d'opportunités pour que les valeurs de ces autres choix d'actions remontent en consatant leur efficacité. Seule la valeur des actions choisies étant mises à jour, la seule façon dont la politique (le choix d'une action en réaction à une observation) de l'agent peut évoluer est lorsque la valeur d'un choix d'action descend suffisamment (après avoir constaté qu'il conduisait à des retours moindres qu'attendus).

Vous pouvez donc grandement accélerer l'apprentissage des agents en implémentant une méthode d'exploration plus intelligente (contrairement à la politique *epsilon-greedy* qui est purement aléatoire sur tous les plans). 

Si vous choisissez cette voie, commencez par expérimenter avec la `SoftmaxSamplingPolicy` fournie dans le module. Cette politique choisit des actions aléatoires avec des probabilités qui dépendent des valeurs estimées pour chaque action. On utilise une variante de la fonction `softmax` pour convertir les valeurs prédites par la fonction de Q-valeur approximée en probabilités de choix : plus la valeur d'une action est élevée, plus sa probabilité d'être choisie est élevée. De plus, cette politique dispose d'un paramètre `epsilon` qui définit l'étalement des probabilités : `epsilon=1` correspond à des probabilités égales entre toutes les actions (donc une politique complètement aléatoire), et `epsilon=0` correspond à une probabilité de 1 pour la meilleure action (donc une politique complètement *greedy*).

Ce paramètre `epsilon` a donc le même sens que pour la politique *epsilon-greedy* lorsqu'il prend les valeurs extrèmes 0 ou 1. En revanche, si `epsilon` prend une valeur intermédiaire, la `SoftmaxSamplingPolicy` permet un choix d'action plus subtil. Par exemple, si face à une observation on estime les valeurs `[-1 , 5.95, 6]` ; alors la première action (celle qui a été évaluée à -1) ne sera presque jamais choisie, et les deux autres actions (qui ont des valeurs largement plus élevées et presque identiques) auront des probabilités de choix d'environ 0.5 chacune (contrairtement à une politique purement *greedy* qui choisirait systématiquement la troisième action qui est estimée à une valeur maximale).

Après avoir constaté l'intérêt d'une politique d'exploration plus riche, vous pourrez implémenter un mécanisme d'exploration de votre choix :

* L'amélioration *Noisy Nets* (qui fait partie de *Rainbow*) a l'avantage d'avoir déjà été introduite au cours de la dernière séance, et son ancienneté relative facilitera probablement la recherche de ressources concernant cet article.
* Les améliorations introduisant une notion de **récompense intrinsèque** ont expérimentalement donné de bons résultats. Cette récompense est dite *intrinsèque* car c'est un mécanisme interne à l'agent qui détermine sa valeur (contrairement à la récompense extrinsèque, qui est la récompense habituelle délivrée par l'environnement). Cette récompense intrinsèque est ajoutée à la récompense extrinsèque avant le traitement de la transition par le mécanisme d'entrainement. Le mécanisme interne qui attribue la récompense intrinsèque est conçu de manière à favoriser l'exploration. Dans le cas tabulaire, on attribue généralement à chaque transition une récompense qui décroit avec le nombre de visites dans l'état résultant de la transition : l'agent va donc naturellement apprendre à essayer d'atteindre des états peu visités (et donc mal connus). On parle alors de mécanisme de **curiosité**. On utilise en *deep reinforcement learning* des mécanismes de curiosité qui ne se basent pas sur un compte de visites (cf limitations du *Q-learning* tabulaire) mais par exemple sur une notion de **surprise** de l'agent. De nombreux articles scientifiques traitent de ce sujet (cf bibliographie ci-dessous)


#### Bibliographie

Pour trouver des articles ayant impacté le domaine du *deep reinforcement learning*, OpenAI propose [une liste d'aticles classés par catégorie](https://spinningup.openai.com/en/latest/spinningup/keypapers.html). Les sections qui vous intéresseront pariculièrement sont ["Deep Q-Learning"](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#a-deep-q-learning) et ["Exploration"](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#exploration).

Les articles scientifiques étant souvent assez pointus en terme de notations mathématiques, vous pouvez préférer lire du code plutôt que des formules. Dans ce cas, ces [implémentations pédagogiques des algorithmes du DQN à *Rainbow*](https://github.com/Curt-Park/rainbow-is-all-you-need) peuvent accompagner vos lectures d'articles. De manière générale, on trouve souvent de nombreuses implémentations des articles les plus importants sur Github. La recherche suivante sur un moteur de recherche compétent vous sera utile (en utilisant le titre de l'article qui vous intéresse) :

```
site:github.com "Curiosity is all you need"
```

### Évaluation et visualisation des modèles

Le module fourni propose des visualisations rudimentaires utilisant `matplotlib` et `tensorboard`. Pour créer vous-même vos propres visualisations, vous pouvez [convertir vos fichiers de logs Tensorboard en *dataframes* Pandas](https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py), ce qui vous permettra ensuite d'utiliser les outils de visualisation de votre choix. Le compte-rendu étant à rendre au format numérique, vous pouvez rendre un *notebook* contenant des visualisations interractives si vous maitrisez les outils nécessaires.

De plus, toutes les classes qui héritent de l'`Agent` de base inclu dans le module disposent de la méthode [`log_data`](https://pcouy.github.io/ESGI-M2-IABD/code_tp/agents/base.html#Agent.log_data) qui permet d'inclure des données supplémentaires dans les logs tensorboard (et les courbes matplotlib générées automatiquement)

La littérature scientifique regorge d'inspirations pour évaluer et visualiser vos agents entrainés. Les expériences les plus intéressantes à réaliser nécessiteront certainement d'instrumenter une partie du code fourni dans le module. Pour cela, vous pouvez partir d'une définition de classe similaire à celle fournie dans l'énoncé du projet :

```python
class IntegratedAgent(Improvement1Agent, Improvement2Agent, ...):
	pass
```

Vous pouvez enrichir la définition de cet agent pour ajouter vos propres méthodes : par exemple, une méthode `record_detailed_episode` pourrait reprendre le code de `run_episode` pour y ajouter l'enregistrement d'informations supplémentaires que vous utiliserez ensuite pour créer vos visualisations. Votre agent ainsi créé disposera de toutes les mêmes interfaces que les agents fournis dans le module + vos méthodes personnalisées.

Vous pouvez également écraser des méthodes existantes. Dans ce cas, veillez bien à ne pas casser de fonctionnalités pré-existantes.

------------

## À propos du `PrioritizedReplayBufferAgent` (13/01/22)

### Erreur sur l'ordre de résolution des méthodes (résolu)

Nous avons rencontré, lors des tests sur le *replay buffer* priorisé, un bug aucours duquel l'agent ne semblait rien apprendre
et où les valeurs prédites (courbe `predicted_values`) atteignaient plusieurs centaines.

Si vous rencontrez ce bug, et que vout faites : `help(VotreClasseDAgentPriorisé)` et comparez la sortie avec celle de
`help(DQNAgent)`, vous constaterez que la méthode `select_action` de votre agent priorisé provient de `QLearningAgent`
alors que le DQN simple utilisait la méthode `select_action` fournie par `ReplayBufferAgent`

**CORRIGÉ** dans le commit [a58021a5eb](https://github.com/pcouy/ESGI-M2-IABD/commit/a58021a5eb67731fbcfe92bc8410938ed2aff80c)

*Ancien Correctif :* En attendant la correction de ce problème dans le module, vous pouvez corriger ce problème en ajoutant le `ReplayBufferAgent`
juste après le `PrioritizedReplayBufferAgent` dans la déclaration de l'héritage de votre agent.

Un correctif prochain permettra d'utiliser `PrioritizedReplayBufferAgent` et `PrioritizedReplayBuffer` de manière identique à
`ReplayBufferAgent` et `ReplayBuffer`

### Hyper-paramètres du *replay buffer* priorisé

Par ailleurs, une autre erreur a été commise sur le module python inclus : les paramètres `alpha` et `beta` du replay buffer
priorisé sont fixés à des valeurs ne correspondant pas à l'expérience que vous devez réaliser.

**CORRIGÉ** dans le commit [63298c3af2]() ([documentation](https://pcouy.github.io/ESGI-M2-IABD/code_tp/agents/prioritized_replay.html#PrioritizedReplayBuffer))

*Ancien Correctif :* En attendant la modification du module Python, vous pouvez utiliser le code suivant entre l'instanciation et
l'entrainement de votre agent pour définir les valeurs de votre choix pour ces hyper-paramètres (prenez les valeurs utilisées
dans la publication comme point de départ) :

```python
# VALABLE UNIQUEMENT SI L'INSTANCE DE VOTRE AGENT S'APPELLE `a`
a.replay_buffer.a = *** # Valeur de alpha
a.replay_buffer.beta = *** # Valeur de beta
a.replay_buffer.beta_increment_per_sampling = 0 # On met 0 car on ne fait pas varier beta durant l'entrainement
```

Un correctif prochain permettra de définir ces paramètres dans le constructeur de `PrioritizedReplayBuffer` (et donc de sauvegarder
leurs valeurs dans `infos.json`) et introduira également le paramètre `alpha_decrement_per_sampling` qui permettra de faire décroitre
le paramètre `alpha` au cours d'un entrainement.
