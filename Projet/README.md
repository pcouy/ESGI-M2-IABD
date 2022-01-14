# Précisions sur le projet

## À propos du `PrioritizedReplayBufferAgent` (13/01/22)

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

-----

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
