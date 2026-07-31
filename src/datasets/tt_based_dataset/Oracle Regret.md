## Trade Off Conversion - Diversité : Borne Inférieure de la Perte en Conversion



Dans notre environnement de simulation, le reward final est fonction d’une interaction (outcome) binaire (0 ou 1) suite à un tirage de Bernoulli ($X < p$). Si on évalue le bandit optimal sur le reward *réel* obtenu, les métriques sont extrêmement bruitées à court terme à cause de la variance de ce tirage.

On souhaite mesurer la performance non pas sur le tirage au sort, mais directement sur **l’espérance mathématique** $\mathbb{E}[R]$ des gains.

Pour un utilisateur $u$  et un bras $a$, les probabilités réelles (espérances) d’activation de chaque événement sont :

- **Clics :** $P(\text{clicked} \mid u, a) = f_{\text{clicked}}(x)$
- **Favoris :** $P(\text{faved} \mid u, a) =  f_{\text{faved}}(x)$
- **Réservations :** $P(\text{booked} \mid u, a) = f_{\text{booked}}(x)$

L’espérance de la récompense de conversion pure, $\mathbb{E}[R_{\text{conv}}(u, a)]$, est la somme pondérée de ces probabilités par leurs poids respectifs $\omega$ :

$$
\mathbb{E}[R_{\text{conv}}(u, a)] = \frac{\omega_1 \cdot P(\text{clicked} \mid u, a) + \omega_2 \cdot P(\text{faved} \mid u, a) + \omega_3 \cdot P(\text{booked} \mid u, a)}{\omega_1 + \omega_2 + \omega_3}
$$

En y injectant le multiplicateur de diversité $D(h_u, a)$, la récompense maximale attendue $\mu^*(u)$ par l’Oracle pour un utilisateur $u$ est :

$$
\mu^*(u) = \max_{a \in \mathcal{A}} \left( \mathbb{E}[R_{\text{conv}}(u, a)] \cdot \Big(1 + \beta \cdot D(h_u, a)\Big) \right)
$$

Dans ce cas, **le bras qui maximise la conversion n’est plus nécessairement le bras optimal**. Un bras offrant une probabilité de conversion légèrement plus faible mais apportant une forte diversité exclusive (car ses labels ne sont pas dans $h_u$) peut générer une espérance supérieure.

## Dans le code : Optimisation par Bitmask

Calculer la baseline de l'Oracle implique de chercher quel bras de l'action space $\mathcal{A}$ maximise $\mathbb{E}[R]$. Une approche brute-force demande d'exécuter une requête de plus proche voisin dans LanceDB pour *chaque bras*. Avec $N$ utilisateurs et $M$ bras, la complexité temporelle est en $\mathcal{O}(N \times M)$, ce qui rend les calculs très lourds à grande échelle.

### La simplification par les classes d'équivalence

Le score de diversité $D(h_u, a)$ ne dépend pas de l'item exact, mais uniquement de la présence ou de l'absence des métadonnées du bras $a = (\text{cat}_a, \text{sub}_a, \text{vt}_a)$ dans l'historique de l'utilisateur $h_u = \{C_u, S_u, V_u\}$.

Il n'existe que $2^3 = 8$ profils d'intersections possibles. On peut indexer ces profils par un **masque de bits (bitmask)** sur 3 bits (un entier de 0 à 7) :

Pour un utilisateur, on genere 8 filtres de requêtes possibles sous forme de masques binaires (de `000` à `111`) :

- `Bit 2` (Catégorie) → `IN` ou `NOT IN (h_u.categories)`
- `Bit 1` (Sous-catégorie) → `IN` ou `NOT IN (h_u.subcategories)`
- `Bit 0` (Type de lieu) → `IN` ou `NOT IN (h_u.venue_types)`

Pour tous les bras appartenant à un même bitmask, le multiplicateur de diversité $(1 + \gamma \cdot D(h_u, a))$ est une **constante**. Pour maximiser le reward au sein d'un profil, il suffit de trouver l'item qui maximise $P(\text{clic})$, et au lieu de demander "quel est le meilleur item pour le bras X ?", on demande à LanceDB "quel est le meilleur item parmi tous les bras qui partagent le profil binaire Y ?".

## Pour un échantillon de 2000 utilisateurs :

On évalue les pertes en conversion minimales pour un panel de valeurs de $\beta$, en considerant la conversion comme maximale pour $\beta = 0$.

<div style="display: flex; justify-content: center;">
  <img src="oracle_regret_u2000_b1.png" width="50%">
</div>

<br>

⚠️ Il s’agit ici d’une borne inférieure en terme de pertes de conversion, atteindre cette borne avec les bandits n'est en aucun cas garanti !