# Installation

- Clone `bandit-demo`, 
- `cd bandit-demo`,
- `uv sync`,
- `source .venv/bin/activate`,
- `gcloud auth application-default login`

EThen, open a python file or notebook and run  :

```python
from src.datasets.factory import get_dataset

pg_dataset = get_dataset("playground_dataset") # < 1 min
tt_dataset = get_dataset("tt_based_dataset") # ~ 6 min
```

Ces deux lignes exécutent toutes les requêtes SQL nécessaires pour avoir l’entièreté des données. `tt_dataset = get_dataset("tt_based_dataset")` Créée aussi une table `lance_db` pour le retrieval, et récupère des données dans un bucket pour fitter le modèle oracle qui associe une probabilité de conversion à un score du Two Tower. Cette dernière tache peut prendre plusieurs minutes.

You're ready to go !

# App

Cette app a pour objectif de tester différentes architectures de bandits pour le système de recommandation.

<br>

# **Pipeline**


```mermaid
graph LR
    %% --- Nodes Definition ---
    Weibull[Weibull]
    DataGen[Data generation]
    PGData[Playground Dataset]
    Dataset[Training Dataset]
    TTData[TT Based Dataset]
    Embeddings[Embeddings]
    TTScore[TT_score --> conversion proba]

    NoContext[No context]
    Context[Context]
    MABNoContext[MAB without context]
    ContextMAB[Context MAB]
    Bandit[Bandit Type]

    ArmSelection[Arm selection]
    Retrieval[Retrieval]
    Recommendation[recommendation]
    Outcome[outcome]
    RewardEval[reward evaluation]
    Feedback[Feedback]

    inv1[ ]:::invisible
    inv2[ ]:::invisible

    %% --- Connections ---
    %% [Indices 0, 1, 2] Purple Inputs
    Weibull --> PGData
    DataGen --> PGData
    PGData --> Dataset

    %% [Indices 3, 4, 5] Green Inputs
    Embeddings --> TTData
    TTScore --> TTData
    TTData --> Dataset

    %% [Indices 6 - 12] Bandits
    Dataset --> NoContext
    Dataset --> Context
    NoContext --> MABNoContext
    Context --> ContextMAB
    MABNoContext --> ArmSelection
    ContextMAB --> ArmSelection
    ArmSelection <--> Bandit

    %% [Indices 13-15] Retrieval Loop
    ArmSelection --> Recommendation
    ArmSelection --> Retrieval
    Retrieval --> Recommendation

    %% [Indices 16, 17] Post-Rec Flow
    Recommendation --> Outcome
    Outcome --> RewardEval

    %% [Indices 18 - 22] Blue Feedback Loop
    RewardEval --> Feedback
    Bandit <--> Feedback
    inv1 ~~~ TTData
    TTData ~~~ inv1
    TTData ~~~ inv1
    inv1 ~~~ Bandit
    Bandit ~~~ Feedback


    %% --- Node Coloring & Styling ---
    classDef purple fill:#2c1b4d,stroke:#a855f7,stroke-width:2px,color:#d8b4fe;
    classDef green fill:#143424,stroke:#22c55e,stroke-width:2px,color:#86efac;
    classDef blue fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#93c5fd;
    classDef defaultNode fill:#1c1c1e,stroke:#444446,stroke-width:2px,color:#ffffff;
    classDef invisible fill:none,stroke:none,color:none;

    class Weibull,DataGen,PGData purple;
    class Embeddings,TTScore,TTData,Retrieval green;
    class Feedback blue;
    class Dataset,NoContext,Context,MABNoContext,ContextMAB,ArmSelection,Recommendation,Outcome,RewardEval,Bandit defaultNode;

    %% --- Arrow Coloring & Styling ---
    %% Purple Arrows (Indices 0, 1, 2)
    linkStyle 0,1,2,13 stroke:#a855f7,stroke-width:2px;

    %% Green Arrows (Indices 3, 4, 5 to Dataset; 13, 14 for Retrieval)
    linkStyle 3,4,5,14,15 stroke:#22c55e,stroke-width:2px;

    %% Blue Feedback Arrows (Indices 17, 18)
    linkStyle 18,19 stroke:#3b82f6,stroke-width:2px;
```

<br>

# 1. Playground Dataset : Générer un dataset cohérent

Dans un premier temps, on veut utiliser un dataset sur lequel on a un contrôle complet, et qui nous permettra de tester des conditions limites de nos algorithmes.
On veut un dataset qui reproduit assez fidèlement les caractéristiques du nôtre, mais facile à manipuler. On veut garder certaines spécificités :
- La plupart des utilisateurs connaissent peu de catégories, sous catégories et venue type.
- Certaines catégories, sous catégories et venue type sont plus populaires que d'autres.
- Les utilisateurs ont tendance à continuer à réserver dans les catégories, sous catégories et venue type qu'ils connaissent déjà.

### On construit un dataset avec ```generate_playground_dataset```, où une recommendation est :
- Une offre : Appartient à une playlist proposée à l'utilisateur, à une catégorie, sous catégories et venue type.
- Un utilisateur : Il a un historique qui correspond à ses réservations passées
- Un outcome, c'est-à-dire le résultat de l'interaction entre l'utilisateur et la reco

|Recommended playlist related data|User related data|Conversion related data|
|:---|:---|:---|
|Playlist reco id <br> Offer id <br>Offer category <br>Offer subcategory <br>Offer venue type <br> Offer venue <br> Offer genre <br>Offer rank in the playlist |User id <br> User history, including : <br> Known categories <br> Known subcategories <br> Known venue types <br>Known venues <br>Known genres | <br> was offer clicked <br>was offer favorited <br> was offer booked |

La **conversion**, calculée dans ```get_conversion``` dépend uniquement de l'historique de l'utilisateur et des caractéristiques de l'offre.

## Distribution de Weibull :

Pour que le dataset généré soit cohérent avec nos données réelles, on veut que les historiques de nos utilisateurs factices suivent les mêmes tendances que ceux des vrais utilisateurs. On fait une analyse des quantités de catégories, sous catégories, venue type et venue ids distinct dans l'historique de nos utilisateurs. <br>
On observe que ces 4 grandeurs suivent des **distributions de Weibull** $$\forall x \geq 0,~ f(x;k, \lambda) = \frac{k}{\lambda} \cdot \left (\frac{x}{\lambda}\right)^{k-1} e^{-\left (\frac{x}{\lambda}\right)^{k}}$$
Avec les paramètres $k$ et $\lambda$ qui varient pour chaque grandeur.

Avec ```generate_user_history```;
**La taille de l'historique des utilisateurs** (le nombre de catégories, sous catégories, venue type et venue ids distinct) suit une loi de Weibull discrétisée. <br>

**Le contenu de l'historique des utilisateurs** suit une loi de probabilités caractérisée par le total des réservations dans chaque catégorie, sous catégories et venue type entre le 01/02/2026 et 01/04/2026.

<img src="documentation/PlaygroundDataset.png" width="100%">

## Training Dataset ```generate_synthetic_training_dataset```

Pour réduire l'action space de nos bandits, on simule des clusters d'offres en associant chaque bras à un triplet (catégorie|sous-catégorie|venue-type). On calcule la conversion en prenant en compte uniquement l'historique de l'utilisateur et ce triplet. <br>
Afin de simuler le processus d'online learning, on s'assure que chaque recommendation possible soit associée à un reward. On a alors un dataset de type :
| User_history | Arm 1 | Arm 2 | ... | Arm x |
|:---:|:---:|:---:|:---:|:---:|
|cats{}, subs{}, loc_types{}| Conversion 1 | Conversion 2 | ... | Conversion x |

<br>

Lors de l'entraînement, le MAB peut utiliser comme contexte l'historique de l'utilisateur, et doit choisir un bras. Il a ensuite à sa disposition la conversion associée au bras qu'il a choisi pour le context donné, directement dans le dataset. L'outcome est associé à un score par le moyen de la fonction de reward, avant d'être converti en feedback pour améliorer notre bandit.

<br>

# 2. TT Based Dataset : S'entraîner sur les logs

### L'**Offline training** sert deux buts : 
- **Le Problème du Bootstrap :** Si on deploie un bandit sans connaissances préalable (initialisation uniforme), il explorera de manière aléatoire dans un premier temps, ce qui aura pour conséquence une expérience utilisateur dégradée.
On veut utiliser les logs historiques (issus de notre précédent système de recommandation) pour initialiser les paramètres du bandit. Au lieu de partir de zéro, le bandit commence avec une « meilleure estimation » de ce que les utilisateurs aiment, entrant ainsi directement dans la phase d'Exploitation tout en affinant ses connaissances grâce à une Exploration contrôlée.

- **L'Evaluation Off-Policy :** C'est notre principale motivation pour l'offline training. On veut pas déployer un bandit sans savoir (ou du moins sans avoir une idée de) la façon dont il va se comporter, ou plus précisément, de la façon dont il se serait comporté s'il avait été déployé.
Puisqu'on ne peut pas « rejouer » l'histoire (on ne peut pas voir ce que l'utilisateur aurait fait si le bandit lui avait présenté un article différent), on doit utiliser des techniques statistiques telles que l'IPS (Inverse Probability Scoring) ou les estimateurs "Doubly Robust". <br>
Pour plus de détail sur ces méthodes, voir [Contextual MAB in recommender systems - Évaluation Hors-Ligne (Off-Policy Evaluation)](https://www.notion.so/passcultureapp/Contextual-MAB-in-recommender-systems-and-adjacent-methods-33bad4e0ff988046afacf3f04cb2d2a1?source=copy_link#34bad4e0ff9880ce8924e1d2a53c759c).

Dans notre cas, on n'a pas suffisamment de données à notre disposition pour s'appuyer sur l'IPS. Pour combler les trous dans la donnée associés à la partie *Exploration* de nos bandits, on va faire appel à la méthode directe : On va exploiter un modèle de régression pour prédire un reward $\hat{r}(x,a)$ pour toute pair (context - action).

Concrètement, on veut :

```mermaid
graph LR
    %% --- Nodes Definition ---

    Context[Contexte]
    Reco[Action]
    Modele[Modèle Oracle]
    Prediction[Prediction]

    %% --- Connections ---

    Context --> Modele
    Reco --> Modele
    Modele --> Prediction

    %% --- Node Coloring & Styling ---
    classDef blue fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#93c5fd;
    classDef defaultNode fill:#1c1c1e,stroke:#444446,stroke-width:2px,color:#ffffff;

    class Modele blue
    class Context,Reco,Prediction defaultNode
```

## Extraction de probabilités de conversion à partir de cosine distance

Concrètement, pour chaque context possible et chaque choix d'action (c'est-à-dire choix de bras) possible, on veut pouvoir estimer une probabilité de conversion (probabilité que la recommendation soit vue, proba qu'elle soit mise en favori et probabilité qu'elle soit réservée). Pour faire cette estimation, on veut si possible s'appuyer sur un modèle déjà existant, afin de ne pas avoir à développer et entraîner un modèle complexe.  <br>
Dans notre cas, un modèle répond à nos attentes : le **Two Tower**. <br>

Dans notre cas, on veut alors :

```mermaid
graph LR
    %% --- Nodes Definition ---

    Context[Contexte]
    Reco[Action : Chosen Arm]
    Retrieval[Retrieval]
    Item[Recommendation]
    Distance[Distance]
    Modele[Probabilités de conversion]
    Reward[Reward]

    %% --- Connections ---

    Context --> Retrieval
    Reco --> Retrieval
    Retrieval --> Item
    Retrieval --> Distance
    subgraph Oracle : fit Two Tower score - Conversion Probabilities
        Distance --> Modele
    end
    Modele --> Reward

    %% --- Node Coloring & Styling ---
    classDef purple fill:#2c1b4d,stroke:#a855f7,stroke-width:2px,color:#d8b4fe;
    classDef blue fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#93c5fd;
    classDef defaultNode fill:#1c1c1e,stroke:#444446,stroke-width:2px,color:#ffffff;

    class Retrieval purple
    class Modele,Distance blue
    class Context,Reco,Item,Reward defaultNode
```

On s'intéresse aux logs de recommendations dont on dispose, au score associé à chaque reco et aux interactions utilisateurs qui en ont découlé.
Le dataset source a été construit avec les données de `past_offer_context_sink` pour les logs de recommandation et de `reco_offer_conversion` pour les événements de conversion. Il contient les colonnes :
|item_id|offer_id|offer_reco_rank|offer_category|context|user_id|conversion|score|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|

<br>

On applique un filtre sur ces données : score <= 1, user_id != "-1", context == "recommendation:user_based", offer_reco_rank < 2
- score <= 1 : Garantit que les cold starts sont exclus.
- user_id != "-1" : Garantit que la route « similar_offer » est exclue.
- context == "recommendation:user_based" : Garantit que les « Top Offres » sont exclues.
- offer_reco_rank < 2 : Ne conserve que les deux premières recommandation de chaque playlist afin de s'assurer que les probabilités de conversion soient calculées sur des recommandations effectivement vues.

<br>

>Il est important de noter pour la suite que le **score** du **Two Tower** est une **distance**, basée sur une métrique de **cosine similarity** évaluée sur des vecteurs **normalisés**.

<br>

On peut maintenant fitter nos données sur des fonctions adaptées. On choisit des exponentielles décroissantes $f(x, k, a, b) = k \cdot e^{-a \cdot x} + b$ (fit pour k, a , b), et on obtient :

<img src="documentation/TTMapping.png" width="100%">

## Context et embeddings

On choisit dans un premier temps comme context les embeddings d'utilisateurs réalisés pour le Two Tower, pour rester dans les conditions les plus simples d'estimation de probabilités de conversion. On aura aussi besoin des embeddings d'items pour pouvoir calculer le score qui nous permettra de déterminer ces probabilités.

## Retrieval

Ce module utilise LanceDB pour récupérer les top-k éléments les plus similaires à l'embedding d'un utilisateur.

* ### Fonctionnement

    - Une table LanceDB est ouverte ou créée via `get_lancedb_table()`.
    - La fonction `retrieve_items()` effectue une recherche vectorielle sur la colonne `item_embedding`.
    - La métrique de recherche est explicitement configurée sur `dot` (produit scalaire).

* ### Sémantique de la métrique

    Le code utilise la métrique `dot` dans LanceDB. Elle est traitée comme une métrique de **distance** et non de similarité. Des valeurs de `_distance` plus faibles correspondent à des vecteurs plus similaires. <br>
    Pour des vecteurs normalisés, le *dot product* est équivalent à la *cosine similarity*, et la distance du produit scalaire correspond donc à la distance cosinus.

## Training Dataset

Le MAB récupère un `user_id`, et un `user_embedding` en guise de context. Il choisit un bras, c'est à dire une catégorie d'offre, sous catégorie et venue type, puis envoie au retrieval le `user_embedding` et le bras choisi. <br>
Le retrieval applique des filtres correspondant au bras choisi, puis récupère l'item qui minimise la distance (`item_embedding`, `user_embedding`) parmi tous les items du catalogue qui correspondent au filtre. <br>
Ensuite, on estime les probabilités de conversion de la recommendation grâce à ce score, et on en déduit un outcome qui est associé à un score par le moyen de la fonction de reward, avant d'être converti en feedback pour améliorer notre bandit.

<br>

# 3. Benchmark des Multi-Armed Bandits

On évalue plusieurs types d'algorithmes :
* Un **Random Bandit**, qui choisit un bras aléatoire à chaque étape -> C'est notre baseline qui servira de comparaison.
* Un (ou plusieurs) **$\epsilon$-greedy** bandit.
* Un (ou plusieurs) **LinUCB**, qui sont des bandits contextuels.
* Un **LinTS**
* Un **GP-TS**

---


# 4. Etude de la fonction de reward

Nos objectifs sont multiples, donc notre fonction de reward doit superviser plusieurs axers d'amélioration. Les fonctions possibles deviennent rapidement complexes, avec de nombreux paramètres a fitter.

* ### Fonction de reward possible, telle que définie dans ```get_reward``` :

    On note $R_{p,u}$ le reward pour une playlist $P$ recommandée à un utilisateur $U$ ayant un historique $H_u$.

    Pour $i \in P$ un item recommandé, on note $r_i$ le rang de $i$ dans $P$.

    On définit alors le reward :

    $$
    R_{p,u} = \sum_{i \in P} \gamma^{r_i} \cdot Conv(u, i) \cdot (1 + \lambda \cdot Div(H_u, i))
    $$

    Où :

    - $\gamma$ est un **facteur de décroissance** pour les items recommandés plus bas dans la playlist ($\gamma < 1$)
    - $\lambda$ est un **paramètre de pondération** pour l'importance de la diversité dans le reward.
    - $Conv(i)$ est une **mesure de la conversion** de l'item $i$ :

        $$
        Conv(u, i) = \frac {\omega_1 \cdot \mathbb{1}_{clic}(u, i) + \omega_2 \cdot \mathbb{1}_{favoris}(u, i) + \omega_3 \cdot \mathbb{1}_{resa}(u, i)}{\omega_1 + \omega_2 + \omega_3}
        $$

    - $Div(H_u, i)$ est une **mesure de la diversité** de l'item $i$ par rapport à l'historique $H_u$ :

        $$
        Div(H_u, i) = \frac{1}{65} \cdot div\_score(H_u, i)
        $$

        ---

        Le **score de diversité** est calculé comme suit :
        Pour chaque réservation, les caractéristiques de diversité observées rapportent des points différents.

        - nouvelle catégorie d’offre : 25 points
        - nouvelle sous-catégorie d’offre : 10 points
        - nouveau type de lieu : 20 points
        - nouveau lieu : 5 points
        - nouveau genre : 5 points

        Chaque utilisateur commence à 0. Une première réservation permet de gagner des points dans chacune des caractéristiques : elle rapporte donc 65 points.
