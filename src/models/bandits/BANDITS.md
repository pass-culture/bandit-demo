# Bandits


## 1. Random Bandit

Sélectionne aléatoirement un bras à chaque étape. Ce bandit n’apprend pas, il sert de baseline (c’est-à-dire qu’il répond à la question : *que se passerait-il si on choisissait l’output de manière purement aléatoire ?*).

---

## 2. $\epsilon$-greedy Bandit et Decaying $\epsilon$-greedy

- **Explore** à une fréquence $\epsilon$ → Le bras est choisi aléatoirement.
- **Exploite** à une fréquence $1-\epsilon$ → Choisit le bras ayant la récompense (*reward*) moyenne la plus haute.

$\epsilon \in [0,1]$ est le paramètre d’exploration à ajuster :

- Si $\epsilon \approx 1$ : Le bandit explore en continu et ne profite pas de ce qu’il apprend.
- Si $\epsilon \approx 0$ : Le bandit exploite en continu et n’apprend jamais.

---

## 3. LinUCB : Linear Upper Confidence Bound

### UCB : Upper Confidence Bound

!image.png

**Upper Confidence Bound**

L’idée derrière l’UCB est “l’optimisme face à l’incertitude” : il essaie d’exploiter les bras qui ont un reward moyen faible, et qui ont été peu exploités par le passé. Le moins on connait un bras ($N(a)$ faible, bras peu sélectionné), plus on a envie de l’explorer.  Ici, l’algorithme choisira le Bras 2 : même si son reward moyen estimé est inférieur au Bras 1, la zone d’incertitude rend le Bras 2 potentiellement meilleur.

### LinUCB : La version contextuelle
    
Un LinUCB utilise des informations supplémentaires (**le contexte**) pour prendre de meilleures décisions.

L’idée centrale est de supposer que la récompense attendue pour un bras donné est une **combinaison linéaire** des caractéristiques du contexte. Pour chaque bras $a$, l’algorithme cherche à estimer un vecteur de poids inconnus $\theta_a^*$. La relation s’exprime ainsi :

$$
\mathbb{E}[r_{t,a} \mid X_{t,a}] = X_{t,a}^T \theta_a^*
$$

- $X_{t,a} \in \mathbb{R}^{d \times 1}$ représente les caractéristiques (le contexte) à l’instant $t$.
- $\theta_a^* \in \mathbb{R}^{d \times 1}$ représente le “profil” idéal du bras que l’on cherche à estimer.

L’**output final** correspond à l’index du bras qui maximise le score de compromis entre **Exploitation** et **Exploration** :

$$
a_t = \arg\max_{a} (\underbrace{\hat{\theta}_a^T X_t}_{\text{Exploitation}} + \underbrace{\alpha \sqrt{X_t^T A_a^{-1} X_t}}_{\text{Exploration}})
$$

1. **La partie Exploitation ($\hat{\theta}_a^T X_t$)** calcule le produit scalaire entre le contexte à l’instant $t$ et l’expérience passée du bras. C’est la récompense moyenne attendue.
2. **La partie Exploration ($\alpha \sqrt{X_t^T A_a^{-1} X_t}$)** détermine la variance (l’incertitude) de cette prédiction.
3. L’algorithme est **optimiste face à l’incertitude** : il choisit le bras qui a soit la meilleure espérance de gain, soit la plus grande zone d’ombre à éclaircir.

> **Note sur le paramètre $\alpha$ :** Tout comme le paramètre $\epsilon$ precedemment, il sert de curseur pour ajuster l’équilibre exloitation-exploration. Plus $\alpha$ est grand, plus l’algorithme valorise l’incertitude et plus il est prêt à explorer. S’il est petit, l’algorithme devient “fait confiance à ses estimations passées.
> 

---

### Composants : Rôles et Évolutions

#### A. À quoi sert la matrice $A$ ? Comment évolue-t-elle et pourquoi ?

- **Son rôle :** La matrice $A_a$ (de taille $d \times d$) représente la matrice de covariance des contextes observés pour le bras $a$. Elle sert à mesurer le niveau de certitude accumulée dans chaque direction de l’espace contextuel.
- **Son évolution :** Elle est initialisée à la matrice identité $A_a = I_d$ (ce qui correspond à une régularisation de type Ridge / L2 pour éviter les divisions par zéro au départ). À chaque fois que le bras $a$ est choisi, on lui ajoute le produit externe du contexte actuel : $A_a \leftarrow A_a + X_t X_t^T$.
- **Pourquoi ?** En accumulant $X_t X_t^T$, les valeurs sur la diagonale de $A_a$ grandissent là où les variables du contexte ont été fréquemment observées. Lors du calcul du score UCB, on utilise son inverse $A_a^{-1}$. Plus on observe un contexte similaire, plus les valeurs correspondantes dans $A_a^{-1}$ deviennent petites, ce qui réduit le terme d’exploration $\alpha \sqrt{X_t^T A_a^{-1} X_t}$. À l’inverse, si un contexte pointe vers une direction encore jamais explorée, la valeur dans $A_a^{-1}$ reste grande, ce qui maximise l’incertitude et forçant l’exploration.

#### B. À quoi sert le vecteur $b$ ?

- **Son rôle :** Le vecteur $b_a$ (de dimension $d \times 1$) est un accumulateur des contextes observés, pondérés par les récompenses **réelles** reçues.
- **Son évolution :** Initialisé à zéro, il est mis à jour uniquement pour le bras choisi via : $b_a \leftarrow b_a + r_t X_t$.
- **Pourquoi ?** Il fait le lien entre la direction du contexte et le succès obtenu ($r_t$). Si un contexte $X_t$ mène à une forte récompense positive, les poids dans $b_a$ vont augmenter dans cette direction. S’il mène à un échec ($r_t = 0$), le vecteur $b_a$ ne bouge pas.

#### C. À quoi sert le vecteur $\theta$ ?

- **Son rôle :** Le vecteur $\hat{\theta}_a$ (de dimension $d \times 1$) représente l’estimation actuelle des coefficients de la régression linéaire pour le bras $a$. C’est le “cerveau” ou le profil prédictif du bras.
- **Comment il est calculé :** Il est façonné en combinant l’historique des contextes et des récompenses : $\hat{\theta}_a = A_a^{-1} b_a$. Il extrait les tendances de performance du bras à partir des données accumulées.

#### **Algorithme LinUCB Disjoint (Li et al., 2010)**

Voici le déroulement précis de l’algorithme pour chaque itération $t$ :

1. **Initialisation** :
Pour chaque bras $a \in \mathcal{A}$ :
    - $A_a \leftarrow I_d$ (Matrice identité de dimension $d \times d$)
    - $b_a \leftarrow 0_{d \times 1}$ (Vecteur nul de dimension $d \times 1$)
2. **À chaque étape $t = 1, 2, 3, \dots$** :
    - Observer le contexte courant $X_t$ (pour tous les bras).
    - Pour chaque bras $a \in \mathcal{A}$ :
        - Calculer l’estimateur des moindres carrés : $\hat{\theta}_a = A_a^{-1} b_a$
        - Calculer le score UCB : $p_{t,a} = \hat{\theta}_a^T X_t + \alpha \sqrt{X_t^T A_a^{-1} X_t}$
    - Sélectionner le bras $a_t = \arg\max_{a} p_{t,a}$
    - Observer la récompense réelle $r_t$.
    - Mettre à jour les variables du bras sélectionné $a_t$ uniquement :
        - $A_{a_t} \leftarrow A_{a_t} + X_t X_t^T$
        - $b_{a_t} \leftarrow b_{a_t} + r_t X_t$

---

### LinUCB dans le code : Optimisation par Sherman-Morrison

Dans la théorie, on doit à chaque étape calculer $A_a^{-1}$. Or, inverser une matrice de taille $(d, d)$ à chaque itération coûte énormément en calcul (complexité en $O(d^3)$).

Dans la méthode `update_by_idx`, on n’inverse jamais de matrice.

| **Code** | **Formule de Sherman-Morrison** |
| --- | --- |
| `A_inv_x = inv @ x`  `denom = 1.0 + (x.T @ A_inv_x)`  `self.A_inv[arm_idx] -= (A_inv_x @ A_inv_x.T) / denom` | La **formule de Sherman-Morrison** permet de calculer l’inverse d’une matrice après lui avoir ajouté un nouveau vecteur ($A_{new} = A_{old} + x x^T$) en utilisant seulement l’inverse de l’étape précédente. |

On passe d’une complexité cubique $O(d^3)$ à une complexité quadratique $O(d^2)$.

---

## 4. LinTS : Linear Thompson Sampling

- Thompson Sampling (TS) & Approche Bayésienne

**Échantillonnage de Thompson Linéaire**

Contrairement à la philosophie “optimiste face à l’incertitude” du LinUCB, le **Thompson Sampling** adopte une **approche bayésienne**. Au lieu de calculer une borne supérieure rigide, il maintient une distribution de probabilité a posteriori sur les paramètres de chaque bras.

À chaque étape, l’algorithme tire aléatoirement un scénario (un vecteur de paramètres) depuis cette distribution. L’exploration est ainsi dictée par le hasard probabiliste : plus un bras est incertain, plus sa distribution a posteriori est large, et plus il a de chances de générer un échantillon élevé par pur hasard.

- LinTS : La version contextuelle

Tout comme LinUCB, LinTS modélise la récompense attendue sous forme d’une combinaison linéaire du contexte :

$$
\mathbb{E}[r_{t,a} \mid X_{t,a}] = X_{t,a}^T \theta_a^*
$$

La différence réside dans le fait que $\theta_a^*$ est vu ici comme une **variable aléatoire** suivant une distribution gaussienne multivariée :

$$
\theta_a \sim \mathcal{N}\left(\hat{\theta}_a, \, v^2 B_a^{-1}\right)
$$

L’**output final** est déterminé en échantillonnant d’abord un vecteur $\tilde{\theta}_{t,a}$ pour chaque bras, puis en choisissant le bras qui maximise la récompense espérée sous ce scénario échantillonné :

$$
\tilde{\theta}_{t,a} \sim \mathcal{N}\left(\hat{\theta}_{a}, \, v^2 B_a^{-1}\right)
$$

$$
a_t = \arg\max_{a} \left( X_t^T \tilde{\theta}_{t,a} \right)
$$

1. **La moyenne a posteriori ($\hat{\theta}_a$)** représente l’estimation de notre maximum a posteriori (MAP) ou des moindres carrés régularisés.
2. **La matrice d’information ($B_a$)** régit la forme de l’ellipsoïde d’incertitude. Le paramètre de variance $v^2$ (noté `v2` dans le code) pilote l’amplitude globale de l’exploration.
3. **Le compromis Exploration/Exploitation** est intrinsèque : si un bras a été peu exploré dans la direction de $X_t$, la variance de la récompense prédite sera grande, ce qui permettra de générer des valeurs de $\tilde{\theta}$ très optimistes et incitera naturellement à l’exploration.

> **Note sur le paramètre $v^2$ :** Cet hyperparamètre joue le rôle du thermostat d’exploration. En théorie (Agrawal & Goyal, 2013), on peut definir $v^2$ de manière à avoir des bornes de regret optimales, mais en pratique il est traité comme un paramètre de régulation. Un $v^2$ élevé étale les gaussiennes, forçant une exploration agressive, tandis qu’un $v^2 \to 0$ rend l’algorithme purement glouton (*greedy*).
> 

---

### Composants : Rôles et Évolutions

---

#### **Le modèle théorique de départ**

Pour un bras donné $a$, on observe un historique de $t$ choix. Notons :

- $X$ la matrice des contextes observés pour ce bras (de taille $t \times d$).
- $Y$ le vecteur des récompenses reçues (de taille $t \times 1$).

On suppose que la vraie relation est $Y = X\theta_a^* + \epsilon$, où le bruit suit une loi normale $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$.

On choisit une distribution *a priori* (prior) elle-même gaussienne sur les paramètres :

$$
\theta_a \sim \mathcal{N}\left(0, \, \sigma_0^2 I_d\right)
$$

---

#### **Loi de Bayes**

La distribution *a posteriori* est proportionnelle au produit de la vraisemblance et du prior :

$$
P(\theta_a \mid X, Y) \propto P(Y \mid X, \theta_a) \cdot P(\theta_a)
$$

En passant aux logarithmes (ce qui revient à maximiser l’exposant de la loi normale), la densité *a posteriori* s’écrit sous la forme d’un problème des moindres carrés pénalisés (Régression Ridge) :

$$
\log P(\theta_a \mid X, Y) = -\frac{1}{2\sigma^2} \|Y - X\theta_a\|^2 - \frac{1}{2\sigma_0^2} \|\theta_a\|^2 + \text{constante}
$$

---

#### **L’équation théorique**

En développant et en regroupant les termes quadratiques en $\theta_a$, on cherche à identifier la distribution sous la forme d’une nouvelle loi normale $\mathcal{N}(\hat{\theta}_a, \Sigma_a)$. Par identification, on obtient les deux piliers de votre algorithme :

**Étape A : Calcul de la matrice de précision ($B_a$)**

On définit d’abord la matrice de précision (l’inverse de la covariance) :

$$
B_a = \frac{1}{\sigma_0^2} I_d + \frac{1}{\sigma^2} X^T X
$$

*(En pratique dans LinTS, on simplifie souvent en posant le bruit $\sigma^2 = 1$ et $\frac{1}{\sigma_0^2} = \lambda$, ce qui donne votre formule : $B_a = \lambda I_d + \sum X_t X_t^T$).*

- **Son rôle :** La matrice $B_a$ (de taille $d \times d$) est la matrice de précision (l’inverse de la matrice de covariance, à un facteur d’échelle $v^2$ près). Elle capture la quantité d’information contextuelle accumulée par le bras $a$.
- **Son évolution :** Initialisée avec une régularisation de Ridge $B_a = \lambda I_d$, elle est incrémentée par : $B_a \leftarrow B_a + X_t X_t^T$.
- **Pourquoi ?** Plus on observe un contexte dans une direction donnée, plus les valeurs propres de $B_a$ augmentent dans cette direction. Par conséquent, lors du calcul de la covariance de notre échantillonnage ($B_a^{-1}$), la variance s’écrase. Les tirages de $\tilde{\theta}$ deviennent de plus en plus déterministes autour de la moyenne $\hat{\theta}_a$.

**Étape B : Calcul du vecteur d’information ($f_a$)**

On accumule les récompenses pondérées par les contextes :

$$
f_a = \frac{1}{\sigma^2} X^T Y = \sum_{t} r_t X_t
$$

- **Son rôle :** Le vecteur $f_a$ (de dimension $d \times 1$) est le pendant du vecteur $b_a$ de LinUCB. Il accumule les caractéristiques des contextes observés, modulées de manière linéaire par le signal de récompense $r_t$ : $f_a \leftarrow f_a + r_t X_t$.

**Étape C : Résolution de la moyenne théorique ($\hat{\theta}_a$)**

Le vecteur $\hat{\theta}_a$ est le sommet de cette cloche de Gauss (le mode et la moyenne confondus). Il est la solution unique du système linéaire :

$$
B_a \hat{\theta}_a = f_a
$$

Ce qui nous donne la formule théorique exacte :

$$
\hat{\theta}_a = B_a^{-1} f_a
$$

- **Son rôle :** Le vecteur $\hat{\theta}_a$ (de dimension $d \times 1$) stocke la moyenne conditionnelle *a posteriori* (la distribution centrale).

---

### LinTS dans le code :

Le code implémente deux choix techniques à des fins d’optimisation :

#### 1. L’échantillonnage par Décomposition de Cholesky et Résolution Triangulaire

Pour échantillonner un vecteur aléatoire selon une distribution $\mathcal{N}(\hat{\theta}, \Sigma)$ avec $\Sigma = v^2 B^{-1}$, on utilise la transformation linéaire d’un bruit blanc gaussien standard $Z \sim \mathcal{N}(0, I_d)$.

Le code calcule la factorisation de Cholesky de l’inverse de la matrice de précision :

$$
\text{Si } L L^T = B, \text{ alors } \tilde{\theta} = \hat{\theta} + v \cdot L^{-1} Z
$$

Cette méthode garantit que le vecteur échantillonné possède exactement la structure de corrélation requise par l’historique du bras, tout en limitant le coût computationnel à une décomposition par itération ($O(d^3)$).

Pour chaque bras, on se contente de resoudre une équation linéaire n’impliquant qu’une matrice triangulaire et deux vecteurs ($O(d^2)$)

$$
L X = Z \implies X = L^{-1}Z
$$

La matrice de covariance de $X$ vaut alors

$$
\mathbb{E}[X X^T] = L^{-1}(L^{-1})^T = (L^T L)^{-1} = B^{-1}
$$

#### 2. Robustesse face aux matrices mal conditionnées (Fallback Ridge)

En cours d’apprentissage, si certains contextes sont fortement colinéaires ou si certaines dimensions ne sont jamais explorées, la matrice $B_a$ peut devenir numériquement singulière (déterminant proche de $0$). L’inversion et la décomposition de Cholesky lèveraient alors une exception (`LinAlgError`).

Le code intègre un mécanisme de sécurité (“Emergency fallback”) en appliquant une perturbation de Ridge locale :

$$
B_{\text{reg}} = B_a + 10^{-6} \cdot I_d
$$

Cette légère augmentation artificielle de la diagonale garantit que la matrice redevient définie positive, assurant la stabilité de l’inversion et empêchant le crash du système en production.

---

## 5. GP-TS : Gaussian Process Thompson Sampling

- Processus Gaussiens & Flexibilité Non-Paramétrique

Contrairement à **LinTS** qui restreint la relation contexte-récompense à une projection strictement linéaire ($X_t^T \theta$), le **GP-TS** s’affranchit de toute hypothèse paramétrique rigide. Il utilise un **Processus Gaussien (GP)** pour modéliser le paysage des récompenses sur l’espace conjoint des contextes et des bras.

L’apprentissage repose sur la notion de **similarité spatiale** (via un noyau ou *kernel*). Si deux configurations [contexte, bras] sont proches dans cet espace latent, l’algorithme suppose que leurs récompenses seront fortement corrélées. L’incertitude (variance) s’accroît naturellement dès que l’on s’éloigne des zones déjà observées.

> Note structurelle : Si le GP est flexible face à la forme de la fonction, il reste dépendant de la bonne spécification de ses hyperparamètres (comme la longueur d’échelle length-scale). Un noyau mal calibré brisera sa capacité de généralisation.
> 
- Le Formalisme Mathématique

On définit le vecteur de caractéristiques conjointes $x = [x_{\text{contexte}}, x_{\text{bras}}]$. La fonction de récompense inconnue $f(x)$ est modélisée comme une réalisation d’un Processus Gaussien :

$$
f(x) \sim \mathcal{GP}\left(m(x), k(x, x')\right)
$$

- $m(x)$ est la fonction de moyenne *a priori*, fixée ici à $0$ après normalisation des données (`normalize_y=True`).
- $k(x, x')$ est la fonction de **noyau RBF (Radial Basis Function)**, définie par :

$$
k(x, x') = \sigma_f^2 \exp\left( -\frac{\|x - x'\|^2}{2 \ell^2} \right)
$$

Où $\sigma_f^2$ représente la variance du signal et $\ell$ est la métrique d’échelle (*length-scale*), dictant le rayon d’influence et de lissage de chaque observation.

Après avoir collecté un historique de $t$ interactions $\mathcal{D}_t = \{(x_1, y_1), \dots, (x_t, y_t)\}$, la distribution *a posteriori* en un nouveau point candidat $x^*$ reste une loi gaussienne univariée : $\mathcal{N}\left(\mu_t(x^*), \sigma_t^2(x^*)\right)$.

Les équations d’inférence s’appuient sur l’algèbre des matrices de corrélation :

$$
\mu_t(x^*) = \mathbf{k}_*^T \left(\mathbf{K} + \sigma_n^2 \mathbf{I}\right)^{-1} \mathbf{y}
$$

$$
\sigma_t^2(x^*) = k(x^*, x^*) - \mathbf{k}_*^T \left(\mathbf{K} + \sigma_n^2 \mathbf{I}\right)^{-1} \mathbf{k}_*
$$

- $\mathbf{K} \in \mathbb{R}^{t \times t}$ est la matrice de Gram historique où $[\mathbf{K}]_{i,j} = k(x_i, x_j)$.
- $\mathbf{k}_* = [k(x_1, x^*), \dots, k(x_t, x^*)]^T \in \mathbb{R}^t$ mesure la covariance entre le candidat et le passé.
- $\mathbf{y} = [y_1, \dots, y_t]^T$ est le vecteur des récompenses observées.
- $\sigma_n^2$ représente la **variance du bruit blanc gaussien d’observation** (paramètre `alpha` du `GaussianProcessRegressor`), indispensable pour régulariser l’inversion de matrice et modéliser le bruit du signal.
- La Politique de Choix

À l’instant $t+1$, pour un contexte donné et un ensemble de bras disponibles, l’algorithme construit le vecteur conjoint $x_a$ pour chaque bras $a$. Le mécanisme de Thompson Sampling effectue un tirage probabiliste (Heuristic TS) dans la distribution *a posteriori* calculée par le GP :

$$
\tilde{f}(x_a) \sim \mathcal{N}\left(\mu_t(x_a), \,\alpha^2 \cdot \sigma_t^2(x_a)\right)
$$

$$
a_t = \arg\max_{a} \tilde{f}(x_a)
$$

> Le paramètre multiplicatif $\alpha$ (facteur d’exploration) permet d’ajuster manuellement le comportement du bandit : un $\alpha > 1$ force une sur-exploration agressive, tandis qu’un $\alpha < 1$ privilégie l’exploitation des connaissances actuelles du GP.
> 

---

### GPTS Dans le code

L’inversion de la matrice de Gram $(\mathbf{K} + \sigma^2 \mathbf{I})^{-1}$ présente une complexité cubique en $\mathcal{O}(t^3)$ et spatiale en $\mathcal{O}(t^2)$. À l’échelle de la Data Science industrielle ($100\,000$ itérations et plus), un calcul exact à chaque étape provoquerait une explosion des temps de calcul.

La classe `ContextualGPTS` implémente trois techniques d’approximation pour contourner cette barrière computationnelle.

#### 1. Fréquence d’Entraînement Adaptative ($K$Schedules)

L’ajustement dynamique de l’intervalle d’update du bandit (méthode `_get_adaptive_k`) module la fréquence d’appel à `gp.fit()` selon la maturité du bandit :

```python
def _get_adaptive_k(self) -> int:
    if self.total_steps < 500:   return 5    # Phase d'incertitude critique
    if self.total_steps < 2000:  return 50   # Consolidation
    return 250                               # Régime de croisière
```

Le coût de calcul n’est pas réduit lors des phases d’inférence (`gp.predict`), mais le coût de calcul lié à la ré-optimisation interne des hyperparamètres du noyau (L-BFGS-B interne) est drastiquement amorti.

#### 2. Élagage de la Mémoire par Fenêtre Glissante (Sliding-Window Memory)

Le coût d’entraînement du GP est strictement borné en fixant une capacité maximale au buffer (`max_memory = 2000`). Cette contrainte évite la dérive cubique infinie en plafonnant le coût opérationnel maximal à une constante déterministe de $\mathcal{O}(N_{\text{max}}^3)$.

> L’abandon des données anciennes implique l’abandon de la certitude associée. Dans les zones géométriques purgées de la mémoire, la variance globale $\sigma_t^2(x)$ remonte naturellement vers sa valeur *a priori* $\sigma_f^2$. L’algorithme s’expose à des phénomènes de **re-exploration cyclique**, revenant tester des configurations lointaines qu’il avait pourtant déjà validées comme sous-optimales des milliers d’étapes auparavant.
> 

#### 3. Préservation des Jalons à Haute Valeur (Heuristic Core Preservation)

Pour pallier les défauts d’un élagage purement temporel FIFO, le code applique une stratégie de partitionnement asymétrique de la mémoire (80% de fraîcheur temporelle, 20% d’ancres historiques) dès que la taille du buffer dépasse le seuil :

```python
cutoff = int(self.max_memory * 0.8)
recent_X = self.X_buffer[-cutoff:] # 80% de fraîcheur temporelle

# Tri et conservation des 20% des meilleurs succès historiques
historical_X = self.X_buffer[:-cutoff]
top_indices = np.argsort(historical_y)[-(self.max_memory - cutoff):]
```

En sauvegardant de manière permanente les configurations ayant généré les plus fortes récompenses, le modèle conserve des points d’ancrage mathématiques. Cette sélection crée un biais de confirmation positif.

> Le GP tend à devenir structurellement optimiste, mémorisant ses victoires et oubliant ses échecs. Si l’environnement est non stationnaire (les performances d’un bras chutent drastiquement), le bandit mettra plus de temps à s’en rendre compte, car il conserve la mémoire de son succès passé au sein de ses ancres.
> 

---

## 6. Sparse GP-TS : Approximations par Inducing Points

### La Barrière Computationnelle des Grands Volumes

Bien que les stratégies de fenêtre glissante (*Sliding-Window*) ou de K-schedules permettent de limiter l’impact de la complexité en $\mathcal{O}(t^3)$ du Processus Gaussien, elles forcent un compromis douloureux : oublier le passé ou ralentir la réactivité du modèle.

Le **Sparse GP-TS** résout ce problème en figeant définitivement la dimension des calculs matriciels. Au lieu de laisser la matrice de Gram grandir avec le nombre d’interactions $t$, on projette le flux continu de données sur un ensemble fini de $M$ **points induisants** (ou *inducing points*), agissant comme des capteurs ou des ancres spatiales permanentes. Les calculs d’inversion sont ainsi réduits à une complexité stricte de $\mathcal{O}(M^3)$ à chaque étape, indépendamment de $t$.

> **Super explication sur ce blog** : [https://ludwigwinkler.github.io/blog/InducingPoints/](https://ludwigwinkler.github.io/blog/InducingPoints/)
> 

---

### Formalisme Mathématique de l'Approximation Sparse (Subset of Regressors)

Soit $\mathbf{U} = [u_1, \dots, u_M]^T \in \mathbb{R}^{M \times d}$ la matrice contenant les $M$ points induisants choisis dans l'espace conjoint des contextes et des bras.

Plutôt que de calculer la matrice de covariance complète $\mathbf{K}_{t \times t}$, l'approche par points induisants (cadre variationnel ou SoR) approxime la fonction de covariance en postulant que toute l'information transite par ces points de passage $\mathbf{U}$.

La mise à jour de la distribution se fait via l'espace de précision (formes duales). On définit la matrice de précision accumulée $\mathbf{A} \in \mathbb{R}^{M \times M}$ et le vecteur d'information $\mathbf{b} \in \mathbb{R}^M$ :

$$\mathbf{A}_t = \frac{1}{\sigma_n^2} \sum_{i=1}^t \mathbf{k}_{\mathbf{U}}(x_i)\mathbf{k}_{\mathbf{U}}(x_i)^T$$

$$\mathbf{b}_t = \frac{1}{\sigma_n^2} \sum_{i=1}^t y_i \mathbf{k}_{\mathbf{U}}(x_i)$$

Où $\mathbf{k}_{\mathbf{U}}(x_i) = [k(u_1, x_i), \dots, k(u_M, x_i)]^T \in \mathbb{R}^M$ est le vecteur de covariance entre le point observé $x_i$ et les ancres de la mémoire sparse.

#### Inférence Prédictive Linéarisée

Lorsqu'un nouveau candidat $x^*$ se présente, la moyenne et la variance se calculent par des factorisations de Cholesky directes sur les matrices de taille $M \times M$ :

$$\mu_t(x^*) = \mathbf{k}_{\mathbf{U}}(x^*)^T \left(\mathbf{K}_{\mathbf{U}\mathbf{U}} + \mathbf{A}_t\right)^{-1} \mathbf{b}_t$$

$$\sigma_t^2(x^*) = \underbrace{k_U(x^*)^TK_{UU}^{-1}k_U(x^*)}_{\text{variance a priori $k(x^*, x^*)$ restreinte à l'espace $U$}} - \mathbf{k}_{\mathbf{U}}(x^*)^T \left( \mathbf{K}_{\mathbf{U}\mathbf{U}} + \mathbf{A}_t \right)^{-1} \mathbf{k}_{\mathbf{U}}(x^*)$$

* $\mathbf{K}_{\mathbf{U}\mathbf{U}} \in \mathbb{R}^{M \times M}$ est la matrice de Gram interne des points induisants.
* $\sigma_n^2$ est le bruit blanc d'observation (`noise_variance`).

### La Problématique de l'Allocation Active des Points Induisants

Le talon d'Achille de tout algorithme de GP Sparse réside dans le positionnement géométrique de sa grille d'ancres $\mathbf{U}$. Dans un système de recommandation séquentiel, on fait face à deux contraintes :

1. **La Divergence Géométrique (Exploration) :** Le modèle doit placer des points induisants dans les zones de faible récompense pour "ancrer" et réduire la variance prédictive. Sans ancres dans ces régions, la variance $\sigma_t^2(x^*)$ du GP Sparse remonterait systématiquement vers la variance *a priori* $\sigma_f^2$. Lors du tirage de Thompson Sampling, le bandit verrait ces zones explorées et décevantes comme "totalement inconnues", provoquant des boucles de ré-exploration cyclique infinies (**piège de la fausse confiance**).
2. **La Résolution Locale (Exploitation) :** Le modèle doit concentrer une forte densité d'ancres dans les zones à haute récompense afin de modéliser avec une extrême précision le comportement des utilisateurs face aux bras optimaux.

Une allocation passive (qui fige comme ancres les $M$ premières interactions observées) ou purement géométrique échoue à résoudre cette tension.

---

### Implémentation et Compromis

### 1. Streaming sans perte à $\mathcal{O}(1)$ Temporel

Contrairement au GP classique qui nécessite de ré-exécuter un entraînement global sur un jeu de données, la méthode `_apply_precision_update` incrémente les matrices $\mathbf{A}$ et $\mathbf{b}$ à chaque interaction. L'opération n'implique qu'un produit externe de vecteurs de taille $M$, s'exécutant en temps strictement constant $\mathcal{O}(M^2)$. L'inversion lourde par Cholesky ($\mathcal{O}(M^3)$) est déportée uniquement au moment de la prise de décision (`select_arm_idx`), et amortie grâce au flag `needs_factorization`.

### 2. Le Risque Algorithmique : Le Gel Géométrique (*Cold-Start Bias*)

La première version utilise une stratégie de **Warm-up passif** : les $M$ premiers points vus par le système deviennent la grille de lecture définitive (`inducing_points`) du modèle.

> ⚠️ **Limitation Critique :** Si la phase de démarrage souffre d'un manque de diversité (par exemple, si un seul bras est tiré de manière répétée ou si les premiers contextes proviennent tous du même segment d'utilisateurs), la matrice $\mathbf{K}_{\mathbf{U}\mathbf{U}}$ sera mathématiquement instable ou mal conditionnée (proche de la singularité). Bien qu'un bruit de structure (`jitter`) soit injecté pour sauver la décomposition de Cholesky, le modèle restera définitivement "aveugle" aux régions de l'espace qui n'ont pas été échantillonnées durant ces $M$ premières étapes.

Cette version risque un mauvais positionnement de sa grille d'ancres $\mathbf{U}$, ce qui peut bloquer la convergence.

---

#### **Solution : Le Reward-Weighted Pivoted Cholesky (RWPC)**

Pour pallier ce problème, notre implémentation s'appuie sur une méthode de sélection par **Cholesky Pivoté Pondéré par la Récompense**.

#### Fondement Théorique : Approximation Greedy de DPP

Idéalement, on voudrait échantillonner les points induisants selon un **Determinantal Point Process (DPP)** combinant Qualité et Diversité. La recherche du sous-ensemble optimal de taille $M$ maximisant le déterminant de la matrice de covariance pondérée est un problème NP-difficile.

Le **Cholesky Pivoté (Pivoted Cholesky)** est un algorithme glouton déterministe qui résout efficacement ce problème. En sélectionnant à chaque étape le point maximisant le résidu diagonal de la matrice de covariance, il maximise de manière équivalente le déterminant de la sous-matrice de Gram.

On introduis la notion de **Qualité (Reward Utility)** en définissant une matrice de covariance pondérée $\tilde{\mathbf{K}}$ :

$$\tilde{k}(x_i, x_j) = w_i \cdot k(x_i, x_j) \cdot w_j$$

Où le poids d'utilité $w_i$ d'un point historique dépend de sa performance observée (ou prédite) $y_i$ :

$$w_i = \exp\left(\frac{y_i - \max(\mathbf{y})}{\tau}\right) + \epsilon$$

* **$\tau$ (Température) :** Règle la sélectivité face aux récompenses. Une température basse focalise le poids sur les meilleurs succès.
* **$\epsilon$ (Ancre de Diversité) :** Garantit que même les zones à récompense nulle conservent un poids minimal non nul ($\epsilon$). Cela permet au Cholesky Pivoté de placer tout de même quelques ancres de sauvegarde dans les déserts de récompense pour neutraliser la remontée de variance prédictive (évitant ainsi le piège de la fausse confiance).

#### Processus en Deux Niveaux (*Two-Tier Strategy*)

Pour maintenir une latence faible en prod, la sélection des points induisants est découplée des inférences temps réel :

1. **Phase de Streaming Ultra-Rapide (Tier 1) :** Pendant que le bandit tourne, les calculs d'inférence s'effectuent via des résolutions de Cholesky sur les matrices $\mathbf{A}$ et $\mathbf{b}$ figées. Chaque nouvelle observation incrémente instantanément $\mathbf{A}$ et $\mathbf{b}$ par une mise à jour de rang 1 en $\mathcal{O}(M^2)$.
2. **Re-optimisation Périodique (Tier 2) :** Toutes les $U_{\text{interval}}$ étapes, l'algorithme déclenche en arrière-plan la sélection RWPC sur un buffer glissant contenant les $N_{\text{buffer}}$ dernières interactions. Une fois les nouvelles ancres $\mathbf{U}$ déterminées en temps $\mathcal{O}(N_{\text{buffer}} M^2)$, les matrices de précision globale $\mathbf{A}$ et $\mathbf{b}$ sont reconstruites de manière vectorisée et ré-injectées instantanément dans le modèle.

---

### Calibration des Hyperparamètres

Le comportement du `SparseGPTS` repose sur le réglage fin de ses hyperparamètres. Contrairement à un GP classique, leur mauvaise configuration ici peut bloquer complètement l’apprentissage en raison de la dimensionnalité induite par les bras.

- **`num_inducing` ($M$) :** Nombre de points pivots (pseudo-inputs) synthétisant l’espace.
    - *Impact avec 500 bras :* Si $M \le 500$, le modèle souffrira d’un **biais d’aveuglement structurel**. Durant la phase de démarrage, chaque interaction fige un point pivot. Si $M=100$, au moins 400 bras n’auront aucun ancrage géométrique dédié.
    - *Recommandation :* Fixer $M \ge \text{nombre de bras}$. Avec 500 bras, un $M$ entre $600$ et $1000$ est un minimum pour garantir que le modèle puisse discriminer les bras tout en conservant l’avantage computationnel du $\mathcal{O}(M^3)$ borné.
- **`length_scale` ($\ell$) :** Rayon d’influence spatiale d’une observation.
    - Une valeur trop faible ($< 0.5$) empêchera la généralisation entre deux contextes proches. Une valeur trop élevée ($> 5.0$) lissera les différences entre les bras : le modèle pensera que le bras $A$ et le bras $B$ fournissent la même récompense face à un contexte donné.
- **`signal_variance` ($\sigma_f^2$) :** Budget initial d’incertitude (variance *a priori* maximale).
    - Contrôle l’amplitude verticale des prédictions. Couplé au paramètre d’exploration $\alpha$, il détermine la force des variations lors du tirage de Thompson. Pour des récompenses bornées (ex: taux de clic entre 0 et 1), $\sigma_f^2 = 1.0$ est une base saine.
- **`noise_variance` ($\sigma_n^2$) :** Estimation du bruit de fond aléatoire (paramètre `alpha` du régresseur).
    - Agit comme un régularisateur (crête de Tikhonov). Plus $\sigma_n^2$ est grand, plus le modèle tolère les fluctuations de récompenses sans altérer sa trajectoire. Une valeur trop faible ($< 1e-4$) peut rendre l’inversion de la matrice $\mathbf{K}_{\mathbf{U}\mathbf{U}} + \mathbf{A}$ instable.

### Tableau Comparatif : Mémoire Glissante vs Architecture Sparse

| Métrique / Comportement | `ContextualGPTS` (Sliding Window) | `SparseGPTS` (Inducing Points) |
| --- | --- | --- |
| **Complexité Inférence** | $\mathcal{O}(W^3)$ où $W$ est la taille de la fenêtre | $\mathcal{O}(M^3)$ où $M$ est le nombre d’ancres |
| **Complexité Update** | $\mathcal{O}(1)$ (simple append au buffer) | $\mathcal{O}(M^2)$ (mise à jour incrémentale) |
| **Rétention du Passé** | Limitative (amnésie totale au-delà de $W$) | Infinie (toutes les données sont condensées dans $\mathbf{A}$ et $\mathbf{b}$) |
| **Phénomène de dérive** | **Re-exploration cyclique** (la variance remonte) | **Biais d’ancrage** (incapacité à apprendre hors de la grille $\mathbf{U}$) |