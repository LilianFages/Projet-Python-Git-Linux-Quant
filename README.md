# Quant Platform — Dashboard de marché, stratégies, portfolio & rapports (http://141.145.217.89/)

## 1) Présentation générale

**Quant Platform** est une application **Streamlit** dédiée à :
- l’**analyse de marchés** (prix, tendances, “market snapshot”),
- le **backtesting** de stratégies quantitatives,
- l’**optimisation de portefeuille** (Markowitz),
- la **génération et consultation de rapports quotidiens**.

L’application vise une expérience utilisateur claire : sélectionner un univers d’actifs, choisir une période, visualiser les données, exécuter des analyses (stratégies/portfolio), puis exporter ou consulter des rapports.

---

## 2) Navigation et expérience utilisateur

L’application est structurée en **4 pages** accessibles via la barre latérale.

### 2.1 Quant A — Analyse Marché
Page orientée “lecture rapide” et exploration d’un actif.

**Ce que l’utilisateur voit :**
- Un bandeau (ticker tape) affichant l’évolution récente d’indices et instruments globaux.
- Des horloges des principales places financières.
- Un bloc de sélection d’actif (classe d’actifs → univers → actif).
- Un choix de période (ex. 1 jour, 5 jours, 1 mois, 6 mois, 1 an, 5 ans, historique).
- Un graphique de prix (adapté à la période).
- Deux choix de visualisations (droite ou bougies)
- En affichage bougie, possibilité d'afficher ou de masquer les volumes

**Objectif :**
- Obtenir une vue synthétique du marché
- Explorer un actif rapidement sur différentes horizons de temps

---

### 2.2 Quant A — Stratégies & Backtest
Page orientée “recherche” et test de stratégies.

**Ce que l’utilisateur fait :**
- Sélectionner une période de backtest :
  - périodes fixes (ex. 1 an, 5 ans, 10 ans)
  - selon la version : sélection manuelle start/end
- Paramétrer une stratégie (ex. SMA, paramètres, règles d’entrée/sortie).
- Optimiser les paramètres de la stratégie automatiquement en définissant les critères d'optimisation
- Lancer le backtest.

**Ce que l’utilisateur obtient :**
- Une **courbe de capital** (stratégie vs buy & hold / benchmark).
- Des **métriques** de performance (ex. rendement, volatilité, drawdown, Sharpe, etc.).
- Une lecture comparative : “est-ce que la stratégie apporte une amélioration par rapport au buy and hold (benchmark) ?”
- bonus : une prévision ML ARIMA pour prédire la valeur future d'un actif selon la stratégie utilisée

**Objectif :**
- Tester des hypothèses quantitatives sur un historique cohérent
- Comparer un modèle de trading à une stratégie passive

---

### 2.3 Quant B — Portfolio
Page orientée “allocation” et optimisation.

**Ce que l’utilisateur fait :**
- Ajouter des actifs au portefeuille (au moins 2 pour l’optimisation).
- Ajuster les pondérations (manuellement) ou demander une optimisation via Markovitz.
- Lancer l’optimisation Markowitz pour maximiser le sharpe ratio ou minimiser la volatilité
- Choisit une stratégie, l'optimise puis lance le backtest

**Ce que l’utilisateur obtient :**
- Une allocation optimisée (poids par actif) selon l’objectif choisi.
- Une cohérence de portefeuille (somme des poids = 100%).
- Des éléments d’aide au rééquilibrage.
- des métriques propres au portefeuille mais aussi au backtest de la stratégie

**Objectif :**
- Construire un portefeuille multi-actifs
- Proposer une allocation “quant” basée sur rendements/covariances historiques
- backtest des stratégies pluri-actifs

---

### 2.4 Rapports
Page orientée “traçabilité” et export.

**Ce que l’utilisateur voit :**
- Une liste de dates disponibles (rapports existants).
- Un rapport sélectionnable par date.
- Trois formats disponibles :
  - **HTML** (visualisation riche)
  - **CSV** (données tabulaires)
  - **Markdown** (lecture simple + export)

**Actions utilisateur :**
- Télécharger HTML / CSV / Markdown.
- Consulter directement le contenu dans l’application via des onglets :
  - Aperçu Markdown
  - Aperçu HTML
  - Table CSV

**Objectif :**
- Disposer d’un journal quotidien automatisé
- Pouvoir exporter et archiver des résultats

---

## 3) Fonctionnalités détaillées

### 3.1 Bandeau “Market Snapshot” (indices globaux)
- Agrège un ensemble d’indices/instruments globaux (ex. indices majeurs, crypto, FX).
- Calcule la variation entre les **deux derniers closes valides**.
- Affiche :
  - le nom lisible de l’instrument
  - le dernier prix
  - la variation (en %)

**Valeur utilisateur :**
- Lire le marché en quelques secondes sans entrer dans des détails.

---

### 3.2 Horloges des marchés
- Affiche l’heure locale de places financières (ex. New York, Londres, Paris, Tokyo, Hong Kong).
- Permet de contextualiser la session (Europe/US/Asie) et comprendre certains comportements intraday.

**Valeur utilisateur :**
- Situer l’état du marché dans la journée, surtout lors d’analyses rapides.

---

### 3.3 Sélection d’actifs et univers
L’application utilise une logique de sélection en cascade :
1. **Classe d’actifs** (ex. Actions, ETF, Crypto, FX… selon configuration)
2. **Univers / indice** (ex. S&P 500, listes internes…)
3. **Actif** (ex. Apple, SPY, BTC-USD, EURUSD=X…)

**Valeur utilisateur :**
- Éviter de chercher un ticker “brut” : l’UI structure le choix.

---

### 3.4 Gestion des périodes
Plusieurs horizons sont proposés selon les pages :
- “Analyse Marché” : horizons rapides (1j → historique)
- “Backtest” : horizons longs et comparables (1/5/10 ans, etc.)
- “Portfolio” : horizon cohérent pour estimer rendements/covariances

**Valeur utilisateur :**
- Adapter la granularité des données à la question : court-terme vs long-terme.

---

### 3.5 Backtesting
Le backtesting comprend typiquement :
- construction d’un signal (ex. croisement de moyennes, etc.)
- simulation des positions
- calcul de la courbe de performance
- comparaison à un benchmark (buy & hold)

**Métriques généralement affichées (selon implémentation) :**
- Rendement cumulé / annualisé
- Volatilité
- Sharpe Ratio
- Max drawdown
- Nombre de trades, win rate (si fourni)
- Autres ratios (si activés)

**Valeur utilisateur :**
- Tester une stratégie de manière reproductible, sur des fenêtres temporelles définies.

---

### 3.6 Optimisation de portefeuille (Markowitz)
Le module portfolio s’appuie sur :
- rendements historiques
- matrice de covariance
- résolution d’un problème d’optimisation (ex. via `cvxpy`)

**Objectifs possibles (selon version) :**
- Maximiser le Sharpe ratio
- (éventuellement) minimiser la variance / viser un rendement cible

**Contraintes usuelles (selon implémentation) :**
- somme des poids = 1
- poids ≥ 0 (portefeuille long-only), ou contraintes personnalisées

**Valeur utilisateur :**
- Proposer une allocation mathématiquement optimisée, compréhensible et exploitable.

---

### 3.7 Rapports quotidiens
Chaque rapport est produit en **3 formats** :
- HTML : rendu lisible, partageable
- CSV : données structurées, analysables dans Excel/Python
- Markdown : lecture rapide, versionnable, export simple

**Contenu typique d’un rapport (selon script) :**
- résumé (“Summary”)
- tableaux (ex. prix/rendements/variations)
- éventuellement : signaux, métriques, highlights

**Valeur utilisateur :**
- Un “snapshot” quotidien conservé, consultable et téléchargeable.

---

## 4) Architecture technique (vue d’ensemble)

### 4.1 Point d’entrée et routing
- `main.py` initialise Streamlit, définit la navigation (radio sidebar) et appelle les UI de chaque page.

### 4.2 Découpage logique
Le projet est structuré par domaines :
- `app/common/` : fonctions partagées (données, normalisation, cache, utilitaires temps)
- `app/quant_a/` : analyse marché + stratégies/backtests
- `app/quant_b/` : portfolio + optimisation
- `reports/` : affichage et gestion des rapports
- `scripts/` : génération des rapports (batch)

---

## 5) Données : comportement, formats et robustesse

### 5.1 Sources de données
- La majorité des prix provient de **yfinance** (actions, ETF, indices, FX, crypto `*-USD`).
- Certains modules peuvent utiliser d’autres sources selon l’implémentation (ex. crypto exchange).

### 5.2 Particularités à connaître (important)
**a) MultiIndex (yfinance)**
`yfinance.download()` peut renvoyer des colonnes **MultiIndex** (ex. `(Close, AAPL)`).
Le projet inclut des fonctions de normalisation/“flatten” pour rendre les colonnes cohérentes côté graphiques et calculs.

**b) Timezones et “edge cases”**
Sur les données daily, des cas limites peuvent apparaître autour de minuit (fenêtres start/end).
Le code inclut des mécanismes de normalisation des bornes et des filtres finaux robustes pour éviter des datasets vides.

**c) Fenêtre demandée vs fenêtre disponible**
Sur certaines périodes, un actif peut avoir :
- peu d’historique
- trous de données
- jours non-tradés (weekends, fériés)
Des fallback et contrôles existent pour éviter des “écrans vides”.

---

## 6) Cache et performance

### 6.1 Cache Streamlit
Certaines fonctions sont mises en cache via `@st.cache_data` pour :
- limiter les appels externes,
- accélérer l’UI,
- réduire les latences de rafraîchissement.

Exemple typique : bandeau indices (TTL court).

### 6.2 Cache disque (daily)
Pour les données daily, le projet peut stocker un cache CSV afin :
- d’éviter de re-télécharger la même plage,
- de stabiliser la performance,
- de réduire les risques de rate limit.

**Bonnes pratiques :**
- si un comportement semble incohérent, vider le cache (Streamlit et/ou cache disque) permet de distinguer :
  - bug data/timezone
  - bug lié à une donnée stale en cache
  - indisponibilité temporaire de la source externe

---

## 7) Génération des rapports : fonctionnement

### 7.1 Génération manuelle
La génération manuelle sert à tester et à produire immédiatement un rapport consultable.

- Script python : `scripts/daily_report.py`
- Wrapper shell : `scripts/run_daily_report.sh`

Le wrapper typiquement :
- se place dans le repo,
- active l’environnement,
- crée `logs/` et `reports/outputs/` si besoin,
- exécute le script et redirige la sortie vers un log horodaté.

### 7.2 Génération automatique (cron)
En production, une tâche cron peut déclencher le wrapper à heure fixe.

Résultat attendu :
- nouveaux fichiers `daily_report_YYYY-MM-DD.(html|csv|md)` dans `reports/outputs/`
- logs correspondants dans `logs/`
- visibilité immédiate dans la page **Rapports** de l’UI (dès que les fichiers existent)

---

## 8) Conseils d’usage (pratiques)

### 8.1 Analyse Marché
- Utiliser les horizons courts (1j, 5j) pour un aperçu récent.
- Passer sur 6 mois/1 an pour contextualiser une tendance.
- Utiliser “tout l’historique” pour repérer régimes long terme.

### 8.2 Stratégies & Backtest
- Comparer systématiquement à buy & hold.
- Tester plusieurs fenêtres (1 an vs 5 ans vs 10 ans) :
  - une stratégie peut “surperformer” sur une période courte mais échouer sur long terme.
- Vérifier la stabilité : nombre de trades, drawdowns, sensibilité aux paramètres.

### 8.3 Portfolio (Markowitz)
- Ajouter au moins 2 actifs valides.
- Préférer des actifs avec historique suffisant et non “incomplet”.
- Interpréter les poids optimisés comme un résultat mathématique dépendant fortement :
  - de la période sélectionnée,
  - des rendements observés,
  - des corrélations estimées.

### 8.4 Rapports
- Consulter Markdown pour lecture rapide.
- Utiliser HTML pour partage et rendu.
- Exporter CSV pour analyses additionnelles (Excel, pandas).

---

## 9) Dépannage (FAQ)

### “Aucune donnée retournée …”
Causes possibles :
- fenêtre temporelle trop restrictive (start/end)
- indisponibilité temporaire de la source de données
- colonnes inattendues (MultiIndex non aplati)
- cache stale ou fichier CSV de cache corrompu

Actions usuelles :
- tester la récupération brute d’un actif (diagnostic simple)
- désactiver/vider le cache temporairement
- vérifier la normalisation des dates daily
- relancer l’application

### “Optimisation impossible / pas assez de données”
- Moins de 2 actifs valides
- Historique insuffisant
- Données vides après nettoyage

Actions :
- ajouter un second actif liquide
- changer la période / vérifier l’historique
- vérifier que les séries de prix sont bien alignées

### “Aucun rapport disponible”
Simplement : aucun fichier dans `reports/outputs/`.

Actions :
- lancer manuellement `scripts/run_daily_report.sh`
- attendre la génération automatique (cron)

---

## 10) Avertissement
Cette application est un outil d’analyse et de démonstration. Les résultats de backtests et d’optimisations ne constituent pas un conseil financier ou une recommandation d’investissement.
