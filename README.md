# Quant Platform — Dashboard de marché, stratégies, portfolio & rapports

**URL :** http://141.145.217.89/

## 1. Présentation générale

Quant Platform est une application Streamlit déployée sur une VM Linux Oracle Cloud, dédiée à l’analyse quantitative de marchés financiers.

La plateforme permet de couvrir plusieurs étapes d’un workflow quantitatif :

- analyse de marché multi-actifs ;
- visualisation de prix et de tendances ;
- backtesting de stratégies quantitatives ;
- optimisation de paramètres ;
- prise en compte des frais et du slippage ;
- analyse détaillée de performance et de risque ;
- optimisation de portefeuille via Markowitz ;
- simulation historique de portefeuille multi-actifs ;
- analyse de diversification et de contribution au risque ;
- backtest de stratégie appliqué à un portefeuille global ;
- génération automatique de rapports quotidiens exportables.

L’objectif est de proposer une plateforme complète de recherche quantitative : sélectionner un univers d’actifs, analyser les prix, tester des stratégies, construire un portefeuille, mesurer le risque, puis générer un rapport quotidien exploitable.

---

## 2. Navigation et expérience utilisateur

L’application est organisée autour de quatre pages principales accessibles via la barre latérale :

1. **Quant A — Analyse Marché**
2. **Quant A — Stratégies & Backtest**
3. **Quant B — Portfolio**
4. **Rapports**

La barre latérale permet également de sélectionner dynamiquement :

- une classe d’actifs ;
- un indice ou univers ;
- un actif ;
- les options propres au module actif.

---

## 3. Quant A — Analyse Marché

## 3.1 Objectif

La page **Analyse Marché** est orientée exploration rapide d’un actif.

Elle permet d’obtenir une vision synthétique du marché, de sélectionner un actif et d’observer son comportement sur plusieurs horizons de temps.

## 3.2 Fonctionnalités principales

La page contient :

- un bandeau de type **ticker tape** affichant plusieurs instruments globaux ;
- les horloges des principales places financières ;
- un module de sélection d’actif ;
- un choix de période ;
- un résumé de l’actif ;
- un graphique de prix ;
- un choix de visualisation entre courbe simple et chandeliers ;
- une option d’affichage des volumes en mode chandeliers.

## 3.3 Market Snapshot

Le bandeau de marché affiche l’évolution récente d’instruments globaux tels que :

- indices actions ;
- crypto-actifs ;
- devises ;
- instruments de marché configurés dans l’application.

Pour chaque instrument, l’application affiche :

- le nom lisible ;
- le dernier prix ;
- la variation récente en pourcentage.

Ce module permet de lire rapidement l’état général du marché avant d’analyser un actif en détail.

## 3.4 Horloges de marché

La page affiche l’heure locale de plusieurs places financières :

- New York ;
- Londres ;
- Paris ;
- Hong Kong ;
- Tokyo.

L’objectif est de contextualiser la session de marché et de distinguer les périodes Europe, US et Asie.

## 3.5 Sélection d’actif

La sélection d’actif se fait en cascade :

1. classe d’actifs ;
2. indice ou univers ;
3. actif.

Exemple :

- classe d’actifs : Actions ;
- indice actions : S&P 500 ;
- actif : Apple.

Cette logique évite une saisie manuelle brute de tickers et rend l’expérience utilisateur plus structurée.

## 3.6 Gestion des périodes

Plusieurs horizons sont disponibles :

- 1 jour ;
- 5 jours ;
- 1 mois ;
- 6 mois ;
- année écoulée ;
- 1 année ;
- 5 années ;
- tout l’historique.

Cette granularité permet d’adapter l’analyse à plusieurs horizons : court terme, moyen terme ou long terme.

## 3.7 Résumé de l’actif

Pour l’actif sélectionné, l’application affiche notamment :

- le dernier prix ;
- la variation récente ;
- le plus haut sur la période ;
- le plus bas sur la période ;
- la date du dernier point disponible ;
- le volume associé quand disponible.

Ce résumé donne une lecture rapide de la position actuelle de l’actif dans son range récent.

## 3.8 Graphique de prix

Deux modes de visualisation sont disponibles :

- graphique en ligne ;
- graphique en chandeliers.

En mode chandeliers, l’utilisateur peut également afficher ou masquer les volumes.

---

## 4. Quant A — Stratégies & Backtest

## 4.1 Objectif

La page **Stratégies & Backtest** permet de tester des stratégies quantitatives sur un actif donné.

Elle combine :

- sélection de période ;
- choix de stratégie ;
- optimisation de paramètres ;
- capital initial paramétrable ;
- frais et slippage ;
- comparaison avec un benchmark buy & hold ;
- analyse détaillée de performance ;
- analyse de risque ;
- visualisation de distributions ;
- prévision expérimentale ARIMA.

Cette page sert à évaluer si une stratégie systématique améliore ou non une exposition passive à l’actif sélectionné.

## 4.2 Sélection de la période

L’utilisateur peut choisir :

- une période fixe ;
- ou une sélection manuelle start/end selon la configuration.

Exemples de périodes fixes :

- 1 an ;
- 5 ans ;
- 10 ans.

L’objectif est de tester les stratégies sur des historiques cohérents et comparables.

## 4.3 Choix de la stratégie

L’utilisateur choisit une stratégie disponible dans l’interface.

Stratégies présentes dans l’application :

- Buy & Hold ;
- SMA Crossover.

La stratégie sélectionnée est appliquée à l’actif courant.

## 4.4 Optimisation des paramètres

L’application permet d’optimiser automatiquement les paramètres de la stratégie selon un objectif donné.

Exemples d’objectifs disponibles :

- Total Return ;
- Sharpe Ratio.

Dans le cas d’une stratégie SMA Crossover, l’utilisateur peut paramétrer ou optimiser :

- SMA courte ;
- SMA longue.

Ce module permet de comparer plusieurs combinaisons de paramètres et d’éviter un paramétrage purement manuel.

## 4.5 Frais et slippage

La page intègre un bloc dédié aux **frais et slippage**.

L’objectif est de rendre le backtest plus réaliste en prenant en compte :

- des frais de transaction ;
- un coût implicite d’exécution ;
- un écart potentiel entre prix théorique et prix réellement exécuté.

Ce module permet d’éviter des backtests excessivement optimistes.

## 4.6 Capital initial

L’utilisateur peut définir un capital initial.

Ce capital sert de base à la simulation de la courbe de capital.

Exemple :

- capital initial : 10 000.

L’objectif est de rendre les résultats plus lisibles en valeur de portefeuille.

## 4.7 Courbe de capital

La page affiche une courbe comparant :

- la stratégie testée ;
- le benchmark buy & hold.

Cette visualisation permet d’observer :

- la performance cumulée ;
- les périodes de surperformance ;
- les périodes de sous-performance ;
- les phases de drawdown ;
- la stabilité de la stratégie dans le temps.

## 4.8 Analyse détaillée de performance

L’application calcule et affiche plusieurs métriques avancées :

- rendement total ;
- CAGR annualisé ;
- volatilité annualisée ;
- max drawdown ;
- ratio de Sharpe ;
- pourcentage de jours positifs ;
- nombre de trades ;
- temps investi.

L’objectif est de ne pas juger une stratégie uniquement sur son rendement, mais aussi sur son risque, sa stabilité et son exposition.

## 4.9 Structure de performance

La page affiche une heatmap de saisonnalité des rendements mensuels.

Cette heatmap permet d’observer :

- les mois fortement positifs ;
- les mois fortement négatifs ;
- la régularité des rendements ;
- les années atypiques ;
- les périodes de stress.

Elle permet d’identifier les périodes qui expliquent une part importante de la performance totale.

## 4.10 Corrélation et bêta glissants

La page affiche une analyse de corrélation et de bêta glissants sur une fenêtre donnée.

Ce module permet de mesurer :

- la sensibilité de l’actif ou de la stratégie au benchmark ;
- l’évolution du risque relatif dans le temps ;
- le degré de dépendance au marché.

## 4.11 Analyse des risques et distribution

La page contient une section dédiée aux risques et à la distribution des rendements.

Elle comprend notamment :

- un **underwater plot** des drawdowns ;
- un histogramme de distribution des rendements journaliers.

Le graphique de drawdown permet d’observer :

- les pertes depuis les plus hauts historiques ;
- la profondeur des phases de baisse ;
- la durée des périodes de récupération.

La distribution des rendements permet d’observer :

- l’asymétrie potentielle ;
- les queues de distribution ;
- la concentration des rendements autour de zéro ;
- les événements extrêmes.

## 4.12 Prévision ARIMA expérimentale

La page intègre un module expérimental de prévision via ARIMA.

L’utilisateur peut :

- activer ou désactiver la prévision ;
- choisir un horizon de prévision ;
- visualiser une trajectoire projetée ;
- lire une tendance projetée sur l’horizon défini.

Le module affiche également un intervalle de confiance.

Cette prévision est expérimentale et sert à tester une logique de forecasting statistique. Elle ne constitue pas une recommandation d’investissement.

---

## 5. Quant B — Portfolio Management

## 5.1 Objectif

La page **Portfolio Management** permet de construire, pondérer, optimiser et analyser un portefeuille multi-actifs.

Elle couvre :

- l’ajout d’actifs ;
- la pondération manuelle ;
- l’optimisation Markowitz ;
- la simulation historique du portefeuille ;
- le calcul de métriques de performance ;
- l’analyse de diversification ;
- la matrice de corrélation ;
- la contribution au risque ;
- le backtest de stratégie appliqué au portefeuille global.

## 5.2 Ajout d’actifs au portefeuille

L’utilisateur peut ajouter des actifs depuis la barre latérale.

La sélection suit la même logique que dans l’analyse marché :

- classe d’actifs ;
- univers ou indice ;
- actif.

Une fois ajoutés, les actifs apparaissent dans la composition du portefeuille.

Exemples d’actifs :

- MC.PA ;
- BNP.PA ;
- NVDA.

## 5.3 Composition et pondération

L’utilisateur peut définir les poids de chaque actif.

La plateforme vérifie que la somme des poids atteint 100 %.

Un bouton d’équilibrage permet de répartir les poids de manière égale entre les actifs.

Ce module permet de construire manuellement un portefeuille et de contrôler la cohérence des pondérations.

## 5.4 Optimisation Markowitz

La plateforme propose une optimisation d’allocation d’actifs via une logique Markowitz.

Objectif disponible dans l’application :

- maximisation du ratio de Sharpe.

L’optimisation utilise notamment :

- les rendements historiques ;
- la matrice de covariance ;
- les contraintes de poids ;
- la contrainte de somme des poids égale à 100 %.

L’objectif est de générer une allocation quantitative cohérente avec les rendements et risques historiques.

## 5.5 Simulation historique du portefeuille

L’utilisateur choisit une date de début et une date de fin puis lance la simulation du portefeuille.

L’application affiche ensuite :

- la performance du portefeuille global en base 100 ;
- la trajectoire de chaque actif ;
- la performance totale du portefeuille.

Ce module permet de comparer la performance du portefeuille global à celle de chacun des actifs qui le composent.

## 5.6 Métriques portefeuille et diversification

La plateforme calcule plusieurs métriques propres au portefeuille :

- total return ;
- CAGR ;
- volatilité annualisée ;
- Sharpe Ratio ;
- max drawdown ;
- rendement annualisé historique ;
- volatilité annualisée historique ;
- diversification ratio ;
- Neff, nombre effectif de positions.

Le nombre effectif de positions permet d’évaluer la concentration réelle du portefeuille, au-delà du simple nombre d’actifs détenus.

## 5.7 Matrice de corrélation

La page affiche une matrice de corrélation des rendements entre les actifs du portefeuille.

Elle permet d’identifier :

- les actifs fortement corrélés ;
- les actifs réellement diversifiants ;
- la structure de dépendance du portefeuille.

## 5.8 Poids du portefeuille

La plateforme affiche les poids de portefeuille sous forme graphique.

Ce graphique permet de visualiser rapidement :

- la répartition du capital ;
- les actifs dominants ;
- la cohérence de l’allocation.

## 5.9 Contribution au risque

La page affiche les contributions au risque de chaque actif.

Cette analyse distingue :

- le poids nominal d’un actif ;
- sa contribution réelle à la volatilité du portefeuille.

Un actif peut avoir un poids modéré mais contribuer fortement au risque si sa volatilité ou sa corrélation avec les autres actifs est élevée.

## 5.10 Backtest de stratégie sur portefeuille global

La plateforme permet d’appliquer une stratégie au portefeuille global.

Exemple de stratégie disponible :

- SMA Crossover.

La page permet :

- d’optimiser les paramètres de la stratégie ;
- de définir une SMA courte ;
- de définir une SMA longue ;
- de comparer portefeuille actif et portefeuille passif.

La page affiche ensuite :

- la courbe de performance ;
- le rendement de la stratégie ;
- le Sharpe Ratio ;
- le max drawdown ;
- le win rate.

Ce module permet de tester une logique active non seulement sur un actif isolé, mais aussi sur un portefeuille multi-actifs.

---

## 6. Rapports

## 6.1 Objectif

La page **Rapports** permet de consulter, visualiser et télécharger les rapports quotidiens générés automatiquement.

Elle sert de module de traçabilité et d’archivage.

## 6.2 Source des rapports

Les rapports actuels sont basés sur les derniers tickers ajoutés dans la section **Portfolio**.

L’univers analysé dans le rapport correspond donc aux actifs suivis ou construits par l’utilisateur dans la page portefeuille.

Ce fonctionnement permet de relier directement les rapports à la composition du portefeuille.

## 6.3 Formats disponibles

Chaque rapport est disponible en plusieurs formats :

- HTML ;
- CSV ;
- Markdown.

L’utilisateur peut :

- télécharger le rapport HTML ;
- télécharger le CSV ;
- télécharger le Markdown ;
- consulter directement le rapport dans l’application.

## 6.4 Visualisation dans l’application

La page propose plusieurs onglets :

- aperçu Markdown ;
- aperçu HTML ;
- table CSV.

L’objectif est de permettre à la fois :

- une lecture rapide ;
- une visualisation enrichie ;
- une récupération tabulaire des données ;
- un export vers Excel ou Python.

## 6.5 Contenu actuel des rapports

Les rapports actuels contiennent notamment :

- un titre avec la date du rapport ;
- les tickers analysés ;
- un bloc de highlights ;
- un résumé ;
- la période de lookback ;
- le statut de couverture des tickers ;
- la date de génération ;
- le temps d’exécution ;
- le meilleur actif du jour ;
- le pire actif du jour ;
- l’actif le plus volatil ;
- l’actif avec le plus fort drawdown ;
- une table complète avec les métriques calculées.

Métriques présentes dans la table :

- ticker ;
- statut ;
- première date disponible ;
- date de dernière observation ;
- nombre d’observations ;
- open ;
- high ;
- low ;
- close ;
- daily return ;
- rendement 5 jours ;
- rendement 20 jours ;
- rendement 252 jours ;
- volatilité annualisée 20 jours ;
- max drawdown.

## 6.6 Valeur utilisateur

Le module de reporting permet de :

- conserver un journal quotidien ;
- suivre les actifs du portefeuille ;
- détecter les meilleurs et pires actifs ;
- archiver les résultats ;
- exporter les données ;
- partager un rendu HTML propre.

---

## 7. Architecture technique

## 7.1 Point d’entrée

Le fichier `main.py` initialise l’application Streamlit.

Il gère notamment :

- la configuration générale ;
- la navigation ;
- le routing entre pages ;
- les appels aux modules fonctionnels.

## 7.2 Découpage logique

Le projet est organisé par domaines fonctionnels.

Structure indicative :

```text
app/
├── common/
├── quant_a/
├── quant_b/
├── reports/
scripts/
reports/
├── outputs/
logs/
```

## 7.3 Dossiers fonctionnels

### app/common/

Contient les fonctions partagées :

- récupération de données ;
- nettoyage ;
- normalisation ;
- gestion des dates ;
- gestion du cache ;
- utilitaires de calcul.

### app/quant_a/

Contient les modules liés à :

- l’analyse marché ;
- les stratégies ;
- le backtesting ;
- les graphiques de performance ;
- les indicateurs ;
- les métriques de risque.

### app/quant_b/

Contient les modules liés à :

- la construction de portefeuille ;
- l’optimisation ;
- la simulation historique ;
- la matrice de corrélation ;
- la contribution au risque ;
- le backtest portefeuille.

### reports/

Contient les fichiers de rapport générés.

### scripts/

Contient les scripts batch, notamment pour la génération automatique des rapports.

---

## 8. Données et robustesse

## 8.1 Sources de données

La majorité des données de marché provient de `yfinance`.

Les actifs couverts peuvent inclure :

- actions ;
- ETF ;
- indices ;
- crypto-actifs ;
- devises ;
- futures ou proxies selon disponibilité.

## 8.2 Normalisation yfinance

`yfinance.download()` peut retourner des colonnes MultiIndex.

Le projet inclut des fonctions de normalisation afin de rendre les données exploitables par les graphiques, les calculs et les backtests.

Objectif :

- éviter les erreurs de structure ;
- rendre les colonnes cohérentes ;
- fiabiliser les calculs multi-actifs.

## 8.3 Gestion des timezones et fenêtres temporelles

Des cas limites peuvent apparaître autour :

- des données daily ;
- des week-ends ;
- des jours fériés ;
- des changements de fuseaux horaires ;
- des fenêtres start/end trop restrictives.

Le projet intègre des contrôles pour :

- éviter les datasets vides ;
- nettoyer les dates ;
- aligner les séries ;
- filtrer les observations finales.

## 8.4 Fenêtre demandée vs fenêtre disponible

Certains actifs peuvent avoir :

- peu d’historique ;
- des données manquantes ;
- des trous de cotation ;
- des jours non-tradés ;
- des changements de format selon la source.

L’application prévoit des contrôles pour éviter des erreurs d’affichage ou de calcul.

---

## 9. Cache et performance

## 9.1 Cache Streamlit

Certaines fonctions utilisent `@st.cache_data`.

Objectifs :

- limiter les appels externes ;
- améliorer la fluidité de l’interface ;
- réduire les temps de chargement ;
- limiter les risques de rate limit.

## 9.2 Cache disque

Le projet peut utiliser un cache CSV local pour les données daily.

Objectifs :

- éviter de re-télécharger les mêmes données ;
- stabiliser les performances ;
- accélérer les calculs ;
- fiabiliser l’application en production.

## 9.3 Diagnostic

Si un comportement semble incohérent, il faut distinguer :

- bug de données ;
- bug de timezone ;
- cache obsolète ;
- problème temporaire de source externe ;
- fichier CSV de cache corrompu.

Actions possibles :

- vider le cache Streamlit ;
- supprimer temporairement le cache disque ;
- tester un ticker simple ;
- vérifier les colonnes renvoyées par yfinance ;
- relancer l’application.

---

## 10. Génération automatique des rapports

## 10.1 Génération manuelle

La génération manuelle permet de produire immédiatement un rapport.

Scripts utilisés :

```text
scripts/daily_report.py
scripts/run_daily_report.sh
```

Le wrapper shell peut notamment :

- se placer dans le repo ;
- activer l’environnement Python ;
- créer les dossiers nécessaires ;
- lancer le script Python ;
- rediriger les logs vers un fichier horodaté.

## 10.2 Génération automatique via cron

En production, une tâche cron lance automatiquement la génération de rapport.

Résultat attendu :

```text
reports/outputs/daily_report_YYYY-MM-DD.html
reports/outputs/daily_report_YYYY-MM-DD.csv
reports/outputs/daily_report_YYYY-MM-DD.md
```

Objectif :

- générer un rapport quotidien sans intervention manuelle ;
- conserver un historique ;
- rendre les nouveaux rapports visibles dans l’application.

## 10.3 Logs

La génération des rapports produit des logs dans le dossier `logs/`.

Les logs permettent de suivre :

- la date d’exécution ;
- les tickers traités ;
- les erreurs éventuelles ;
- le temps d’exécution ;
- le statut des fichiers générés.

---

## 11. Déploiement

## 11.1 Environnement

La plateforme est déployée sur une VM Linux Oracle Cloud.

L’application est accessible en ligne via une adresse publique.

Le déploiement repose sur :

- un environnement Python ;
- Streamlit ;
- les dépendances du projet ;
- des scripts de génération de rapports ;
- une tâche cron pour l’automatisation.

## 11.2 Oracle Always Free

La plateforme est hébergée sur Oracle Cloud dans un environnement compatible Always Free.

Objectif :

- maintenir le site accessible sans coût serveur significatif ;
- conserver un environnement Linux complet ;
- permettre l’exécution de scripts cron ;
- garder la main sur le runtime Python.

---

## 12. Conseils d’usage

## 12.1 Analyse Marché

Utiliser :

- 1 jour / 5 jours pour observer les mouvements récents ;
- 1 mois / 6 mois pour contextualiser une tendance ;
- 1 an / 5 ans / historique complet pour analyser les régimes long terme.

## 12.2 Stratégies & Backtest

Bonnes pratiques :

- comparer systématiquement la stratégie au buy & hold ;
- tester plusieurs périodes ;
- regarder le max drawdown autant que le rendement ;
- analyser la volatilité ;
- vérifier le nombre de trades ;
- activer les frais et slippage pour une simulation réaliste ;
- interpréter ARIMA comme un module expérimental.

## 12.3 Portfolio

Bonnes pratiques :

- ajouter au moins deux actifs ;
- vérifier que les poids totalisent 100 % ;
- comparer allocation manuelle et allocation optimisée ;
- analyser la corrélation entre actifs ;
- regarder les contributions au risque ;
- ne pas interpréter Markowitz comme une vérité stable, mais comme une allocation dépendante des inputs historiques.

## 12.4 Rapports

Bonnes pratiques :

- générer régulièrement les rapports ;
- vérifier que les tickers du portefeuille sont bien ceux que l’on souhaite suivre ;
- utiliser HTML pour une lecture propre ;
- utiliser CSV pour analyse complémentaire ;
- utiliser Markdown pour archivage ou versioning.

---

## 13. Dépannage

## 13.1 Aucune donnée retournée

Causes possibles :

- ticker invalide ;
- source externe temporairement indisponible ;
- fenêtre temporelle trop courte ;
- problème de timezone ;
- cache obsolète ;
- colonnes MultiIndex non normalisées.

Actions :

- tester un ticker très liquide ;
- élargir la période ;
- vider le cache ;
- vérifier la donnée brute yfinance ;
- relancer l’application.

## 13.2 Optimisation impossible

Causes possibles :

- moins de deux actifs valides ;
- historique insuffisant ;
- données manquantes ;
- matrice de covariance instable ;
- poids incohérents ;
- contraintes trop restrictives.

Actions :

- ajouter des actifs ;
- choisir une période plus longue ;
- vérifier les données disponibles ;
- équilibrer les poids ;
- relancer l’optimisation.

## 13.3 Aucun rapport disponible

Cause probable :

- aucun fichier présent dans `reports/outputs/`.

Actions :

- lancer `scripts/run_daily_report.sh` manuellement ;
- vérifier la tâche cron ;
- vérifier les logs ;
- vérifier les droits d’écriture du dossier.

## 13.4 Rapport généré mais incomplet

Causes possibles :

- certains tickers du portefeuille ne retournent pas de données ;
- yfinance indisponible temporairement ;
- données trop courtes ;
- erreur de cache ;
- erreur dans la génération HTML/CSV/Markdown.

Actions :

- lire les logs ;
- vérifier le statut des tickers dans le CSV ;
- tester les tickers individuellement ;
- relancer la génération.

---

## 14. Limites actuelles

La plateforme est fonctionnelle mais certaines limites existent encore :

- le rapport quotidien est actuellement centré sur les tickers du portefeuille ;
- l’analyse macro cross-asset n’est pas encore intégrée ;
- le rapport ne contient pas encore de narrative automatique avancée ;
- les signaux multi-stratégies ne sont pas encore agrégés dans une page dédiée ;
- le simulateur d’ordres n’est pas encore intégré ;
- la robustesse des stratégies pourrait être renforcée par une analyse walk-forward ;
- les rapports pourraient intégrer davantage de contributions portefeuille et de scoring macro.

---

## 15. Avertissement

Quant Platform est une application d’analyse, de recherche quantitative et de démonstration.

Les résultats affichés, backtests, optimisations, prévisions ARIMA et rapports générés ne constituent pas un conseil financier, une recommandation d’investissement ou une incitation à acheter ou vendre un instrument financier.

Les performances passées ne préjugent pas des performances futures.
