# Analyses et Prédictions TOP 14

Projet de statistiques et de Machine Learning appliqué au championnat de France de rugby **TOP 14**.

## 🎯 Objectif

Étudier l’évolution du championnat depuis 2005 à travers :
1. **Collecte et nettoyage des données** (Wikipedia, reconstruction des tableaux manquants).
2. **Visualisations** pour suivre budgets, performances et classements.
3. **Prédictions** avec deux axes :
   - Prédire le **champion** d’une saison à partir des données disponibles.
   - Prédire le **classement de la saison suivante** grâce à un modèle d’Elo enrichi.

---

## 📊 Données

Les données proviennent des pages Wikipédia du TOP 14 (2005–2025), complétées et uniformisées.  
Cinq tableaux principaux sont exploités pour chaque saison :

- **Présentation** (budget, entraîneur, stade, classement précédent, etc.)
- **Classement** (points, victoires/défaites, bonus, différentiel de points)
- **Évolution** (rang par journée)
- **Forme** (séquences victoire/nul/défaite)
- **Résultats** (scores domicile/extérieur)

Nettoyage réalisé :
- Uniformisation des noms de clubs (≈150 noms ramenés à 30 équipes).
- Suppression des notes et colonnes parasites.
- Reconstruction de données manquantes (ex : évolution/forme avant 2015-2016).

---

## 📈 Visualisations

Quelques analyses descriptives :
- Évolution du **budget moyen** des clubs (+100% en 15 ans).
- Suivi du **classement des clubs dominants** (Toulouse, Toulon, La Rochelle, Bordeaux).
- Étude de la **forme saisonnière** (impact limité du Tournoi des 6 Nations pour Toulouse).
- Périodes de domination : Toulon (2013–2015), Toulouse (dernières saisons).

---

## 🏆 Prédire le champion

L’objectif : déterminer si on peut prévoir le vainqueur des phases finales en connaissant la saison régulière.

### Approches testées
- **Baseline (BASE)** : le 1er de la saison régulière gagne.
  - 11 fois sur 19 (57.8%), finaliste 73.7%.
- **Régression logistique** : ≈42% de précision (champion), < BASE.
- **Random Forest** : ≈58%, mais revient souvent à prédire simplement le 1er.
- **XGBoost** :
  - 68.4% de précision (13/19 saisons).
  - 100% de précision pour les **2 finalistes**.
  - Amélioration intéressante mais statistiquement limitée (p-value ~0.41 faute de plus de données).

**Conclusion** : XGBoost apporte un vrai signal au-delà de BASE, mais davantage de données (joueurs, phases finales) seraient nécessaires.

---

## 📅 Prédire la saison

### Méthode : modèle d’**Elo enrichi**
- Pondération des saisons passées (70% dernière, 30% avant-dernière).
- Ajustement par la **forme récente** et le **mercato**.
- Compression en 3 blocs : top clubs, poursuivants, promu.

### Saison 2024/2025
- Top 3 prédit correctement (Toulouse, Bordeaux, Toulon).
- Sous-estimation de Bayonne (mercato sous-évalué).

### Saison 2025/2026 (projection)
- Classement prédit :  
  1. Toulouse  
  2. Bordeaux  
  3. La Rochelle  
  4. Toulon  
  5. Bayonne  
  6. Clermont  
  7. Pau  
  …  
  14. Montauban  

- Attribution de **probabilités de victoire** par match (modèle de Davidson avec nul).
- Elo mis à jour à chaque journée → permet de recalculer le classement dynamique.

---

## 🚀 Utilisation

Notebook principal : **`main.ipynb`**

### Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Exécution
Ouvrir le notebook et lancer les cellules pour :
- Charger les données nettoyées.
- Visualiser budgets, classements et performances.
- Tester les modèles de prédiction.

---

## 🔮 Perspectives
- Intégrer des données **joueurs/transferts** plus fines.
- Étendre aux compétitions européennes.
- Publier les prédictions mises à jour à chaque journée sur une page dédiée.

---

✍️ *Mathieu Roig — Septembre 2024 / Août 2025*
