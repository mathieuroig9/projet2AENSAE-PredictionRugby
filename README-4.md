# Analyses et Prédictions TOP 14

Projet de statistiques et de Machine Learning appliqué au championnat de France de rugby **TOP 14**.

## 🎯 Objectif

Étudier l’évolution du championnat à travers un projet de bout en bout qui passe par :
1. **Collecte et nettoyage des données** (webscrapping, reconstruction des données manquantes).
2. **Visualisations** pour suivre budgets, performances et classements.
3. **Prédictions** selon deux axes :
   - Prédire le **champion** d’une saison à la fin des phases régulières.
   - Prédire le **résultat des prochaines rencontres** grâce à un modèle d’Elo.

---

## 📊 Données

Les données proviennent des pages Wikipédia du TOP 14 (2005–2025). Le scrapping des données a demandé des étapes de nettoyage, d'uniformisation et de reconstruction.
Cinq tableaux principaux sont exploités pour chaque saison :

- **Présentation** (budget, entraîneur, stade, classement précédent, etc.)
- **Classement** (points, victoires/défaites, bonus, différentiel de points)
- **Évolution** (rang par journée)
- **Forme** (séquences victoire/nul/défaite)
- **Résultats** (scores domicile/extérieur)

---

## 📈 Visualisations

Quelques analyses descriptives :
- Évolution du **budget moyen** des clubs (+100% en 15 ans).
- Suivi du **classement des clubs dominants** (Toulouse, Toulon, La Rochelle, Bordeaux).
- Étude de la **forme saisonnière** (impact limité des compétitions internationales sur Toulouse).
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
- Pronostics de (victoire,nul,défaite) pour les rencontres de l'année en cours.
- Elo mis à jour à chaque journée.
- Prédictions publiées hebdomadairement sur X.


---

✍️ *Mathieu Roig — Septembre 2024 / Août 2025*
