# 📊 Disney Revenue & Strategy Analysis
 <a href="https://github.com/YilanRuan/AnalysisOfDisneysAndHistoricalPerforamnce/blob/main/README.md#-project-overview">Project Overview</a>
 <a href="https://github.com/YilanRuan/AnalysisOfDisneysAndHistoricalPerforamnce/blob/main/README.md#-what-this-project-does">what-this-project-does</a> 
 <a href="https://github.com/YilanRuan/AnalysisOfDisneysAndHistoricalPerforamnce/blob/main/README.md#-impact-of-this-analysis">Impact of this analysis</a> 
 <a href="https://github.com/YilanRuan/AnalysisOfDisneysAndHistoricalPerforamnce/blob/main/README.md#-final-results-summary">Final Results Summary</a> 
 <a href="https://github.com/YilanRuan/AnalysisOfDisneysAndHistoricalPerforamnce/blob/main/README.md#%EF%B8%8F-technologies-used">Technologies Used</a> 
 <a href="https://github.com/YilanRuan/AnalysisOfDisneysAndHistoricalPerforamnce/blob/main/README.md#-future-work">Future Work</a> 
 <a href="https://github.com/YilanRuan/AnalysisOfDisneysAndHistoricalPerforamnce/blob/main/README.md#-how-to-run-the-analysis">How to run the analysis</a> 
 <a href="https://github.com/YilanRuan/AnalysisOfDisneysAndHistoricalPerforamnce/blob/main/README.md#-acknowledgements">Acknowledgements</a>
## 🎬 Project Overview

This repository contains a comprehensive data analytics project that investigates **Disney's historical revenue drivers** and evaluates **long-term strategic insights** through data analysis, statistical testing, and causal inference. The project explores:

- What factors contribute to Disney's movie box office success
- The impact of genre on revenue
- Audience preferences for heroes vs villains
- The effect of the Disney+ recommendation system on subscription growth

## 📈 What This Project Does

Using a combination of **hypothesis testing, ANOVA, causal analysis (DoWhy)**, and **regression models**, this project answers the following core questions:

1. **Movie Revenue Performance**: Do Disney's pre-2016 movies perform as well as post-2017 blockbusters?
2. **Genre Impact**: Which movie genres contribute most to revenue?
3. **Audience Preference Analysis**: Do age demographics influence preferences for heroes or villains?
4. **Disney+ Strategy**: Does the recommendation system increase Disney+ subscriber counts?

## 💡 Impact of This Analysis

- Provided Disney with **evidence-backed insights** to prioritize top-performing genres like **Action, Adventure, Musical, Romantic Comedy, Thriller/Suspense, and Western**.
- Demonstrated that **audience age groups (under/over 44)** do **not significantly differ** in their preference for heroes vs villains, guiding more generalized marketing strategies.
- Assessed the **causal impact of the Disney+ recommendation system**, revealing nuanced insights:
  - Linear regression suggested a **negative or inconclusive effect**.
  - However, using **Propensity Score Matching via DoWhy**, a **positive causal impact of +2,495 subscribers** was discovered, suggesting areas for optimization in recommendation algorithms.

## ✅ Final Results Summary

| Research Question                  | Method                             | Outcome                                                                                  |
| ---------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------- |
| Box Office Revenue Comparison      | Hypothesis Testing                 | Pre-2016 movies do **not outperform** post-2017 releases                                 |
| Genre Performance                  | Welch ANOVA + Tukey                | Top 6 genres identified for higher revenue                                               |
| Age Preference for Heroes/Villains | Heterogeneity Test                 | No significant difference across age groups                                              |
| Disney+ Recommendation System      | Regression + DoWhy Causal Analysis | Weak evidence from regression, but causal analysis showed **positive subscriber growth** |

## ⚙️ Technologies Used

- Python (Pandas, NumPy, SciPy, statsmodels, DoWhy)
- Statistical Methods: Hypothesis Testing, Welch's ANOVA, Games-Howell Post-Hoc
- Causal Inference: DoWhy library
- Data Visualization: Matplotlib

## 🔮 Future Work

- Implement a **two-factor ANOVA** to explore how **movie ratings + genres** interact to affect revenue.
- Conduct **price sensitivity analysis** on Disney+ subscription models.
- Gather **data on younger audiences (<18)** for more robust preference studies.
- Refine the recommendation system impact with **larger, more granular datasets** and **additional covariates**.

## 👩‍💻 How to Run the Analysis

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install pandas numpy scipy statsmodels dowhy matplotlib pingouin
   ```
3. Run `Disney Analysis Code.py` to replicate the analysis results.

## 📚 Acknowledgements

- Data sources: Kaggle, Disney's public data, Professor Orkun Baycik's datasets
- Tools: Boston University BA472 Course

