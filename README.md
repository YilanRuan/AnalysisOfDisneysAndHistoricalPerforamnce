# 📊 Disney Revenue & Strategy Analysis
## 📌 Table of Contents
- [🎬 Project Overview](#-project-overview)
- [📈 What This Project Does](#-what-this-project-does)
- [💡 Impact of This Analysis](#-impact-of-this-analysis)
- [✅ Final Results Summary](#-final-results-summary)
- [⚙️ Technologies Used](#️-technologies-used)
- [🔮 Future Work](#-future-work)
- [👩‍💻 How to Run the Analysis](#-how-to-run-the-analysis)
- [📚 Acknowledgements](#-acknowledgements)
## 🎬 Project Overview

This repository contains a comprehensive data analytics project that investigates **Disney's historical revenue drivers** and evaluates **long-term strategic insights** through data analysis, statistical testing, and causal inference. The project explores:

- What factors contribute to Disney's movie box office success
- The impact of genre on revenue
- Audience preferences for heroes vs villains
- The effect of the Disney+ recommendation system on subscription growth

[Back to the Table of Content](#-table-of-contents)
## 📈 What This Project Does

Using a combination of **hypothesis testing, ANOVA, causal analysis (DoWhy)**, and **regression models**, this project answers the following core questions:

1. **Movie Revenue Performance**: Do Disney's pre-2016 movies perform as well as post-2017 blockbusters?
2. **Genre Impact**: Which movie genres contribute most to revenue?
3. **Audience Preference Analysis**: Do age demographics influence preferences for heroes or villains?
4. **Disney+ Strategy**: Does the recommendation system increase Disney+ subscriber counts?

[Back to the Table of Content](#-table-of-contents)
## 💡 Impact of This Analysis

- Provided Disney with **evidence-backed insights** to prioritize top-performing genres like **Action, Adventure, Musical, Romantic Comedy, Thriller/Suspense, and Western**.
- Demonstrated that **audience age groups (under/over 44)** do **not significantly differ** in their preference for heroes vs villains, guiding more generalized marketing strategies.
- Assessed the **causal impact of the Disney+ recommendation system**, revealing nuanced insights:
  - Linear regression suggested a **negative or inconclusive effect**.
  - However, using **Propensity Score Matching via DoWhy**, a **positive causal impact of +2,495 subscribers** was discovered, suggesting areas for optimization in recommendation algorithms.

[Back to the Table of Content](#-table-of-contents)
## ✅ Final Results Summary

| Research Question                  | Method                             | Outcome                                                                                  |
| ---------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------- |
| Box Office Revenue Comparison      | Hypothesis Testing                 | Pre-2016 movies do **not outperform** post-2017 releases                                 |
| Genre Performance                  | Welch ANOVA + Tukey                | Top 6 genres identified for higher revenue                                               |
| Age Preference for Heroes/Villains | Heterogeneity Test                 | No significant difference across age groups                                              |
| Disney+ Recommendation System      | Regression + DoWhy Causal Analysis | Weak evidence from regression, but causal analysis showed **positive subscriber growth** |

[Back to the Table of Content](#-table-of-contents)
## ⚙️ Technologies Used

- Python (Pandas, NumPy, SciPy, statsmodels, DoWhy)
- Statistical Methods: Hypothesis Testing, Welch's ANOVA, Games-Howell Post-Hoc
- Causal Inference: DoWhy library
- Data Visualization: Matplotlib
- 
[Back to the Table of Content](#-table-of-contents)
## 🔮 Future Work

- Implement a **two-factor ANOVA** to explore how **movie ratings + genres** interact to affect revenue.
- Conduct **price sensitivity analysis** on Disney+ subscription models.
- Gather **data on younger audiences (<18)** for more robust preference studies.
- Refine the recommendation system impact with **larger, more granular datasets** and **additional covariates**.

[Back to the Table of Content](#-table-of-contents)
## 👩‍💻 How to Run the Analysis

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install pandas numpy scipy statsmodels dowhy matplotlib pingouin
   ```
3. Run `Disney Analysis Code.py` to replicate the analysis results.

[Back to the Table of Content](#-table-of-contents)
## 📚 Acknowledgements

- Data sources: Kaggle, Disney's public data, Professor Orkun Baycik's datasets
- Tools: Boston University BA472 Course

[Back to the Table of Content](#-table-of-contents)

