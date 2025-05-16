# Student Habits and Academic Performance

This project analyzes the relationship between student lifestyle habits and academic outcomes using a publicly available dataset from Kaggle. The goal was to identify which personal behaviors most strongly correlate with—and predict—exam performance.

---

## Purpose

To explore how study habits, sleep, exercise, diet, attendance, and mental health contribute to academic success. Specifically, the project aimed to:

- Identify the strongest behavioral predictors of exam score
- Quantify both individual and joint effects of multiple lifestyle variables
- Use regression modeling to evaluate how these variables interact

---

## Tools & Techniques

**Languages & Libraries:**
- Python: pandas, seaborn, matplotlib, scipy, statsmodels, stargazer

**Techniques:**
- Exploratory Data Analysis (EDA)
- Outlier filtering and distribution smoothing
- Spearman correlation analysis
- Multiple linear regression
- Visualization and interpretation of regression results

---

## Key Insights

- **Study Hours per Day** showed the strongest individual correlation with exam score (rho ≈ 0.78), and remained the most impactful predictor in the regression model (β = 9.86, p < 0.01).
- **Mental Health Rating**, **Exercise Frequency**, and **Sleep Hours** all had statistically significant and moderately strong effects on performance when analyzed together, despite weaker individual correlations.
- **Attendance Percentage** was statistically significant but contributed less meaningfully to score variance.
- **Diet Quality** had no statistically significant effect—neither as a standalone variable nor in the full model.
- The final regression model explained approximately 83% of the variance in exam scores (Adjusted R² = 0.828).

---

## Business Value

This project can inform educational policy and student support services by:

- Emphasizing mental health, sleep, and exercise as core academic performance factors alongside study habits
- Encouraging personalized interventions for at-risk students based on behavioral profiles
- Supporting multivariate modeling as a more accurate way to understand academic predictors

---

## Personal Takeaways

- Gained experience in balancing EDA with statistical rigor using both bivariate and multivariate methods
- Developed practical intuition for how variable relationships can shift between correlation and regression contexts
- Improved ability to present statistical findings clearly for decision-making audiences

---

## Contact

Ryan Nesbitt  
rmnesbitt@gmail.com  
www.linkedin.com/in/rmnesbitt