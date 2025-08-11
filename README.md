
# Data Preparation  
- **Aggregate and Clean Data (2018–2022):** Combine annual missing-persons records for 2018–2022 at the district (and state) level into a unified table. Harmonize age groups and gender fields across years, noting that 2018–20 data use seven age bands (<5, 5–14, 14–18, 18–30, 30–45, 45–60, 60+) while 2021–22 consolidate to four bands (<12, 12–16, 16–18, 18+). Drop any summary or “total” rows (e.g., a “Total_Districts” row per state) that can skew analysis. Handle missing or inconsistent entries, and standardize feature names. If the dataset is labeled (e.g. districts tagged as “hotspot” or “non-hotspot”), ensure labels are correctly aligned with records. Consider enriching the dataset with external covariates, such as district population (to compute per-capita rates), socioeconomic indicators (poverty rates, literacy), or crime rates, to provide context for modeling.  

- **Outlier Treatment and Scaling:** Identify extreme values (e.g. unusually large missing counts in urban districts) and decide whether to winsorize or transform them. The presentation noted “extreme outliers in age-gender counts” that can distort analysis. Standardize or normalize features so that variables on different scales do not dominate (especially important for clustering).

# Feature Engineering  
- **Key Demographic Features:** Based on prior work, reduce collinearity by summarizing age/gender categories. For example, the previous analysis distilled 13 original features into four independent counts (males below 18, males 18+, females below 18, females 18+). You can similarly aggregate into broad groups (e.g. *Male_Below18*, *Male_18Plus*, *Female_Below18*, *Female_18Plus*, possibly combining all transgender counts). Alternatively, compute ratios like the proportion of missing persons who are minors vs adults in each district.  
- **Per-Capita and Rate Features:** Divide missing-person counts by district population to get rates. Also compute trends (e.g. year-over-year growth) if using time-series data or multi-year averages.  
- **Derived Socioeconomic Indicators:** Merge in features such as population density, literacy rate, unemployment, or regional crime statistics (e.g. violent crime rate), as these can help explain spatial risk. Such covariates may improve supervised models.  
- **Dimensionality Reduction (for Exploration):** For visualization or as a pre-step to clustering, apply PCA or non-linear embeddings (t-SNE, UMAP). The team found that linear PCA was overwhelmed by the *Grand_Total* feature ([UML_LabExam_FinalPresentation.pdf](file://file-CST6XZVTWN6JhMgMmEhR3e#:~:text=The%20feature%2C%20,based%20grouping)), so it may be prudent to exclude grand totals or high-correlation features before PCA. Nonlinear methods (t-SNE/UMAP) can reveal local groupings but may distort global structure ([UML_LabExam_FinalPresentation.pdf](file://file-CST6XZVTWN6JhMgMmEhR3e#:~:text=The%20feature%2C%20,based%20grouping)), so use them mainly for exploratory plotting rather than formal clustering.

# Unsupervised Analysis (Clustering)  
- **Feature Selection:** Use the engineered features (e.g. the four age/gender counts) for clustering. Remove redundant or highly correlated variables to avoid noisy clusters (as done previously by selecting four independent features).  
- **Clustering Algorithms:** Apply multiple clustering methods to identify spatial patterns of missing cases. For example:  
  - **K-Means:** Good for finding compact, globular clusters. Determine *k* via the Elbow Method or gap statistics. Evaluate cluster quality with the Silhouette Score (ranges −1 to 1; values >0.7 indicate strong clustering and the Davies–Bouldin Index (lower values are better) . The prior analysis found K-Means with *k=3* yielded a silhouette ~0.94, indicating very tight clusters.  
  - **DBSCAN or HDBSCAN:** Density-based clustering can detect irregular hotspot shapes and outliers. Tune the `eps` parameter (via a k-distance graph) and `min_samples` to identify dense regions. As previously observed, DBSCAN effectively flagged outlier districts (noise) and complex clusters ([UML_LabExam_FinalPresentation.pdf](file://file-CST6XZVTWN6JhMgMmEhR3e#:~:text=2.%20%20DBSCAN%20%28Density,spotting%20outliers%20and%20dense%20areas)). Report its Silhouette and DB scores too.  
  - **Agglomerative Hierarchical Clustering:** Use linkage (e.g. Ward’s method) to capture hierarchical relations (e.g. districts within states). Set a reasonable number of clusters or distance threshold. This complements K-Means by revealing nested structure ([UML_LabExam_FinalPresentation.pdf](file://file-CST6XZVTWN6JhMgMmEhR3e#:~:text=3,Means)).  
- **Cluster Validation:** Besides internal metrics (Silhouette, DBI), examine cluster stability under different parameters. Map clusters geographically (scatterplot or GIS) to see if they correspond to contiguous regions or known hotspots. A high silhouette means each district is well-matched to its cluster and far from others. Seek a configuration where silhouette is high and DBI is low, as done in prior work (DBSCAN had DBI ~0.40, indicating clear separation ([UML_LabExam_FinalPresentation.pdf](file://file-CST6XZVTWN6JhMgMmEhR3e#:~:text=2.%20%20DBSCAN%20%28Density,spotting%20outliers%20and%20dense%20areas))).  

# Supervised Modeling (Classification/Risk Prediction)  
- **Define Targets:** If the new dataset provides labels (e.g. “hotspot” vs “non-hotspot” district), frame a classification problem. If not, you may create a binary or ordinal target by thresholding missing-person rates (e.g. top quartile as hotspots). Alternatively, predict a continuous risk score.  
- **Train-Test Split:** Split data by time or geography. For static spatial risk, a random stratified split might suffice, but consider preserving a hold-out set of states. Use k-fold cross-validation (stratified by label) to robustly estimate performance.  
- **Algorithms:** Use powerful ensemble classifiers that handle mixed data and nonlinearity:  
  - **Random Forests:** Ensemble of decision trees, robust to overfitting, and naturally handles feature interactions.  
  - **XGBoost (Gradient Boosting Trees):** Often outperforms RF on structured data by sequentially focusing on hard cases. Has efficient implementations and regularization.  
  - **Logistic Regression / SVM:** Use as simple baselines, possibly with L1/L2 regularization.  
  - **Semi-Supervised (Self-Training):** If labels are scarce, apply a Self-Training classifier. This iterative method trains a base classifier, labels high-confidence unlabeled points, and retrains. It leverages all data distribution, which can improve the model if unlabeled data is abundant.
- **Evaluation Metrics:** For imbalance or binary risk classification, focus on precision/recall and AUC rather than accuracy. Compute:  
  - **F1-Score:** The harmonic mean of precision and recall, reflecting balance between false positives and false negatives. A high F1 indicates good overall detection of hotspots without too many false alarms.  
  - **ROC-AUC:** Probability that a random hotspot ranks above a random non-hotspot. AUC=1 is perfect, 0.5 is random. Use ROC-AUC to compare models and tune probability thresholds.  
  - **Precision-Recall AUC:** If hotspots are rare, PR-AUC can be more informative than ROC-AUC (as noted in comparable hotspot studies.
  - **Silhouette for Clustering (again):** If you cluster as a pseudo-classifier (e.g. assign cluster IDs as labels), report silhouette and DBI of the final clustering.  
- **Hyperparameter Tuning:** Perform grid search or Bayesian optimization for key hyperparameters (number of trees, depth, learning rate, etc.), using cross-validation. Monitor metrics like AUC or F1. For example, XGBoost often uses `max_depth`, `eta`, and `subsample` tuning.  
- **Feature Importance and Interpretability:** After training, analyze which features drive predictions. Use:  
  - **Gini Importance / Permutation Importance:** From tree ensembles, to rank features by reduction in impurity or performance drop.  
  - **SHAP Values:** For local/global interpretability. SHAP values quantify each feature’s contribution to each prediction. A summary plot can highlight the most influential factors for hotspot risk. 
  
  - **Partial Dependence Plots:** Show how predicted risk changes with one feature, holding others fixed. Useful for understanding nonlinear effects.

# Visualization and Interpretation  
- **Geographic Mapping:** Create district-level maps to communicate results. For instance:  
  - **Choropleth Maps:** Shade each district by the predicted risk score or class (hotspot vs not). A choropleth “colors each area by an aggregate statistic”. Use consistent color scales (e.g. red for high-risk, green for low).  
  - **Cluster Maps:** If using K-Means or similar, map cluster memberships by color to see spatial groupings. Overlay cluster centroids or boundaries for clarity.  
  - **Intensity/Heat Maps:** If you have coordinates of missing-person events (point data), a heatmap or kernel density estimate can show “hot” zones. Even without point data, one can plot densities per district.  
- **Interpretation Aids:**  
  - Use **legend and annotation** to clarify categories or thresholds.  
  - If using shapefiles or GeoJSON, tools like GeoPandas or Folium can generate interactive maps. Static maps can be saved as images.  
  - Overlay clusters on state outlines to see if high-risk clusters cross state borders or align with known regions.  
- **Evaluation of Spatial Predictions:** Verify that predicted hotspots correspond to known problematic districts (if any ground truth exists). Use geographic cross-validation (e.g. leave-one-state-out) to test robustness.  
- **Model Monitoring:** Plot ROC and PR curves, and confusion matrices. For spatial analysis, consider plotting residuals (predicted minus actual) on a map to spot systematic errors (e.g. rural districts underpredicted).  

# Hotspot Risk Map Construction  
- **Aggregate Risk Scores:** From the classifier, use predicted probabilities of being a hotspot as risk scores. Rank districts by this score.  
- **Risk Categories:** Define bins (e.g. “High”, “Medium”, “Low” risk) based on quantiles or decision thresholds that balance precision/recall. Label each district accordingly.  
- **Final Map:** Produce a final choropleth (or thematic) map where districts are colored by risk category. Include state boundaries for context. This “Hotspot Risk Map” can guide policymakers: high-risk areas might need targeted intervention.  
- **Communicating Uncertainty:** Optionally, show confidence (e.g. semi-transparent shading for uncertain predictions). Alternatively, map predictive probability surface.  
- **Validation Layers:** Overlay other relevant layers (e.g. population centers, police jurisdictions) to interpret risk in context.  

**In summary**, this methodology integrates careful **data cleaning and feature selection** (removing outliers like aggregate totals and redundant features), exploratory **clustering** (with metrics like silhouette and Davies–Bouldin to assess compactness/separation , and **supervised classification** (using Random Forests/XGBoost with evaluation via AUC and F1. Semi-supervised **Self-Training** can exploit unlabeled data. Finally, results are made transparent with model interpretability (feature importances, SHAP) and **geographic visualizations** (choropleth and cluster maps. The ultimate output is a data-driven “Hotspot Risk Map” highlighting districts by predicted missing-person risk level, aiding targeted action. 

