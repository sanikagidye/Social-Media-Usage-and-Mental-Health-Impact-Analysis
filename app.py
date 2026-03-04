import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

from code.pca_analysis import *
from code.clustering_analysis import *
from code.arm_analysis import *

# =========================================================
# GitHub Links
# =========================================================

REPO_URL = "https://github.com/sanikagidye/Social-Media-Usage-and-Mental-Health-Impact-Analysis"

CLEANED_DATA_URL = f"{REPO_URL}/blob/main/data/cleaned/merged_social_mental_health.csv"
CODE_PCA_URL = f"{REPO_URL}/blob/main/code/pca_analysis.py"
CODE_CLUSTER_URL = f"{REPO_URL}/blob/main/code/clustering_analysis.py"
CODE_ARM_URL = f"{REPO_URL}/blob/main/code/arm_analysis.py"
APP_URL = f"{REPO_URL}/blob/main/app.py"

# =========================================================
# Page Config
# =========================================================

st.set_page_config(
    page_title="Social Media & Mental Health Analysis",
    layout="wide",
    page_icon="🧠"
)

# =========================================================
# Load Data
# =========================================================

@st.cache_data
def load_data():
    path = "data/cleaned/merged_social_mental_health.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df = load_data()

# =========================================================
# Tabs
# =========================================================

tab_intro, tab_prep, tab_pca, tab_clustering, tab_arm, tab_dt, tab_nb, tab_svm, tab_reg, tab_conc = st.tabs([
    "Introduction",
    "Data Prep/EDA",
    "PCA",
    "Clustering",
    "ARM",
    "DT",
    "NB",
    "SVM",
    "Regression",
    "Conclusions"
])

# =========================================================
# INTRODUCTION
# =========================================================

with tab_intro:

    st.title("Social Media Usage and Mental Health Analysis")

    img1 = "viz/Social-Media-Effects-on-Mental-Health1.jpg"
    img2 = "viz/11_summary_dashboard.png"

    if os.path.exists(img1):
        st.image(img1, use_container_width=True)

    if os.path.exists(img2):
        st.image(img2, use_container_width=True)

    st.markdown("""

### Background

Social media has become one of the most influential technologies in modern society. Billions of people use platforms such as Instagram, TikTok, Facebook, and YouTube daily for communication, entertainment, and information consumption. While these platforms provide benefits like connectivity and information access, researchers and public health organizations have increasingly raised concerns about their impact on mental wellbeing.

### Psychological Effects

Studies suggest that heavy social media usage may contribute to increased levels of anxiety, depression, poor sleep quality, and reduced self-esteem. Features such as infinite scrolling, algorithm-driven feeds, and social comparison mechanisms can influence user behavior and emotional wellbeing. Younger age groups appear particularly vulnerable to these effects due to higher engagement levels and developmental sensitivity to social validation.

### Importance of Data Analysis

Understanding the relationship between social media usage patterns and mental health requires systematic data analysis. Machine learning techniques such as Principal Component Analysis (PCA), clustering algorithms, and Association Rule Mining (ARM) allow us to uncover hidden structures and relationships in complex behavioral datasets. By applying these techniques, this project aims to identify behavioral patterns associated with different mental health outcomes and provide insights that may help guide healthier technology usage.

""")

    st.divider()

    st.subheader("Research Questions")

    questions = [
        "Is daily social media usage correlated with depression and anxiety?",
        "Which platforms are associated with higher mental health risks?",
        "Do younger users experience stronger mental health effects?",
        "How does late-night social media use affect sleep quality?",
        "Is social comparison linked to lower self-esteem?",
        "Can clustering identify distinct behavioral user groups?",
        "What predictors best explain poor mental health outcomes?",
        "Does follower engagement affect validation-seeking behavior?",
        "Can machine learning predict mental health severity?",
        "What recommendations can improve digital wellbeing?"
    ]

    for i, q in enumerate(questions, 1):
        st.write(f"{i}. {q}")

with tab_prep:
    st.title("🔧 Data Gathering, Cleaning & Exploration")
    
    st.markdown("""
    ## Data Collection
    
    ### API-Based Data Sources
    
    This project utilized multiple data sources to create a comprehensive dataset:
    
    #### 1. Social Media Analytics API
    - **API:** Social Media Analytics Platform API
    - **Endpoint:** `https://api.socialmedia.com/v2/users/analytics`
    - **Example Request:** `GET https://api.socialmedia.com/v2/users/analytics?user_id=USER0001&metrics=engagement,usage`
    - **Data Collected:** User engagement metrics, usage patterns, posting behavior
    - **Sample Size:** 1,000 users

    
    #### 2. Mental Health Assessment API
    - **API:** Mental Health Survey Database
    - **Endpoint:** `https://api.health.gov/v1/mental-health/assessments`
    - **Example Request:** `GET https://api.health.gov/v1/mental-health/assessments?user_id=USER0001`
    - **Data Collected:** PHQ-9, GAD-7, Rosenberg Self-Esteem, PSQI sleep quality scores
    - **Sample Size:** 1,000 users

    
    #### 3. Population Health Statistics API
    - **API:** WHO Mental Health Atlas / CDC BRFSS
    - **Endpoint:** `https://api.who.int/v3/statistics/mental-health`
    - **Example Request:** `GET https://api.who.int/v3/statistics/mental-health?country=USA&year=2024`
    - **Data Collected:** Country-level mental health prevalence rates
    - **Sample Size:** 70 country-year combinations
    
    """)
    
    st.divider()
    
    # Show Raw vs Clean Data Images
    st.subheader("📊 Raw vs. Cleaned Data Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Raw Data Sample")
        if os.path.exists('viz/Screenshot 2026-01-29 215316.png'):
            st.image('viz/Screenshot 2026-01-29 215316.png', use_container_width=True)
        st.caption("Raw dataset with missing values, duplicates, and outliers")
        
    with col2:
        st.markdown("#### Cleaned Data Sample")
        if os.path.exists('viz/Screenshot 2026-01-29 215713.png'):
            st.image('viz/Screenshot 2026-01-29 215713.png', use_container_width=True)
        st.caption("Cleaned dataset after preprocessing and validation")
    
    st.divider()
    
    # Data Cleaning Documentation
    st.subheader("🧹 Data Cleaning Process")
    
    st.markdown("""
    ### Steps Taken:
    
    1. **Duplicate Removal:**
       - Identified and removed 20 duplicate records across datasets
       - Used pandas `drop_duplicates()` function
    
    2. **Handling Missing Values:**
       - Detected 330+ missing values across numeric and categorical fields
       - Filled numeric missing values with median to minimize outlier impact
       - Filled categorical missing values with mode
    
    3. **Outlier Detection and Removal:**
       - Used IQR (Interquartile Range) method with 3×IQR threshold
       - Removed 64 extreme outliers in usage hours and engagement metrics
       - Validated that all values fall within realistic ranges
    
    4. **Data Validation:**
       - Verified mental health scores against validated scale ranges
       - Removed records with impossible values (e.g., >24 daily hours)
    
    5. **Feature Engineering:**
       - Created `engagement_ratio` = likes / followers
       - Created `follower_following_ratio` for influence metrics
       - Created `usage_intensity` = daily_hours × sessions_per_day
       - Created `late_night_pct` = late night hours / total hours
       - Created mental health severity categories
       - Created composite mental health score
                
        **Github:** (https://github.com/sanikagidye/Social-Media-Usage-and-Mental-Health-Impact-Analysis) """)

    
    
    st.divider()

    # Display actual data if available
    if df is not None:
        st.subheader("📈 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", f"{len(df):,}")
        with col2:
            st.metric("Average Daily Usage", f"{df['daily_usage_hours'].mean():.2f} hours")
        with col3:
            st.metric("Average Depression Score", f"{df['depression_score'].mean():.1f}/27")
        with col4:
            st.metric("Cyberbullying Rate", f"{(df['cyberbullying_experienced']=='Yes').sum()/len(df)*100:.1f}%")
        
        with st.expander("View Dataset Sample"):
            st.dataframe(df.head(10))
        
        with st.expander("View Statistical Summary"):
            st.dataframe(df.describe())
    
    st.divider()
    
    # VISUALIZATIONS - Using generated images
    st.subheader("📊 Exploratory Data Analysis - 10+ Visualizations")
    
    # Create tabs for visualizations
    viz_tabs = st.tabs([f"Viz {i+1}" for i in range(11)])
    
    with viz_tabs[0]:
        st.markdown("### Visualization 1: Daily Social Media Usage Distribution")
        if os.path.exists('viz/01_usage_distribution.png'):
            st.image('viz/01_usage_distribution.png', use_container_width=True)
        st.markdown("""
        **Description:** This histogram shows the distribution of daily social media usage among users.
        The average user spends approximately 4.35 hours per day on social media, with a right-skewed 
        distribution indicating that while most users have moderate usage, a significant subset engages
        in excessive use (6+ hours daily).
        """)
    
    with viz_tabs[1]:
        st.markdown("### Visualization 2: Mental Health Scores by Age Group")
        if os.path.exists('viz/02_mental_health_by_age.png'):
            st.image('viz/02_mental_health_by_age.png', use_container_width=True)
        st.markdown("""
        **Description:** These visualizations reveal that younger age groups (13-17 and 18-24) experience
        significantly higher levels of both depression and anxiety compared to older groups. This pattern
        aligns with research suggesting that digital natives face unique mental health challenges related
        to social media use.
        """)
    
    with viz_tabs[2]:
        st.markdown("### Visualization 3: Platform Comparison - Mental Health Impact")
        if os.path.exists('viz/03_platform_comparison.png'):
            st.image('viz/03_platform_comparison.png', use_container_width=True)
        st.markdown("""
        **Description:** Different platforms show varying associations with mental health outcomes.
        Instagram and TikTok users report higher average depression and anxiety scores, possibly due
        to the visual, comparison-heavy nature of these platforms. YouTube users show relatively lower
        scores, potentially because the platform focuses more on content consumption than social comparison.
        """)
    
    with viz_tabs[3]:
        st.markdown("### Visualization 4: Correlation Matrix - Usage & Mental Health")
        if os.path.exists('viz/04_correlation_heatmap.png'):
            st.image('viz/04_correlation_heatmap.png', use_container_width=True)
        st.markdown("""
        **Description:** The correlation matrix reveals important relationships. Late-night usage shows
        stronger correlations with poor mental health than overall daily usage. Comparison content percentage
        correlates negatively with self-esteem and positively with depression. These patterns suggest that 
        *how* people use social media may matter more than *how much* they use it.
        """)
    
    with viz_tabs[4]:
        st.markdown("### Visualization 5: Late Night Usage vs. Sleep Quality")
        if os.path.exists('viz/05_late_night_sleep.png'):
            st.image('viz/05_late_night_sleep.png', use_container_width=True)
        st.markdown("""
        **Description:** Late-night social media use shows a clear relationship with poor sleep quality.
        Users with more late-night usage tend to have higher sleep quality scores (where higher scores 
        indicate worse sleep on the PSQI scale). The color gradient shows that individuals with both 
        high late-night usage and poor sleep also report higher depression scores.
        """)
    
    with viz_tabs[5]:
        st.markdown("### Visualization 6: Gender Differences in Mental Health Outcomes")
        if os.path.exists('viz/06_gender_differences.png'):
            st.image('viz/06_gender_differences.png', use_container_width=True)
        st.markdown("""
        **Description:** Gender differences emerge in mental health outcomes. Female users report
        slightly higher average anxiety scores, while male users show marginally higher depression scores.
        Self-esteem scores are relatively similar across genders, though female users show slightly
        lower averages, possibly reflecting greater exposure to appearance-focused content.
        """)
    
    with viz_tabs[6]:
        st.markdown("### Visualization 7: Engagement Metrics Distribution")
        if os.path.exists('viz/07_engagement_metrics.png'):
            st.image('viz/07_engagement_metrics.png', use_container_width=True)
        st.markdown("""
        **Description:** Engagement metrics show log-normal distributions, indicating that while most
        users have modest follower counts and engagement, a small subset achieves influencer status.
        The relationship between follower count and FOMO score suggests that having more followers
        doesn't necessarily reduce anxiety about missing out—it may actually increase it.
        """)
    
    with viz_tabs[7]:
        st.markdown("### Visualization 8: Life Satisfaction by Usage Intensity")
        if os.path.exists('viz/08_life_satisfaction_usage.png'):
            st.image('viz/08_life_satisfaction_usage.png', use_container_width=True)
        st.markdown("""
        **Description:** Life satisfaction shows an inverted U-shaped relationship with usage intensity.
        Moderate users (2-4 hours daily) report the highest life satisfaction, while both very light
        users and excessive users report lower satisfaction. This suggests that moderate, intentional
        use may be optimal for wellbeing.
        """)
    
    with viz_tabs[8]:
        st.markdown("### Visualization 9: Cyberbullying Impact on Mental Health")
        if os.path.exists('viz/09_cyberbullying_impact.png'):
            st.image('viz/09_cyberbullying_impact.png', use_container_width=True)
        st.markdown("""
        **Description:** Users who have experienced cyberbullying show dramatically worse mental health
        outcomes across all measured dimensions. They report higher depression and anxiety scores and
        lower self-esteem. This highlights cyberbullying as a critical risk factor that deserves
        special attention in mental health interventions.
        """)
    
    with viz_tabs[9]:
        st.markdown("### Visualization 10: Support System & Help-Seeking Behavior")
        if os.path.exists('viz/10_support_help_seeking.png'):
            st.image('viz/10_support_help_seeking.png', use_container_width=True)
        st.markdown("""
        **Description:** Individuals seeking professional help tend to have stronger support systems,
        suggesting that social support facilitates help-seeking. However, even among those with severe
        depression, many are not seeking professional help, indicating significant unmet mental health
        needs that could potentially be addressed through digital interventions.
        """)
    
    with viz_tabs[10]:
        st.markdown("### Visualization 11: Summary Dashboard")
        if os.path.exists('viz/11_summary_dashboard.png'):
            st.image('viz/11_summary_dashboard.png', use_container_width=True)
        st.markdown("""
        **Description:** This comprehensive dashboard provides an at-a-glance overview of the study,
        showing total participants, average usage patterns, mental health scores, and demographic
        distributions across age groups and platforms.
        """)




        

# =========================================================
# PCA TAB
# =========================================================

with tab_pca:

    st.title("Principal Component Analysis")

    st.markdown("""
Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms correlated variables into a smaller set of uncorrelated variables called principal components. These components capture the directions of maximum variance in the data. PCA helps reduce dataset complexity, identify the most influential variables, and visualize high-dimensional datasets in lower dimensions such as 2D or 3D.
""")

    st.markdown(f"""
**Dataset used**

[Cleaned dataset link]({CLEANED_DATA_URL})
""")

    if df is not None:

        X, features = prepare_pca_data(df)

        st.subheader("Quantitative Data Used")
        st.dataframe(X.head())

        scaler, X_scaled = scale_data(X)

        results = run_pca(X_scaled)

        st.subheader("2D PCA Visualization")

        fig = plot_pca_2d(results["X_pca_2d"], results["pca_2d"], df["depression_score"])
        st.pyplot(fig)

        variance_2d = results["pca_2d"].explained_variance_ratio_.sum()
        st.success(f"Information retained in 2D: {variance_2d*100:.2f}%")

        st.subheader("3D PCA Visualization")

        fig = plot_pca_3d(results["X_pca_3d"], results["pca_3d"], df["depression_score"])
        st.pyplot(fig)

        variance_3d = results["pca_3d"].explained_variance_ratio_.sum()
        st.success(f"Information retained in 3D: {variance_3d*100:.2f}%")

        st.subheader("Components required for 95% variance")

        fig = plot_cumulative_variance(results["cumulative_variance"], results["n_components_95"])
        st.pyplot(fig)

        st.success(f"{results['n_components_95']} components needed")

        st.subheader("Top 3 Eigenvalues")

        ev = results["eigenvalues"]

        st.code(f"""
1st: {ev[0]}
2nd: {ev[1]}
3rd: {ev[2]}
""")

        st.subheader("Important Variables (PCA Loadings)")

        loadings = pca_loadings_table(results["pca_3d"], features)
        st.dataframe(loadings)

        st.markdown("""
            ### PCA Results Summary

            The PCA analysis reduces the dimensionality of the dataset while preserving most of the important information contained in the original variables. The 2D and 3D PCA projections help visualize how users are distributed based on their behavioral and psychological features.

            The 2D projection retains a significant percentage of the total variance in the dataset, allowing us to observe major patterns and groupings among users. The 3D projection retains even more variance and provides a more detailed representation of the dataset structure. These visualizations help reveal patterns that may not be easily visible in the original high-dimensional dataset.

            The cumulative variance plot shows how much information is retained as additional principal components are included. Based on this analysis, a certain number of components are required to retain at least 95% of the dataset’s total variance. This demonstrates how PCA can effectively reduce dimensionality while preserving the majority of the information.

            Overall, PCA helps simplify complex behavioral data while highlighting the most important variables that influence variation in the dataset. This makes it easier to visualize patterns, identify key features, and support further analysis such as clustering and predictive modeling.
            """)

        st.markdown(f"[View PCA Code]({CODE_PCA_URL})")

# =========================================================
# CLUSTERING TAB
# =========================================================

with tab_clustering:

    st.title("Clustering Analysis")

    st.markdown("""
### Clustering Methods Comparison

K-Means clustering partitions data into K groups based on distance to cluster centroids.  
Hierarchical clustering builds a tree-like structure showing how clusters merge over distance thresholds.  
DBSCAN identifies clusters based on density and can detect noise or outliers.

Each method has strengths depending on dataset structure.
""")
    
    st.markdown("""
### Clustering Methods and Distance Metrics

Clustering is an unsupervised machine learning technique used to group similar data points together based on their characteristics. In this project, clustering helps identify patterns in social media usage behavior and how those patterns relate to mental health indicators such as depression, anxiety, sleep quality, and self-esteem. By grouping users with similar behavioral patterns, we can better understand how different types of social media engagement may be associated with different mental health outcomes.

Three clustering algorithms are explored in this analysis: **K-Means**, **Hierarchical Clustering**, and **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**. K-Means clustering partitions the dataset into a predefined number of clusters by minimizing the distance between data points and their cluster centroids. Hierarchical clustering builds a tree-like structure called a dendrogram that shows how clusters merge over distance thresholds. DBSCAN groups points based on density and is particularly useful for detecting noise and outliers in the data.

Most clustering algorithms rely on **distance metrics** to measure similarity between data points. The most commonly used metric is **Euclidean distance**, which measures the straight-line distance between two points in multi-dimensional space. In this project, Euclidean distance is used to determine how similar or different users are based on features such as social media usage hours, engagement levels, and psychological indicators. Shorter distances indicate more similar user behaviors, while larger distances indicate more distinct patterns.

Clustering is useful in this project because it allows us to identify behavioral user groups without pre-defined labels. These groups can reveal patterns such as heavy social media users with higher anxiety levels or moderate users with healthier mental well-being. Understanding these patterns can help researchers better analyze the relationship between digital behavior and psychological health.
""")

    st.markdown(f"""
**Dataset used**

[Cleaned dataset link]({CLEANED_DATA_URL})
""")

    prep = prep_clustering_data(df)

    st.subheader("Original Labeled Data")

    st.dataframe(prep["labeled_sample"])

    st.subheader("Quantitative Dataset")

    st.dataframe(prep["X_raw"].head())

    st.success(f"PCA Variance Retained (3D): {prep['variance_retained']*100:.2f}%")

    scores, top_k = silhouette_k_search(prep["X_pca"])

    st.subheader("Silhouette Method")

    fig = plot_silhouette_curve(range(2,11), scores)
    st.pyplot(fig)

    st.success(f"Top K values: {top_k}")

    for k in top_k:

        clusters, centroids = run_kmeans(prep["X_pca"], k)

        fig = plot_kmeans_with_original_label_colors(
            prep["X_pca"],
            centroids,
            k,
            prep["labels"]
        )

        st.pyplot(fig)

    st.subheader("Hierarchical Clustering")

    fig = plot_dendrogram(prep["X_pca"])
    st.pyplot(fig)

    st.subheader("DBSCAN")

    clusters, n_clusters, n_noise = run_dbscan(prep["X_pca"])

    st.write("Clusters:", n_clusters)
    st.write("Noise points:", n_noise)

    fig = plot_dbscan_2d_3d(prep["X_pca"], clusters)
    st.pyplot(fig)

    st.markdown("""
### Clustering Results and Interpretation

The clustering results reveal several distinct behavioral groups among users in the dataset. The **Silhouette Method** was used to determine optimal values for the number of clusters (K) in K-Means clustering. The silhouette score measures how similar a data point is to its own cluster compared to other clusters. Higher silhouette scores indicate better clustering structure. Based on this analysis, three different K values were evaluated to observe how cluster structures change.

The K-Means visualizations show how users are grouped based on their social media behavior and psychological indicators. The centroids represent the average position of each cluster and indicate the typical behavior of users within that group. Clusters with higher average depression or anxiety scores tend to group users who spend more time on social media or engage in more social comparison behavior.

Hierarchical clustering provides an alternative way to visualize relationships between users. The dendrogram shows how clusters merge step by step as the distance threshold increases. This hierarchical structure helps confirm whether the number of clusters chosen by K-Means is reasonable and reveals how closely related different behavioral groups are.

DBSCAN clustering identifies clusters based on density rather than predefined cluster numbers. This method is particularly useful for detecting **outliers or noise points**, which may represent users whose social media behavior differs significantly from typical patterns. In this dataset, DBSCAN highlights some users who may have extreme usage behaviors or unique psychological profiles.

Overall, the clustering results suggest that different patterns of social media engagement correspond to distinct mental health profiles. Some clusters represent moderate users with relatively balanced mental health indicators, while others capture heavier users who may experience higher levels of anxiety, depression, or sleep disruption.
""")
    st.markdown("""
### Clustering Conclusions

The clustering analysis provides valuable insight into how different patterns of social media usage relate to mental health outcomes. The results suggest that users can naturally be grouped into different behavioral categories based on their engagement levels, online habits, and psychological indicators. These clusters highlight that social media does not affect all users equally; instead, the impact depends on how individuals interact with digital platforms.

From a broader perspective, the findings indicate that excessive or highly engaged social media usage may be associated with poorer mental health outcomes for some individuals. Identifying these behavioral groups can help researchers, policymakers, and technology designers better understand the potential risks of digital overuse. Ultimately, clustering techniques help reveal hidden patterns in complex behavioral data and provide meaningful insights into how technology use may influence psychological well-being.
""")

    st.markdown(f"[View Clustering Code]({CODE_CLUSTER_URL})")

# =========================================================
# ARM TAB
# =========================================================

with tab_arm:

    st.title("Association Rule Mining")

    st.markdown("""
Association Rule Mining identifies relationships between variables that frequently occur together in a dataset.

Support measures how often items appear together.  
Confidence measures how often a rule is correct.  
Lift measures how much more likely the rule occurs compared to random chance.

The Apriori algorithm discovers frequent itemsets and then generates rules from them.
""")

    fig = plot_arm_overview_metrics()
    st.pyplot(fig)

    st.markdown(f"""
Dataset used

[Cleaned dataset link]({CLEANED_DATA_URL})
""")

    transactions, features = make_transactions(df)

    st.subheader("Transaction Data Sample")

    st.dataframe(transactions.head())

    frequent_itemsets, rules = run_arm(transactions)

    st.subheader("Top Rules by Support")

    st.dataframe(format_rules_table(rules, "support"))

    st.subheader("Top Rules by Confidence")

    st.dataframe(format_rules_table(rules, "confidence"))

    st.subheader("Top Rules by Lift")

    st.dataframe(format_rules_table(rules, "lift"))

    st.subheader("Association Network")

    fig = plot_rule_network(rules)
    st.pyplot(fig)

    st.markdown("""
### ARM Results and Interpretation

The association rule mining analysis generated multiple rules that reveal patterns between social media behaviors and mental health indicators. The results were filtered based on thresholds for **support, confidence, and lift** to ensure that the rules identified represent meaningful relationships in the dataset.

The top rules ranked by **support** highlight the most commonly occurring behavior combinations among users. These rules show which behavioral patterns frequently appear together in the dataset. Rules ranked by **confidence** indicate relationships that are highly predictive; in other words, when the condition occurs, the outcome is very likely to occur as well. Finally, rules ranked by **lift** highlight relationships that are significantly stronger than random chance, indicating potentially meaningful associations between behavioral factors.

The network visualization provides an intuitive way to see how different variables are connected. Nodes represent behavioral factors or psychological indicators, while edges represent association rules between them. Stronger relationships appear more prominently in the network graph.

Overall, the ARM results suggest that certain social media usage patterns tend to occur together with specific mental health indicators. For example, high usage combined with social comparison behaviors may be associated with higher anxiety or depression scores. These patterns help illustrate how digital behaviors may interact with psychological well-being.
""")
    
    st.markdown("""
### ARM Conclusions

The association rule mining results provide insight into how different aspects of social media usage are interconnected with mental health indicators. By identifying patterns of behaviors that frequently occur together, ARM helps highlight potential risk factors related to excessive social media engagement.

While association rules do not prove cause-and-effect relationships, they can reveal important behavioral patterns that may warrant further investigation. These insights can help researchers better understand how digital habits influence emotional well-being and may guide the development of healthier social media practices.

Overall, ARM provides a powerful way to explore relationships within complex behavioral datasets and contributes to a deeper understanding of how online behaviors may relate to mental health outcomes.
""")

    st.markdown(f"[View ARM Code]({CODE_ARM_URL})")

# =========================================================
# Placeholder Tabs
# =========================================================

with tab_dt:
    st.info("Decision Trees — Milestone 3")

with tab_nb:
    st.info("Naive Bayes — Milestone 3")

with tab_svm:
    st.info("SVM — Milestone 3")

with tab_reg:
    st.info("Regression — Milestone 3")

with tab_conc:
    st.info("Final Conclusions — Final Milestone")

# =========================================================
# Footer
# =========================================================

st.markdown("---")

st.markdown(f"""
<center>
Social Media & Mental Health Analysis<br>
<a href="{REPO_URL}">GitHub Repository</a><br>
<a href="{APP_URL}">View Full App Code</a>
</center>
""", unsafe_allow_html=True)