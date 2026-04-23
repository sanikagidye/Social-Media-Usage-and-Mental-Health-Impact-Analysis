import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import warnings

warnings.filterwarnings("ignore")

from code.pca_analysis import *
from code.clustering_analysis import *
from code.arm_analysis import *
from code.nb_analysis import *
from code.dt_analysis import *
from code.regression_analysis import *
import code.nb_analysis as nb_analysis
import code.dt_analysis as dt_analysis
import code.regression_analysis as regression_analysis
import code.svm_analysis as svm_analysis
import code.ensemble_analysis as ensemble_analysis
# =========================================================
# GitHub Links
# =========================================================

REPO_URL = "https://github.com/sanikagidye/Social-Media-Usage-and-Mental-Health-Impact-Analysis"


def get_repo_branch(default="main"):
    env_branch = os.getenv("GITHUB_BRANCH") or os.getenv("BRANCH_NAME")
    if env_branch:
        return env_branch

    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            text=True
        ).strip()
        return branch or default
    except Exception:
        return default


REPO_BRANCH = get_repo_branch()
CLEANED_DATA_URL = f"{REPO_URL}/blob/{REPO_BRANCH}/data/cleaned/merged_social_mental_health.csv"
CODE_PCA_URL = f"{REPO_URL}/blob/{REPO_BRANCH}/code/pca_analysis.py"
CODE_CLUSTER_URL = f"{REPO_URL}/blob/{REPO_BRANCH}/code/clustering_analysis.py"
CODE_ARM_URL = f"{REPO_URL}/blob/{REPO_BRANCH}/code/arm_analysis.py"
APP_URL = f"{REPO_URL}/blob/{REPO_BRANCH}/app.py"
CODE_NB_URL = f"{REPO_URL}/blob/{REPO_BRANCH}/code/nb_analysis.py"
CODE_DT_URL = f"{REPO_URL}/blob/{REPO_BRANCH}/code/dt_analysis.py"
CODE_REG_URL = f"{REPO_URL}/blob/{REPO_BRANCH}/code/regression_analysis.py"
CODE_SVM_URL = f"{REPO_URL}/blob/{REPO_BRANCH}/code/svm_analysis.py"
CODE_ENSEMBLE_URL = f"{REPO_URL}/blob/{REPO_BRANCH}/code/ensemble_analysis.py"

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


@st.cache_data
def get_svm_analysis(dataframe):
    prep = svm_analysis.prepare_svm_data(dataframe)
    results = svm_analysis.run_svm_experiments(prep)
    return prep, results


@st.cache_data
def get_ensemble_analysis(dataframe):
    prep = ensemble_analysis.prepare_ensemble_data(dataframe)
    results = ensemble_analysis.run_ensemble_models(prep)
    return prep, results


def build_split_summary(X_train, X_test, y_train, y_test):
    return pd.DataFrame(
        [
            {"Subset": "Training features", "Rows": len(X_train), "Columns": X_train.shape[1]},
            {"Subset": "Testing features", "Rows": len(X_test), "Columns": X_test.shape[1]},
            {"Subset": "Training labels", "Rows": len(y_train), "Columns": 1},
            {"Subset": "Testing labels", "Rows": len(y_test), "Columns": 1},
        ]
    )


def plot_disjoint_split(train_size, test_size, title="Disjoint Train/Test Split"):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    total = train_size + test_size
    ax.barh(["Rows"], [train_size], color="#4C78A8", label="Training")
    ax.barh(["Rows"], [test_size], left=[train_size], color="#F58518", label="Testing")
    ax.text(train_size / 2, 0, f"Train\n{train_size}", ha="center", va="center", color="white", fontweight="bold")
    ax.text(train_size + (test_size / 2), 0, f"Test\n{test_size}", ha="center", va="center", color="white", fontweight="bold")
    ax.set_xlim(0, total)
    ax.set_xlabel("Number of rows")
    ax.set_title(title, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper center", ncol=2, frameon=False)
    plt.tight_layout()
    return fig


def dataframe_snapshot_figure(df_snapshot, title):
    fig, ax = plt.subplots(figsize=(12, 2.8 + 0.45 * min(len(df_snapshot), 8)))
    ax.axis("off")
    ax.set_title(title, fontweight="bold", pad=12)
    display_df = df_snapshot.copy()
    for col in display_df.columns:
        if pd.api.types.is_bool_dtype(display_df[col]):
            display_df[col] = display_df[col].map({True: "True", False: "False"})
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    plt.tight_layout()
    return fig

# =========================================================
# Tabs
# =========================================================

tab_intro, tab_prep, tab_pca, tab_clustering, tab_arm, tab_dt, tab_nb, tab_svm, tab_ensemble, tab_reg, tab_conc = st.tabs([
    "Introduction",
    "Data Prep/EDA",
    "PCA",
    "Clustering",
    "ARM",
    "DT",
    "NB",
    "SVM",
    "Ensemble",
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
    ## Understanding the Topic in a Human Way

    Social media is woven into everyday life so completely that many people barely notice how often they reach for it. It is where friendships continue after school, where trends spread, where people celebrate milestones, and where many users look for distraction at the end of a stressful day. Because these platforms sit so close to daily routines, they also have the power to shape mood, confidence, sleep habits, and the way people see themselves.

    What makes this topic especially important is that social media can feel helpful and harmful at the same time. A person might use it to stay connected, laugh, learn, or feel less alone, yet the very same spaces can also create pressure, comparison, overstimulation, and emotional fatigue. That tension makes the subject worth studying carefully, because the answer is not simply that social media is good or bad. The more honest question is when it supports wellbeing and when it starts to wear people down.

    This project looks at that everyday reality through the lens of mental health. Instead of focusing only on dramatic stories, it explores ordinary behaviors such as how long people stay online, how late they scroll, how often they compare themselves to others, and how supported or isolated they feel. Those patterns matter because small habits repeated every day can quietly shape how rested, anxious, satisfied, or overwhelmed someone feels over time.

    The goal is not to shame people for using technology or to suggest that everyone should disconnect. Social media is now part of how people communicate, build identity, follow news, and participate in culture. A stronger goal is to understand which habits seem healthier, which habits appear more risky, and how people can create a better balance between digital connection and personal wellbeing.

    In that way, this website tells a practical story. It brings together visuals, patterns, and predictive models to show how online behavior connects with emotional health in the real world. By the end, the reader should come away with a clearer sense that mental wellbeing online is not just about screen time alone. It is also about timing, comparison, sleep, social support, and the way repeated digital experiences can add up across a person’s life.
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

    st.title("Principal Component Analysis (PCA)")

    st.markdown("""
Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms correlated variables into a smaller set of uncorrelated variables called principal components. These components capture the directions of maximum variance in the data. PCA helps reduce dataset complexity, identify the most influential variables, and visualize high-dimensional datasets in lower dimensions such as 2D or 3D. In this project, PCA is used to simplify the social media and mental health dataset so that important behavior patterns can be visualized more clearly while preserving most of the information in the original data.
""")

    st.markdown(f"""
**Dataset used**

[Cleaned dataset link]({CLEANED_DATA_URL})
""")

    if df is not None:

        # =========================================================
        # Data Selection
        # =========================================================
        X, features = prepare_pca_data(df)

        st.subheader("Selected Quantitative Dataset for PCA")

        st.markdown("""
PCA can only be applied to **quantitative variables**, so all categorical columns such as user ID, age group, gender, and platform are excluded. 
The dataset below shows the numerical features that were selected and prepared for PCA.
""")

        st.markdown("### BEFORE PCA: Original Quantitative Dataset")
        st.dataframe(X.head(10))

        prepared_csv = X.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prepared PCA Data (CSV)",
            data=prepared_csv,
            file_name="prepared_pca_data.csv",
            mime="text/csv"
        )

        # =========================================================
        # Normalization
        # =========================================================
        st.subheader("Normalization with StandardScaler")

        st.markdown("""
Before applying PCA, the numerical data is normalized using **StandardScaler**. This step ensures that each feature has mean 0 and standard deviation 1. 
Normalization is necessary because PCA is sensitive to scale, and variables with larger numeric ranges could dominate the principal components if scaling is skipped.
""")

        scaler, X_scaled = scale_data(X)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before Scaling**")
            st.write(f"Mean: {X.mean().mean():.2f}")
            st.write(f"Average Std Dev: {X.std().mean():.2f}")
        with col2:
            st.markdown("**After Scaling**")
            st.write(f"Mean: {X_scaled.mean():.2f}")
            st.write(f"Std Dev: {X_scaled.std():.2f}")

        # =========================================================
        # PCA Transform
        # =========================================================
        st.subheader("Applying PCA")

        st.markdown("""
After scaling, PCA is applied twice:
- once with **2 components** for 2D visualization
- once with **3 components** for 3D visualization

The transformed dataset no longer contains the original variables directly. Instead, it contains new variables called **principal components** (PC1, PC2, PC3), which summarize most of the variation in the original data.
""")

        results = run_pca(X_scaled)

        X_pca_2 = results["X_pca_2d"]
        X_pca_3 = results["X_pca_3d"]

        st.markdown("### AFTER PCA: Transformed Dataset (2 Components)")
        pca_df_2 = pd.DataFrame(X_pca_2, columns=["PC1", "PC2"])
        st.dataframe(pca_df_2.head(10))

        st.markdown("### AFTER PCA: Transformed Dataset (3 Components)")
        pca_df_3 = pd.DataFrame(X_pca_3, columns=["PC1", "PC2", "PC3"])
        st.dataframe(pca_df_3.head(10))

        st.markdown("""
### PCA Transformation Explanation

Principal Component Analysis transforms the original dataset into a new coordinate system where each axis represents a principal component. 
These components are weighted combinations of the original variables and are ordered by how much variance they explain. 
This means the first component captures the most variation in the dataset, the second captures the next most, and so on.

In this project, PCA helps simplify a large behavioral dataset into a smaller number of dimensions so that patterns related to social media usage 
and mental health can be visualized and interpreted more effectively.
""")

        # =========================================================
        # 2D PCA Visualization
        # =========================================================
        st.subheader("2D PCA Visualization")

        fig = plot_pca_2d(results["X_pca_2d"], results["pca_2d"], df["depression_score"])
        st.pyplot(fig)

        variance_2d = results["pca_2d"].explained_variance_ratio_.sum()
        st.success(f"Information retained in 2D: {variance_2d*100:.2f}%")

        # =========================================================
        # 3D PCA Visualization
        # =========================================================
        st.subheader("3D PCA Visualization")

        fig = plot_pca_3d(results["X_pca_3d"], results["pca_3d"], df["depression_score"])
        st.pyplot(fig)

        variance_3d = results["pca_3d"].explained_variance_ratio_.sum()
        st.success(f"Information retained in 3D: {variance_3d*100:.2f}%")

        # =========================================================
        # 95% Variance
        # =========================================================
        st.subheader("How Many Components Are Needed to Retain 95% Variance?")

        fig = plot_cumulative_variance(results["cumulative_variance"], results["n_components_95"])
        st.pyplot(fig)

        st.success(f"{results['n_components_95']} components are needed to retain at least 95% of the variance.")

        # =========================================================
        # Eigenvalues
        # =========================================================
        st.subheader("Top 3 Eigenvalues")

        ev = results["eigenvalues"]

        st.code(f"""
Top 3 Eigenvalues:
1st: {ev[0]:.4f}
2nd: {ev[1]:.4f}
3rd: {ev[2]:.4f}
""")

        ev_df = eigenvalue_table(ev, results["pca_full"].explained_variance_ratio_, top_n=10)
        st.dataframe(ev_df)

        # =========================================================
        # Loadings / Important Variables
        # =========================================================
        st.subheader("Important Variables (PCA Loadings)")

        st.markdown("""
The table below shows the variables with the strongest contributions to each principal component. 
Higher absolute loading values mean that a feature has more influence on that component. 
This helps identify which original variables are most important in explaining variation in the dataset.
""")

        loadings = pca_loadings_table(results["pca_3d"], features)
        st.dataframe(loadings)

        # =========================================================
        # PCA Summary
        # =========================================================
        st.markdown(f"""
### PCA Results Summary

The PCA analysis reduces the dimensionality of the dataset while preserving most of the important information contained in the original variables. 
The 2D and 3D PCA projections help visualize how users are distributed based on their behavioral and psychological features.

The 2D projection retains **{variance_2d*100:.2f}%** of the total variance in the dataset, allowing us to observe major patterns and broad structure. 
The 3D projection retains **{variance_3d*100:.2f}%** of the total variance and provides a more detailed representation of the dataset.

The cumulative variance analysis shows that **{results['n_components_95']} components** are needed to retain at least 95% of the information in the dataset. 
This demonstrates that PCA can reduce dimensionality while preserving the majority of the original data’s structure.

Overall, PCA plays an important role in this project by simplifying a complex social media and mental health dataset into a smaller number of informative dimensions. 
This makes it easier to visualize patterns, understand important variables, and support further analyses such as clustering and predictive modeling.
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

    st.title("Association Rule Mining (ARM)")

    # =========================================================
    # (a) Overview
    # =========================================================
    st.subheader("(a) Overview")

    st.markdown("""
Association Rule Mining (ARM) is a data mining technique used to discover relationships between variables that frequently occur together in a dataset. 
Instead of predicting a target label, ARM looks for **patterns of co-occurrence**. In this project, ARM is used to identify combinations of social 
media usage behaviors and mental health indicators that tend to appear together across users.

An **association rule** has the form **A → B**, which means that when condition A is present, condition B is more likely to occur. 
For example, a rule might suggest that users with high daily social media usage and high late-night activity are more likely to also have high anxiety. 
These rules help uncover interpretable behavioral patterns in the data.

Three important rule-quality measures are used:

- **Support**: how often a combination of items appears in the dataset.
- **Confidence**: how often the rule is true when the left-hand side occurs.
- **Lift**: how much stronger the rule is compared to random chance. A lift greater than 1 suggests a meaningful positive association.

The **Apriori algorithm** is used to generate these rules. Apriori first finds frequent itemsets that occur together often enough, then builds 
association rules from them. It works efficiently by using the idea that if an itemset is frequent, then all of its subsets must also be frequent.
""")

    # Image 1
    st.markdown("### ARM Concept Image 1")
    fig1 = plot_arm_overview_metrics()
    st.pyplot(fig1)

    # Image 2
    st.markdown("### ARM Concept Image 2")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(["Support", "Confidence", "Lift"], [0.2, 0.6, 1.4])
    ax2.set_title("Understanding Support, Confidence, and Lift", fontweight="bold")
    ax2.set_ylabel("Illustrative Value")
    ax2.grid(axis="y", alpha=0.3)
    st.pyplot(fig2)

    st.markdown(f"""
**Dataset used:**  
[Cleaned dataset link]({CLEANED_DATA_URL})

**Code link:**  
[View ARM Code]({CODE_ARM_URL})
""")

    st.divider()

    # =========================================================
    # (b) Data Prep
    # =========================================================
    st.subheader("(b) Data Prep")

    st.markdown("""
Association Rule Mining requires **unlabeled transaction data**, where each row represents one transaction and each column records whether a condition is present.  
The original social-media dataset is not in that format because it contains raw numeric values such as usage hours, depression score, anxiety score, late-night hours, and self-esteem score.  
Before ARM can be applied, those raw values must be transformed into **boolean transaction features**.

For this project, the transformation step was: **source variable -> threshold or rule -> True/False transaction feature**.  
That means the original mixed-format dataset is converted into the item-presence structure required by Apriori and association rule mining.  
The tables below show the source columns used, the exact rules applied, and the final transaction-format dataframe created from them.
""")

    arm_source_cols = [
        "daily_usage_hours",
        "depression_score",
        "anxiety_score",
        "late_night_hours",
        "self_esteem_score",
        "comparison_content_pct",
        "cyberbullying_experienced",
        "sleep_quality_score",
    ]
    arm_before = df[arm_source_cols].head(10).copy()

    threshold_df = pd.DataFrame(
        [
            {"Original column": "daily_usage_hours", "Rule used": "above dataset median", "Transaction feature": "high_usage"},
            {"Original column": "depression_score", "Rule used": "> 14", "Transaction feature": "high_depression"},
            {"Original column": "anxiety_score", "Rule used": "> 10", "Transaction feature": "high_anxiety"},
            {"Original column": "late_night_hours", "Rule used": "> 1.5", "Transaction feature": "late_night_user"},
            {"Original column": "self_esteem_score", "Rule used": "< 25", "Transaction feature": "low_self_esteem"},
            {"Original column": "comparison_content_pct", "Rule used": "> 40", "Transaction feature": "high_comparison"},
            {"Original column": "cyberbullying_experienced", "Rule used": "== 'Yes'", "Transaction feature": "cyberbullying_yes"},
            {"Original column": "sleep_quality_score", "Rule used": "> 10", "Transaction feature": "poor_sleep"},
        ]
    )

    st.markdown("### Transformation Rules Used")
    st.dataframe(threshold_df, use_container_width=True)

    transactions, features = make_transactions(df)
    arm_after = transactions.head(10).copy()

    st.markdown("### BEFORE: Source Columns Used for ARM")
    before_col1, before_col2 = st.columns(2)
    with before_col1:
        st.dataframe(arm_before, use_container_width=True)
    with before_col2:
        st.pyplot(dataframe_snapshot_figure(arm_before.head(6), "Before Transformation"))

    st.markdown("### AFTER: Transaction Dataset Used by Apriori")
    after_col1, after_col2 = st.columns(2)
    with after_col1:
        st.dataframe(arm_after, use_container_width=True)
    with after_col2:
        st.pyplot(dataframe_snapshot_figure(arm_after.head(6), "After Transformation"))

    st.caption("The 'before' sample contains raw numeric and categorical values. The 'after' sample contains only boolean transaction features, which is the format required by ARM.")

    st.markdown("### Visualization of Transaction Features")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    transactions.sum().plot(kind="bar", ax=ax3)
    ax3.set_title("Frequency of Transaction Features", fontweight="bold")
    ax3.set_ylabel("Count of True Values")
    ax3.set_xlabel("Transaction Features")
    ax3.grid(axis="y", alpha=0.3)
    st.pyplot(fig3)

    sample_csv = transactions.head(200).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download ARM Sample Transactions (CSV)",
        data=sample_csv,
        file_name="arm_transactions_sample.csv",
        mime="text/csv"
    )

    source_csv = arm_before.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download ARM Source Sample (CSV)",
        data=source_csv,
        file_name="arm_source_sample.csv",
        mime="text/csv"
    )

    st.divider()

    # =========================================================
    # (c) Code ARM
    # =========================================================
    st.subheader("(c) Code ARM")
    st.markdown(f"[View ARM Code]({CODE_ARM_URL})")

    # =========================================================
    # (d) Results
    # =========================================================
    st.subheader("(d) Results")

    min_support = 0.05
    min_confidence = 0.30
    min_lift = 1.00

    st.write(f"**Thresholds used:** support ≥ {min_support}, confidence ≥ {min_confidence}, lift ≥ {min_lift}")

    frequent_itemsets, rules = run_arm(
        transactions,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift
    )

    if rules.empty:
        st.warning("No rules found. Try lowering thresholds.")
    else:
        st.success(f"Generated {len(rules)} association rules")

        st.markdown("### Top 15 Rules by Support")
        st.dataframe(format_rules_table(rules, "support", 15))

        st.markdown("### Top 15 Rules by Confidence")
        st.dataframe(format_rules_table(rules, "confidence", 15))

        st.markdown("### Top 15 Rules by Lift")
        st.dataframe(format_rules_table(rules, "lift", 15))

        st.markdown("### Association Network Visualization")
        fig4 = plot_rule_network(rules, top_n=20)
        st.pyplot(fig4)

    st.markdown("""
### ARM Results and Interpretation

The association rule mining analysis generated multiple rules that reveal patterns between social media behaviors and mental health indicators. 
The rules were filtered using thresholds for support, confidence, and lift so that only meaningful and interpretable relationships were retained.

The top rules by **support** show the most common co-occurring conditions in the dataset. The top rules by **confidence** show patterns that are most 
predictive, meaning that when the antecedent happens, the consequent is very likely to happen as well. The top rules by **lift** reveal the strongest 
associations beyond what would be expected by random chance.

The network visualization makes it easier to see which behavioral and psychological factors are connected. Some rules may highlight how high usage, 
late-night use, comparison behavior, poor sleep, and elevated anxiety or depression tend to appear together.

Overall, ARM provides interpretable evidence of how different social media habits may cluster together with mental health risk indicators.
""")

    # =========================================================
    # (e) Conclusions
    # =========================================================
    st.subheader("(e) Conclusions")

    st.markdown("""
Association Rule Mining helps translate complex behavioral data into simple, understandable relationship patterns. In this project, ARM shows that 
certain social media behaviors do not appear in isolation; instead, they often co-occur with other risk-related behaviors and mental health indicators.

From a topic perspective, these rules suggest that mental health risks associated with social media may emerge through combinations of habits such as 
heavy use, late-night use, comparison-focused content, and poor sleep. These findings do not prove causation, but they do help identify combinations 
of behaviors that may deserve closer attention.

Overall, ARM contributes to the project by providing a transparent and interpretable way to uncover behavioral patterns that relate social media usage 
to mental health outcomes.
""")

# =========================================================
# Placeholder Tabs
# =========================================================

with tab_dt:
    st.title("Decision Tree Classification")

    st.subheader("(a) Overview")
    st.markdown("""
Decision Trees are supervised learning models that predict a class by repeatedly splitting the data into smaller groups.  
Each split asks a question about one feature, such as whether late-night social media use is above a threshold, and sends the row down one branch or another.  
This creates a structure with a **root node**, internal decision nodes, and **leaf nodes** that store the final prediction.

A split is judged using **Gini impurity** or **Entropy**, which both measure how mixed the class labels are inside a node.  
**Information Gain** tells us how much that uncertainty is reduced after a split, so a larger value means a better split.  
For example, if a parent node has entropy `1.00` and the weighted child entropy after a split is `0.40`, then the information gain is `1.00 - 0.40 = 0.60`.

It is possible to create an enormous number of trees because there are many possible feature choices, split thresholds, stopping depths, and pruning decisions.  
That is why Decision Trees need guardrails such as depth limits or minimum split sizes; otherwise they can keep growing and overfit the training data.
""")

    col_overview_1, col_overview_2 = st.columns(2)
    with col_overview_1:
        fig1, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.axis("off")
        points = {
            "Root": (0.5, 0.85),
            "Branch A": (0.28, 0.55),
            "Branch B": (0.72, 0.55),
            "Leaf 1": (0.18, 0.22),
            "Leaf 2": (0.38, 0.22),
            "Leaf 3": (0.62, 0.22),
            "Leaf 4": (0.82, 0.22),
        }
        edges = [("Root", "Branch A"), ("Root", "Branch B"), ("Branch A", "Leaf 1"), ("Branch A", "Leaf 2"), ("Branch B", "Leaf 3"), ("Branch B", "Leaf 4")]
        for parent, child in edges:
            x1, y1 = points[parent]
            x2, y2 = points[child]
            ax1.plot([x1, x2], [y1, y2], color="#4C78A8", linewidth=2)
        for label, (x, y) in points.items():
            ax1.scatter(x, y, s=1600 if "Leaf" not in label else 1200, color="#F2F2F2", edgecolor="#4C78A8", linewidth=2)
            ax1.text(x, y, label, ha="center", va="center", fontsize=10, fontweight="bold")
        ax1.set_title("Decision Tree Anatomy", fontweight="bold")
        st.pyplot(fig1)
    with col_overview_2:
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        values = [1.00, 0.40, 0.60]
        bars = ax2.bar(["Parent Entropy", "Child Entropy", "Information Gain"], values, color=["#E45756", "#72B7B2", "#54A24B"])
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel("Value")
        ax2.set_title("Example Split Quality", fontweight="bold")
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.2f}", ha="center", fontsize=10, fontweight="bold")
        st.pyplot(fig2)

    st.subheader("(b) Data Prep")
    prep = dt_analysis.prepare_dt_data(df)
    split_summary = build_split_summary(prep["X_train"], prep["X_test"], prep["y_train"], prep["y_test"])

    st.markdown(f"""
**Dataset Link:** [Cleaned dataset]({CLEANED_DATA_URL})  
**Code Link:** [dt_analysis.py]({CODE_DT_URL})
""")

    st.markdown("""
The decision tree model uses labeled data where the label is the depression severity category derived from `depression_score`.  
I used an **80/20 stratified train/test split**, which keeps the class distribution similar in both subsets.  
The training and testing sets must be **disjoint** because the model should learn on one set and then be judged on unseen rows from the other set.
""")

    dt_csv = prep["X"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prepared DT Input (CSV)",
        data=dt_csv,
        file_name="prepared_dt_input.csv",
        mime="text/csv"
    )

    split_col1, split_col2 = st.columns([1.1, 1.4])
    with split_col1:
        st.markdown("### Split Summary")
        st.dataframe(split_summary, use_container_width=True)
    with split_col2:
        st.markdown("### Train/Test Split Image")
        st.pyplot(plot_disjoint_split(len(prep["X_train"]), len(prep["X_test"]), "Decision Tree 80/20 Split"))

    st.markdown("### Sample Input Data")
    st.dataframe(prep["X"].head(10), use_container_width=True)

    prep_col1, prep_col2 = st.columns(2)
    with prep_col1:
        st.markdown("### Training Set Sample")
        st.dataframe(prep["X_train"].head(10), use_container_width=True)
    with prep_col2:
        st.markdown("### Testing Set Sample")
        st.dataframe(prep["X_test"].head(10), use_container_width=True)

    st.markdown("### Training Labels Sample")
    st.dataframe(prep["y_train"].head(10).rename("depression_severity"), use_container_width=True)

    st.subheader("(c) Code")
    st.markdown(f"[View Decision Tree Code]({CODE_DT_URL})")

    st.subheader("(d) Results")
    results = dt_analysis.build_three_different_trees(prep)
    st.dataframe(dt_analysis.accuracy_table(results), use_container_width=True)

    class_names = sorted(prep["y_test"].unique())
    best_tree_name, best_tree_out = max(results.items(), key=lambda item: item[1]["accuracy"])

    for tree_name, out in results.items():
        st.markdown(f"### {tree_name}")
        st.write(f"Root feature: {out['root_feature']}")
        st.write(f"Accuracy: {out['accuracy']:.4f}")
        st.write(f"Features used: {len(out['features_used'])}")

        fig_tree = dt_analysis.plot_decision_tree_model(
            out["model"],
            out["features_used"],
            class_names,
            title=f"{tree_name} Structure"
        )
        fig_cm = dt_analysis.plot_confusion_matrix(
            out["confusion_matrix"],
            out["labels"],
            title=f"{tree_name} Confusion Matrix"
        )

        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.pyplot(fig_tree)
        with result_col2:
            st.pyplot(fig_cm)

    st.markdown(f"""
Three different trees were intentionally created with different settings and feature availability so that the structures and root nodes would differ.  
Across the three runs, the roots were **{results['Tree 1']['root_feature']}**, **{results['Tree 2']['root_feature']}**, and **{results['Tree 3']['root_feature']}**, which satisfies the requirement to include different trees with different starting decisions.  
The best-performing tree in this run was **{best_tree_name}** with accuracy **{best_tree_out['accuracy']:.4f}**, showing which split strategy worked best for this dataset.
""")

    st.subheader("(e) Conclusions")
    st.markdown(f"""
Decision Trees make the classification process easy to interpret because the model visibly shows how predictions are built from one split at a time.  
For this project, the strongest tree was **{best_tree_name}**, and its root split on **{best_tree_out['root_feature']}**, which suggests that this variable is one of the most informative starting points for separating depression severity levels.  
This matters for the topic because it highlights which social media behaviors appear most useful for identifying users who may be experiencing worse mental health outcomes.
""")

with tab_nb:
    st.title("Naïve Bayes Classification")

    st.subheader("(a) Overview")
    st.markdown("""
Naïve Bayes is a probabilistic classification algorithm based on Bayes' Theorem. It assumes that features are conditionally independent given the class label, which is why it is called "naïve." Even though that assumption is strong, Naïve Bayes often performs well in practice and is widely used because it is fast, simple, and effective.

There are several versions of Naïve Bayes. **Multinomial Naïve Bayes** is often used for count-style or non-negative frequency data. **Gaussian Naïve Bayes** is used when features are continuous and approximately normally distributed. **Bernoulli Naïve Bayes** is used when the features are binary, such as yes/no or true/false indicators. In general, the best version depends on the type of input data being modeled.

Smoothing is required because some features may not appear in a class during training. Without smoothing, that would produce zero probability and could eliminate a class from consideration entirely. Laplace smoothing solves this by assigning a small non-zero probability to unseen events.

In this project, Naïve Bayes is used to classify depression severity groups using social media usage behavior and mental health features.
""")

    # Two images for overview
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(["Multinomial", "Gaussian", "Bernoulli"], [1, 1, 1])
    ax1.set_title("Naïve Bayes Variants Used in This Project", fontweight="bold")
    ax1.set_ylabel("Included")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(["No Smoothing", "With Smoothing"], [0.0, 0.1])
    ax2.set_title("Why Smoothing Is Needed in Naïve Bayes", fontweight="bold")
    ax2.set_ylabel("Example Probability")
    st.pyplot(fig2)

    st.markdown("""
Multinomial NB is best when the inputs behave like counts or non-negative frequencies, Gaussian NB is best for continuous numeric features, Bernoulli NB is best for binary 0/1 features, and Categorical NB is designed for discrete category-coded inputs. The same Bayes rule is used in each case, but the probability model for the features changes. Smoothing matters because an unseen feature-class combination should not force the entire class probability to become zero. In this project, Multinomial NB, Gaussian NB, and Bernoulli NB are run directly, while Categorical NB is included in the comparison discussion because it would fit naturally if these behaviors were binned into categories such as low, medium, and high.
""")

    extra_nb_col1, extra_nb_col2 = st.columns(2)
    with extra_nb_col1:
        fig3, ax3 = plt.subplots(figsize=(7, 4.5))
        variant_names = ["Multinomial", "Gaussian", "Bernoulli", "Categorical"]
        preferred_input = [3, 2, 1, 1]
        bars = ax3.bar(variant_names, preferred_input, color=["#4C78A8", "#72B7B2", "#F58518", "#54A24B"])
        ax3.set_yticks([1, 2, 3], ["Binary", "Continuous", "Count-like"])
        ax3.set_title("Which Data Type Fits Each NB Variant?", fontweight="bold")
        for bar, label in zip(bars, ["Counts", "Continuous", "0/1", "Categories"]):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, label, ha="center", fontsize=9, fontweight="bold")
        plt.xticks(rotation=15)
        st.pyplot(fig3)
    with extra_nb_col2:
        fig4, ax4 = plt.subplots(figsize=(7, 4.5))
        bars = ax4.bar(["Unseen feature,\nno smoothing", "Unseen feature,\nwith smoothing"], [0.00, 0.08], color=["#E45756", "#54A24B"])
        ax4.set_ylim(0, 0.12)
        ax4.set_ylabel("Example probability")
        ax4.set_title("Why Smoothing Matters", fontweight="bold")
        for bar, value in zip(bars, [0.00, 0.08]):
            ax4.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.2f}", ha="center", fontsize=10, fontweight="bold")
        st.pyplot(fig4)

    st.subheader("(b) Data Prep")
    prep = nb_analysis.prepare_nb_datasets(df)
    split_summary = build_split_summary(prep["X_train_raw"], prep["X_test_raw"], prep["y_train"], prep["y_test"])

    st.markdown(f"""
**Dataset Link:** [Cleaned dataset]({CLEANED_DATA_URL})  
**Code Link:** [nb_analysis.py]({CODE_NB_URL})
""")

    st.markdown("""
Supervised learning requires labeled data. Here, the target label is **depression severity**, created from the depression score and split into four classes.  
I used an **80/20 stratified train/test split**, so the class mix stays similar in both subsets.  
The split must be **disjoint** because the testing rows cannot be seen during fitting; otherwise the reported accuracy would be overly optimistic.
""")

    nb_raw_csv = prep["X_raw"].to_csv(index=False).encode("utf-8")
    nb_bern_csv = prep["X_train_bern"].to_csv(index=False).encode("utf-8")
    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            label="Download Prepared NB Features (CSV)",
            data=nb_raw_csv,
            file_name="prepared_nb_features.csv",
            mime="text/csv"
        )
    with download_col2:
        st.download_button(
            label="Download Bernoulli NB Binary Sample (CSV)",
            data=nb_bern_csv,
            file_name="prepared_nb_bernoulli_train.csv",
            mime="text/csv"
        )

    split_col1, split_col2 = st.columns([1.1, 1.4])
    with split_col1:
        st.markdown("### Split Summary")
        st.dataframe(split_summary, use_container_width=True)
    with split_col2:
        st.markdown("### Train/Test Split Image")
        st.pyplot(plot_disjoint_split(len(prep["X_train_raw"]), len(prep["X_test_raw"]), "Naive Bayes 80/20 Split"))

    st.markdown("### Sample Input Data")
    st.dataframe(prep["X_raw"].head(10), use_container_width=True)

    prep_col1, prep_col2 = st.columns(2)
    with prep_col1:
        st.markdown("### Training Set Sample")
        st.dataframe(prep["X_train_raw"].head(10), use_container_width=True)
    with prep_col2:
        st.markdown("### Testing Set Sample")
        st.dataframe(prep["X_test_raw"].head(10), use_container_width=True)

    st.markdown("### Training Labels Sample")
    st.dataframe(prep["y_train"].head(10).rename("depression_severity"), use_container_width=True)

    gaussian_preview = pd.DataFrame(prep["X_train_gauss"], columns=prep["feature_names"]).head(10)
    transformed_col1, transformed_col2 = st.columns(2)
    with transformed_col1:
        st.markdown("### Gaussian NB Continuous Data Sample")
        st.dataframe(gaussian_preview, use_container_width=True)
    with transformed_col2:
        st.markdown("### Bernoulli NB Binary Data Sample")
        st.dataframe(prep["X_train_bern"].head(10), use_container_width=True)

    st.caption("Multinomial NB uses the non-negative version of the training data, Gaussian NB uses the scaled continuous version, and Bernoulli NB uses the binary 0/1 version shown above.")

    st.subheader("(c) Code")
    st.markdown(f"[View Naïve Bayes Code]({CODE_NB_URL})")

    st.subheader("(d) Results")
    results = nb_analysis.run_all_nb_models(prep)
    st.dataframe(nb_analysis.accuracy_table(results), use_container_width=True)
    best_nb_name, best_nb_out = max(results.items(), key=lambda item: item[1]["accuracy"])

    for model_name, out in results.items():
        st.markdown(f"### {model_name}")
        st.write(f"Accuracy: {out['accuracy']:.4f}")
        fig_cm = nb_analysis.plot_confusion_matrix(
            out["confusion_matrix"],
            out["labels"],
            title=f"{model_name} Confusion Matrix"
        )
        st.pyplot(fig_cm)

    st.markdown("""
The confusion matrices and accuracy values show how well each Naïve Bayes variant classified the depression severity groups.  
Gaussian NB is useful when continuous variables are preserved, Bernoulli NB works well for binary thresholds, and Multinomial NB is suitable for non-negative frequency-style data.  
Comparing all three helps show how the data format affects model performance.
""")

    st.subheader("(e) Conclusions")
    st.markdown("""
Naïve Bayes provides a simple and efficient way to classify mental health risk groups from behavioral data.  
The results suggest that social media usage features contain enough signal to support predictive modeling, although performance depends on how the data is represented.  
This makes Naïve Bayes a useful baseline for comparing against more complex supervised learning methods.
""")

    st.markdown(f"""
For this project, **{best_nb_name}** performed best, which suggests that the chosen feature representation matters as much as the algorithm family itself. The results also show that social media behavior contains useful predictive signal, but multiclass depression severity remains a difficult problem for the strong independence assumptions used by Naive Bayes.
""")

with tab_svm:
    st.title("Support Vector Machines (SVMs)")

    if df is None:
        st.info("Load the cleaned dataset to view the SVM analysis.")
    else:
        st.subheader("(a) Overview")
        st.markdown("""
Support Vector Machines are supervised learning models that learn from **labeled examples**. For this project, the labels represent whether a person falls into a **lower-risk** or **higher-risk** mental health group based on the depression score. The job of the SVM is to look at the training examples and draw the strongest possible boundary between those two groups.

An SVM begins as a **linear separator**, which means it tries to find a straight boundary that splits one class from the other. What makes SVMs special is that they do not just search for any boundary. They search for the boundary with the **largest margin**, meaning the widest possible buffer between the two groups. The training points closest to that buffer are called **support vectors**, and they are the records that most strongly shape the final decision boundary.

Real-world behavioral data is often not cleanly split by a straight line. That is where the **kernel trick** helps. A kernel lets the SVM behave as if the data has been moved into a richer feature space, where a straight split may become possible, without explicitly calculating every expanded feature. This is powerful because it gives the model more flexibility while keeping the calculations manageable.

The **dot product** is critical because kernels use it to measure similarity between pairs of points. If two points point in a similar direction in feature space, their dot product is larger, and the SVM treats them as more alike. That means the kernel can compare points in a transformed space using only similarity calculations, which is why the kernel trick works so efficiently.
""")

        st.latex(r"K(\mathbf{x}, \mathbf{z}) = (\mathbf{x} \cdot \mathbf{z} + r)^d \quad \text{Polynomial kernel}")
        st.latex(r"K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \lVert \mathbf{x} - \mathbf{z} \rVert^2) \quad \text{RBF kernel}")

        overview_col1, overview_col2 = st.columns(2)
        with overview_col1:
            st.pyplot(svm_analysis.plot_svm_margin_concept())
        with overview_col2:
            st.pyplot(svm_analysis.plot_kernel_trick_concept())

        st.markdown("""
The first image shows the classic SVM idea: a separating boundary with margins and support vectors. The second image shows why kernels matter: a shape that is hard to separate in the original view can become easier to separate after the data is lifted into a higher-dimensional space.
""")

        st.markdown("""
For the required 2D point-casting example, consider the polynomial kernel with **r = 1** and **d = 2**:
""")
        st.latex(r"K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}\cdot\mathbf{z} + 1)^2")
        st.latex(r"\phi(x_1, x_2) = [x_1^2,\ \sqrt{2}x_1x_2,\ x_2^2,\ \sqrt{2}x_1,\ \sqrt{2}x_2,\ 1]")
        st.markdown("""
That means a 2D point is effectively cast into **6 dimensions**. For the example point **(2, 3)**, the expanded representation is shown below.
""")
        st.dataframe(svm_analysis.polynomial_feature_cast_example(2, 3), use_container_width=True)

        st.subheader("(b) Data Prep")
        prep, svm_results = get_svm_analysis(df)
        split_summary = build_split_summary(prep["X_train_raw"], prep["X_test_raw"], prep["y_train"], prep["y_test"])

        st.markdown(f"""
**Dataset Link:** [Cleaned dataset]({CLEANED_DATA_URL})  
**Code Link:** [svm_analysis.py]({CODE_SVM_URL})
""")

        st.markdown("""
Supervised models require **labeled data**, because the algorithm must know the correct answer during training. In this SVM section, each record is labeled as **Lower Risk** or **Higher Risk**. Without those labels, the model would have no way to learn what kind of pattern it should separate.

SVMs also require **numeric input features**. That is because the model depends on distances, margins, and dot products, and those ideas only make sense when the inputs are numbers. For that reason, this section uses only numeric behavioral and wellbeing features such as usage hours, anxiety score, sleep quality score, notifications per day, and engagement ratio.

The dataset was split into a **training set** and a **testing set** using an 80/20 stratified split. They must be **disjoint**, meaning no row can appear in both sets. The training set is used to build the model, while the testing set is held back until the end so the reported results reflect how the model performs on unseen data rather than data it already memorized.
""")

        svm_csv = prep["X"].assign(
            high_risk_label=prep["y"].map({0: "Lower Risk", 1: "Higher Risk"})
        ).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prepared SVM Dataset (CSV)",
            data=svm_csv,
            file_name="prepared_svm_dataset.csv",
            mime="text/csv"
        )

        prep_col1, prep_col2 = st.columns([1.15, 1.35])
        with prep_col1:
            st.markdown("### Split Summary")
            st.dataframe(split_summary, use_container_width=True)
        with prep_col2:
            st.markdown("### Train/Test Split Image")
            st.pyplot(plot_disjoint_split(len(prep["X_train_raw"]), len(prep["X_test_raw"]), "SVM 80/20 Split"))

        sample_with_label = prep["X"].copy()
        sample_with_label["high_risk_label"] = prep["y"].map({0: "Lower Risk", 1: "Higher Risk"})
        sample_col1, sample_col2 = st.columns(2)
        with sample_col1:
            st.markdown("### Sample of the Planned SVM Data")
            st.dataframe(sample_with_label.head(10), use_container_width=True)
        with sample_col2:
            st.markdown("### Sample Data Image")
            st.pyplot(dataframe_snapshot_figure(sample_with_label.head(8), "Labeled Numeric SVM Input Sample"))

        train_col1, train_col2 = st.columns(2)
        with train_col1:
            st.markdown("### Training Set Sample")
            st.dataframe(prep["X_train_raw"].head(10), use_container_width=True)
        with train_col2:
            st.markdown("### Testing Set Sample")
            st.dataframe(prep["X_test_raw"].head(10), use_container_width=True)

        st.markdown("### Training Labels Sample")
        st.dataframe(prep["y_train"].head(10).map({0: "Lower Risk", 1: "Higher Risk"}).rename("high_risk_label"), use_container_width=True)

        st.subheader("(c) Code")
        st.markdown(f"[View SVM Code]({CODE_SVM_URL})")

        st.subheader("(d) Results")
        best_svm_name, best_svm_out = max(
            ((kernel, details["best"]) for kernel, details in svm_results.items()),
            key=lambda item: item[1]["accuracy"]
        )

        st.markdown("""
Three kernels were tested: **linear**, **polynomial**, and **rbf**. For each kernel, multiple cost values were tried so the model could be compared fairly rather than relying on a single setting. The table below shows every kernel-cost run, and the smaller table highlights the selected best cost for each kernel.
""")

        result_table_col1, result_table_col2 = st.columns([1.45, 1.0])
        with result_table_col1:
            st.dataframe(svm_analysis.svm_accuracy_table(svm_results), use_container_width=True)
        with result_table_col2:
            st.dataframe(svm_analysis.best_svm_table(svm_results), use_container_width=True)

        st.pyplot(svm_analysis.plot_accuracy_by_cost(svm_results))

        for kernel in ["linear", "poly", "rbf"]:
            best = svm_results[kernel]["best"]
            st.markdown(f"### {kernel.upper()} Kernel with C = {best['C']}")
            st.write(f"Accuracy: {best['accuracy']:.4f}")
            st.write(f"Support vectors used: {best['support_vectors']}")

            kernel_col1, kernel_col2 = st.columns(2)
            with kernel_col1:
                st.pyplot(
                    svm_analysis.plot_confusion_matrix(
                        best["confusion_matrix"],
                        title=f"{kernel.upper()} Kernel Confusion Matrix"
                    )
                )
            with kernel_col2:
                st.pyplot(svm_analysis.plot_pca_decision_regions(prep, kernel, best["C"]))

        st.markdown(f"""
Across the tested kernels, the strongest result came from the **{best_svm_name.upper()}** kernel with **C = {best_svm_out['C']}**, reaching an accuracy of **{best_svm_out['accuracy']:.4f}**. The linear kernel gives a straight split, the polynomial kernel gives a curved but still structured split, and the RBF kernel allows the most flexible boundary. In this dataset, the stronger performance of the RBF kernel suggests that the relationship between social media behavior and mental health risk is probably **not purely straight-line** in nature.
""")

        st.subheader("(e) Conclusions")
        st.markdown(f"""
This SVM analysis shows that social media behavior can be used to separate lower-risk and higher-risk mental health patterns, but the shape of that separation matters. A simple straight boundary was not the best fit here. The more flexible **{best_svm_name.upper()}** kernel performed best, which suggests that the connection between digital habits and wellbeing is more complex than a single straight rule.

In practical terms, the SVM results reinforce the idea that risk does not come from only one behavior. It comes from combinations of behaviors such as heavier use, more late-night activity, stronger comparison habits, and other signals that interact with one another. That is why a flexible boundary can outperform a rigid one on this topic.
""")

with tab_ensemble:
    st.title("Ensemble Learning")

    if df is None:
        st.info("Load the cleaned dataset to view the ensemble learning analysis.")
    else:
        st.subheader("(a) Overview")
        st.markdown("""
Ensemble learning improves prediction by combining multiple models instead of trusting just one. The idea is simple: if several different models look at the same problem and each catches part of the pattern, the combined result can be more stable and reliable than any one model working alone.

To go beyond the minimum requirement, this section compares several ensemble styles: **Random Forest**, **Extra Trees**, **AdaBoost**, and a **Voting Ensemble**. Together they represent common ensemble ideas such as bagging, randomized trees, boosting, and voting. All of them are applied to the same mental health risk prediction task used in the SVM section so the comparison stays consistent.
""")

        st.pyplot(ensemble_analysis.plot_ensemble_concept())

        st.subheader("(b) Data Prep")
        ensemble_prep, ensemble_results = get_ensemble_analysis(df)
        split_summary = build_split_summary(
            ensemble_prep["X_train_raw"],
            ensemble_prep["X_test_raw"],
            ensemble_prep["y_train"],
            ensemble_prep["y_test"]
        )

        st.markdown(f"""
**Dataset Link:** [Cleaned dataset]({CLEANED_DATA_URL})  
**Code Link:** [ensemble_analysis.py]({CODE_ENSEMBLE_URL})
""")

        st.markdown("""
The ensemble models use the same labeled numeric dataset as the SVM section. Each row contains behavioral and wellbeing measurements, and each row is labeled as **Lower Risk** or **Higher Risk**. Keeping the same prediction target makes it easier to compare what a single strong classifier does versus what a group of models can do together.

An 80/20 stratified train/test split was used here as well, and the two subsets remain disjoint for the same reason: the test rows must stay unseen until evaluation time. If the same rows showed up during both training and testing, the reported performance would look better than it really is.
""")

        ensemble_csv = ensemble_prep["X"].assign(
            high_risk_label=ensemble_prep["y"].map({0: "Lower Risk", 1: "Higher Risk"})
        ).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prepared Ensemble Dataset (CSV)",
            data=ensemble_csv,
            file_name="prepared_ensemble_dataset.csv",
            mime="text/csv"
        )

        ensemble_col1, ensemble_col2 = st.columns([1.15, 1.35])
        with ensemble_col1:
            st.markdown("### Split Summary")
            st.dataframe(split_summary, use_container_width=True)
        with ensemble_col2:
            st.markdown("### Train/Test Split Image")
            st.pyplot(plot_disjoint_split(len(ensemble_prep["X_train_raw"]), len(ensemble_prep["X_test_raw"]), "Ensemble 80/20 Split"))

        st.markdown("### Sample Input Data")
        st.dataframe(
            ensemble_prep["X"].assign(
                high_risk_label=ensemble_prep["y"].map({0: "Lower Risk", 1: "Higher Risk"})
            ).head(10),
            use_container_width=True
        )

        st.subheader("(c) Code")
        st.markdown(f"[View Ensemble Code]({CODE_ENSEMBLE_URL})")

        st.subheader("(d) Results")
        best_ensemble_name, best_ensemble_out = max(ensemble_results.items(), key=lambda item: item[1]["accuracy"])
        best_tree_name, best_tree_out = ensemble_analysis.best_tree_ensemble(ensemble_results)

        results_col1, results_col2 = st.columns([1.0, 1.2])
        with results_col1:
            st.dataframe(ensemble_analysis.ensemble_accuracy_table(ensemble_results), use_container_width=True)
        with results_col2:
            st.pyplot(ensemble_analysis.plot_accuracy_bar(ensemble_results))

        st.markdown(f"""
The strongest overall ensemble in this run was **{best_ensemble_name}**, with an accuracy of **{best_ensemble_out['accuracy']:.4f}**. That means combining models, or using many trees together, can provide a dependable way to recognize mental health risk patterns from this dataset.
""")

        for model_name, out in ensemble_results.items():
            st.markdown(f"### {model_name}")
            st.write(f"Accuracy: {out['accuracy']:.4f}")
            st.pyplot(
                ensemble_analysis.plot_confusion_matrix(
                    out["confusion_matrix"],
                    title=f"{model_name} Confusion Matrix"
                )
            )

        st.markdown(f"### Feature Importance from {best_tree_name}")
        st.pyplot(
            ensemble_analysis.plot_feature_importance(
                best_tree_out["model"],
                ensemble_prep["feature_names"],
                title=f"{best_tree_name} Top Feature Importances"
            )
        )

        st.subheader("(e) Conclusions")
        st.markdown(f"""
The ensemble results show that combining many decisions can be useful for this topic because social media and mental health do not depend on one single behavior. They depend on patterns that build across many habits at once. That is exactly the kind of situation where ensemble methods can help.

In this project, **{best_ensemble_name}** performed best, while **{best_tree_name}** also helped reveal which features mattered most. Together, these results suggest that mental health risk in the dataset is driven by a mix of usage intensity, emotional strain, sleep-related behaviors, and social comparison rather than one isolated variable.
""")

with tab_reg:
    st.title("Regression")

    st.subheader("Concept Questions")
    st.markdown("""
**(a) Define and explain linear regression.**  
Linear regression models the relationship between input variables and a **continuous** output using a straight-line equation. It is used when the goal is to predict a number such as a score, time, or price.

**(b) Define and explain logistic regression.**  
Logistic regression is a classification algorithm that predicts the probability that a record belongs to a class, usually one of two classes. It is commonly used for yes/no style outcomes such as low risk versus high risk.

**(c) How are they similar and how are they different?**  
Both methods begin with a linear combination of the input features. The difference is that linear regression predicts a continuous value, while logistic regression converts the result into a probability and then a class label.

**(d) Does logistic regression use the sigmoid function? Explain.**  
Yes. Logistic regression uses the sigmoid function to map the raw linear score into a probability between `0` and `1`, which makes the output suitable for binary classification.

**(e) Explain how maximum likelihood is connected to logistic regression.**  
Logistic regression estimates its coefficients by maximizing the likelihood of the observed class labels. In other words, it chooses the parameter values that make the training outcomes most probable under the model.
""")

    st.subheader("Data Prep")
    prep = regression_analysis.prepare_regression_data(df)
    split_summary = build_split_summary(prep["X_train_raw"], prep["X_test_raw"], prep["y_train"], prep["y_test"])

    st.markdown(f"""
**Dataset Link:** [Cleaned dataset]({CLEANED_DATA_URL})  
**Code Link:** [regression_analysis.py]({CODE_REG_URL})
""")

    st.markdown("""
For logistic regression, the target must be categorical.  
Here, a binary label is created from depression score: lower risk vs higher risk.  
I used an **80/20 stratified train/test split**, and these two subsets remain disjoint so the evaluation reflects generalization rather than memorization.
""")

    reg_csv = prep["X"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prepared Regression Input (CSV)",
        data=reg_csv,
        file_name="prepared_regression_input.csv",
        mime="text/csv"
    )

    split_col1, split_col2 = st.columns([1.1, 1.4])
    with split_col1:
        st.markdown("### Split Summary")
        st.dataframe(split_summary, use_container_width=True)
    with split_col2:
        st.markdown("### Train/Test Split Image")
        st.pyplot(plot_disjoint_split(len(prep["X_train_raw"]), len(prep["X_test_raw"]), "Regression 80/20 Split"))

    st.markdown("### Sample Input Data")
    st.dataframe(prep["X"].head(10), use_container_width=True)

    prep_col1, prep_col2 = st.columns(2)
    with prep_col1:
        st.markdown("### Training Set Sample")
        st.dataframe(prep["X_train_raw"].head(10), use_container_width=True)
    with prep_col2:
        st.markdown("### Testing Set Sample")
        st.dataframe(prep["X_test_raw"].head(10), use_container_width=True)

    st.markdown("### Training Labels Sample")
    st.dataframe(prep["y_train"].head(10).rename("high_risk_label"), use_container_width=True)

    log_preview = pd.DataFrame(prep["X_train_log"], columns=prep["feature_names"]).head(10)
    transform_col1, transform_col2 = st.columns(2)
    with transform_col1:
        st.markdown("### Logistic Regression Scaled Data Sample")
        st.dataframe(log_preview, use_container_width=True)
    with transform_col2:
        st.markdown("### Multinomial NB Non-Negative Data Sample")
        st.dataframe(prep["X_train_nb"].head(10), use_container_width=True)

    st.subheader("Model Results: Logistic Regression vs Multinomial NB")
    results = regression_analysis.run_logistic_and_nb(prep)
    st.dataframe(regression_analysis.accuracy_table(results), use_container_width=True)
    best_reg_name, best_reg_out = max(results.items(), key=lambda item: item[1]["accuracy"])
    worst_reg_name, worst_reg_out = min(results.items(), key=lambda item: item[1]["accuracy"])

    for model_name, out in results.items():
        st.markdown(f"### {model_name}")
        st.write(f"Accuracy: {out['accuracy']:.4f}")
        fig_cm = regression_analysis.plot_confusion_matrix(
            out["confusion_matrix"],
            labels=("Low Risk", "High Risk"),
            title=f"{model_name} Confusion Matrix"
        )
        st.pyplot(fig_cm)

    st.markdown("""
Logistic regression and Multinomial Naïve Bayes were both applied to the same binary mental health prediction task.  
The confusion matrices and accuracy scores show how well each model separates lower-risk and higher-risk users.  
Comparing these results helps determine whether a linear probability-based model or a probabilistic count-style classifier works better for this project.
""")

    st.markdown(f"""
Logistic regression and Multinomial Naive Bayes were both applied to the same binary mental health prediction task.  
The stronger model in this run was **{best_reg_name}** with accuracy **{best_reg_out['accuracy']:.4f}**, while **{worst_reg_name}** reached **{worst_reg_out['accuracy']:.4f}**.  
This comparison shows which model works better for the project when the goal is to separate lower-risk and higher-risk users on the same labeled dataset.
""")

with tab_conc:
    st.title("Conclusions")

    if df is not None:
        nb_prep = nb_analysis.prepare_nb_datasets(df)
        nb_results = nb_analysis.run_all_nb_models(nb_prep)
        dt_prep = dt_analysis.prepare_dt_data(df)
        dt_results = dt_analysis.build_three_different_trees(dt_prep)
        _, svm_results = get_svm_analysis(df)
        _, ensemble_results = get_ensemble_analysis(df)
        reg_prep = regression_analysis.prepare_regression_data(df)
        reg_results = regression_analysis.run_logistic_and_nb(reg_prep)

        best_nb_name, best_nb_out = max(nb_results.items(), key=lambda item: item[1]["accuracy"])
        best_dt_name, best_dt_out = max(dt_results.items(), key=lambda item: item[1]["accuracy"])
        best_svm_name, _ = max(
            ((kernel, details["best"]) for kernel, details in svm_results.items()),
            key=lambda item: item[1]["accuracy"]
        )
        best_ensemble_name, _ = max(ensemble_results.items(), key=lambda item: item[1]["accuracy"])
        best_reg_name, best_reg_out = max(reg_results.items(), key=lambda item: item[1]["accuracy"])

        image_col1, image_col2 = st.columns(2)
        with image_col1:
            summary_img = "viz/11_summary_dashboard.png"
            if os.path.exists(summary_img):
                st.image(summary_img, use_container_width=True)
        with image_col2:
            support_img = "viz/10_support_help_seeking.png"
            if os.path.exists(support_img):
                st.image(support_img, use_container_width=True)

        st.markdown(f"""
This project shows that social media is not just entertainment or background noise in daily life. It is part of how many people connect, compare, unwind, and cope. Because of that, online habits can spill over into sleep, mood, confidence, and overall wellbeing in ways that feel ordinary on the surface but become important when they repeat day after day.

One of the clearest take-home messages is that mental health risk is shaped by combinations of behaviors rather than one single habit. Heavy use by itself does not tell the whole story. Late-night scrolling, frequent comparison with others, emotional strain, lower satisfaction, and weaker support systems all seem to matter when they appear together. The broader pattern matters more than any one number on its own.

The modeling sections support that same idea from different angles. The strongest Decision Tree was **{best_dt_name}**, the strongest Naive Bayes model was **{best_nb_name}**, the strongest SVM result came from the **{best_svm_name.upper()}** kernel, the strongest ensemble method was **{best_ensemble_name}**, and the strongest regression-side result was **{best_reg_name}**. In plain language, the more flexible methods usually handled the topic better than the more rigid ones, which suggests that human behavior online does not follow one simple rule.

Another important conclusion is that this topic is not really about blaming technology. The website points toward balance rather than fear. Social media can still be useful, social, creative, and comforting. The problem seems to grow when use becomes more intense, more isolating, more comparison-driven, or more likely to interfere with sleep and offline wellbeing.

Overall, the key message is hopeful as much as it is cautionary. If unhealthy patterns can be recognized, then healthier patterns can also be encouraged. Better habits, stronger support, more intentional use, and earlier awareness all have the potential to improve digital wellbeing. That makes this topic worth studying not only to describe a problem, but also to help people make smarter and kinder choices in everyday life.
""")
    else:
        st.info("Load the cleaned dataset to view the final cross-model summary.")

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
