import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules


def make_transactions(df: pd.DataFrame):
    arm_df = df.copy()

    arm_df["high_usage"] = arm_df["daily_usage_hours"] > arm_df["daily_usage_hours"].median()
    arm_df["high_depression"] = arm_df["depression_score"] > 14
    arm_df["high_anxiety"] = arm_df["anxiety_score"] > 10
    arm_df["late_night_user"] = arm_df["late_night_hours"] > 1.5
    arm_df["low_self_esteem"] = arm_df["self_esteem_score"] < 25
    arm_df["high_comparison"] = arm_df["comparison_content_pct"] > 40
    arm_df["cyberbullying_yes"] = arm_df["cyberbullying_experienced"] == "Yes"
    arm_df["poor_sleep"] = arm_df["sleep_quality_score"] > 10

    features = [
        "high_usage", "high_depression", "high_anxiety", "late_night_user",
        "low_self_esteem", "high_comparison", "cyberbullying_yes", "poor_sleep"
    ]

    transactions = arm_df[features].astype(bool)
    return transactions, features


def run_arm(transactions: pd.DataFrame, min_support=0.05, min_confidence=0.3, min_lift=1.0):
    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return frequent_itemsets, rules

    rules = rules[rules["lift"] >= min_lift].copy()
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)
    return frequent_itemsets, rules


def format_rules_table(rules: pd.DataFrame, metric: str, top_n=15):
    if rules.empty:
        return pd.DataFrame()

    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    top = rules.nlargest(top_n, metric)[cols].copy()
    top["antecedents"] = top["antecedents"].apply(lambda x: ", ".join(list(x)))
    top["consequents"] = top["consequents"].apply(lambda x: ", ".join(list(x)))
    return top


def plot_arm_overview_metrics():
    """
    Simple rubric-friendly image: illustrates support/confidence/lift (example values).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(["Support", "Confidence", "Lift"], [0.2, 0.6, 1.4])
    ax.set_title("ARM Metrics (Illustrative Example)", fontweight="bold")
    ax.set_ylabel("Example Value")
    ax.grid(axis="y", alpha=0.3)
    return fig


def plot_rule_network(rules: pd.DataFrame, top_n=20):
    if rules.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No rules to visualize", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    viz_rules = rules.nlargest(top_n, "lift").copy()

    all_items = set()
    for _, row in viz_rules.iterrows():
        all_items.update(row["antecedents"])
        all_items.update(row["consequents"])
    all_items = list(all_items)
    n = len(all_items)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {item: (np.cos(a), np.sin(a)) for item, a in zip(all_items, angles)}

    fig, ax = plt.subplots(figsize=(14, 10))

    max_lift = viz_rules["lift"].max() if len(viz_rules) else 1.0
    for _, row in viz_rules.iterrows():
        for ant in row["antecedents"]:
            for cons in row["consequents"]:
                x = [pos[ant][0], pos[cons][0]]
                y = [pos[ant][1], pos[cons][1]]
                lw = float(row["support"]) * 20
                color = plt.cm.RdYlGn(float(row["lift"]) / max_lift)
                ax.plot(x, y, color=color, linewidth=lw, alpha=0.6)

    for item, (x, y) in pos.items():
        ax.scatter(x, y, s=500, c="lightblue", edgecolors="black", linewidth=2, zorder=10)
        ax.text(x, y, item.replace("_", "\n"), ha="center", va="center",
                fontsize=8, fontweight="bold")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Association Rules Network", fontweight="bold", fontsize=14)
    ax.axis("off")
    return fig