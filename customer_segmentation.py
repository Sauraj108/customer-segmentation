"""
====================================================
  Customer Segmentation Project — K-Means Clustering
  Tools: Python, scikit-learn, Pandas, Seaborn, Matplotlib
====================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ── Plot Style ────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

print("=" * 55)
print("   CUSTOMER SEGMENTATION — K-MEANS CLUSTERING")
print("=" * 55)

# ─────────────────────────────────────────────────────────────
# STEP 1: Load Data
# ─────────────────────────────────────────────────────────────
print("\n[1/6] Data load ho rahi hai...")

df = pd.read_csv("customer_data.csv")
print(f"     Total customers: {len(df)}")
print(f"     Columns: {list(df.columns)}")

# ─────────────────────────────────────────────────────────────
# STEP 2: Data Cleaning & EDA
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Data clean aur explore kar rahe hain...")

# Missing values check
print(f"     Missing values:\n{df.isnull().sum()}")

# Fill missing values if any
df.fillna(df.median(numeric_only=True), inplace=True)

# Gender encode karo (Male=0, Female=1)
df["Gender_Encoded"] = df["Gender"].map({"Male": 0, "Female": 1})

print(f"\n     Basic Stats:")
print(df[["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency"]].describe().round(1))

# ─────────────────────────────────────────────────────────────
# STEP 3: Feature Selection & Scaling
# ─────────────────────────────────────────────────────────────
print("\n[3/6] Features select aur scale kar rahe hain...")

features = ["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency"]
X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"     Features used: {features}")
print("     StandardScaler applied ✓")

# ─────────────────────────────────────────────────────────────
# STEP 4: Elbow Method + Silhouette Score → Best K dhundho
# ─────────────────────────────────────────────────────────────
print("\n[4/6] Best K dhundh rahe hain (Elbow + Silhouette)...")

inertia_list   = []
silhouette_list = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia_list.append(km.inertia_)
    silhouette_list.append(silhouette_score(X_scaled, km.labels_))

best_k = k_range[silhouette_list.index(max(silhouette_list))]
print(f"     Best K = {best_k} (Silhouette Score: {max(silhouette_list):.3f})")

# Plot 1: Elbow + Silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Optimal K Selection", fontsize=15, fontweight="bold")

ax1.plot(list(k_range), inertia_list, "bo-", linewidth=2, markersize=7)
ax1.axvline(x=best_k, color="red", linestyle="--", label=f"Best K = {best_k}")
ax1.set_title("Elbow Method")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia")
ax1.legend()

ax2.plot(list(k_range), silhouette_list, "go-", linewidth=2, markersize=7)
ax2.axvline(x=best_k, color="red", linestyle="--", label=f"Best K = {best_k}")
ax2.set_title("Silhouette Score")
ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("Silhouette Score")
ax2.legend()

plt.tight_layout()
plt.savefig("plot1_elbow_silhouette.png", bbox_inches="tight")
plt.show()
print("     plot1_elbow_silhouette.png saved ✓")

# ─────────────────────────────────────────────────────────────
# STEP 5: Final K-Means Model
# ─────────────────────────────────────────────────────────────
print(f"\n[5/6] K={best_k} se final model train kar rahe hain...")

final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = final_model.fit_predict(X_scaled)

# Cluster profiles banao
profile = df.groupby("Cluster")[features].mean().round(1)
profile["Count"] = df["Cluster"].value_counts().sort_index()

# Smart labels assign karo based on Spending_Score & Income
labels_map = {}
for c in range(best_k):
    income  = profile.loc[c, "Annual_Income"]
    spend   = profile.loc[c, "Spending_Score"]
    freq    = profile.loc[c, "Purchase_Frequency"]
    age     = profile.loc[c, "Age"]

    if spend >= 70 and income >= 70000:
        label = "💎 Premium Spenders"
    elif spend >= 70 and income < 50000:
        label = "🛍️ Big Spenders (Low Income)"
    elif spend <= 35 and income >= 70000:
        label = "💰 High Income, Low Spend"
    elif spend <= 35 and income < 50000:
        label = "🧾 Budget Conscious"
    elif freq >= 7 and spend >= 60:
        label = "❤️ Loyal Customers"
    elif age >= 55 and spend <= 30:
        label = "⚠️ At-Risk Customers"
    else:
        label = f"📊 Segment {c+1}"

    labels_map[c] = label

df["Segment"] = df["Cluster"].map(labels_map)
profile["Segment"] = profile.index.map(labels_map)

print("\n     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("     CLUSTER PROFILES:")
print("     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
for c in range(best_k):
    p = profile.loc[c]
    print(f"\n     Cluster {c} — {labels_map[c]}")
    print(f"       Customers  : {int(p['Count'])}")
    print(f"       Avg Age    : {p['Age']}")
    print(f"       Avg Income : ₹{int(p['Annual_Income']):,}")
    print(f"       Spend Score: {p['Spending_Score']}")
    print(f"       Purchase/mo: {p['Purchase_Frequency']}")

# ─────────────────────────────────────────────────────────────
# STEP 6: Visualizations
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Charts bana rahe hain...")

colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
          "#9B59B6", "#1ABC9C", "#E67E22", "#34495E"]
palette = {labels_map[i]: colors[i % len(colors)] for i in range(best_k)}

# Plot 2: PCA 2D Scatter
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

plt.figure(figsize=(10, 7))
for seg, grp in df.groupby("Segment"):
    plt.scatter(grp["PCA1"], grp["PCA2"],
                label=seg, alpha=0.7, s=60,
                color=palette.get(seg, "gray"))
plt.title("Customer Segments — 2D View (PCA)", fontsize=14, fontweight="bold")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("plot2_clusters_2d.png", bbox_inches="tight")
plt.show()
print("     plot2_clusters_2d.png saved ✓")

# Plot 3: Income vs Spending Scatter
plt.figure(figsize=(10, 7))
for seg, grp in df.groupby("Segment"):
    plt.scatter(grp["Annual_Income"], grp["Spending_Score"],
                label=seg, alpha=0.75, s=70,
                color=palette.get(seg, "gray"))
plt.title("Annual Income vs Spending Score", fontsize=14, fontweight="bold")
plt.xlabel("Annual Income (₹)")
plt.ylabel("Spending Score")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("plot3_income_vs_spend.png", bbox_inches="tight")
plt.show()
print("     plot3_income_vs_spend.png saved ✓")

# Plot 4: Box plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Feature Distribution by Segment", fontsize=15, fontweight="bold")

for ax, feat in zip(axes.flatten(), features):
    order = df.groupby("Segment")[feat].median().sort_values().index
    sns.boxplot(data=df, x="Segment", y=feat, ax=ax,
                order=order, palette=palette)
    ax.set_title(feat, fontsize=12)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("plot4_boxplots.png", bbox_inches="tight")
plt.show()
print("     plot4_boxplots.png saved ✓")

# Plot 5: Segment size bar chart
seg_counts = df["Segment"].value_counts()
plt.figure(figsize=(10, 5))
bars = plt.bar(seg_counts.index, seg_counts.values,
               color=[palette.get(s, "gray") for s in seg_counts.index],
               edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, seg_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.title("Number of Customers per Segment", fontsize=14, fontweight="bold")
plt.ylabel("Customer Count")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("plot5_segment_sizes.png", bbox_inches="tight")
plt.show()
print("     plot5_segment_sizes.png saved ✓")

# ─────────────────────────────────────────────────────────────
# BUSINESS INSIGHTS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("   BUSINESS INSIGHTS & MARKETING STRATEGY")
print("=" * 55)

strategies = {
    "💎 Premium Spenders":
        "→ VIP membership, exclusive offers, loyalty rewards. Retain at all cost!",
    "🛍️ Big Spenders (Low Income)":
        "→ EMI offers, buy-now-pay-later, flash sales. Price-sensitive but eager!",
    "💰 High Income, Low Spend":
        "→ Premium product showcase, personalized recommendations. Untapped potential!",
    "🧾 Budget Conscious":
        "→ Discount coupons, combo deals, value packs. Win with affordability!",
    "❤️ Loyal Customers":
        "→ Referral programs, early access, thank-you rewards. Brand ambassadors!",
    "⚠️ At-Risk Customers":
        "→ Re-engagement emails, special comeback offers, feedback surveys.",
}

for seg in df["Segment"].unique():
    strategy = strategies.get(seg, "→ Analyze further and create targeted campaigns.")
    count = len(df[df["Segment"] == seg])
    print(f"\n  {seg}  ({count} customers)")
    print(f"  {strategy}")

# Save final segmented data
df.to_csv("customer_segments_result.csv", index=False)
print("\n" + "=" * 55)
print("  customer_segments_result.csv saved ✓")
print("  5 charts saved as PNG files ✓")
print("  Project complete! 🎉")
print("=" * 55)