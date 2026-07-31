from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import entropy


def analyze_reward_performance(
    scenarios_df: pd.DataFrame, reward_func: Callable
) -> pd.DataFrame:
    """
    Runs the reward function and calculates key performance indicators (KPIs)
    for each playlist scenario.
    KPIs include:
    - reward_score: The reward calculated by the reward function
    - total_clicks: Total number of clicks in the outcomes
    - total_bookings: Total number of bookings in the outcomes
    - playlist_cat_count: Number of unique categories in the playlist
    - discovery_rate: Proportion of items in the playlist that are from categories
      not seen in the user's history
    """

    scenarios_df["reward_score"] = scenarios_df.apply(
        lambda r: reward_func(r["playlist"], r["user_history"], r["outcomes"]), axis=1
    )

    scenarios_df["total_clicks"] = scenarios_df["outcomes"].apply(
        lambda x: sum(item["clicked"] for item in x)
    )
    scenarios_df["total_bookings"] = scenarios_df["outcomes"].apply(
        lambda x: sum(item["booked"] for item in x)
    )

    scenarios_df["playlist_cat_count"] = scenarios_df["playlist"].apply(
        lambda x: len(set(item["offer_category"] for item in x))  # noqa: C401
    )

    def calculate_discovery(row):
        new_items = 0
        known_categories = row["user_history"]["categories"]
        for item in row["playlist"]:
            if item["offer_category"] not in known_categories:
                new_items += 1
        return new_items / len(row["playlist"])

    scenarios_df["discovery_rate"] = scenarios_df.apply(calculate_discovery, axis=1)

    return scenarios_df


def plot_bandit_performances(
    bandits: list,
    axhline: float | None = None,
    save_to: str | None = None,
) -> None:
    """
    Plots the average reward over time and the distribution of chosen arms for a list of bandit instances.
    Each bandit instance is expected to have a 'logs' attribute which is a DataFrame containing 'reward' and 'chosen_arm' columns.
    """  # noqa: E501, W505
    bandit_names = [f"{type(b).__name__} ({i})" for i, b in enumerate(bandits)]
    colors = sns.color_palette("magma", n_colors=len(bandits))

    f, axs = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [5, 3]})

    for i, (name, bandit) in enumerate(zip(bandit_names, bandits, strict=True)):
        sns.lineplot(
            data=bandit.logs,
            x=bandit.logs.index,
            y=bandit.logs["reward"].expanding().mean(),
            label=name,
            ax=axs[0],
            color=colors[i],
        )
    axs[0].set_title("Average Reward Over Time")
    axs[0].set_xlabel("Steps")

    if axhline is not None:
        axs[0].axhline(
            y=axhline,
            color="black",
            linestyle="--",
            label="Mean Expected Reward",
        )

    all_data = pd.concat(
        [
            bandit.logs.assign(bandit_id=name)
            for name, bandit in zip(bandit_names, bandits, strict=True)
        ]
    )
    counts = all_data.groupby(["chosen_arm", "bandit_id"]).size().unstack(fill_value=0)
    counts = counts[bandit_names]

    counts["total"] = counts.sum(axis=1)
    top_10_counts = (
        counts.sort_values("total", ascending=False)
        .head(10)
        .drop(columns="total")
        .iloc[::-1]
    )

    top_10_counts.plot(
        kind="barh",
        stacked=True,
        ax=axs[1],
        color=colors,
    )

    axs[1].set_title("Top 10 Most Frequent Arms")
    axs[1].set_xlabel("Frequency")
    axs[1].set_ylabel("Arms")
    axs[1].tick_params(axis="y", labelsize=6)

    axs[1].legend(
        title="Bandit Instances",
        fontsize="x-small",
        loc="best",
    )

    f.tight_layout()
    if save_to:
        plt.savefig(f"{save_to}")
    plt.draw()


def lineplot_conv_events(
    ax1: plt.Axes,
    ax2: plt.Axes,
    data: pd.DataFrame,
    events: list,
    title: str,
    CI_wrapper=False,
) -> None:
    """
    Plots the conversion events (viewed, clicked, faved, booked) against the model score.
    Uses two y-axes to differentiate between view/click events and fav/book events.
    If CI_wrapper is True, it expects the data to contain mean and sem columns for each event to plot confidence intervals.
    """  # noqa: E501, W505
    ax1_max = (
        data[["viewed", "clicked"]].max().max()
        if "viewed" in events
        else data["clicked"].max().max()
    )
    ax2_max = (
        data[["faved", "booked"]].max().max()
        if "faved" in events
        else data["booked"].max().max()
    )
    if ax1_max == 0:
        ax1.set_ylim(0, 1)
    else:
        ax1.set_ylim(0 - ax1_max * 0.1, ax1_max * 1.1)
    if ax2_max == 0:
        ax2.set_ylim(0, 1)
    else:
        ax2.set_ylim(0 - ax2_max * 0.1, ax2_max * 1.1)

    colors = {
        "viewed": "tab:blue",
        "clicked": "tab:green",
        "faved": "tab:orange",
        "booked": "tab:red",
    }

    for event in events:
        ax = ax1 if event in ["viewed", "clicked"] else ax2
        linestyle = "-" if event in ["viewed", "clicked"] else "--"

        if CI_wrapper:
            y_mean = data[(event, "mean")]
            y_sem = data[(event, "sem")]
            x = data["score_bin"]

            ax.plot(
                x,
                y_mean,
                label=event,
                marker="o",
                linestyle=linestyle,
                color=colors[event],
                markersize=4,
            )

            ax.fill_between(
                x.astype(float),
                (y_mean - 1.96 * y_sem).where((y_mean - 1.96 * y_sem) > 0, 0),
                y_mean + 1.96 * y_sem,
                color=colors[event],
                alpha=0.2,
            )

        else:
            sns.lineplot(
                data=data,
                x="score_bin",
                y=event,
                label=event,
                marker="o",
                ax=ax,
                linestyle=linestyle,
                color=colors[event],
            )

    ax1.set_title(title, fontsize=10, fontweight="bold")
    ax1.set_xlabel("Model Score")
    ax1.set_ylabel("Prob (View/Click)")
    ax2.set_ylabel("Prob (Fav/Book)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax2.grid(False)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()


def evaluate_bandits(bandits, num_last_steps=100):
    summary_stats = []

    for i, b in enumerate(bandits):
        df = b.logs.copy()
        total_steps = len(df)

        if total_steps == 0:
            continue

        outcomes_df = pd.json_normalize(df["outcome"])
        df = pd.concat([df.drop(columns=["outcome"]), outcomes_df], axis=1)

        df_last = df.tail(num_last_steps)
        actual_last_steps = len(df_last)

        arm_counts = df["chosen_arm"].value_counts()
        arm_probs = arm_counts / total_steps

        # Metrics
        stats = {
            "Bandit Name": f"{type(b).__name__} ({i})",
            "Total Steps": total_steps,
            # Rewards
            "Total Reward": df["reward"].sum(),
            "Avg Reward/Step": df["reward"].mean(),
            f"Total Reward (Last {actual_last_steps})": df_last["reward"].sum(),
            f"Avg Reward/Step (Last {actual_last_steps})": df_last["reward"].mean(),
            # Arm Diversity
            "Unique Arms Picked": df["chosen_arm"].nunique(),
            f"Unique Arms Picked (Last {actual_last_steps})": df_last[
                "chosen_arm"
            ].nunique(),
            "Arm Choice Entropy": entropy(arm_probs),
            "Top Arm Dominance (%)": (arm_counts.max() / total_steps) * 100,
            # Funnel Metrics (Global)
            "CTR (%)": df["clicked"].mean() * 100,
            "Fave Rate (%)": df["faved"].mean() * 100,
            "Booking Rate (%)": df["booked"].mean() * 100,
            # Funnel Metrics (Recent)
            f"CTR Last {actual_last_steps} (%)": df_last["clicked"].mean() * 100,
            f"Fave Rate Last {actual_last_steps} (%)": df_last["faved"].mean() * 100,
            f"Booking Rate Last {actual_last_steps} (%)": df_last["booked"].mean()
            * 100,
        }

        summary_stats.append(stats)

    return pd.DataFrame(summary_stats)
