def get_conversion_reward(
    outcome: dict, w1: float = 1, w2: float = 7, w3: float = 12, **kwargs
) -> float:
    """
    Calculates the normalized conversion reward
    """
    return (
        w1 * outcome.get("clicked", 0)
        + w2 * outcome.get("faved", 0)
        + w3 * outcome.get("booked", 0)
    ) / (w1 + w2 + w3)


def get_diversity_score(h_u: dict, cat: str, sub: str, lt: str, **kwargs) -> float:
    """
    Calculates diversity gain based on point system:
    - New category: 25 | New subcat: 10 | New venue Type: 20 | New venue: 5
    """
    score = 0
    if cat not in h_u.get("categories"):
        score += 25
    if sub not in h_u.get("subcategories"):
        score += 10
    if lt not in h_u.get("venue_types"):
        score += 20

    return score / 55


def get_item_reward(
    outcome: dict,
    h_u: dict,
    chosen_arm_key: str,
    w1: float = 1,
    w2: float = 7,
    w3: float = 12,
    beta: float = 0.7,
    **kwargs,
) -> float:
    cat, sub, lt = chosen_arm_key.split("__")
    conv_val = get_conversion_reward(outcome, w1, w2, w3)
    div_val = get_diversity_score(h_u, cat=cat, sub=sub, lt=lt)
    return conv_val * (1 + (beta * div_val))


def get_playlist_reward(
    P: list,
    h_u: dict,
    g: list,
    gamma: float = 0.9,
    beta: float = 0.7,
    w1: float = 1,
    w2: float = 7,
    w3: float = 12,
) -> float:
    r"""
    Reward function of shape :
    \sum_{i \in P} \gamma^{r_i} \cdot Conv(i) \cdot (1 + \lambda \cdot Div(H_u, i))

    with :
    Conv(i) = \frac {\omega_1 * 1_{clic}(i) + \omega_2 * 1_{faved}(i) + \omega_3 * 1_{booked}(i)}{\omega_1 + \omega_2 + \omega_3}

    P: List of offer dicts (the playlist)
    h_u: User history dict (sets of known attributes)
    g: List of outcome dicts (clicked, faved, booked)
    gamma: Rank discount factor (e.g., 0.9^(rank-1))
    beta: Weight for the diversity component
    """  # noqa: E501, W505
    total_reward = 0

    for offer, outcome in zip(P, g, strict=True):
        rank = offer["offer_rank"]

        conv_val = get_conversion_reward(outcome, w1, w2, w3)

        div_val = get_diversity_score(
            h_u,
            cat=offer["offer_category"],
            sub=offer["offer_subcategory"],
            lt=offer["offer_venue_type"],
        )

        item_reward = conv_val * (1 + (beta * div_val)) * (gamma ** (rank - 1))

        total_reward += item_reward

    return total_reward
