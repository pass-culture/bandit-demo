SELECT b.user_id,
    STRUCT(
        ARRAY_AGG(distinct b.offer_category_id) AS categories,
        ARRAY_AGG(distinct b.offer_subcategory_id) AS subcategories,
        ARRAY_AGG(distinct b.venue_type_label) AS venue_types,
        ARRAY_AGG(distinct b.venue_id) AS venue_ids
    ) AS user_history,
    u.user_embedding
FROM `passculture-data-prod.analytics_prod.global_booking` AS b
INNER JOIN `passculture-data-prod.ml_feat_prod.two_tower_last_user_embedding` AS u
ON u.user_id = b.user_id
GROUP BY b.user_id, u.user_embedding
