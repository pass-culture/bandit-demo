SELECT DISTINCT
    o.offer_category_id AS cat,
    o.offer_subcategory_id AS sub,
    o.venue_type_label AS lt
FROM `passculture-data-prod.analytics_prod.global_offer` AS o
INNER JOIN `passculture-data-prod.ml_feat_prod.two_tower_last_item_embedding` AS tt
ON tt.item_id = o.item_id
GROUP BY o.item_id, o.offer_category_id, o.offer_subcategory_id, o.venue_type_label
