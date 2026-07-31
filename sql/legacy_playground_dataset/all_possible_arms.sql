SELECT DISTINCT offer_category_id AS cat, offer_subcategory_id AS sub, venue_type_label AS lt
FROM `passculture-data-prod.analytics_prod.global_offer`
WHERE offer_is_bookable
