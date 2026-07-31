WITH import AS (
  SELECT  distinct user_id, offer_category_id, offer_subcategory_id, venue_type_label, venue_id
  FROM `passculture-data-prod.analytics_prod.global_booking`
),

history as (
  SELECT user_id,
    COUNT(distinct offer_category_id) AS nb_categories,
    COUNT(distinct offer_subcategory_id) AS nb_subcategories,
    COUNT(distinct venue_type_label) AS nb_venue_types,
    COUNT(distinct venue_id) AS nb_venues
  FROM import
  GROUP BY user_id
),

cats AS (
  SELECT "categories" as type,
  nb_categories AS amount,
  COUNT(DISTINCT user_id) AS users
  FROM history
  GROUP BY type, amount
),

subcats AS (
  SELECT "subcategories" as type,
  nb_subcategories AS amount,
  COUNT(DISTINCT user_id) AS users
  FROM history
  GROUP BY type, amount
),

venue_type AS (
  SELECT "venue_types" as type,
  nb_venue_types AS amount,
  COUNT(DISTINCT user_id) AS users
  FROM history
  GROUP BY type, amount
),

venues AS (
  SELECT "venues" as type,
  nb_venues AS amount,
  COUNT(DISTINCT user_id) AS users
  FROM history
  GROUP BY type, amount
)

SELECT * FROM cats UNION ALL
SELECT * FROM subcats UNION ALL
SELECT * FROM venue_type UNION ALL
SELECT * FROM venues
ORDER BY type, amount
