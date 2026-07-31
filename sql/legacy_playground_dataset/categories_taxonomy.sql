WITH import AS (
    SELECT  distinct offer_category_id, offer_subcategory_id, count(booking_id) AS bookings
    FROM `passculture-data-prod.analytics_prod.global_booking`
    WHERE booking_created_at BETWEEN "2026-02-01" AND "2026-04-01"
    GROUP BY offer_category_id, offer_subcategory_id
)

SELECT
    offer_category_id,
    sum(bookings) as cat_bookings,
    ARRAY_AGG(distinct offer_subcategory_id) AS subcategories,
    ARRAY_AGG(bookings) AS subcat_bookings
FROM import
GROUP BY offer_category_id
ORDER BY offer_category_id
