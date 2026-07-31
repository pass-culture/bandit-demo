WITH
    weights as (
    SELECT  distinct venue_type_label, count(booking_id) AS bookings, count(distinct venue_id)
    FROM `passculture-data-prod.analytics_prod.global_booking`
    WHERE booking_created_at BETWEEN "2026-02-01" AND "2026-04-01"
    GROUP BY venue_type_label
),

venues as (
    SELECT venue_type_label, count(venue_id) AS venues
    FROM `passculture-data-prod.analytics_prod.global_venue`
    GROUP BY venue_type_label
)

SELECT v.venue_type_label, w.bookings, v.venues
FROM venues v
LEFT JOIN weights w on v.venue_type_label = w.venue_type_label
ORDER BY venue_type_label
