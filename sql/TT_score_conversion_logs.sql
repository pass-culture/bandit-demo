WITH

conversion AS (
  SELECT
    reco_call_id,
    offer_id,
    item_id,
    offer_reco_rank,
    offer_category,
    STRUCT(
      offer_viewed AS viewed,
      offer_consulted AS clicked,
      offer_faved AS faved,
      offer_booked AS booked
    ) AS conversion
  FROM `passculture-data-prod.int_pcreco_prod.reco_offer_conversion`
  WHERE event_date >= DATE_ADD(CURRENT_DATE('Europe/Paris'), INTERVAL - 3 month)
),

score AS (
  SELECT reco_call_id, offer_id, user_id, offer_context.offer_item_score AS score
  FROM `passculture-data-prod.int_pcreco_prod.past_offer_context_sink`
  WHERE event_date >=  DATE_ADD(CURRENT_DATE('Europe/Paris'), INTERVAL - 3 month)
)

SELECT
  c.item_id,
  c.offer_id,
  c.offer_reco_rank,
  c.offer_category,
  s.user_id,
  c.conversion,
  s.score
FROM conversion AS c
INNER JOIN score AS s
ON c.reco_call_id = s.reco_call_id
AND c.offer_id = s.offer_id
