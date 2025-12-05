{{
  config(
    materialized='table',
    tags=['dimension', 'olist']
  )
}}

with date_spine as (
  -- Generate dates from the earliest order to latest expected date
  select
    cast(d as date) as calendar_date
  from (
    select
      generate_series(
        (select min(date_trunc('day', order_purchase_datetime))::date from {{ ref('stg_orders') }}),
        (select max(date_trunc('day', order_purchase_datetime))::date from {{ ref('stg_orders') }}) + interval '1 year',
        '1 day'::interval
      ) as d
  ) as date_series
),

date_details as (
  select
    calendar_date,

    -- Year and Month
    extract(year from calendar_date)::int as calendar_year,
    extract(month from calendar_date)::int as calendar_month,
    extract(quarter from calendar_date)::int as calendar_quarter,

    -- Day of month/week
    extract(day from calendar_date)::int as day_of_month,
    extract(dow from calendar_date)::int as day_of_week_numeric,  -- 0=Sunday, 6=Saturday
    extract(isodow from calendar_date)::int as iso_day_of_week,   -- 1=Monday, 7=Sunday
    extract(doy from calendar_date)::int as day_of_year,

    -- Week numbers
    extract(week from calendar_date)::int as week_of_year,
    extract(isoyear from calendar_date)::int as iso_year,
    extract(isoyear from calendar_date)::text || '-W' ||
      lpad(extract(week from calendar_date)::text, 2, '0') as iso_week_number,

    -- Day names and abbreviations
    to_char(calendar_date, 'Day')::varchar as day_name,
    to_char(calendar_date, 'Dy')::varchar as day_abbreviation,
    to_char(calendar_date, 'Month')::varchar as month_name,
    to_char(calendar_date, 'Mon')::varchar as month_abbreviation,

    -- Fiscal year (assume fiscal year = calendar year for now, can adjust if needed)
    extract(year from calendar_date)::int as fiscal_year,
    extract(quarter from calendar_date)::int as fiscal_quarter,
    extract(month from calendar_date)::int as fiscal_month,

    -- Boolean flags
    case when extract(isodow from calendar_date) in (6, 7) then 1 else 0 end as is_weekend_flag,
    case when extract(isodow from calendar_date) not in (6, 7) then 1 else 0 end as is_weekday_flag,

    -- Business day flag (simple: not weekend; doesn't account for holidays yet)
    case when extract(isodow from calendar_date) not in (6, 7) then 1 else 0 end as is_business_day_flag,

    -- Month boundary flags
    case when extract(day from calendar_date) = 1 then 1 else 0 end as is_first_day_of_month_flag,
    case when extract(day from calendar_date) = extract(day from (date_trunc('month', calendar_date) + interval '1 month' - interval '1 day')) then 1 else 0 end as is_last_day_of_month_flag,

    -- Week boundary flags
    case when extract(isodow from calendar_date) = 1 then 1 else 0 end as is_monday_flag,
    case when extract(isodow from calendar_date) = 5 then 1 else 0 end as is_friday_flag,
    case when extract(isodow from calendar_date) = 1 then 1 else 0 end as is_week_start_flag,
    case when extract(isodow from calendar_date) = 7 then 1 else 0 end as is_week_end_flag,

    -- Quarter boundary flags
    case when extract(month from calendar_date) in (1, 4, 7, 10) and extract(day from calendar_date) = 1 then 1 else 0 end as is_quarter_start_flag,

    -- Year boundary flags
    case when extract(month from calendar_date) = 1 and extract(day from calendar_date) = 1 then 1 else 0 end as is_year_start_flag,
    case when extract(month from calendar_date) = 12 and extract(day from calendar_date) = 31 then 1 else 0 end as is_year_end_flag,

    -- Days since/until epoch (useful for certain calculations)
    (calendar_date - '1970-01-01'::date)::int as days_since_epoch

  from date_spine
)

select
  calendar_date,
  calendar_year,
  calendar_month,
  calendar_quarter,
  day_of_month,
  day_of_week_numeric,
  iso_day_of_week,
  day_of_year,
  week_of_year,
  iso_year,
  iso_week_number,
  day_name,
  day_abbreviation,
  month_name,
  month_abbreviation,
  fiscal_year,
  fiscal_quarter,
  fiscal_month,
  is_weekend_flag,
  is_weekday_flag,
  is_business_day_flag,
  is_first_day_of_month_flag,
  is_last_day_of_month_flag,
  is_monday_flag,
  is_friday_flag,
  is_week_start_flag,
  is_week_end_flag,
  is_quarter_start_flag,
  is_year_start_flag,
  is_year_end_flag,
  days_since_epoch
from date_details
order by calendar_date
