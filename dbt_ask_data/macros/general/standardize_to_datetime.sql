{% macro standardize_to_datetime(column_name) %}
    case
        when {{ column_name }} is null or trim({{ column_name }}) = '' then null
        else 
            -- Clean and standardize the datetime string once
            case
                when {{ clean_datetime_string(column_name) }} ~ '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'
                    then to_timestamp(
                        {{ clean_datetime_string(column_name) }},
                        'YYYY-MM-DD HH24:MI:SS'
                    )
                when {{ clean_datetime_string(column_name) }} ~ '^\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}$'
                    then to_timestamp(
                        {{ clean_datetime_string(column_name) }},
                        'MM-DD-YYYY HH24:MI:SS'
                    )
                -- Handle date-only inputs (no time)
                when {{ clean_datetime_string(column_name) }} ~ '^\d{4}-\d{2}-\d{2}$'
                    then to_date(
                        {{ clean_datetime_string(column_name) }},
                        'YYYY-MM-DD'
                    )
                when {{ clean_datetime_string(column_name) }} ~ '^\d{2}-\d{2}-\d{4}$'
                    then to_date(
                        {{ clean_datetime_string(column_name) }},
                        'MM-DD-YYYY'
                    )
                else null
            end
    end
{% endmacro %}