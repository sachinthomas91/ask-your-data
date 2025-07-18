{% macro clean_datetime_string(column_name) %}
    case
        when {{ column_name }} is null or trim({{ column_name }}) = '' then null
        else 
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        -- Handle missing time parts and pad dates
                        case
                            -- If no time part exists, add 00:00:00
                            when not ({{ column_name }} ~ '\\d{1,4}[/\\-]\\d{1,2}[/\\-]\\d{1,4}\\s+\\d{1,2}:')
                            then 
                                concat(
                                    regexp_replace(
                                        regexp_replace(
                                            regexp_replace(
                                                {{ column_name }},
                                                '^([0-9])([/\\-])',
                                                '0\\1\\2'
                                            ),
                                            '[/\\-]([0-9])[/\\-]',
                                            '-0\\1-'
                                        ),
                                        '[/\\]',
                                        '-'
                                    ),
                                    ' 00:00:00'
                                )
                            -- If only hour exists (H:), add 00:00
                            when {{ column_name }} ~ '\\d{1,4}[/\\-]\\d{1,2}[/\\-]\\d{1,4}\\s+\\d{1,2}:$'
                            then 
                                concat(
                                    regexp_replace(
                                        regexp_replace(
                                            regexp_replace(
                                                {{ column_name }},
                                                '^([0-9])([/\\-])',
                                                '0\\1\\2'
                                            ),
                                            '[/\\-]([0-9])[/\\-]',
                                            '-0\\1-'
                                        ),
                                        '[/\\]',
                                        '-'
                                    ),
                                    '00:00'
                                )
                            -- If only hour:minute exists (HH:MM), add :00
                            when {{ column_name }} ~ '\\d{1,4}[/\\-]\\d{1,2}[/\\-]\\d{1,4}\\s+\\d{1,2}:\\d{1,2}$'
                            then 
                                concat(
                                    regexp_replace(
                                        regexp_replace(
                                            regexp_replace(
                                                {{ column_name }},
                                                '^([0-9])([/\\-])',
                                                '0\\1\\2'
                                            ),
                                            '[/\\-]([0-9])[/\\-]',
                                            '-0\\1-'
                                        ),
                                        '[/\\]',
                                        '-'
                                    ),
                                    ':00'
                                )
                            else 
                                regexp_replace(
                                    regexp_replace(
                                        regexp_replace(
                                            {{ column_name }},
                                            '^([0-9])([/\\-])',
                                            '0\\1\\2'
                                        ),
                                        '[/\\-]([0-9])[/\\-]',
                                        '-0\\1-'
                                    ),
                                    '[/\\]',
                                    '-'
                                )
                        end,
                        -- Pad single-digit hours
                        '(\\s)(\\d)(:)' ,
                        '\\10\\2\\3'
                    ),
                    -- Pad single-digit minutes
                    ':(\\d)($|:)' ,
                    ':0\\1\\2'
                ),
                -- Pad single-digit seconds
                ':(\\d)$' ,
                ':0\\1'
            )
    end
{% endmacro %}
