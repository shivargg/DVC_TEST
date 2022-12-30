import snowflake.connector
import pandas as pd
from collections import defaultdict


def remove_comments(contents):
    s = []
    for line in contents.split('\n'):
        if not line.strip().startswith('-') and line.strip():
            s.append(line)
    return '\n'.join(s)


def create_procedure(filepath, filename, database, schema, procedure_name):
    print(f'Creating {database}.{schema}.{procedure_name}')
    #exit()
    procedure_header = f"""
    USE DATABASE {database};
    USE SCHEMA {schema};
    CREATE OR REPLACE PROCEDURE {database}.{schema}.{procedure_name}()
    RETURNS VARCHAR
    LANGUAGE javascript
    EXECUTE AS CALLER
    AS
    """

    with open(f'{filepath}/{filename}', 'r') as fp:
        content = remove_comments(fp.read())
        queries = map(lambda x: x.strip(), content.split(';'))
        execution_strings = []
        for query in queries:
            if len(query.strip()):
                statement = '{sqlText: `' + query + '`}'
                #'{sqlText: `{0}`}'.format(query)
                statement = 'rs = snowflake.execute({0})'.format(statement)
                execution_strings.append(statement)
        # statement = '{sqlText: `' + content + '`}'
        # statement = 'rs = snowflake.execute({0})'.format(statement)
        # execution_strings.append(statement)
        execution_string = ';\n'.join(execution_strings)
        execution_string = execution_string.replace('$', '\$')
        #content = 'sqlText: "{0}"'.format(content)
        procedure_body = """
        $$
        var rs = 0;
        {0};
        return 'Done.';
        $$;
        """.format(execution_string)
        procedure_code = f"{procedure_header}{procedure_body}"
        return procedure_code


def upload_procedure(acc, database, schema, procedure_name, filepath, filename):
    procedure_code = create_procedure(filepath, filename, database, schema, procedure_name)
    with open(f'/home/shiv/Github/DVC_TEST/sql_procedures/{procedure_name}_impl.sql', 'w') as fp:
        fp.write(procedure_code)
    # acc.execute_query(procedure_code)
