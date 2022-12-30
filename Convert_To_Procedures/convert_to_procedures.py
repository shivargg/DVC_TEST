import json
import snowflake_acc
import procedure_utils


SPECS = [
    ('00_REF.sql', 'CONSTRUCT_00_REF_TABLES_TOP_AGENT', 'DS_GLOC_DEV_DB', '"00_REF"'),
#    ('02_INT.sql', 'CONSTRUCT_02_INT_TABLES_TOP_AGENT', 'DS_GLOC_DEV_DB', '"02_INT"'),
#    ('03_PRI.sql', 'CONSTRUCT_03_PRI_TABLES_TOP_AGENT', 'DS_GLOC_DEV_DB', '"03_PRI"'),
#    ('04_FEA_TOP_AGENT.sql', 'CONSTRUCT_04_FEA_TABLES_TOP_AGENT', 'DS_GLOC_DEV_DB', '"04_FEA_TOP_AGENT"'),
#    ('05_MODEL_INPUT_TOP_AGENT.sql', 'CONSTRUCT_05_MODEL_INPUT_TABLES_TOP_AGENT', 'DS_GLOC_DEV_DB', '"05_MODEL_INPUT_TOP_AGENT"')
]


def main():
    file_path = '/home/shiv/Github/DVC_TEST/sql/'
    with open('/home/shiv/Github/DVC_TEST/src/uat_creds.json', 'r') as fp:
        params = json.load(fp)
        acc = snowflake_acc.create_snowflake_accessor(params, globals())
        for input_file, procedure_name, database, schema in SPECS:
            print('Converting code in ', input_file)
            acc.switch_database(database)
            acc.switch_schema(schema)
            procedure_utils.upload_procedure(acc, database, schema, procedure_name, file_path, input_file)

if __name__ == '__main__':
    main()
