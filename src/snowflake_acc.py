from this import d
from click import echo
import pandas as pd 
from IPython.core.magic import (register_line_magic, register_cell_magic,
                                register_line_cell_magic)

import sys

# from https://stackoverflow.com/questions/49870594/pip-main-install-fails-with-module-object-has-no-attribute-main
# def import_or_install(package):
#     try:
#         __import__(package)
#     except:
#         import sys
#         import subprocess
#         subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
#         __import__(package)


try:
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas, pd_writer
    from pandasql import sqldf
    from sqlalchemy import create_engine
except:
    packages = ['snowflake-sqlalchemy', 'pandasql']
    import sys
    import subprocess
    for package in packages:
        subprocess.check_call(['pip', 'install', package])
    import snowflake.connector
    from pandasql import sqldf
    from sqlalchemy import create_engine
    from snowflake.connector.pandas_tools import write_pandas, pd_writer
    #pip.main(['install', 'snowflake-sqlalchemy', 'pandasql'])



pysqldf = lambda q: sqldf(q, globals())

SQL_CACHE = {
    'res': None,
    'acc': None
}


class SnowflakeTableAccessor(object):
    res = None 
    acc = None
    global_namespace = None

    def __init__(self, params):
        self.warehouse = params['WAREHOUSE']
        self.role = params['ROLE']
        self.schema = params['SCHEMA']
        self.database = params['DATABASE']
        self.username = params['USERNAME']
        self.password = params['PASSWORD']
        self.account = params['ACCOUNT']
        self.params = params

    def switch_schema(self, schema):
        self.schema = schema

    def switch_database(self, database):
        self.database = database

    def get_conection(self, params):
        user = params.get('USERNAME', self.username)
        password = params.get('PASSWORD', self.password)
        account = params.get('ACCOUNT', self.account)
        warehouse = params.get('WAREHOUSE', self.warehouse)
        role = params.get('ROLE', self.role)
        schema = params.get('SCHEMA', self.schema)
        database = params.get('DATABASE', self.database)
        conn = snowflake.connector.connect(
            user=user,
            password=password,
            account=account,
            database=database,
            warehouse=warehouse,
            role=role,
            schema=schema)
        return conn

    def get_engine(self, params):
        user = params.get('USERNAME', self.username)
        password = params.get('PASSWORD', self.password)
        account = params.get('ACCOUNT', self.account)
        warehouse = params.get('WAREHOUSE', self.warehouse)
        role = params.get('ROLE', self.role)
        schema = params.get('SCHEMA', self.schema)
        database = params.get('DATABASE', self.database)
        connection_string = f"""
            snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}&role={role}'
        """.strip()
        engine = create_engine(connection_string, echo=True)
        return engine

    def insert_dataframe(self, table_name, df, **kwargs):
        table_statement = pd.io.sql.get_schema(df, table_name).replace('"', '').replace('CREATE', 'CREATE OR REPLACE')
        table_statement = table_statement.replace(table_name, f'"{table_name}"')
        print('Executing ', table_statement)
        self.execute_query(table_statement, **kwargs)
        connection = self.get_conection(kwargs)
        success, nchunks, nrows, _ = write_pandas(connection, df, table_name)
        print('Wrote ', nrows, ' rows of data')
        # res = None
        # try:
        #     engine = self.get_engine(kwargs)
        #     schema = kwargs.get('SCHEMA', self.schema)
        #     print('Writing to ', table_name, ' in ', schema)
        #     res = df.to_sql(name=table_name.lower(), con=engine, 
        #         schema=schema, method=pd_writer, index=False, if_exists='replace')
        # finally:
        #     engine.dispose()
        #     print(res)
        #     return res
        
    def get_table_info(self, table, is_view=False, name_qualified=False, **kwargs):
        conn = self.get_conection(kwargs)
        table_name = table 
        if not name_qualified:
            table_name = f'{self.database}.{self.schema}.{table}'
        print('Getting info for ', table_name)
        query = f'DESCRIBE {"VIEW" if is_view else "TABLE"} {table_name}'
        curr = conn.cursor()
        print('Executing ', query)
        curr.execute(query)
        rows = []
        columns = [d[0] for d in curr.description]
        print('Processing table with ',len(columns), 'columns')
        for row in curr:
            rows.append(row)
        print('Processed ', len(rows), ' rows')
        curr.close()
        conn.close()
        df = pd.DataFrame(rows, columns=columns)
        return df
    
    def retrieve_query_res(self, query, **kwargs):
        conn = self.get_conection(kwargs)
        curr = conn.cursor()
        print('Executing')
        curr.execute(query)
        rows = []
        columns = [d[0] for d in curr.description]
        print('Processing table with ',len(columns), 'columns')
        for row in curr:
            rows.append(row)
        print('Processed ', len(rows), ' rows')
        curr.close()
        conn.close()
        df = pd.DataFrame(rows, columns=columns)
        return df
    
    
    def execute_query(self, query, **kwargs):
        print(f'Executing')
        conn = self.get_conection(kwargs)
        curr = conn.cursor()
        curr.execute(query)
        curr.close()
        conn.close()


def create_snowflake_accessor(params, gl):
    acc = SnowflakeTableAccessor(params)
    SnowflakeTableAccessor.acc = acc
    SnowflakeTableAccessor.global_namespace = gl
    #SQL_CACHE['acc'] = acc 
    return acc

def get_last_result():
    return SnowflakeTableAccessor.res


def retr_from_sf(line, cell):
    acc = SnowflakeTableAccessor.acc#SQL_CACHE['acc']
    SnowflakeTableAccessor.res = acc.retrieve_query_res(cell)
    return cell

def send_to_sf(line, cell):
    acc = SnowflakeTableAccessor.acc 
    res = sqldf(cell, SnowflakeTableAccessor.global_namespace)
    print('Writing result of ', cell, ' to ', line)
    acc.insert_dataframe(line, res)


def execute_on_sf(line, cell):
    acc = SnowflakeTableAccessor.acc#SQL_CACHE['acc']
    acc.execute_query(cell)
    return cell

def pandas_sql(line, cell):
    res = sqldf(cell, SnowflakeTableAccessor.global_namespace)
    SnowflakeTableAccessor.res = res
    return cell

def load_ipython_extension(ipython):
    ipython.register_magic_function(retr_from_sf, 'cell')
    ipython.register_magic_function(execute_on_sf, 'cell')
    ipython.register_magic_function(pandas_sql, 'cell')
    ipython.register_magic_function(send_to_sf, 'cell')