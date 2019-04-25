import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Connection:
    def __init__(self, 
                 dbname="DEVELOPMENT", 
                 dbhost="10.2.51.116", 
                 dbport="5480",
                 dbuser="ANL_PDM",
                 dbpass="DSPDM123"):
        self.dbname = dbname
        self.dbhost = dbhost
        self.dbport = dbport
        self.dbuser = dbuser
        self.dbpass = dbpass
        self.conn, self.cursor = self.create_connection_to_netezza()
       
    def create_connection_to_netezza(self):
        connection_string = "Driver={};servername={}"\
                              ";port={};database={}"\
                              ";username={};password={};".format(
                                  "{NetezzaSQL}",self.dbhost, self.dbport, 
                                  self.dbname, self.dbuser, self.dbpass)
        conn_engine = pyodbc.connect(connection_string)
        db_cursor = conn_engine.cursor()
        return (conn_engine, db_cursor)
    
class Query:
    def __init__(self, query_string=None, source_file=None):
        self.source_file = source_file
        if source_file is not None:
            self.query = self.read_file()
        else:
            self.query = query_string
    
    def execute(self, connection):
        if self.query is not None and len(self.query)>0:
            return pd.read_sql(self.query, connection.conn)
        else:
            print("Error: Query string cannot be empty")
    
    def read_file(self):
        f = open(self.source_file, 'r')
        query_string = ""
        for row in f:
            query_string += row.replace('\n',' ')
        f.close()
        return query_string