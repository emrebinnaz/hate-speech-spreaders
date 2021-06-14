import psycopg2


def connectDatabase():

    connection = psycopg2.connect(user="postgres",
                          password="postgres",
                          host="localhost",
                          port="5432",
                          database="hate-speech-spreaders")

    return connection


def insert(records, connection, cursor):


    sql_insert_query = """ INSERT INTO hashtag(name, fetching_time) VALUES (%s,%s) """

    cursor.executemany(sql_insert_query, records)
    connection.commit()
    print(cursor.rowcount, "Record inserted successfully into table")



