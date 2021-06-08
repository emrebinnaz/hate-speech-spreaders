import psycopg2


def connectDatabase():

    db = psycopg2.connect(user="postgres",
                          password="postgres",
                          host="localhost",
                          port="5432",
                          database="hate-speech-spreaders")

    return db


db = connectDatabase()
imlec = db.cursor()

db.close()