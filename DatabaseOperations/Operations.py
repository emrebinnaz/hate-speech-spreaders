import psycopg2


def connectDatabase():

    connection = psycopg2.connect(user="postgres",
                          password="postgres",
                          host="localhost",
                          port="5432",
                          database="hate-speech-spreaders")

    return connection


def insertHashtag(records, connection, cursor):

    sql_insert_query = """ INSERT INTO hashtag(name, fetching_time) VALUES (%s, %s) """

    cursor.executemany(sql_insert_query, records)
    connection.commit()
    print(cursor.rowcount, "Record inserted successfully into table")


def insertTweetOwner(records, connection, cursor):

    sql_insert_query = """ INSERT INTO tweet_owner(id, fetching_time, followers, following, image_url, 
                                            name, type_of_spreader, username) 
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING"""

    cursor.executemany(sql_insert_query, records)
    connection.commit()
    print(cursor.rowcount, "Record inserted successfully into table")
    connection.close()


def insertTweet(records):

    connection = connectDatabase()
    cursor = connection.cursor()

    sql_insert_query = """ INSERT INTO tweet(id, fetching_time, fav_count, label, place_of_tweet, 
                                            preprocessed_text, rt_count, text, owner_id) 
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) """

    cursor.executemany(sql_insert_query, records)
    connection.commit()
    print(cursor.rowcount, "Record inserted successfully into table")
    connection.close()


def insertTweetsOfHashtag(records):

    connection = connectDatabase()
    cursor = connection.cursor()

    sql_insert_query = """ INSERT INTO tweets_of_hashtag(fetching_time, hashtag_id, tweet_id) 
                               VALUES (%s, %s, %s) """

    cursor.executemany(sql_insert_query, records)
    connection.commit()
    print(cursor.rowcount, "Record inserted successfully into table")
    connection.close()

def getHashtags(date, connection, cursor):

    sql_select_query = """ SELECT *  FROM hashtag WHERE fetching_time = %s """

    cursor.execute(sql_select_query, (date,))
    connection.commit()
    print(cursor.rowcount, "Records were fetched successfully")

    return cursor.fetchall()


def getOwners():

    connection = connectDatabase()
    cursor = connection.cursor()

    sql_select_query = """ SELECT *  FROM tweet_owner"""

    cursor.execute(sql_select_query)
    connection.commit()
    print(cursor.rowcount, "Records were fetched successfully")

    return cursor.fetchall()


def getTweets(date, connection, cursor):

    sql_select_query = """ SELECT *,MAX(rt_count + fav_count) FROM tweets ORDER BY MAX(rt_count + fav_count) DESC limit 10 """

    cursor.execute(sql_select_query, (date,))
    connection.commit()
    print(cursor.rowcount, "Records were fetched successfully")

    return cursor.fetchall()