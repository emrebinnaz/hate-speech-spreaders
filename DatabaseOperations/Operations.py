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
    print(cursor.rowcount, "Records were inserted to hashtag table successfully")


def insertTweetOwner(records, connection, cursor):

    sql_insert_query = """ INSERT INTO tweet_owner(id, fetching_time, followers, following, image_url, 
                                            name, type_of_spreader, username) 
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING"""

    cursor.executemany(sql_insert_query, records)
    connection.commit()
    print(cursor.rowcount, "Records were inserted to tweet_owners table successfully")
    connection.close()


def insertTweet(records):

    connection = connectDatabase()
    cursor = connection.cursor()

    sql_insert_query = """ INSERT INTO tweet(id, fetching_time, fav_count, label, place_of_tweet, 
                                            rt_count, text, owner_id, created_date) 
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) do update set place_of_tweet = 'BOTH'"""

    cursor.executemany(sql_insert_query, records)
    connection.commit()
    print(cursor.rowcount, "Records weere inserted to tweet table successfully")
    connection.close()


def insertTweetsOfHashtag(records):

    connection = connectDatabase()
    cursor = connection.cursor()

    sql_insert_query = """ INSERT INTO tweets_of_hashtag(fetching_time, hashtag_id, tweet_id) 
                               VALUES (%s, %s, %s) """

    cursor.executemany(sql_insert_query, records)
    connection.commit()
    print(cursor.rowcount, "Record were inserted to tweets_of_hashtag table successfully")
    connection.close()


def getHashtags(date, connection, cursor):

    sql_select_query = """ SELECT *  FROM hashtag WHERE fetching_time = %s """

    cursor.execute(sql_select_query, (date,))
    connection.commit()
    print(cursor.rowcount, "Records were fetched from hashtag table successfully")

    return cursor.fetchall()


def getOwners():

    connection = connectDatabase()
    cursor = connection.cursor()

    sql_select_query = """ SELECT *  FROM tweet_owner"""

    cursor.execute(sql_select_query)
    connection.commit()
    print(cursor.rowcount, "Records were fetched from tweet_owner table successfully")

    return cursor.fetchall()


def getTweets(date, connection, cursor):

    sql_select_query = """ SELECT *,MAX(rt_count + fav_count) FROM tweets ORDER BY MAX(rt_count + fav_count) DESC limit 10 """

    cursor.execute(sql_select_query, (date,))
    connection.commit()
    print(cursor.rowcount, "Records were fetched from tweets table successfully")

    return cursor.fetchall()


def getMostInteractedTweetOwnerIds():

    connection = connectDatabase()
    cursor = connection.cursor()

    sql_select_query = """select owner_id, fav_count,rt_count
                          from (
                                    select distinct on (owner_id) owner_id,fav_count,rt_count
                                    from (
                                            select distinct on (text) text, fav_count, rt_count, owner_id
                                            from tweet) as uniqueTweets) as uniqueOwners
                          order by fav_count + rt_count desc limit 10;"""

    cursor.execute(sql_select_query)
    connection.commit()

    print(cursor.rowcount, " most interacted tweet owners were fetched from tweet table successfully")
    records = cursor.fetchall()

    mostInteractedTweetOwnerIds = []

    for record in records:
        mostInteractedTweetOwnerIds.append(record[0])

    print(mostInteractedTweetOwnerIds)

    return mostInteractedTweetOwnerIds