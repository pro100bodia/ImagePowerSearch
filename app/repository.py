import os

import psycopg2
from fastapi.responses import Response

conn = psycopg2.connect(
    host="localhost",
    database="image_data_storage",
    user=os.environ['PS_USER'],
    password=os.environ['PS_PASS']
)

cursor = conn.cursor()


def insert_image(image):
    insert_query = "INSERT INTO \"images\"(\"binary\") VALUES (decode(%s, 'hex'));"

    data = (image.hex(),)

    cursor.execute(insert_query, data)
    conn.commit()


def get_last_image_id():
    select_query = "SELECT MAX(\"id\") AS \"image_id\" FROM \"images\""

    cursor.execute(select_query)
    conn.commit()
    result_tuple = cursor.fetchone()

    return result_tuple[0]


def insert_caption(image_id, image_model):
    insert_query = "INSERT INTO \"captions\"(\"caption\", \"image_id\") VALUES (%s, %s);"
    data = (image_model.captions, image_id)

    cursor.execute(insert_query, data)
    conn.commit()


def insert_objects(image_model):
    objects = image_model.objects

    insert_query = "INSERT INTO \"objects\"(\"object\") VALUES (%s);"

    for _object in objects:
        data = (_object,)

        cursor.execute(insert_query, data)
        conn.commit()


def get_last_object_ids(image_model):
    objects_number = len(image_model.objects)

    select_query = "SELECT \"id\" FROM \"objects\" ORDER BY \"id\" DESC LIMIT %s"
    data = (objects_number,)

    cursor.execute(select_query, data)
    conn.commit()
    return cursor.fetchall()


def insert_into_objects_images(object_ids, image_id):
    for object_id in object_ids:
        insert_query = "INSERT INTO \"objects_images\"(\"object_id\", \"image_id\") VALUES(%s, %s)"
        data = (object_id, image_id)

        cursor.execute(insert_query, data)
        conn.commit()


def insert_faces(image_model):
    faces = image_model.faces

    for face in faces:
        insert_query = "INSERT INTO \"faces\"(\"person\") VALUES (%s);"
        data = (face,)

        cursor.execute(insert_query, data)
        conn.commit()


def get_last_face_ids(image_model):
    objects_number = len(image_model.faces)

    select_query = "SELECT \"id\" FROM \"faces\" ORDER BY \"id\" DESC LIMIT %s"
    data = (objects_number,)

    cursor.execute(select_query, data)
    conn.commit()
    return cursor.fetchall()


def insert_into_faces_images(face_ids, image_id):
    for face_id in face_ids:
        insert_query = "INSERT INTO \"faces_images\"(\"face_id\", \"image_id\") VALUES(%s, %s)"
        data = (face_id, image_id)

        cursor.execute(insert_query, data)
        conn.commit()


def save_information(image, image_model):
    # insert image
    insert_image(image)

    # get id of the image
    image_id = get_last_image_id()

    # insert captions with image id
    insert_caption(image_id, image_model)

    # insert objects
    insert_objects(image_model)
    # get ids of inserted objects
    object_ids = get_last_object_ids(image_model)

    # insert objects ids with image id
    insert_into_objects_images(object_ids, image_id)

    # insert faces
    insert_faces(image_model)

    # get faces ids
    face_ids = get_last_face_ids(image_model)
    print("FACE_IDS" + str(face_ids))

    # insert faces ids with image id
    insert_into_faces_images(face_ids, image_id)

    cursor.close()
    conn.close()
    return None


def alter_person(person_id, name):
    update_query = 'UPDATE "faces" SET "person"=%s	WHERE "id"=%s;'
    data = (name, person_id)
    cursor.execute(update_query, data)

    conn.commit()
    cursor.close()
    conn.close()
    return {"Result": "The " + name + " was assigned to person #" + str(person_id)}


def search(query):
    data = ('%' + query + '%',)

    captions_search_query = "SELECT \"images\".\"id\" FROM \"captions\"" \
                            "INNER JOIN \"images\" ON \"captions\".\"image_id\" = \"images\".\"id\"" \
                            "WHERE \"captions\".\"caption\" LIKE %s;"

    cursor.execute(captions_search_query, data)
    conn.commit()
    image_ids = cursor.fetchall()

    objects_search_query = "SELECT \"images\".\"id\" FROM \"objects\"" \
                           "INNER JOIN \"objects_images\" ON \"objects\".\"id\" = \"objects_images\".\"object_id\"" \
                           "INNER JOIN \"images\" ON \"objects_images\".\"image_id\" = \"images\".\"id\"" \
                           "WHERE \"objects\".\"object\" LIKE %s;"

    cursor.execute(objects_search_query, data)
    conn.commit()
    image_ids.extend(cursor.fetchall())

    faces_search_query = "SELECT \"images\".\"id\" FROM \"faces\"" \
                         "INNER JOIN \"faces_images\" ON \"faces\".\"id\" = \"faces_images\".\"face_id\"" \
                         "INNER JOIN \"images\" ON \"faces_images\".\"image_id\" = \"images\".\"id\"" \
                         "WHERE \"faces\".\"person\" LIKE %s;"

    cursor.execute(faces_search_query, data)
    conn.commit()
    image_ids.extend(cursor.fetchall())

    image_ids = [image_id[0] for image_id in image_ids]

    image_ids = list(set(image_ids))

    image_links = ["http://localhost:8000/image/" + str(image_id) for image_id in image_ids]

    return {
        "results": image_links
    }


def get_image(image_id):
    get_query = "SELECT \"binary\" FROM \"images\" WHERE \"id\"=%s;"
    data = (image_id,)

    cursor.execute(get_query, data)
    conn.commit()
    image = cursor.fetchone()[0]

    return Response(content=image.tobytes(), media_type="image/jpeg")
