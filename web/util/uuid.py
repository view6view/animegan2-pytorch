import uuid


def uuid_file_name(file_name):
    return str(uuid.uuid1()) + "_" + file_name
