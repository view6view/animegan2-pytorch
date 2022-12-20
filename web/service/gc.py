import os


def delete_temp_file_run(file_path_arr):
    for path in file_path_arr:
        os.remove(path)