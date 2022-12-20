from web.util import const
from PIL import Image


def allow_image_types(image_file_name):
    return '.' in image_file_name and image_file_name.rsplit('.', 1)[1] in const.allow_image_types


def load_image(image_path, x32=False):
    new_image = Image.open(image_path).convert("RGB")

    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32

        w, h = new_image.size
        new_image = new_image.resize((to_32s(w), to_32s(h)))

    return new_image
