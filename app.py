import os.path

from flask import Flask
from flask import request, jsonify, make_response
from web.service import pytorch, gc
from concurrent.futures import ThreadPoolExecutor
from web.util import uuid, image_util

app = Flask(__name__)


@app.route("/test")
def hello():
    return "hello world"


# 使用pytorch框架的aniMegan2模型预测图像
@app.route("/pytorch/aniMegan2", methods=['post'])
def pytorch_ani_megan2():
    input_image = request.files.get("input_image")
    file_name = uuid.uuid_file_name(input_image.filename)
    if input_image and image_util.allow_image_types(file_name):
        input_path = os.path.join("./temp", file_name)
        input_image.save(input_path)
        output_image = pytorch.aniMegan2_run(input_path)
        output_path = os.path.join("./temp", "new" + file_name)
        output_image.save(output_path)

        output_image = open(output_path, "rb").read()

        pool.submit(gc.delete_temp_file_run, [input_path, output_path])

        response = make_response(output_image)
        response.headers['Content-Type'] = 'image/png'

        return response
    else:
        return jsonify({"status": 400, "msg": "非法的请求图片"})


if __name__ == '__main__':
    pool = ThreadPoolExecutor(2)
    app.run(debug=True)
