import json
import os.path
from concurrent.futures import ThreadPoolExecutor
from flask import Flask
from flask import request, jsonify, make_response

from web.service import pytorch, gc
from web.service import pytorch_aniMegan2_map, pytorch_aniMegan2_temp_dir
from web.util import uuid, image_util

app = Flask(__name__)


@app.route("/test")
def hello():
    return "hello world"


@app.route("/pytorch/aniMegan2/list", methods=['get'])
def pytorch_ani_megan2_list():
    keys = sorted(list(pytorch_aniMegan2_map.keys()))
    return jsonify({"status": 200, "msg": "SUCCESS",
                    "data": {"model_list": json.loads(json.dumps(keys, ensure_ascii=False))}})


# 使用pytorch框架的aniMegan2模型预测图像
@app.route("/pytorch/aniMegan2", methods=['post'])
def pytorch_ani_megan2():
    input_image = request.files.get("input_image")
    model_name = request.args.get("model")
    if model_name is None or model_name == "" or model_name not in pytorch_aniMegan2_map:
        return jsonify({"status": 400, "msg": "非法的model名称"})
    file_name = uuid.uuid_file_name(input_image.filename)
    if input_image and image_util.allow_image_types(file_name):
        input_path = os.path.join(pytorch_aniMegan2_temp_dir, file_name)
        input_image.save(input_path)

        output_image = pytorch.aniMegan2_run(input_path, model_name)

        output_path = os.path.join(pytorch_aniMegan2_temp_dir, "new" + file_name)
        output_image.save(output_path)

        output_image = open(output_path, "rb").read()

        pool.submit(gc.delete_temp_file_run, [input_path, output_path])

        response = make_response(output_image)
        response.headers['Content-Type'] = 'image/png'

        return response
    else:
        return jsonify({"status": 400, "msg": "非法的请求图片"})


if __name__ == '__main__':
    pool = ThreadPoolExecutor(1)
    app.run(debug=True)
