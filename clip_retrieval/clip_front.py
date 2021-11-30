"""clip front"""

from flask import Flask, send_from_directory, request
import json
import fire


def add_static_endpoints(app, default_backend=None, default_index=None, url_column="url"):
    """add static endpoints to the flask app"""
    import pkg_resources  # pylint: disable=import-outside-toplevel

    front_path = pkg_resources.resource_filename("clip_retrieval", "../front/build")

    def static_dir_():
        return send_from_directory(front_path, "index.html")

    app.route("/")(static_dir_)

    def config_json():
        back = default_backend if default_backend is not None else request.host_url
        index = default_index if default_index is not None else ""
        config = {"defaultBackend": back, "defaultIndex": index, "urlColumn": url_column}
        return json.dumps(config)

    app.route("/config.json")(config_json)

    def static_dir(path):
        return send_from_directory(front_path, path)

    app.route("/<path:path>")(static_dir)


def clip_front(default_backend=None, default_index=None, url_column="url"):
    app = Flask(__name__)
    add_static_endpoints(app, default_backend, default_index, url_column)
    app.run(host="0.0.0.0", port=1235, debug=False)


if __name__ == "__main__":
    fire.Fire(clip_front)
