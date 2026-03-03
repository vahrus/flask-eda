import os
import uuid
from pathlib import Path

from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from main import analyze_dish, generate_recipe_with_gigachat, load_env


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def create_app() -> Flask:
    load_env()

    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/analyze", methods=["POST"])
    def analyze():
        file = request.files.get("image")
        if not file or not file.filename:
            return redirect(url_for("index"))

        original_name = secure_filename(file.filename)
        _, ext = os.path.splitext(original_name)
        filename = f"{uuid.uuid4().hex}{ext.lower()}"

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        analysis = analyze_dish(
            save_path,
            model="gpt-4.1-mini",
            detail="high",
        )

        return render_template(
            "index.html",
            analysis=analysis,
            image_filename=filename,
        )

    @app.route("/gigachat", methods=["POST"])
    def gigachat():
        analysis_text = request.form.get("analysis_text", "")
        if not analysis_text:
            return redirect(url_for("index"))

        recipe = generate_recipe_with_gigachat(analysis_text)

        return render_template(
            "index.html",
            analysis=analysis_text,
            recipe=recipe,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)

