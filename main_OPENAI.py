import argparse
import base64
import mimetypes
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI


def load_api_key() -> str:
    """
    Загружает OPENAI_API_KEY из .env или окружения.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Ошибка: переменная окружения OPENAI_API_KEY не найдена.\n"
            "Добавьте её в файл .env или в системные переменные.",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def image_to_data_url(image_path: str) -> str:
    """
    Кодирует локальный файл изображения в data URL (base64),
    подходящий для передачи в OpenAI Responses API как input_image.
    """
    if not os.path.isfile(image_path):
        print(f"Ошибка: файл '{image_path}' не найден.", file=sys.stderr)
        sys.exit(1)

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        # По умолчанию считаем JPEG, если определить не удалось
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def build_prompt() -> str:
    """
    Возвращает текстовый промпт для анализа блюда.
    """
    return (
        "Ты ассистент по анализу блюд по фотографии.\n\n"
        "Твои задачи:\n"
        "1) Определить, какое блюдо изображено на фото.\n"
        "2) Предположить состав (ингредиенты) и способ приготовления.\n"
        "3) Оценить примерную калорийность и пищевую ценность.\n\n"
        "Важно:\n"
        "- Если точный состав неизвестен, делай разумные предположения на основе внешнего вида.\n"
        "- Калорийность и БЖУ оценивай приблизительно, исходя из типичных рецептур.\n"
        "- Если часть данных невозможно определить по фото, указывай «не могу определить».\n\n"
        "Под \"анализом блюда\" подразумеваются:\n"
        "- название блюда;\n"
        "- категория (суп, салат, десерт, фастфуд, напиток и т.п.);\n"
        "- предполагаемые ингредиенты;\n"
        "- способ приготовления (если можно определить: жареное, запечённое, варёное и т.д.);\n"
        "- примерный вес порции;\n"
        "- примерная калорийность (ккал);\n"
        "- ориентировочные БЖУ (белки, жиры, углеводы);\n"
        "- возможные аллергены;\n"
        "- предполагаемая кухня (итальянская, японская, домашняя и т.п.);\n"
        "- степень уверенности в распознавании.\n\n"
        "Оформи ответ как чётко структурированный отчёт на русском языке:\n"
        "- сначала перечисли ключевые факты по пунктам (заголовок + значение);\n"
        "- затем отдельным блоком дай краткую рекомендацию (например, по полезности блюда, частоте употребления и т.п.).\n"
    )


def analyze_dish(image_path: str, model: str, detail: str) -> str:
    """
    Отправляет изображение в OpenAI Responses API и возвращает текстовый отчёт.
    """
    load_api_key()
    client = OpenAI()

    data_url = image_to_data_url(image_path)
    prompt = build_prompt()

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": detail,
                    },
                ],
            }
        ],
    )

    # В новом SDK есть удобное свойство output_text
    try:
        return response.output_text
    except AttributeError:
        # На всякий случай резервный вариант
        outputs = getattr(response, "output", None) or []
        text_chunks = []
        for item in outputs:
            if getattr(item, "type", None) == "output_text":
                text_chunks.append(getattr(item, "content", ""))
        return "\n".join(text_chunks) if text_chunks else str(response)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI для анализа блюд по фотографии с помощью OpenAI."
    )
    parser.add_argument(
        "image",
        help="Путь к изображению блюда (PNG, JPG, JPEG, WEBP, GIF).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Имя модели OpenAI для анализа (по умолчанию: gpt-4.1-mini).",
    )
    parser.add_argument(
        "--detail",
        choices=["low", "high", "auto"],
        default="high",
        help="Уровень детализации обработки изображения (по умолчанию: high).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = analyze_dish(args.image, args.model, args.detail)
    print("\n=== Анализ блюда ===\n")
    print(report)


if __name__ == "__main__":
    main()

