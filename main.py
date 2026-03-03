import argparse
import base64
import mimetypes
import os
import sys

from dotenv import load_dotenv
from gigachat import GigaChat
from openai import OpenAI


def load_env() -> None:
    """
    Загружает переменные окружения из .env один раз.
    """
    load_dotenv()


def ensure_openai_key() -> None:
    """
    Проверяет, что OPENAI_API_KEY задан.
    """
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Ошибка: переменная окружения OPENAI_API_KEY не найдена.\n"
            "Добавьте её в файл .env или в системные переменные.",
            file=sys.stderr,
        )
        sys.exit(1)


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
    ensure_openai_key()
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


def generate_recipe_with_gigachat(analysis_text: str) -> str:
    """
    Отправляет текстовый анализ из OpenAI в GigaChat
    и просит придумать новый рецепт на основе упомянутых продуктов.
    Использует GIGACHAT_CREDENTIALS и связанные настройки из окружения.
    """
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        print(
            "Предупреждение: GIGACHAT_CREDENTIALS не задан, "
            "рецепт от GigaChat сгенерировать нельзя.",
            file=sys.stderr,
        )
        return ""

    scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    model = os.getenv("GIGACHAT_MODEL", "GigaChat")
    verify_ssl_env = os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "true").lower()
    verify_ssl_certs = verify_ssl_env not in ("false", "0", "no")
    ca_bundle_file = os.getenv("GIGACHAT_CA_BUNDLE_FILE") or None

    prompt = (
        "Ниже приведён отчёт об анализе блюда и список предполагаемых ингредиентов.\n"
        "Используя указанные продукты, придумай новый интересный рецепт блюда.\n\n"
        "Требования к ответу:\n"
        "- кратко укажи название нового блюда;\n"
        "- перечисли список ингредиентов с примерным количеством;\n"
        "- опиши пошаговый способ приготовления;\n"
        "- укажи примерное время приготовления;\n"
        "- добавь короткий совет по подаче.\n\n"
        "Текст анализа блюда:\n"
        "---------------------\n"
        f"{analysis_text}\n"
        "---------------------\n"
    )

    with GigaChat(
        credentials=credentials,
        scope=scope,
        model=model,
        verify_ssl_certs=verify_ssl_certs,
        ca_bundle_file=ca_bundle_file,
    ) as client:
        response = client.chat(prompt)

    try:
        return response.choices[0].message.content
    except Exception:
        return str(response)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CLI для анализа блюд по фотографии с помощью OpenAI "
            "и генерации рецепта в GigaChat."
        )
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
    parser.add_argument(
        "--gigachat-recipe",
        action="store_true",
        help=(
            "Дополнительно отправить анализ блюда в GigaChat "
            "и получить рецепт нового блюда."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    load_env()
    args = parse_args(argv)
    report = analyze_dish(args.image, args.model, args.detail)
    print("\n=== Анализ блюда ===\n")
    print(report)

    if args.gigachat_recipe:
        recipe = generate_recipe_with_gigachat(report)
        if recipe:
            print("\n=== Рецепт от GigaChat ===\n")
            print(recipe)


if __name__ == "__main__":
    main()

