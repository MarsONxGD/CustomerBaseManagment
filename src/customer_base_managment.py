import csv
import os
import subprocess
import sys
from pathlib import Path


def run_email_handler(force_mode=False):
    mode_text = "в форсированном режиме" if force_mode else "макс. 10 писем"
    print(f"\n🔄 Запуск обработки новых писем ({mode_text})...")
    try:
        if force_mode:
            result = subprocess.run(
                [sys.executable, "email_handler.py", "--force"],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        else:
            result = subprocess.run(
                [sys.executable, "email_handler.py"],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )

        print("✅ Обработка писем завершена")

        if not force_mode:
            for line in result.stdout.split("\n"):
                if "РЕЗУЛЬТАТ:" in line:
                    print(f"{line.replace('РЕЗУЛЬТАТ:', '📊 ').strip()}")
                elif "ОСТАЛОСЬ:" in line:
                    print(f"{line.replace('ОСТАЛОСЬ:', '📭 ').strip()}")
            if result.stderr:
                error_lines = [
                    line for line in result.stderr.split("\n") if "ERROR" in line
                ]
                if error_lines:
                    print("\n❌ Ошибки (ERROR):")
                    for error_line in error_lines:
                        clean_line = (
                            error_line.split("- ERROR - ")[-1]
                            if "- ERROR - " in error_line
                            else error_line
                        )
                        print(f"   {clean_line}")
        else:
            if result.stdout:
                print("\n📋 Вывод программы:")
                print(result.stdout)
            if result.stderr:
                print("\n⚠️ Предупреждения и ошибки:")
                print(result.stderr)

    except Exception as e:
        print(f"❌ Ошибка при запуске обработки писем: {e}")


def show_results():
    print("\n📊 Результаты обработки заявок:")

    results_dir = Path("../temp/results")
    csv_file = results_dir / "applications.csv"

    if csv_file.exists():
        with open(csv_file, "r", encoding="utf-8") as f:
            applications = list(csv.DictReader(f))

        if applications:
            print(f"Найдено заявок: {len(applications)}")
            print("\n" + "=" * 102)
            print(f"{'Email':<50} {'Дата':<35} {'Тема'}")
            print("=" * 102)

            for app in applications:
                print(
                    f"{app['Email'][:48]:<50} {app['Date'][:33]:<35} {app['Subject'][:15]}"
                )
        else:
            print("Заявки не найдены")
    else:
        print("Файл с результатами не найден")


def show_articles():
    print("\n🔍 Загруженные артикулы:")
    try:
        from article_matcher import ArticleMatcher

        matcher = ArticleMatcher()
        if matcher.articles:
            articles_list = sorted(list(matcher.articles))
            print(f"Всего артикулов: {len(articles_list)}")
            print("Список артикулов:")
            for i, article in enumerate(articles_list, 1):
                print(f"  {i:2d}. {article}")
        else:
            print("Артикулы не загружены")
    except Exception as e:
        print(f"Ошибка при загрузке артикулов: {e}")


def clear_logs():
    print("\n🗑️  Очистка логов...")
    log_dir = Path("../log")
    if log_dir.exists():
        try:
            log_count = 0
            for log_file in log_dir.glob("*.log"):
                log_file.unlink()
                print(f"   Удален: {log_file.name}")
                log_count += 1
            print(f"✅ Логи очищены. Удалено файлов: {log_count}")
        except Exception as e:
            print(f"❌ Ошибка при очистке логов: {e}")
    else:
        print("ℹ️ Папка логов не существует")


def clear_data():
    print("\n🗂️  Очистка данных...")
    data_dir = Path("../temp/email")
    if data_dir.exists():
        try:
            json_count = 0
            attachment_count = 0

            for json_file in data_dir.glob("*.json"):
                json_file.unlink()
                print(f"   Удален JSON: {json_file.name}")
                json_count += 1

            attachments_dir = data_dir / "attachments"
            if attachments_dir.exists():
                for attachment_file in attachments_dir.glob("*"):
                    attachment_file.unlink()
                    print(f"   Удален файл: {attachment_file.name}")
                    attachment_count += 1

            print(
                f"✅ Данные очищены. Удалено: {json_count} JSON, {attachment_count} вложений"
            )
        except Exception as e:
            print(f"❌ Ошибка при очистке данных: {e}")
    else:
        print("ℹ️ Папка данных не существует")


def clear_all():
    clear_data()
    clear_logs()


def show_status():
    print("\n📊 Статус системы:")

    log_dir = Path("../log")
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        if log_files:
            print(f"\n📁 Логи ({len(log_files)} файлов):")
            for log_file in log_files:
                size = log_file.stat().st_size
                print(f"   📄 {log_file.name} - {size} байт")
        else:
            print("\n📁 Логи: нет файлов")
    else:
        print("\n📁 Логи: папка не существует")

    data_dir = Path("../temp/email")
    if data_dir.exists():
        json_files = list(data_dir.glob("*.json"))
        attachments_dir = data_dir / "attachments"
        attachment_files = (
            list(attachments_dir.glob("*")) if attachments_dir.exists() else []
        )

        print(f"\n📁 Данные:")
        print(f"   📄 JSON файлов: {len(json_files)}")
        print(f"   📎 Вложений: {len(attachment_files)}")

        if json_files:
            print(f"\n   Последние JSON файлы:")
            for json_file in sorted(json_files, key=os.path.getmtime, reverse=True)[:5]:
                size = json_file.stat().st_size
                print(f"      📄 {json_file.name} - {size} байт")

    else:
        print("\n📁 Данные: папка не существует")


def print_help():
    print("\n" + "=" * 50)
    print("📧 SoftwareCBM - КОМАНДЫ")
    print("=" * 50)
    print("receive-mail      - 📥 Получить новые письма")
    print("receive-mail-force - ⚡ Принудительно получить все письма")
    print("show-results      - 📋 Показать найденные заявки")
    print("show-articles     - 🔍 Показать загруженные артикулы")
    print("cleanup-logs      - 🗑️  Очистить логи")
    print("cleanup-temp      - 🗂️  Очистить данные")
    print("cleanup-all       - ⚠️  Очистить всё")
    print("status            - 📊 Показать статус системы")
    print("help              - 📖 Показать эту справку")
    print("exit              - ❌ Выход")
    print("=" * 50)


def main():
    if not os.path.exists("email_handler.py"):
        print("❌ Файл email_handler.py не найден!")
        print("Убедитесь, что он находится в той же папке, что и этот скрипт")
        input("Нажмите Enter для выхода...")
        return

    print("🚀 Запуск SoftwareCBM...")
    print_help()

    while True:
        try:
            command = input("\nSoftwareCBM > ").strip().lower()

            if command == "receive-mail":
                run_email_handler(force_mode=False)
            elif command == "receive-mail-force":
                run_email_handler(force_mode=True)
            elif command == "show-results":
                show_results()
            elif command == "show-articles":
                show_articles()
            elif command == "cleanup-logs":
                clear_logs()
            elif command == "cleanup-temp":
                clear_data()
            elif command == "cleanup-all":
                clear_all()
            elif command == "status":
                show_status()
            elif command == "help":
                print_help()
            elif command in ["exit", "quit", "q"]:
                print("\n👋 До свидания!")
                break
            elif command == "":
                continue
            else:
                print("❌ Неизвестная команда. Введите 'help' для списка команд.")

        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Произошла ошибка: {e}")


if __name__ == "__main__":
    main()
