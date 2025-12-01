import csv
import subprocess
import sys
import time
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_email_handler_force():
    print(
        f"\n🔄 [{datetime.now().strftime('%H:%M:%S')}] Запуск обработки новых писем..."
    )
    try:
        email_handler_path = PROJECT_ROOT / "src" / "tools" / "email_handler.py"

        result = subprocess.run(
            [sys.executable, str(email_handler_path), "--force"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] Обработка писем завершена")

        for line in result.stdout.split("\n"):
            if "РЕЗУЛЬТАТ:" in line:
                print(f"📊 {line.replace('РЕЗУЛЬТАТ:', '').strip()}")
            elif "ОСТАЛОСЬ:" in line:
                print(f"📭 {line.replace('ОСТАЛОСЬ:', '').strip()}")

        if result.stderr:
            error_lines = [
                line for line in result.stderr.split("\n") if "ERROR" in line
            ]
            if error_lines:
                print("❌ Ошибки при обработке:")
                for error_line in error_lines:
                    clean_line = (
                        error_line.split("- ERROR - ")[-1]
                        if "- ERROR - " in error_line
                        else error_line
                    )
                    print(f"   {clean_line}")

        return True

    except Exception as e:
        print(f"❌ Ошибка при запуске обработки писем: {e}")
        return False


def show_apps(days=180):
    week_range = 7

    csv_file = PROJECT_ROOT / "temp" / "results" / "applications.csv"
    if not csv_file.exists():
        print("❌ Файл с заявками не найден")
        return

    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            applications = list(csv.DictReader(f))

        if not applications:
            print("ℹ️ Заявки не найдены")
            return

        now = datetime.now().replace(tzinfo=None)

        start_date = now - timedelta(days=days + week_range)
        end_date = now - timedelta(days=days)

        print(
            f"📆 Период: с {start_date.strftime('%d.%m.%Y')} по {end_date.strftime('%d.%m.%Y')}"
        )

        email_stats = {}

        for app in applications:
            try:
                email_date = parsedate_to_datetime(app["Date"]).replace(tzinfo=None)

                if start_date <= email_date <= end_date:
                    email = app["Email"]
                    days_ago = (now - email_date).days

                    if email not in email_stats:
                        email_stats[email] = {
                            "min_days": days_ago,
                            "max_days": days_ago,
                            "count": 0,
                        }

                    email_stats[email]["count"] += 1
                    email_stats[email]["min_days"] = min(
                        email_stats[email]["min_days"], days_ago
                    )
                    email_stats[email]["max_days"] = max(
                        email_stats[email]["max_days"], days_ago
                    )

            except Exception as e:
                print(f"⚠️ Ошибка обработки записи: {e}")
                continue

        if not email_stats:
            print(f"ℹ️ В указанном периоде заявок не найдено")
            return

        print(f"📧 Найдено уникальных отправителей: {len(email_stats)}")
        print("=" * 60)

        for email, stats in sorted(email_stats.items()):
            if stats["min_days"] == stats["max_days"]:
                days_text = f"{stats['min_days']} дней"
            else:
                days_text = f"{stats['min_days']}-{stats['max_days']} дней"

            count_text = f" ({stats['count']} сообщ.)" if stats["count"] > 1 else ""

            print(f"📨 {email}, было прислано {days_text} назад{count_text}")

        print("=" * 60)
        total_messages = sum(stats["count"] for stats in email_stats.values())
        print(f"📊 Всего сообщений в периоде: {total_messages}")

    except Exception as e:
        print(f"❌ Ошибка при чтении файла заявок: {e}")


def auto_monitor():
    print("🚀 Запуск автоматического мониторинга заявок")
    print("=" * 50)

    while True:
        try:
            interval_input = input("Введите интервал проверки в минутах (по умолчанию 120): ").strip()
            if not interval_input:
                interval_minutes = 120
                break
            interval_minutes = int(interval_input)
            if interval_minutes <= 0:
                print("❌ Интервал должен быть положительным числом")
                continue
            break
        except ValueError:
            print("❌ Пожалуйста, введите целое число")

    while True:
        try:
            days_input = input(
                "Введите количество дней для анализа заявок (по умолчанию 180): "
            ).strip()
            if not days_input:
                days = 180
                break
            days = int(days_input)
            if days <= 0:
                print("❌ Количество дней должно быть положительным числом")
                continue
            break
        except ValueError:
            print("❌ Пожалуйста, введите целое число")

    print(f"\n⚙️  Настройки мониторинга:")
    print(f"   • Интервал проверки: {interval_minutes} минут")
    print(f"   • Анализ заявок за: {days} дней")
    print(f"   • Следующая проверка через: {interval_minutes} мин")
    print("=" * 50)

    cycle_count = 0

    try:
        while True:
            cycle_count += 1
            print(
                f"\n🔄 ЦИКЛ #{cycle_count} - {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}"
            )
            print("-" * 50)

            success = run_email_handler_force()

            if success:
                show_apps(days)

            if cycle_count > 1:
                print(f"\n⏰ Ожидание следующей проверки...")

            print(f"\n⏳ Следующая проверка через {interval_minutes} минут...")
            for remaining in range(interval_minutes * 60, 0, -1):
                mins, secs = divmod(remaining, 60)
                time_str = f"{mins:02d}:{secs:02d}"
                print(f"\r🕒 Осталось: {time_str}", end="", flush=True)
                time.sleep(1)
            print("\r" + " " * 50 + "\r", end="", flush=True)

    except KeyboardInterrupt:
        print(f"\n\n⏹️  Автоматический мониторинг остановлен")
        print(f"📊 Всего выполнено циклов: {cycle_count}")
        print("👋 До свидания!")


if __name__ == "__main__":
    email_handler_path = PROJECT_ROOT / "src" / "tools" / "email_handler.py"
    if not email_handler_path.exists():
        print("❌ Файл email_handler.py не найден!")
        print("Убедитесь, что он находится в правильной папке")
        input("Нажмите Enter для выхода...")
        sys.exit(1)

    auto_monitor()
