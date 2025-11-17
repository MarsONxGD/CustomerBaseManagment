# SoftwareCBM

Программа для автоматического распознания заявок клиентов для дальнейшей работы с ними

## Настройка почтового клиента

Для настройки почты создайте файл `<ProgramPath>/config/email_config.py` со следующим содержимым:

```python
# email_config.py
def credentials():
    config = {
        'email': 'example@email.com',
        'password': '0123456789abcdefghij',
        'imap_server': 'imap.email.com',
        'imap_port': 993
    }
    return config
```

### Описание параметров

- `email` - адрес электронной почты
- `password` - пароль от почтового ящика или специальный пароль приложения
- `imap_server` - адрес IMAP сервера для получения почты
- `imap_port` - порт IMAP сервера (обычно 993 для SSL)

_P.S. Замените значения параметров на актуальные для вашего почтового сервиса._
