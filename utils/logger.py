import sys
import os
from loguru import logger
from typing import Dict, Any

# создаем директорию для логов
os.makedirs("logs", exist_ok=True)

def setup_logger():
    # удаляем существующие хендлеры
    logger.remove()
    
    # консольный вывод
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    
    # файловый вывод
    logger.add(
        "logs/app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="1 month",
    )
    
    return logger

# инициализация логгера
app_logger = setup_logger()

# декоратор для логирования функций
def log_function(func):
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        
        # фильтруем длинные аргументы
        filtered_args = [
            str(arg) if len(str(arg)) < 100 else f"{str(arg)[:97]}..." 
            for arg in args
        ]
        filtered_kwargs = {
            k: str(v) if len(str(v)) < 100 else f"{str(v)[:97]}..." 
            for k, v in kwargs.items()
        }
        
        app_logger.debug(
            f"Вызов '{func_name}' с args: {filtered_args} и kwargs: {filtered_kwargs}"
        )
        
        try:
            result = func(*args, **kwargs)
            
            # не логируем большие значения
            if isinstance(result, (str, dict, list)) and len(str(result)) > 100:
                log_result = f"{str(result)[:97]}..."
            else:
                log_result = result
                
            app_logger.debug(f"Функция '{func_name}' вернула: {log_result}")
            return result
        except Exception as e:
            app_logger.error(f"Ошибка в функции '{func_name}': {str(e)}")
            raise
    
    return wrapper 