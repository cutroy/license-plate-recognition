#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import requests
import json
from typing import Optional, List, Dict, Any
import time

def process_video_local(video_path: str) -> None:
    if not os.path.exists(video_path):
        print(f"Ошибка: Файл {video_path} не найден")
        sys.exit(1)
        
    if not video_path.lower().endswith('.mp4'):
        print("Ошибка: Поддерживаются только MP4 файлы")
        sys.exit(1)
        
    try:
        # запуск обработки видео локально
        print(f"Обработка видео: {video_path}")
        cmd = ["python", "license_plate_recognition.py", video_path]
        subprocess.run(cmd, check=True)
        print("Обработка успешно завершена")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка обработки видео: {e}")
        sys.exit(1)

def process_image_local(img_path: str, save_output: bool = True) -> None:
    if not os.path.exists(img_path):
        print(f"Ошибка: Файл {img_path} не найден")
        sys.exit(1)
        
    if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        print("Ошибка: Поддерживаются только JPG/PNG файлы")
        sys.exit(1)
        
    try:
        import cv2
        from license_plate_recognition import find_plates, read_plate
        
        # чтение и обработка изображения
        print(f"Обработка изображения: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print("Ошибка: Не удалось прочитать изображение")
            sys.exit(1)
            
        plates = find_plates(img)
        out = img.copy()
        
        if not plates:
            print("Номера не обнаружены")
            return
            
        print(f"Найдено {len(plates)} возможных номеров")
        
        found = []
        
        # проверяем каждую область с номером
        for i, plate in enumerate(plates):
            x1, y1, x2, y2 = plate['coords']
            plate_img = img[y1:y2, x1:x2]
            
            if plate_img.size == 0:
                continue
                
            # распознаем текст номера
            text, conf = read_plate(plate_img)
            
            if len(text) > 2 and conf > 0.03:
                total = plate['score'] * conf
                print(f"Номер {i+1}: {text} (уверенность: {total:.2f}, метод: {plate['method']})")
                
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(out, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                found.append({
                    "text": text,
                    "confidence": total,
                    "method": plate['method'],
                    "coords": [x1, y1, x2, y2]
                })
        
        # выводим лучший результат
        if found:
            best = max(found, key=lambda x: x['confidence'])
            print(f"\nЛучший результат: {best['text']} (уверенность: {best['confidence']:.2f}, метод: {best['method']})")
            
            if save_output:
                name = os.path.basename(img_path)
                out_path = f"detected_{name}"
                cv2.imwrite(out_path, out)
                print(f"\nСохранено изображение с рамками: {out_path}")
        else:
            print("Читаемые номера не найдены")
            
    except Exception as e:
        print(f"Ошибка обработки изображения: {e}")
        sys.exit(1)

def start_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    try:
        # запуск сервера на указанном хосте и порту
        print(f"Запуск сервера на {host}:{port}")
        cmd = ["uvicorn", "api:app", f"--host={host}", f"--port={port}"]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка запуска сервера: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Сервер остановлен")

def call_api(endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None, files: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_url = "http://localhost:8000"
    url = f"{base_url}/{endpoint.lstrip('/')}"
    
    try:
        # выполняем запрос к API
        if method.upper() == "GET":
            response = requests.get(url)
        elif method.upper() == "POST":
            response = requests.post(url, data=data, files=files)
        elif method.upper() == "DELETE":
            response = requests.delete(url)
        else:
            print(f"Неподдерживаемый метод: {method}")
            sys.exit(1)
            
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка API: {e}")
        print("Запущен ли сервер?")
        sys.exit(1)

def get_session_results(session_id: int) -> None:
    try:
        # получаем результаты обработки по ID сессии
        data = call_api(f"api/results/{session_id}")
        
        print(f"Идентификатор сессии: {session_id}")
        print(f"Статус: {data['session']['status']}")
        if data['session']['output_file']:
            print(f"Выходной файл: {data['session']['output_file']}")
            
        if data['plates']:
            print("\nОбнаруженные номера:")
            for plate in data['plates']:
                print(f"- {plate['text']} (уверенность: {plate['conf']:.2f})")
        else:
            print("\nНомера не обнаружены")
    except Exception as e:
        print(f"Ошибка получения результатов: {e}")

def get_status(session_id: int) -> None:
    try:
        # получаем статус обработки
        data = call_api(f"api/status/{session_id}")
        
        print(f"Идентификатор сессии: {session_id}")
        print(f"Статус: {data['status']}")
        print(f"Обработано кадров: {data['frames_processed']}")
    except Exception as e:
        print(f"Ошибка получения статуса: {e}")

def upload_video(video_path: str) -> None:
    if not os.path.exists(video_path):
        print(f"Ошибка: Файл {video_path} не найден")
        sys.exit(1)
        
    if not video_path.lower().endswith('.mp4'):
        print("Ошибка: Поддерживаются только MP4 файлы")
        sys.exit(1)
        
    try:
        # загружаем видео через API
        with open(video_path, "rb") as f:
            files = {"file": (os.path.basename(video_path), f)}
            data = call_api("api/video", method="POST", files=files)
            
            session_id = data["session_id"]
            print(f"Загрузка успешна. ID сессии: {session_id}")
            
            # отслеживаем прогресс обработки
            print("Обработка видео... Это может занять время.")
            while True:
                status_data = call_api(f"api/status/{session_id}")
                status = status_data["status"]
                
                if status in ['completed', 'failed']:
                    print(f"Обработка {status}")
                    break
                    
                print(".", end="", flush=True)
                time.sleep(2)
            
            if status == 'completed':
                get_session_results(session_id)
    except Exception as e:
        print(f"Ошибка загрузки видео: {e}")

def upload_photo(img_path: str) -> None:
    if not os.path.exists(img_path):
        print(f"Ошибка: Файл {img_path} не найден")
        sys.exit(1)
        
    if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        print("Ошибка: Поддерживаются только JPG/PNG файлы")
        sys.exit(1)
        
    try:
        # загружаем фото через API
        with open(img_path, "rb") as f:
            files = {"file": (os.path.basename(img_path), f)}
            data = call_api("api/photo", method="POST", files=files)
            
            if 'error' in data:
                print(f"Ошибка: {data['error']}")
                return
            
            # выводим результаты
            if data['plates']:
                print("Обнаруженные номера:")
                for plate in data['plates']:
                    print(f"- {plate['text']} (уверенность: {plate['conf']:.2f}, метод: {plate['method']})")
            else:
                print("Номера не обнаружены")
                
            print(f"URL изображения: http://localhost:8000{data['image']}")
    except Exception as e:
        print(f"Ошибка загрузки фото: {e}")

def show_stats() -> None:
    try:
        # получаем статистику работы системы
        stats = call_api("api/stats")
        
        print("Статистика системы:")
        print(f"Всего сессий: {stats['sessions']['total']}")
        print(f"Успешных сессий: {stats['sessions']['completed']}")
        print(f"Ошибочных сессий: {stats['sessions']['failed']}")
        print(f"Всего распознано номеров: {stats['total_plates']}")
    except Exception as e:
        print(f"Ошибка получения статистики: {e}")

def main():
    # настройка аргументов командной строки
    parser = argparse.ArgumentParser(description='Система распознавания автомобильных номеров')
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
    
    # обработка видео локально
    video_parser = subparsers.add_parser('video', help='Обработать видео локально')
    video_parser.add_argument('path', help='Путь к видео файлу')
    
    # обработка фото локально
    img_parser = subparsers.add_parser('photo', help='Обработать фото локально')
    img_parser.add_argument('path', help='Путь к файлу изображения')
    
    # запуск сервера
    server_parser = subparsers.add_parser('server', help='Запустить API сервер')
    server_parser.add_argument('--host', default='0.0.0.0', help='Хост (по умолчанию: 0.0.0.0)')
    server_parser.add_argument('--port', type=int, default=8000, help='Порт (по умолчанию: 8000)')
    
    # api загрузка видео
    api_video_parser = subparsers.add_parser('api-video', help='Загрузить видео через API')
    api_video_parser.add_argument('path', help='Путь к видео файлу')
    
    # api загрузка фото
    api_img_parser = subparsers.add_parser('api-photo', help='Загрузить фото через API')
    api_img_parser.add_argument('path', help='Путь к файлу изображения')
    
    # статистика
    subparsers.add_parser('stats', help='Показать статистику системы')
    
    # результаты сессии
    results_parser = subparsers.add_parser('results', help='Получить результаты сессии')
    results_parser.add_argument('session_id', type=int, help='Идентификатор сессии')
    
    # статус сессии
    status_parser = subparsers.add_parser('status', help='Проверить статус сессии')
    status_parser.add_argument('session_id', type=int, help='Идентификатор сессии')
    
    # обработка аргументов
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    # выполнение соответствующей команды
    if args.command == 'video':
        process_video_local(args.path)
    elif args.command == 'photo':
        process_image_local(args.path)
    elif args.command == 'server':
        start_server(args.host, args.port)
    elif args.command == 'api-video':
        upload_video(args.path)
    elif args.command == 'api-photo':
        upload_photo(args.path)
    elif args.command == 'stats':
        show_stats()
    elif args.command == 'results':
        get_session_results(args.session_id)
    elif args.command == 'status':
        get_status(args.session_id)

if __name__ == '__main__':
    main() 