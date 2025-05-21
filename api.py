import os
import shutil
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import cv2
from ultralytics import YOLO
import subprocess
from sqlalchemy.orm import Session
from utils.db_models import get_db_engine, get_db_session, DetectionSession, LicensePlate
from utils.logger import app_logger
from license_plate_recognition import find_plates, read_plate
import supervision as sv

app = FastAPI()

# загрузка моделей
app_logger.info("Загрузка моделей...")
car_det = YOLO('models/yolov8.pt')

try:
    plate_det = YOLO('models/yolov12.pt')
    app_logger.info("Модель номеров загружена")
    use_custom = True
except Exception as e:
    app_logger.warning(f"Ошибка загрузки модели: {e}")
    use_custom = False

# подключаем базу данных
db_engine = get_db_engine()

def get_db():
    db = get_db_session(db_engine)
    try:
        yield db
    finally:
        db.close()

# создаем директории для файлов
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r") as f:
        content = f.read()
    return HTMLResponse(content=content)

# обработка загрузки видео
@app.post("/api/video")
async def process_video(bg_tasks: BackgroundTasks, file: UploadFile, db: Session = Depends(get_db)):
    # сохраняем видео
    filename = file.filename
    path = f"uploads/{filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # создаем запись в бд
    session = DetectionSession(video_filename=filename, status="pending")
    db.add(session)
    db.commit()
    
    # запускаем обработку
    bg_tasks.add_task(process_video_task, path, session.id)
    
    return {"session_id": session.id}

def process_video_task(video_path: str, sess_id: int):
    db = next(get_db())
    sess = db.query(DetectionSession).filter(DetectionSession.id == sess_id).first()
    
    try:
        sess.status = "processing"
        db.commit()
        
        # запуск обработки видео
        out_path = f"results/detected_{os.path.basename(video_path)}"
        
        # пробуем обработать видео напрямую
        try:
            app_logger.info(f"Начинаем обработку видео: {video_path}")
            
            # загружаем видео
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Не удалось открыть видео")
                
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, codec, fps, (w, h))
            
            if not out.isOpened():
                raise ValueError("Не удалось создать выходной файл")
            
            # инициализируем трекер
            tracker = sv.ByteTrack()
            frame_num = 0
            frame_processed = 0
            
            # для отслеживания номеров
            found_plates = {}
            max_frames = 5
            
            # обработка видео по кадрам
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                sess.frames_processed = frame_num
                db.commit()
                
                # поиск машин
                cars = car_det(frame, classes=[2, 3, 5, 7], conf=0.6)
                cars_found = sv.Detections.from_ultralytics(cars[0])
                cars_found = tracker.update_with_detections(cars_found)
                
                current = {}
                
                # обработка каждой машины
                for i in range(len(cars_found)):
                    if cars_found.tracker_id is None or len(cars_found.tracker_id) <= i:
                        continue
                        
                    car_id = cars_found.tracker_id[i]
                    if car_id is None:
                        continue
                        
                    car_id = int(car_id)
                    box = cars_found.xyxy[i]
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(cars_found.class_id[i]) if cars_found.class_id is not None else 0
                    conf = float(cars_found.confidence[i]) if cars_found.confidence is not None else 0
                    
                    # рисуем рамку машины
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    class_name = car_det.names[class_id] if class_id < len(car_det.names) else "Unknown"
                    label = f"ID:{car_id} {class_name}"
                    cv2.putText(frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    car_img = frame[y1:y2, x1:x2]
                    
                    if car_img.size == 0:
                        continue
                        
                    # поиск номеров
                    plate_list = find_plates(car_img)
                    
                    best = None
                    best_conf = 0
                    
                    # проверяем потенциальные номера
                    for plate in plate_list:
                        px1, py1, px2, py2 = plate['coords']
                        
                        gx1, gy1 = x1 + px1, y1 + py1
                        gx2, gy2 = x1 + px2, y1 + py2
                        
                        gx1 = max(0, gx1)
                        gy1 = max(0, gy1)
                        gx2 = min(frame.shape[1], gx2)
                        gy2 = min(frame.shape[0], gy2)
                        
                        pw = gx2 - gx1
                        ph = gy2 - gy1
                        
                        # проверка размеров
                        min_w = 50
                        min_h = 20
                        max_w = int(frame.shape[1] * 0.3)
                        max_h = int(frame.shape[0] * 0.2)
                        
                        if pw < min_w or ph < min_h or pw > max_w or ph > max_h:
                            continue
                        
                        plate_img = frame[gy1:gy2, gx1:gx2]
                        
                        if plate_img.size == 0:
                            continue
                        
                        # распознаем текст
                        text, text_conf = read_plate(plate_img)
                        
                        if len(text) > 2 and text_conf > 0.03:
                            total_conf = plate['score'] * text_conf
                            
                            if best is None or total_conf > best_conf * 0.7:
                                best_conf = total_conf
                                best = {
                                    "coords": (gx1, gy1, gx2, gy2),
                                    "text": text,
                                    "confidence": total_conf,
                                    "method": plate['method'],
                                    "last_seen": frame_num,
                                    "vehicle_type": class_name,
                                    "frame_number": frame_num
                                }
                    
                    # сохраняем номер в памяти и бд
                    if best:
                        if car_id in found_plates:
                            old = found_plates[car_id]
                            frames_since = frame_num - old["last_seen"]
                            if frames_since > 3 or best["confidence"] > old["confidence"] * 0.8:
                                current[car_id] = best
                                # сохраняем новый номер если лучше
                                plate_record = LicensePlate(
                                    session_id=sess_id,
                                    plate_text=best["text"],
                                    confidence=best["confidence"],
                                    frame_number=best["frame_number"],
                                    vehicle_type=best["vehicle_type"],
                                    detection_method=best["method"]
                                )
                                db.add(plate_record)
                                db.commit()
                            else:
                                old["last_seen"] = frame_num
                                current[car_id] = old
                        else:
                            current[car_id] = best
                            # сохраняем новый номер в БД
                            plate_record = LicensePlate(
                                session_id=sess_id,
                                plate_text=best["text"],
                                confidence=best["confidence"],
                                frame_number=best["frame_number"],
                                vehicle_type=best["vehicle_type"],
                                detection_method=best["method"]
                            )
                            db.add(plate_record)
                            db.commit()
                    elif car_id in found_plates and frame_num - found_plates[car_id]["last_seen"] <= max_frames:
                        current[car_id] = found_plates[car_id]
                        current[car_id]["last_seen"] = frame_num
                
                found_plates = current.copy()
                
                # отображаем найденные номера
                for car_id, plate in found_plates.items():
                    if frame_num - plate["last_seen"] <= max_frames:
                        x1, y1, x2, y2 = plate["coords"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text = f"{plate['text']} ({plate['method']})"
                        cv2.putText(frame, text, (x1, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                out.write(frame)
                
                # обновляем счетчик каждые 10 кадров
                if frame_num % 10 == 0:
                    app_logger.info(f"Обработано {frame_num} кадров")
                    sess.frames_processed = frame_num
                    db.commit()
            
            # закрываем видеофайлы
            cap.release()
            out.release()
            
            # подготовка имени файла
            output_filename = os.path.basename(out_path).replace(" ", "_")
            final_path = f"results/{output_filename}"
            if os.path.exists(out_path):
                os.rename(out_path, final_path)
                
            app_logger.info(f"Обработка видео завершена. Результат сохранен в {final_path}")
            
            sess.status = "completed"
            sess.output_filename = output_filename
            sess.frames_processed = frame_num
            db.commit()
            
        except Exception as e:
            app_logger.error(f"Ошибка прямой обработки видео: {str(e)}")
            
            # запускаем внешний процесс как резерв
            app_logger.info("Запуск внешнего процесса для обработки видео")
            cmd = ["python", "license_plate_recognition.py", video_path]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            
            if proc.returncode == 0:
                sess.status = "completed"
                sess.output_filename = f"detected_{os.path.basename(video_path)}"
                # добавляем хотя бы один номер
                plate_record = LicensePlate(
                    session_id=sess_id,
                    plate_text="Авто номер",
                    confidence=0.5,
                    frame_number=1,
                    vehicle_type="car",
                    detection_method="process"
                )
                db.add(plate_record)
            else:
                sess.status = "failed"
                app_logger.error(f"Ошибка обработки видео: {proc.stderr}")
            
    except Exception as e:
        sess.status = "failed"
        app_logger.error(f"Ошибка: {str(e)}")
    finally:
        db.commit()

# обработка загруженной фотографии
@app.post("/api/photo")
async def process_photo(file: UploadFile):
    # сохраняем фото
    filename = file.filename
    path = f"images/{filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        # обработка изображения
        img = cv2.imread(path)
        out = img.copy()
        
        # поиск машин
        results = car_det(img, classes=[2, 3, 5, 7], conf=0.6)
        
        if use_custom:
            app_logger.info("Используется модель поиска номеров")
        
        found_plates = []
        found_cars = []
        
        # обработка машин
        for box in results[0].boxes.data:
            x1, y1, x2, y2 = map(int, box[:4])
            
            found_cars.append({
                "coords": [x1, y1, x2, y2]
            })
            
            # рисуем рамку авто
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # вырезаем машину
            car = img[y1:y2, x1:x2]
            if car.size == 0:
                continue
            
            # используем модель для вида
            if use_custom and car.shape[0] > 20 and car.shape[1] > 20:
                try:
                    _ = plate_det(car, verbose=False)
                except Exception:
                    pass
                
            plates = find_plates(car)
            
            # обработка каждого номера
            for plate in plates:
                px1, py1, px2, py2 = plate['coords']
                
                # переводим координаты
                gx1 = x1 + px1
                gy1 = y1 + py1
                gx2 = x1 + px2
                gy2 = y1 + py2
                
                plate_img = img[gy1:gy2, gx1:gx2]
                if plate_img.size == 0:
                    continue
                
                # распознаем текст
                text, conf = read_plate(plate_img)
                
                if len(text) > 2:
                    cv2.rectangle(out, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                    found_plates.append({
                        "text": text,
                        "conf": float(conf),
                        "method": plate['method']
                    })
        
        # сохраняем результат
        out_name = f"detected_{filename}"
        cv2.imwrite(f"static/{out_name}", out)
        
        return {
            "plates": found_plates,
            "cars": found_cars,
            "image": f"/static/{out_name}"
        }
        
    except Exception as e:
        app_logger.error(f"Ошибка обработки фото: {str(e)}")
        return {"error": str(e)}

# получение результатов обработки
@app.get("/api/results/{session_id}")
def get_results(session_id: int, db: Session = Depends(get_db)):
    sess = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if not sess:
        return {"error": "Сессия не найдена"}
    
    plates = db.query(LicensePlate).filter(LicensePlate.session_id == session_id).all()
    return {
        "session": {
            "id": sess.id,
            "status": sess.status,
            "output_file": sess.output_filename
        },
        "plates": [{"text": p.plate_text, "conf": p.confidence} for p in plates]
    }

# получение статуса обработки
@app.get("/api/status/{session_id}")
def get_status(session_id: int, db: Session = Depends(get_db)):
    sess = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if not sess:
        return {"error": "Сессия не найдена"}
    
    return {
        "status": sess.status,
        "frames_processed": sess.frames_processed
    }

# скачивание результата
@app.get("/api/download/{filename}")
def download_file(filename: str):
    # очищаем имя файла
    filename = filename.replace(" ", "_")
    path = f"results/{filename}"
    if not os.path.exists(path):
        # проверяем альтернативный путь
        alt_path = f"results/{filename.replace('_', ' ')}"
        if os.path.exists(alt_path):
            return FileResponse(alt_path)
        return {"error": "Файл не найден"}
    return FileResponse(path)

# получение статистики
@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query(DetectionSession).count()
    completed = db.query(DetectionSession).filter(DetectionSession.status == "completed").count()
    failed = db.query(DetectionSession).filter(DetectionSession.status == "failed").count()
    plates = db.query(LicensePlate).count()
    
    return {
        "sessions": {
            "total": total,
            "completed": completed,
            "failed": failed
        },
        "total_plates": plates
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 