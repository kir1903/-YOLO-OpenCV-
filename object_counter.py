"""
Real-time Object Detection & Counting
--------------------------------------
Детектирует и трекает объекты на видео (YOLOv8 + OpenCV) и считает,
сколько объектов пересекло заданную линию (например, вход/выход из кадра).

Использование:
    python object_counter.py --source 0                    # веб-камера
    python object_counter.py --source video.mp4             # видеофайл
    python object_counter.py --source video.mp4 --classes person car
    python object_counter.py --source video.mp4 --save out.mp4

Требуется: pip install -r requirements.txt
"""

import argparse
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO object counter")
    parser.add_argument("--source", type=str, default="0",
                         help="Путь к видео, 0 для веб-камеры, или URL")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="Веса модели YOLO (n/s/m/l/x)")
    parser.add_argument("--classes", type=str, nargs="*", default=None,
                         help="Список классов для детекции (напр. person car). "
                              "По умолчанию — все классы COCO")
    parser.add_argument("--conf", type=float, default=0.4,
                         help="Порог уверенности детекции")
    parser.add_argument("--line-y", type=float, default=0.5,
                         help="Позиция счётной линии по вертикали (0.0-1.0 от высоты кадра)")
    parser.add_argument("--save", type=str, default=None,
                         help="Путь для сохранения обработанного видео (напр. out.mp4)")
    parser.add_argument("--show", action="store_true", default=True,
                         help="Показывать окно с видео в реальном времени")
    return parser.parse_args()


class LineCounter:
    """Считает объекты, пересекающие горизонтальную линию, по трек-ID."""

    def __init__(self, line_y: int):
        self.line_y = line_y
        self.track_prev_y = {}       # track_id -> предыдущая координата y центра
        self.counted_ids = set()
        self.count_in = 0            # сверху вниз
        self.count_out = 0           # снизу вверх

    def update(self, track_id: int, center_y: float):
        prev_y = self.track_prev_y.get(track_id)
        self.track_prev_y[track_id] = center_y

        if prev_y is None or track_id in self.counted_ids:
            return

        crossed_down = prev_y < self.line_y <= center_y
        crossed_up = prev_y > self.line_y >= center_y

        if crossed_down:
            self.count_in += 1
            self.counted_ids.add(track_id)
        elif crossed_up:
            self.count_out += 1
            self.counted_ids.add(track_id)


def resolve_source(source: str):
    """Позволяет передавать индекс камеры как строку '0'."""
    return int(source) if source.isdigit() else source


def main():
    args = parse_args()
    source = resolve_source(args.source)

    model = YOLO(args.model)

    # Если заданы конкретные классы — переводим имена в индексы COCO
    class_ids = None
    if args.classes:
        names_to_id = {v: k for k, v in model.names.items()}
        class_ids = [names_to_id[name] for name in args.classes if name in names_to_id]
        missing = set(args.classes) - set(names_to_id.keys())
        if missing:
            print(f"[!] Неизвестные классы, пропущены: {missing}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть источник видео: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    line_y = int(height * args.line_y)
    counter = LineCounter(line_y)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))

    class_counts = defaultdict(int)
    prev_time = time.time()

    print("[*] Запуск. Нажмите 'q' для выхода.")

    # model.track сохраняет ID объектов между кадрами (встроенный ByteTrack)
    for result in model.track(
        source=source,
        conf=args.conf,
        classes=class_ids,
        stream=True,
        persist=True,
        verbose=False,
    ):
        frame = result.orig_img.copy()

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, cls_ids, confs):
                x1, y1, x2, y2 = box.astype(int)
                center_y = (y1 + y2) / 2
                center_x = (x1 + x2) // 2

                counter.update(track_id, center_y)

                label = f"{model.names[cls_id]} #{track_id} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                cv2.circle(frame, (center_x, int(center_y)), 3, (0, 0, 255), -1)

            # Пересчитываем итоговую статистику по классам среди учтённых треков
            class_counts.clear()
            for cls_id in cls_ids:
                class_counts[model.names[cls_id]] += 1

        # Линия подсчёта
        cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)

        # FPS
        now = time.time()
        fps_display = 1 / (now - prev_time) if now != prev_time else 0
        prev_time = now

        # Информационная панель
        cv2.rectangle(frame, (0, 0), (300, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"In: {counter.count_in}  Out: {counter.count_out}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"On screen: {sum(class_counts.values())}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps_display:.1f}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if writer:
            writer.write(frame)

        if args.show:
            cv2.imshow("Object Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n[*] Итог: In={counter.count_in}, Out={counter.count_out}")


if __name__ == "__main__":
    main()
