import os
import json
import cv2
from ultralytics import YOLO
from typing import Any, Optional, Dict, List

# Cấu hình môi trường để tránh lỗi trùng lặp thư viện
os.environ['KMP_DUPLICATE_LIB_OK'] = 'true'

# Tải mô hình YOLO
model = YOLO('yolov8m.pt')


class JSONSink:
    def __init__(self, filename: str = 'output.json'):
        self.filename: str = filename
        self.file: Optional[open] = None
        self.data: List[Dict[str, Any]] = []

    def __enter__(self) -> 'JSONSink':
        self.open()
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        self.write_and_close()

    def open(self) -> None:
        self.file = open(self.filename, 'w')

    def write_and_close(self) -> None:
        if self.file:
            json.dump(self.data, self.file, indent=4)
            self.file.close()

    def append(self, detection: Dict[str, Any]) -> None:
        self.data.append(detection)


def process_video(video_path, output_path, output_frames_dir):
    # Mở video
    cap = cv2.VideoCapture(video_path)

    # Tạo thư mục để lưu khung hình nếu chưa tồn tại
    os.makedirs(output_frames_dir, exist_ok=True)

    # Lấy thông tin về kích thước và FPS của video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tạo đối tượng VideoWriter để lưu video đầu ra
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0

    # Sử dụng JSONSink để lưu kết quả phát hiện
    with JSONSink('detections.json') as json_sink:
        while cap.isOpened():
            # Đọc frame từ video
            success, frame = cap.read()

            if success:
                # Phát hiện đối tượng trong frame với YOLOv8
                results = model(frame)

                # Xử lý từng kết quả phát hiện
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Lấy thông tin bounding box và lớp
                        class_id = int(box.cls[0].item())  # ID lớp đối tượng
                        confidence = float(box.conf[0].item())  # Mức độ tin cậy
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Tọa độ bounding box

                        # Tính thời gian xuất hiện (giây)
                        time_appear = frame_count / fps  # Thời gian xuất hiện trong giây

                        # Tạo thông tin phát hiện
                        detection_data = {
                            "x_min": float(x1),
                            "y_min": float(y1),
                            "x_max": float(x2),
                            "y_max": float(y2),
                            "class_id": class_id,
                            "confidence": confidence,
                            "tracker_id": 0,  # Hoặc giá trị tracker_id nếu có
                            "class_name": ["person", "bicycle", "car", "motorcycle", "unknown", "bus", "truck"][class_id] if class_id < 7 else "unknown",
                            "frame_number": frame_count,
                            "time_appear": time_appear  # Thêm thông tin time_appear
                        }

                        # Thêm thông tin phát hiện vào JSONSink
                        json_sink.append(detection_data)

                # Vẽ các khung xung quanh đối tượng lên frame
                annotated_frame = results[0].plot()

                # Ghi frame đã được xử lý vào file video đầu ra
                out.write(annotated_frame)

                # Lưu từng khung hình vào thư mục
                frame_filename = os.path.join(output_frames_dir, f'frame_{frame_count:04d}.jpg')
                cv2.imwrite(frame_filename, annotated_frame)  # Lưu khung hình

                # Cập nhật tiến trình
                frame_count += 1
                progress = (frame_count / total_frames) * 100
                print(f"\rProcessing: {progress:.2f}% complete", end="")
            else:
                # Dừng vòng lặp nếu video đã hết
                break

    # Giải phóng bộ nhớ
    cap.release()
    out.release()

    print(f"\nKết quả phát hiện đã được lưu tại detections.json")
    print(f"Các khung hình đã được lưu tại thư mục {output_frames_dir}")


# Gọi hàm process_video
video_path = 'video2.mp4'  
output_path = 'output_video2.mp4'  
output_frames_dir = 'output_frames'  
process_video(video_path, output_path, output_frames_dir)
