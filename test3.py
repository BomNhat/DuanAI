import os
import json
import cv2
from ultralytics import YOLO

# Cấu hình môi trường để tránh lỗi trùng lặp thư viện
os.environ['KMP_DUPLICATE_LIB_OK'] = 'true'

# Tải mô hình YOLO
model = YOLO('yolov8m.pt')

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

    # Danh sách để lưu kết quả phát hiện
    results_list = []

    # Từ điển để theo dõi sự xuất hiện và biến mất của các đối tượng
    tracking_objects = {}

    frame_count = 0

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
                    class_id = box.cls[0].item()  # ID lớp đối tượng
                    confidence = box.conf[0].item()  # Mức độ tin cậy
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Tọa độ bounding box

                    # Thời gian xuất hiện
                    time_appear = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Thời gian (giây)

                    # Tạo tên lớp từ class_id
                    class_name_list = ["person", "bicycle", "car", "motorcycle", "unknown", "bus", "truck"]
                    class_index = int(class_id)  # Chuyển đổi class_id thành số nguyên
                    class_name = class_name_list[class_index] if class_index < len(class_name_list) else "unknown"

                    # Lưu thông tin vào tracking_objects
                    if class_id not in tracking_objects:
                        tracking_objects[class_id] = {
                            "frame_number": frame_count,
                            "class_name": class_name,
                            "time_appear": time_appear,
                            "confidence": float(confidence),
                            "bounding_box": [float(x1), float(y1), float(x2), float(y2)]
                        }
                    else:
                        # Cập nhật thời gian biến mất
                        tracking_objects[class_id]["time_disappear"] = time_appear
                        tracking_objects[class_id]["bounding_box"] = [float(x1), float(y1), float(x2), float(y2)]

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

    # Chuyển đổi tracking_objects thành danh sách kết quả
    for obj in tracking_objects.values():
        results_list.append({
            "time_appear": obj.get("time_appear", None),  # Tránh KeyError
            "time_disappear": obj.get("time_disappear", None),
            "class_name": obj["class_name"],
            "frame_number": obj["frame_number"],
            "bounding_box": obj["bounding_box"],
            "confidence": obj["confidence"]
        })

    # Lưu kết quả vào tệp JSON
    json_output_path = 'detections.json'
    with open(json_output_path, 'w') as json_file:
        json.dump(results_list, json_file, indent=4)

    print(f"\nSave on {output_path}")
    print(f"Save on {json_output_path}")
    print(f"Save on {output_frames_dir}")


video_path = 'video_test_2.mp4'  
output_path = 'output_video2.mp4'  
output_frames_dir = 'output_frames'  
process_video(video_path, output_path, output_frames_dir)
