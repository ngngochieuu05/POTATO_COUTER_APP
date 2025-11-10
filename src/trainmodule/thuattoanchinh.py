# ===================== THUẬT TOÁN CHÍNH =====================

# 1️⃣ Convolutional Neural Network (CNN)
# Mô phỏng hoạt động chính của CNN: trích xuất đặc trưng từ ảnh
def convolution_operation(input_image, kernel):
    """
    Thuật toán CNN (Convolution): trích xuất đặc trưng ảnh
    Công thức: y = f(W * x + b)
    """
    return cv2.filter2D(input_image, -1, kernel)

# 2️⃣ Non-Max Suppression (NMS)
def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Thuật toán NMS: loại bỏ bounding box trùng nhau
    """
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        others = indices[1:]

        ious = []
        for o in others:
            x1 = max(boxes[current][0], boxes[o][0])
            y1 = max(boxes[current][1], boxes[o][1])
            x2 = min(boxes[current][2], boxes[o][2])
            y2 = min(boxes[current][3], boxes[o][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (boxes[current][2] - boxes[current][0]) * (boxes[current][3] - boxes[current][1])
            area2 = (boxes[o][2] - boxes[o][0]) * (boxes[o][3] - boxes[o][1])
            union = area1 + area2 - inter
            iou = inter / union if union > 0 else 0
            ious.append(iou)

        indices = [i for i, iou in zip(others, ious) if iou < iou_threshold]
    return keep

# 3️⃣ Confidence Thresholding
def confidence_filter(predictions, threshold=0.5):
    """
    Thuật toán Confidence Thresholding:
    Lọc các dự đoán có độ tin cậy >= ngưỡng người dùng đặt
    """
    filtered = [p for p in predictions if p['confidence'] >= threshold]
    return filtered

# 4️⃣ Counting Algorithm
def count_objects(predictions, threshold=0.5):
    """
    Thuật toán đếm: N_potato = Σ [Conf_i >= T_conf]
    """
    count = sum(1 for p in predictions if p['confidence'] >= threshold)
    return count
