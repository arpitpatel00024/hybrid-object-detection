def extract_detections(results):

    detections = []

    for r in results:

        boxes = r.boxes

        for box in boxes:

            detections.append({
                "class": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy.tolist()
            })

    return detections