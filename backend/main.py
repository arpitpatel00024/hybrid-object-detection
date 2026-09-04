import os
import base64
import random
import time

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from yolo_module.yolo_model import YOLODetector
from yolo_module.detection import extract_detections
from fuzzy_system.inference import FuzzyPriorityEvaluator
from genetic_algorithm.ga_optimizer import optimize


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Hybrid Object Detection API",
    description="YOLO + Fuzzy Logic + Genetic Algorithm",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

frontend_url = os.getenv(
    "FRONTEND_URL",
    "http://localhost:3000"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        frontend_url,
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# COCO Class Names
# ============================================================

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


# ============================================================
# Load Models Once
# ============================================================

detector = YOLODetector()

fuzzy_evaluator = FuzzyPriorityEvaluator()


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "message": "Hybrid Object Detection API",
        "pipeline": "YOLO + Fuzzy Logic + Genetic Algorithm"
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# Detection API
# ============================================================

@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    request_start = time.perf_counter()

    # --------------------------------------------------------
    # Check uploaded file
    # --------------------------------------------------------

    if not file.content_type or not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    try:

        # ----------------------------------------------------
        # Read Image
        # ----------------------------------------------------

        contents = await file.read()

        image_array = np.frombuffer(
            contents,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            raise HTTPException(
                status_code=400,
                detail="Could not read the uploaded image."
            )

        # ----------------------------------------------------
        # YOLO Detection
        # ----------------------------------------------------

        yolo_start = time.perf_counter()

        results = detector.predict(image)

        detections = extract_detections(results)

        yolo_time = time.perf_counter() - yolo_start

        print(f"YOLO time: {yolo_time:.2f} seconds")

        # ----------------------------------------------------
        # Fuzzy Logic
        # ----------------------------------------------------

        fuzzy_filtered = []

        image_height, image_width = image.shape[:2]

        image_area = image_width * image_height

        for det in detections:

            # Get bounding box
            x1, y1, x2, y2 = map(
                float,
                det["bbox"][0]
            )

            # Calculate object size
            box_width = max(
                0,
                x2 - x1
            )

            box_height = max(
                0,
                y2 - y1
            )

            box_area = (
                box_width *
                box_height
            )

            # Calculate size ratio
            if image_area > 0:

                size_ratio = (
                    box_area /
                    image_area
                )

            else:

                size_ratio = 0.0

            # Keep value between 0 and 1
            size_ratio = max(
                0.0,
                min(1.0, size_ratio)
            )

            # Fuzzy evaluation
            fuzzy_score = fuzzy_evaluator.evaluate(
                float(det["confidence"]),
                float(size_ratio)
            )

            det["fuzzy"] = float(
                fuzzy_score
            )

            det["size_ratio"] = float(
                size_ratio
            )

            # Keep detections with positive priority
            if fuzzy_score > 0:

                fuzzy_filtered.append(det)

        fuzzy_time = time.perf_counter() - yolo_start - yolo_time
        print(f"Fuzzy/processing time: {fuzzy_time:.2f} seconds")

        # ----------------------------------------------------
        # Genetic Algorithm
        # ----------------------------------------------------

        ga_start = time.perf_counter()

        if fuzzy_filtered:

            best_threshold, history = optimize(
                fuzzy_filtered
            )

            best_threshold = float(
                best_threshold
            )

        else:

            best_threshold = 0.5

            history = []

        ga_time = time.perf_counter() - ga_start
        print(f"GA time: {ga_time:.2f} seconds")

        # ----------------------------------------------------
        # Final Hybrid Filtering
        # ----------------------------------------------------

        final_results = []

        for det in fuzzy_filtered:

            if float(det["confidence"]) > best_threshold:

                final_results.append(det)

        # ----------------------------------------------------
        # Draw Bounding Boxes
        # ----------------------------------------------------

        output_image = image.copy()

        for det in final_results:

            x1, y1, x2, y2 = map(
                int,
                det["bbox"][0]
            )

            cls_id = int(
                det["class"]
            )

            confidence = float(
                det["confidence"]
            )

            fuzzy_score = float(
                det["fuzzy"]
            )

            # Class name
            if 0 <= cls_id < len(CLASS_NAMES):

                label = CLASS_NAMES[cls_id]

            else:

                label = f"class_{cls_id}"

            # Random box color
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )

            # Label
            text = (
                f"{label} "
                f"{confidence:.2f} "
                f"F:{fuzzy_score:.1f}"
            )

            # Rectangle
            cv2.rectangle(
                output_image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Text
            cv2.putText(
                output_image,
                text,
                (
                    x1,
                    max(y1 - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # ----------------------------------------------------
        # Convert Image to Base64
        # ----------------------------------------------------

        encoding_start = time.perf_counter()

        success, encoded_image = cv2.imencode(
            ".jpg",
            output_image
        )

        if not success:

            raise HTTPException(
                status_code=500,
                detail="Could not encode output image."
            )

        output_base64 = base64.b64encode(
            encoded_image.tobytes()
        ).decode("utf-8")

        encoding_time = time.perf_counter() - encoding_start
        print(f"Encoding time: {encoding_time:.2f} seconds")

        # ----------------------------------------------------
        # Prepare Detection Results
        # ----------------------------------------------------

        result_objects = []

        for det in final_results:

            cls_id = int(
                det["class"]
            )

            if 0 <= cls_id < len(CLASS_NAMES):

                label = CLASS_NAMES[cls_id]

            else:

                label = f"class_{cls_id}"

            result_objects.append({

                "class": cls_id,

                "label": label,

                "confidence": round(
                    float(det["confidence"]),
                    4
                ),

                "fuzzy_score": round(
                    float(det["fuzzy"]),
                    4
                ),

                "size_ratio": round(
                    float(det["size_ratio"]),
                    4
                ),

                "bbox": det["bbox"][0]
            })

        # ----------------------------------------------------
        # Return Response
        # ----------------------------------------------------

        total_time = time.perf_counter() - request_start
        print(f"TOTAL REQUEST TIME: {total_time:.2f} seconds")

        return {

            "success": True,

            "metrics": {

                "yolo": len(detections),

                "fuzzy": len(
                    fuzzy_filtered
                ),

                "hybrid": len(
                    final_results
                )
            },

            "best_threshold": round(
                best_threshold,
                4
            ),

            "ga_history": [

                round(
                    float(x),
                    4
                )

                for x in history
            ],

            "detections": result_objects,

            "image": (
                "data:image/jpeg;base64,"
                + output_base64
            )
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )