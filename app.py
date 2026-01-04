from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import imutils
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import lru_cache


# ---------- Optimized Card Scanner ----------
class OptimizedNationalIdReader:
    def __init__(self):
        # Load models with optimized settings
        self.card_segmentor = YOLO("card_finder_seg.pt")
        self.card_dividor = YOLO("card_divider_model.pt")

        # Configure YOLO for speed
        self.card_segmentor.overrides['verbose'] = False
        self.card_dividor.overrides['verbose'] = False

        self.reader = easyocr.Reader(['ar'], gpu=True)

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)

    # ========== Optimized Edge Detection ==========

    def order_points_fast(self, pts):
        """نسخة محسنة من ترتيب النقاط"""
        rect = np.zeros((4, 2), dtype="float32")

        # استخدام العمليات المتجهة بدلاً من الحلقات
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    def four_point_transform_fast(self, image, pts):
        """نسخة محسنة من التحويل المنظوري"""
        rect = self.order_points_fast(pts)
        (tl, tr, br, bl) = rect

        # حساب الأبعاد بطريقة محسنة
        width1 = np.linalg.norm(br - bl)
        width2 = np.linalg.norm(tr - tl)
        maxWidth = max(int(width1), int(width2))

        height1 = np.linalg.norm(tr - br)
        height2 = np.linalg.norm(tl - bl)
        maxHeight = max(int(height1), int(height2))

        # تحديد حد أقصى للأبعاد لتجنب الصور الضخمة
        max_dim = 1000
        if maxWidth > max_dim or maxHeight > max_dim:
            scale = min(max_dim / maxWidth, max_dim / maxHeight)
            maxWidth = int(maxWidth * scale)
            maxHeight = int(maxHeight * scale)

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def detect_card_edges_fast(self, image, min_area_ratio=0.1):
        """نسخة محسنة وأسرع من كشف الحواف"""
        try:
            # تقليل حجم الصورة للمعالجة السريعة
            height, width = image.shape[:2]
            if width > 800:
                scale = 800.0 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                resized = cv2.resize(image, (new_width, new_height))
            else:
                resized = image
                scale = 1.0

            # تحويل للرمادي
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized

            # معالجة أسرع
            blurred = cv2.medianBlur(gray, 5)  # أسرع من GaussianBlur

            # كشف الحواف بمعاملات محسنة
            edged = cv2.Canny(blurred, 75, 200)

            # عمليات مورفولوجية مبسطة
            kernel = np.ones((3, 3), np.uint8)
            edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

            # العثور على المحيطات
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # ترتيب حسب المساحة (أخذ أكبر 5 فقط للسرعة)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            image_area = resized.shape[0] * resized.shape[1]

            for contour in contours:
                # تبسيط المحيط
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                if len(approx) == 4:
                    area = cv2.contourArea(approx)
                    area_ratio = area / image_area

                    if area_ratio > min_area_ratio:
                        # إعادة تحويل النقاط للحجم الأصلي
                        if scale != 1.0:
                            approx = approx / scale
                        return approx.reshape(4, 2).astype(np.float32)

            return None

        except Exception:
            return None

    def crop_card_with_edges_fast(self, image):
        """نسخة أسرع من قطع البطاقة"""
        try:
            card_corners = self.detect_card_edges_fast(image)

            if card_corners is not None:
                cropped_card = self.four_point_transform_fast(image, card_corners)
                return cropped_card, "Edge_Detection"

            return None, "Failed"

        except Exception:
            return None, "Error"

    # ========== Optimized Original Methods ==========

    @lru_cache(maxsize=32)
    def get_resize_dimensions(self, field):
        """كاش لأبعاد تغيير الحجم"""
        dimensions = {
            "name": (358, 53),
            "firstName": (106, 49),
            "city": (104, 47),
            "gov": (196, 90),
            "idNo": (451, 53)
        }
        return dimensions.get(field, (100, 50))

    def cropLocations(self, frame, bbox):
        try:
            x1, y1, w, h = [float(coord) for coord in bbox]
            x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            return frame[y1:h, x1:w] if frame is not None else None
        except Exception:
            return None

    def cropImage(self, frame, bbox):
        try:
            x1, y1, w, h = [float(coord) for coord in bbox[0][0:4]]
            x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            return frame[y1:h, x1:w] if frame is not None else None
        except Exception:
            return None

    def Get_the_Id_bbox_fast(self, frame):
        try:
            # تقليل دقة الصورة للكشف السريع
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640.0 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                resized = cv2.resize(frame, (new_width, new_height))

                card = self.card_segmentor.predict(resized, conf=0.7, verbose=False, imgsz=640)

                if card and len(card[0].boxes.data.tolist()) > 0:
                    # إعادة تحويل الإحداثيات للحجم الأصلي
                    bbox = card[0].boxes.data.tolist()[0]
                    # تحويل جميع القيم إلى float ثم int
                    bbox[0:4] = [float(coord) / scale for coord in bbox[0:4]]
                    return [bbox]
            else:
                card = self.card_segmentor.predict(frame, conf=0.7, verbose=False, imgsz=640)
                if card and len(card[0].boxes.data.tolist()) > 0:
                    # تحويل جميع القيم إلى Python native types
                    bbox_list = card[0].boxes.data.tolist()[0:3]
                    # تأكد من أن جميع القيم هي Python native types
                    clean_bbox_list = []
                    for bbox in bbox_list:
                        clean_bbox = [float(x) for x in bbox]
                        clean_bbox_list.append(clean_bbox)
                    return clean_bbox_list
        except Exception:
            pass
        return None

    def Get_the_Id_content_fast(self, frame):
        try:
            # استخدام دقة أقل للكشف السريع
            card = self.card_dividor.predict(frame, verbose=False, imgsz=640, conf=0.5)

            if card and len(card[0].boxes.data.tolist()) > 0:
                name = self.card_dividor.names
                location_of_slots = {}

                for detection in card[0].boxes.data.tolist():
                    class_id = int(detection[5])
                    if class_id in name:
                        # تحويل جميع الإحداثيات إلى Python native types
                        bbox = [float(x) for x in detection[0:4]]
                        location_of_slots[name[class_id]] = bbox

                return location_of_slots if location_of_slots else None
        except Exception:
            pass
        return None

    def FilterStack_fast(self, frame):
        """نسخة محسنة من الفلترة"""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # استخدام فلتر أسرع
        frame = cv2.medianBlur(frame, 3)
        frame = cv2.adaptiveThreshold(frame, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 7, 2)
        return frame

    def FilterStack2Id_fast(self, frame):
        """نسخة محسنة من فلتر الرقم القومي"""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = cv2.equalizeHist(frame)
        frame = cv2.inRange(frame, 0, 25)  # توسيع النطاق قليلاً
        return frame

    async def safe_ocr_async(self, frame, **kwargs):
        """OCR غير متزامن"""
        if frame is None:
            return None

        try:
            # تشغيل OCR في thread منفصل
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.reader.readtext(frame, detail=0, paragraph=True, **kwargs)
            )
            # تأكد من إرجاع string عادي
            return str(result[0]) if result and len(result) > 0 else ""
        except Exception:
            return ""

    def detect_and_crop_card_ultra_fast(self, image):
        """كشف وقطع البطاقة بأقصى سرعة"""
        # الطريقة 1: YOLO السريع
        try:
            bbox = self.Get_the_Id_bbox_fast(image)
            if bbox:
                cropped_card = self.cropImage(image, bbox)
                if cropped_card is not None:
                    return cropped_card, "YOLO"
        except Exception:
            pass

        # الطريقة 2: كشف الحواف السريع (فقط إذا فشل YOLO)
        try:
            cropped_card, method = self.crop_card_with_edges_fast(image)
            if cropped_card is not None:
                return cropped_card, method
        except Exception:
            pass

        # الطريقة 3: الصورة الأصلية
        return image, "Original"

    async def extract_information_fast(self, frame):
        """استخراج المعلومات بأقصى سرعة"""
        try:
            # كشف البطاقة السريع
            cropped_card, detection_method = self.detect_and_crop_card_ultra_fast(frame)

            if cropped_card is None:
                return {"error": "No card detected", "method": detection_method}

            # كشف محتويات البطاقة
            location = self.Get_the_Id_content_fast(cropped_card)
            if not location:
                return {"error": "No fields detected", "method": detection_method}

            # معالجة متوازية للحقول
            slots = {}
            for field in ["firstName", "name", "city", "gov", "idNo"]:
                if field in location:
                    cropped_slot = self.cropLocations(cropped_card, location[field])
                    if cropped_slot is not None:
                        # تغيير الحجم والفلترة
                        w, h = self.get_resize_dimensions(field)
                        resized = cv2.resize(cropped_slot, (w, h))

                        if field == "idNo":
                            slots[field] = self.FilterStack2Id_fast(resized)
                        else:
                            slots[field] = self.FilterStack_fast(resized)

            # OCR متوازي للحقول المهمة فقط
            ocr_tasks = []

            # اختيار الحقول الأساسية فقط
            essential_fields = ["name", "firstName", "idNo"]
            for field in essential_fields:
                if field in slots and slots[field] is not None:
                    if field == "idNo":
                        task = self.safe_ocr_async(slots[field],
                                                   rotation_info=[0, 90, 180, 270],  # تقليل الزوايا
                                                   text_threshold=0.2,
                                                   low_text=0.2)
                    else:
                        task = self.safe_ocr_async(slots[field])
                    ocr_tasks.append((field, task))

            # تنفيذ OCR بشكل متوازي
            ocr_results = {}
            if ocr_tasks:
                results = await asyncio.gather(*[task for _, task in ocr_tasks], return_exceptions=True)
                for (field, _), result in zip(ocr_tasks, results):
                    if not isinstance(result, Exception):
                        ocr_results[field] = result

            # تجميع النتائج
            name = str(ocr_results.get("name", ""))
            firstName = str(ocr_results.get("firstName", ""))
            id_number = str(ocr_results.get("idNo", ""))

            # معالجة الحقول الثانوية بشكل مشروط
            city = ""
            gov = ""
            if "city" in slots and slots["city"] is not None:
                city_result = await self.safe_ocr_async(slots["city"])
                city = str(city_result) if city_result else ""
            if "gov" in slots and slots["gov"] is not None:
                gov_result = await self.safe_ocr_async(slots["gov"])
                gov = str(gov_result) if gov_result else ""

            # تنظيف وتجميع الاسم الكامل
            full_name = f"{firstName} {name}".strip()
            if not full_name:
                full_name = name or firstName or ""

            return {
                "name": str(full_name),
                "city": str(city),
                "governorate": str(gov),
                "national_id": str(id_number),
                "detection_method": str(detection_method)
            }

        except Exception as e:
            return {"error": f"Processing failed: {str(e)}", "method": "Error"}


# ---------- Optimized FastAPI Application ----------
app = FastAPI(title="Ultra-Fast ID Scanner API", version="3.0.0")
ocr_model = OptimizedNationalIdReader()


@app.post("/extract-id/")
async def extract_id_fast(file: UploadFile = File(...)):
    """
    Extract ID information with maximum speed optimization
    """
    try:
        # قراءة الملف
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # معالجة سريعة
        data = await ocr_model.extract_information_fast(image)

        if "error" in data:
            return {"status": "failed", "reason": data["error"], "method": data.get("method", "Unknown")}

        return {"status": "success", "data": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Ultra-Fast Egyptian ID Scanner", "version": "3.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)