from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import pytesseract
import easyocr
from ultralytics import YOLO


# ---------- Utility Functions ----------
def extract_birth_date(id_number: str):
    """Extract birth date from Egyptian ID number"""
    if not id_number or len(id_number) < 7:
        return "Invalid ID"
    id_number = ''.join(c for c in id_number if c.isdigit())
    if len(id_number) < 7:
        return "Invalid ID"
    try:
        if id_number[0] == '2':
            year = '19' + id_number[1:3]
        else:
            year = '20' + id_number[1:3]
        month = id_number[3:5]
        day = id_number[5:7]
        return f"{year}/{month}/{day}"
    except:
        return "Invalid ID format"


def get_governorate(id_number: str):
    gov_codes = {
        '01': 'القاهرة', '02': 'الإسكندرية', '03': 'بورسعيد', '04': 'السويس',
        '11': 'دمياط', '12': 'الدقهلية', '13': 'الشرقية', '14': 'القليوبية',
        '15': 'كفر الشيخ', '16': 'الغربية', '17': 'المنوفية', '18': 'البحيرة',
        '19': 'الإسماعيلية', '21': 'الجيزة', '22': 'بني سويف', '23': 'الفيوم',
        '24': 'المنيا', '25': 'أسيوط', '26': 'سوهاج', '27': 'قنا',
        '28': 'أسوان', '29': 'الأقصر', '31': 'البحر الأحمر', '32': 'الوادي الجديد',
        '33': 'مطروح', '34': 'شمال سيناء', '35': 'جنوب سيناء', '88': 'خارج الجمهورية'
    }
    if len(id_number) >= 9:
        gov_code = id_number[7:9]
        return gov_codes.get(gov_code, 'غير محدد')
    return 'غير محدد'


# ---------- Main Reader Class ----------
class NationalIdReader:
    def __init__(self):
        self.card_segmentor = YOLO("card_finder_seg.pt")
        self.card_dividor = YOLO("card_divider_model.pt")
        self.reader = easyocr.Reader(['ar'], gpu=False)

    def cropLocations(self, frame, bbox):
        try:
            x1, y1, w, h = bbox
            return frame[int(y1):int(h), int(x1):int(w)]
        except Exception:
            return None

    def cropImage(self, frame, bbox):
        try:
            x1, y1, w, h = bbox[0][0:4]
            return frame[int(y1):int(h), int(x1):int(w)]
        except Exception:
            return None

    def Get_the_Id_bbox(self, frame):
        card = self.card_segmentor.predict(frame, conf=0.8, verbose=False)
        if card and len(card[0].boxes.data.tolist()) > 0:
            return card[0].boxes.data.tolist()[0:3]
        return None

    def Get_the_Id_content(self, frame):
        name = self.card_dividor.names
        card = self.card_dividor.predict(frame, verbose=False)
        location_of_slots = {}
        if card and len(card[0].boxes.data.tolist()) > 0:
            for i in card[0].boxes.data.tolist():
                location_of_slots[name[int(i[5])]] = i[0:4]
        return location_of_slots if location_of_slots else None

    def safe_ocr(self, frame, **kwargs):
        try:
            result = self.reader.readtext(frame, detail=0, paragraph=True, **kwargs)
            return result[0] if result else None
        except Exception:
            return None

    def ocr_id_number(self, frame):
        """OCR for numeric ID with EasyOCR + Tesseract fallback"""
        if frame is None:
            return None
        # EasyOCR attempt
        id_easy = self.safe_ocr(frame, rotation_info=[i for i in range(0, 270, 10)])
        # Tesseract attempt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        id_tess = pytesseract.image_to_string(
            gray, lang="eng", config="--psm 8 -c tessedit_char_whitelist=0123456789"
        )
        candidates = ["".join(c for c in x if c.isdigit()) for x in [id_easy, id_tess]]
        for c in candidates:
            if len(c) == 14:  # Egyptian ID should be 14 digits
                return c
        return id_easy or id_tess

    def extract_information(self, frame):
        bbox = self.Get_the_Id_bbox(frame)
        if not bbox:
            return {"error": "No ID card detected"}

        croped = self.cropImage(frame, bbox)
        if croped is None:
            return {"error": "Cropping failed"}

        location = self.Get_the_Id_content(croped)
        if not location:
            return {"error": "Could not detect ID fields"}

        slots = {}
        for field in ["firstName", "name", "city", "gov", "idNo"]:
            if field in location:
                slots[field] = self.cropLocations(croped, location[field])

        # OCR text fields
        firstName = self.safe_ocr(slots.get("firstName"))
        name = self.safe_ocr(slots.get("name"))
        city = self.safe_ocr(slots.get("city"))
        gov = self.safe_ocr(slots.get("gov"))
        id_number = self.ocr_id_number(slots.get("idNo"))

        results = {
            "name": f"{firstName} {name}" if firstName and name else (name or firstName),
            "city": city,
            "governorate": gov,
            "national_id": id_number,
        }

        # Extra info from ID number
        if id_number and len(id_number) == 14:
            results["birth_date"] = extract_birth_date(id_number)
            results["gov_from_id"] = get_governorate(id_number)
        return results


# ---------- FastAPI App ----------
app = FastAPI()
ocr_model = NationalIdReader()


@app.post("/extract-id/")
async def extract_id(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        data = ocr_model.extract_information(image)
        if not any(data.values()):
            return {"status": "failed", "data": data}
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "failed", "error": str(e), "data": {}}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
