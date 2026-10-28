from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List
import numpy as np
import shutil
import os
import json

app = FastAPI(title="Monishka Raddayagala's Physics LMS", version="4.0.0")
@app.get("/")
def home():
    return FileResponse("index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

DB_FILE = "lms_data.json"

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "students": [],
        "attendance": [],
        "submissions": [],
        "slips": [],
        "resources": [
            {"title": "Model Paper 01 - Mechanics", "category": "Paper", "link": "https://example.com/paper1.pdf", "description": "යාන්ත්‍ර විද්‍යාව පුනරීක්ෂණ ප්‍රශ්න පත්‍රය"},
            {"title": "Day 01 - Rotational Motion Recording", "category": "Video", "link": "https://youtube.com", "description": "භ්‍රමණ චලිතය සජීවී පන්ති වීඩියෝව"}
        ],
        "mcq_exams": [
            {
                "exam_id": "EXAM-01",
                "title": "Mechanics Live MCQ Test",
                "questions": [
                    {"q": "ප්‍රවේශයේ ממය (Dimension) කුමක්ද?", "options": ["LT^-1", "MLT^-2", "ML^2T", "L^2T^-1"], "answer": 0},
                    {"q": "ප්‍රවේග කාල ප්‍රස්ථාරයක යටින් ඇති වර්ගඵලයෙන් දැක්වෙන්නේ?", "options": ["ත්වරණය", "විස්ථාපනය", "බලය", "ශක්තිය"], "answer": 1}
                ]
            }
        ]
    }

def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

db = load_db()

class StudentRegister(BaseModel):
    full_name: str
    school: str
    batch_year: str
    center: str
    class_type: str
    phone: str
    password: str

class StudentLogin(BaseModel):
    student_index: str
    password: str

class StudentAttendance(BaseModel):
    student_index: str
    student_name: str
    class_date: str
    status: str

class FullPaperSubmission(BaseModel):
    student_index: str
    student_name: str
    paper_id: str
    mcq_marks: float
    essay_marks: float
    unit_breakdown: Dict[str, float]

class ResourceItem(BaseModel):
    title: str
    category: str
    link: str
    description: str

@app.post("/students/register")
def register_student(student: StudentRegister):
    year_short = student.batch_year[-2:] if len(student.batch_year) >= 2 else "27"
    count = len(db["students"]) + 1
    generated_index = f"MR{year_short}-{count:03d}"

    new_student = {
        "student_index": generated_index,
        **student.dict()
    }
    db["students"].append(new_student)
    save_db(db)
    return {"status": "Success", "student": new_student}

@app.post("/students/login")
def login_student(creds: StudentLogin):
    student = next((s for s in db["students"] if s["student_index"].upper() == creds.student_index.upper() and s["password"] == creds.password), None)
    if not student:
        raise HTTPException(status_code=401, detail="Index අංකය හෝ Password එක වැරදියි!")
    return build_student_profile(student["student_index"])

@app.get("/students/profile/{student_index}")
def build_student_profile(student_index: str):
    student = next((s for s in db["students"] if s["student_index"].upper() == student_index.upper()), None)
    if not student:
        raise HTTPException(status_code=404, detail="ශිෂ්‍යයා හමු නොවීය.")

    student_History = []
    for sub in db["submissions"]:
        if sub["student_index"].upper() == student_index.upper():
            paper_subs = sorted([s for s in db["submissions"] if s["paper_id"] == sub["paper_id"]], key=lambda x: x["total_marks"], reverse=True)
            rank = next((i + 1 for i, s in enumerate(paper_subs) if s["student_index"].upper() == student_index.upper()), 1)
            totals = np.array([s["total_marks"] for s in paper_subs])
            mean = float(np.mean(totals))
            sd = float(np.std(totals))
            z_score = round((sub["total_marks"] - mean) / sd, 4) if sd > 0 else 0.0

            student_History.append({
                **sub,
                "rank": rank,
                "z_score": z_score,
                "class_average": round(mean, 2)
            })

    att_records = [a for a in db["attendance"] if a["student_index"].upper() == student_index.upper()]
    slip_records = [sl for sl in db["slips"] if sl["student_index"].upper() == student_index.upper()]

    return {
        "profile": {k: v for k, v in student.items() if k != "password"},
        "exam_history": student_History,
        "attendance_count": len(att_records),
        "attendance_list": att_records,
        "fee_paid": len(slip_records) > 0,
        "slips": slip_records
    }

@app.get("/students/all")
def get_all_students():
    return [{k: v for k, v in s.items() if k != "password"} for s in db["students"]]

@app.post("/paper/submit")
def submit_full_paper(sub: FullPaperSubmission):
    total_marks = round(sub.mcq_marks + sub.essay_marks, 2)
    record = {
        "student_index": sub.student_index.upper(),
        "student_name": sub.student_name,
        "paper_id": sub.paper_id.upper(),
        "mcq_marks": sub.mcq_marks,
        "essay_marks": sub.essay_marks,
        "total_marks": total_marks,
        "unit_breakdown": sub.unit_breakdown
    }
    existing = next((s for s in db["submissions"] if s["student_index"].upper() == sub.student_index.upper() and s["paper_id"].upper() == sub.paper_id.upper()), None)
    if existing:
        existing.update(record)
    else:
        db["submissions"].append(record)
    save_db(db)
    return {"status": "Success", "result": record}

@app.get("/paper/{paper_id}/analytics")
def get_paper_analytics(paper_id: str):
    paper_subs = [s for s in db["submissions"] if s["paper_id"].upper() == paper_id.upper()]
    if not paper_subs:
        raise HTTPException(status_code=404, detail="No submissions found.")
    totals = np.array([s["total_marks"] for s in paper_subs])
    mean_score = float(np.mean(totals))
    std_dev = float(np.std(totals))
    sorted_subs = sorted(paper_subs, key=lambda x: x["total_marks"], reverse=True)
    
    leaderboard = []
    top_10 = []
    for rank, sub in enumerate(sorted_subs, start=1):
        z_score = round((sub["total_marks"] - mean_score) / std_dev, 4) if std_dev > 0 else 0.0
        item = {"rank": rank, **sub, "z_score": z_score}
        leaderboard.append(item)
        if rank <= 10:
            top_10.append(item)

    return {
        "paper_id": paper_id,
        "class_average": round(mean_score, 2),
        "standard_deviation": round(std_dev, 2),
        "leaderboard": leaderboard,
        "top_10": top_10
    }

@app.get("/exams/all")
def get_exams():
    return db["mcq_exams"]

@app.post("/attendance/mark")
def mark_attendance(record: StudentAttendance):
    db["attendance"].append(record.dict())
    save_db(db)
    return {"status": "Success"}

@app.post("/resources/add")
def add_resource(res: ResourceItem):
    db["resources"].append(res.dict())
    save_db(db)
    return {"status": "Success"}

@app.get("/resources/all")
def get_resources():
    return db["resources"]

@app.post("/fees/upload-slip")
def upload_payment_slip(
    student_index: str = Form(...),
    student_name: str = Form(...),
    month: str = Form(...),
    file: UploadFile = File(...)
):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    slip_record = {
        "student_index": student_index.upper(),
        "student_name": student_name,
        "month": month,
        "file_url": f"/uploads/{file.filename}"
    }
    db["slips"].append(slip_record)
    save_db(db)
    return {"status": "Success", "record": slip_record}

@app.get("/fees/slips")
def get_all_slips():
    return db["slips"]
