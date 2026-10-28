from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List
import numpy as np
import shutil
import os
import hashlib
from pymongo import MongoClient

app = FastAPI(title="Monishka Raddayagala's Physics Class LMS", version="5.0.0")

# MongoDB Atlas Connection with connect=False to prevent build-time timeout
MONGO_URL = os.environ.get("MONGO_URL")
if not MONGO_URL:
    raise RuntimeError("MONGO_URL environment variable is not set!")

# Secure Admin Password from Environment Variables (Fallback provided for local testing)
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "monishka2027")

client = MongoClient(MONGO_URL, connect=False)
db = client["mcr_physics_lms"]

# Collections
students_col = db["students"]
attendance_col = db["attendance"]
submissions_col = db["submissions"]
slips_col = db["slips"]
resources_col = db["resources"]
mcq_exams_col = db["mcq_exams"]

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/logo.jpeg")
def get_logo():
    return FileResponse("logo.jpeg")

@app.get("/bg.PNG")
def get_bg():
    return FileResponse("bg.PNG")    

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mcr-physics.onrender.com", "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

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

class AdminLogin(BaseModel):
    password: str

@app.post("/students/register")
def register_student(student: StudentRegister):
    year_short = student.batch_year[-2:] if len(student.batch_year) >= 2 else "27"
    count = students_col.count_documents({}) + 1
    generated_index = f"MR{year_short}-{count:03d}"

    hashed_password = hashlib.sha256(student.password.encode()).hexdigest()

    new_student = {
        "student_index": generated_index,
        "full_name": student.full_name,
        "school": student.school,
        "batch_year": student.batch_year,
        "center": student.center,
        "class_type": student.class_type,
        "phone": student.phone,
        "password": hashed_password
    }
    students_col.insert_one(new_student)
    
    new_student.pop("_id", None)
    new_student.pop("password", None)
    return {"status": "Success", "student": new_student}

@app.post("/students/login")
def login_student(creds: StudentLogin):
    hashed_password = hashlib.sha256(creds.password.encode()).hexdigest()
    student = students_col.find_one({
        "student_index": {"$regex": f"^{creds.student_index.strip()}$", "$options": "i"},
        "password": hashed_password
    })
    if not student:
        raise HTTPException(status_code=401, detail="Index අංකය හෝ Password එක වැරදියි!")
    return build_student_profile(student["student_index"])

@app.get("/students/profile/{student_index}")
def build_student_profile(student_index: str):
    student = students_col.find_one({"student_index": {"$regex": f"^{student_index.strip()}$", "$options": "i"}})
    if not student:
        raise HTTPException(status_code=404, detail="ශිෂ්‍යයා හමු නොවීය.")

    student.pop("_id", None)
    student.pop("password", None)

    sub_cursor = submissions_col.find({"student_index": {"$regex": f"^{student_index.strip()}$", "$options": "i"}})
    student_History = []
    
    for sub in sub_cursor:
        sub.pop("_id", None)
        paper_subs = list(submissions_col.find({"paper_id": sub["paper_id"]}))
        paper_subs.sort(key=lambda x: x["total_marks"], reverse=True)
        
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

    att_records = list(attendance_col.find({"student_index": {"$regex": f"^{student_index.strip()}$", "$options": "i"}}, {"_id": 0}))
    slip_records = list(slips_col.find({"student_index": {"$regex": f"^{student_index.strip()}$", "$options": "i"}}, {"_id": 0}))

    return {
        "profile": student,
        "exam_history": student_History,
        "attendance_count": len(att_records),
        "attendance_list": att_records,
        "fee_paid": len(slip_records) > 0,
        "slips": slip_records
    }

@app.get("/students/all")
def get_all_students():
    students = list(students_col.find({}, {"_id": 0, "password": 0}))
    return students

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
    submissions_col.update_one(
        {"student_index": sub.student_index.upper(), "paper_id": sub.paper_id.upper()},
        {"$set": record},
        upsert=True
    )
    return {"status": "Success", "result": record}

@app.get("/paper/{paper_id}/analytics")
def get_paper_analytics(paper_id: str):
    paper_subs = list(submissions_col.find({"paper_id": {"$regex": f"^{paper_id.strip()}$", "$options": "i"}}, {"_id": 0}))
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
    exams = list(mcq_exams_col.find({}, {"_id": 0}))
    return exams

@app.post("/attendance/mark")
def mark_attendance(record: StudentAttendance):
    attendance_col.insert_one(record.dict())
    return {"status": "Success"}

@app.post("/resources/add")
def add_resource(res: ResourceItem):
    resources_col.insert_one(res.dict())
    return {"status": "Success"}

@app.get("/resources/all")
def get_resources():
    resources = list(resources_col.find({}, {"_id": 0}))
    return resources

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
    slips_col.insert_one(slip_record)
    return {"status": "Success", "record": slip_record}

@app.get("/fees/slips")
def get_all_slips():
    slips = list(slips_col.find({}, {"_id": 0}))
    return slips

@app.post("/admin/verify")
def verify_admin(admin: AdminLogin):
    if admin.password == ADMIN_PASSWORD:
        return {"status": "Authorized"}
    raise HTTPException(status_code=401, detail="වැරදි Admin Password එකක්!")
