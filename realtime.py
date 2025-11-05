import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, date, time
import time as time_sleep
import os
import hashlib
import pytz
from transformers import pipeline
import calendar

# --- App Configuration ---
st.set_page_config(page_title="AI-Powered Timesheet & Payroll Tool", layout="wide")

# --- Timezone Configuration ---
IST = pytz.timezone('Asia/Kolkata')

# --- Database Setup ---
DB_FILE = "company_data.db"
LAST_UPDATE_FILE = "last_update.txt"
ADMIN_PASSWORD = "admin"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_db_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT UNIQUE NOT NULL, name TEXT NOT NULL, password TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS timesheet (
            id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id TEXT NOT NULL,
            project_name TEXT NOT NULL, task_description TEXT NOT NULL,
            hours_worked REAL NOT NULL, submission_date DATE NOT NULL,
            submission_time TIME NOT NULL,
            FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            attendance_date DATE NOT NULL,
            status TEXT NOT NULL, -- Present, Half-day, Absent, Leave
            reason TEXT,
            UNIQUE(employee_id, attendance_date)
        )
    """)
    conn.commit()
    conn.close()

# --- AI Model Loading ---
@st.cache_resource
def get_classification_pipeline():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def suggest_project_name(task_description, project_list):
    if not task_description or not project_list: return None
    classifier = get_classification_pipeline()
    result = classifier(task_description, candidate_labels=project_list)
    return result['labels'][0]

# --- MISSING FUNCTION ADDED BACK HERE ---
def add_employee(employee_id, name, password):
    """Adds a new employee to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO employees (employee_id, name, password) VALUES (?, ?, ?)",
                       (employee_id, name, hash_password(password)))
        conn.commit()
        st.success(f"Employee {name} ({employee_id}) added successfully.")
    except sqlite3.IntegrityError:
        st.error(f"Employee ID {employee_id} already exists.")
    finally:
        conn.close()

# --- Core Logic Functions ---
def log_attendance(employee_id, attendance_date, status, reason=""):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO attendance_log (employee_id, attendance_date, status, reason)
        VALUES (?, ?, ?, ?)
    """, (employee_id, attendance_date, status, reason))
    conn.commit()
    conn.close()

def add_timesheet_entry(employee_id, project_name, task_description, hours_worked, entry_date):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now(IST)
    cursor.execute("INSERT INTO timesheet (employee_id, project_name, task_description, hours_worked, submission_date, submission_time) VALUES (?, ?, ?, ?, ?, ?)",
                   (employee_id, project_name, task_description, hours_worked, entry_date, now.strftime("%H:%M:%S")))
    conn.commit()
    conn.close()
    
    now_time = now.time()
    status = "Present"
    if now_time >= time(13, 0):
        conn = get_db_connection()
        prev_entry = pd.read_sql_query("SELECT * FROM attendance_log WHERE employee_id = ? AND attendance_date = ?", conn, params=(employee_id, str(entry_date)))
        conn.close()
        if prev_entry.empty or prev_entry.iloc[0]['status'] != 'Present':
            status = "Half-day"
    log_attendance(employee_id, entry_date, status, "Work Submitted")
    
    with open(LAST_UPDATE_FILE, "w") as f:
        f.write(str(now.timestamp()))

# --- Data Retrieval for Reports ---
def get_all_employees():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT employee_id, name FROM employees", conn)
    conn.close()
    return df

def generate_monthly_report(year, month):
    employees = get_all_employees()
    if employees.empty:
        return pd.DataFrame(), pd.DataFrame()

    conn = get_db_connection()
    query = "SELECT employee_id, attendance_date, status, reason FROM attendance_log WHERE strftime('%Y', attendance_date) = ? AND strftime('%m', attendance_date) = ?"
    df = pd.read_sql_query(query, conn, params=(str(year), f'{month:02d}'))
    conn.close()

    summary = df.groupby('employee_id')['status'].value_counts().unstack(fill_value=0)
    summary = pd.merge(employees, summary, on='employee_id', how='left').fillna(0).astype({col: int for col in summary.columns if col not in ['employee_id', 'name']})

    _, num_days = calendar.monthrange(year, month)
    working_days = sum(1 for i in range(1, num_days + 1) if date(year, month, i).weekday() < 5)
    
    summary['Total Logged'] = summary.get('Present', 0) + summary.get('Half-day', 0) + summary.get('Leave', 0)
    summary['Absent'] = working_days - summary['Total Logged']
    summary['Absent'] = summary['Absent'].clip(lower=0)
    summary = summary.drop(columns=['Total Logged'], errors='ignore')

    dates = [date(year, month, i) for i in range(1, num_days + 1)]
    detailed_report = pd.DataFrame(index=employees['employee_id'], columns=dates).fillna('Absent')

    for _, row in df.iterrows():
        try:
            attendance_dt = pd.to_datetime(row['attendance_date']).date()
            if row['employee_id'] in detailed_report.index:
                detailed_report.loc[row['employee_id'], attendance_dt] = row['status']
        except Exception:
            pass
    
    detailed_report = pd.merge(employees, detailed_report, on='employee_id', how='left')

    return summary, detailed_report

# --- Streamlit UI Views ---
def login_page():
    st.header("Employee Login")
    with st.form("login_form"):
        employee_id = st.text_input("Employee ID")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if check_employee_credentials(employee_id, password):
                st.session_state["logged_in"] = True
                st.session_state["employee_id"] = employee_id
                st.rerun()
            else:
                st.error("Invalid credentials. Please contact your manager.")

def check_employee_credentials(employee_id, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM employees WHERE employee_id = ?", (employee_id,))
    result = cursor.fetchone()
    conn.close()
    return result and result['password'] == hash_password(password)

def employee_view():
    st.header(f"Employee Portal: {st.session_state['employee_id']}")
    page = st.sidebar.radio("Menu", ["Submit Task", "Mark Leave / Absence"])
    
    if page == "Submit Task":
        st.subheader("Timesheet Entry")
        now_time = datetime.now(IST).time()
        if not (time(8, 30) <= now_time <= time(10, 0) or now_time >= time(13, 0)):
            st.warning("You can only submit tasks between 8:30 AM - 10:00 AM or after 1:00 PM.")
            return

        with st.form("task_form"):
            entry_date = st.date_input("Date", value=datetime.now(IST).date())
            st.session_state.task_description = st.text_area("Task Description", value=st.session_state.get('task_description', ''))
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.session_state.project_name = st.text_input("Project Name", value=st.session_state.get('project_name', ''))
            with col2:
                st.write("")
                st.write("")
                if st.form_submit_button("💡 Suggest Project"):
                    project_list = get_unique_project_names()
                    if project_list and st.session_state.task_description:
                        st.session_state.project_name = suggest_project_name(st.session_state.task_description, project_list)
                        st.rerun()
                    else:
                        st.warning("Please enter a task description first.")
            
            hours_worked = st.number_input("Hours Worked", min_value=0.5, step=0.5)
            
            if st.form_submit_button("Submit Task"):
                if st.session_state.project_name and st.session_state.task_description and hours_worked > 0:
                    add_timesheet_entry(st.session_state['employee_id'], st.session_state.project_name, st.session_state.task_description, hours_worked, entry_date)
                    st.success("Task submitted and attendance logged!")
                    st.session_state.project_name, st.session_state.task_description = "", ""
                else:
                    st.error("Please fill all fields.")

    elif page == "Mark Leave / Absence":
        st.subheader("Submit Leave or Reason for Absence")
        with st.form("leave_form", clear_on_submit=True):
            leave_date = st.date_input("Date", value=datetime.now(IST).date())
            status = st.selectbox("Type of Leave", ["Leave", "Half-day"])
            reason = st.text_area("Reason (e.g., Sick Leave, Personal Emergency)")
            
            if st.form_submit_button("Submit"):
                if reason:
                    log_attendance(st.session_state['employee_id'], leave_date, status, reason)
                    st.success(f"Your status for {leave_date} has been logged as '{status}'.")
                else:
                    st.error("A reason is required.")

def get_unique_project_names():
    conn = get_db_connection()
    try: return pd.read_sql_query("SELECT DISTINCT project_name FROM timesheet", conn)['project_name'].tolist()
    finally: conn.close()

def admin_view():
    page = st.sidebar.selectbox("Admin Menu", ["Monthly Report", "Manage Employees"])

    if page == "Manage Employees":
        st.header("Manage Employees")
        with st.form("add_employee_form", clear_on_submit=True):
            employee_id = st.text_input("Employee ID")
            name = st.text_input("Employee Name")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Add Employee"):
                if employee_id and name and password: add_employee(employee_id, name, password)
                else: st.error("Please provide all details.")
        st.subheader("All Employees")
        st.dataframe(get_all_employees(), use_container_width=True)

    elif page == "Monthly Report":
        st.header("Monthly Attendance Report")
        current_year = datetime.now(IST).year
        col1, col2 = st.columns(2)
        with col1: year = st.selectbox("Select Year", range(current_year - 5, current_year + 1), index=5)
        with col2: month = st.selectbox("Select Month", range(1, 13), index=datetime.now(IST).month - 1)
            
        summary_df, detailed_df = generate_monthly_report(year, month)
        
        if summary_df.empty:
            st.warning("No data found for the selected period or no employees in the system.")
        else:
            st.subheader("Monthly Summary")
            st.dataframe(summary_df.set_index('employee_id'), use_container_width=True)
            
            csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Summary as CSV", csv, f'attendance_summary_{year}_{month:02d}.csv', 'text/csv')

            st.subheader("Day-by-Day Detailed Report")
            st.dataframe(detailed_df.set_index('employee_id'), use_container_width=True)

# --- Main App Logic ---
def main():
    initialize_database()
    st.title("AI-Powered Timesheet & Payroll Tool")

    if "logged_in" not in st.session_state: st.session_state.logged_in = False
    if "admin_logged_in" not in st.session_state: st.session_state.admin_logged_in = False

    if st.session_state.admin_logged_in:
        admin_view()
        if st.sidebar.button("Logout Admin"):
            st.session_state.admin_logged_in = False
            st.rerun()
    elif st.session_state.logged_in:
        employee_view()
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
    else:
        role = st.sidebar.radio("Choose your portal", ["Employee Login", "Admin/Manager"])
        if role == "Employee Login": login_page()
        else:
            password = st.sidebar.text_input("Enter Admin Password", type="password")
            if st.sidebar.button("Access Admin Panel"):
                if password == ADMIN_PASSWORD:
                    st.session_state.admin_logged_in = True
                    st.rerun()
                else:
                    st.sidebar.error("Incorrect password.")

if __name__ == "__main__":
    main()
