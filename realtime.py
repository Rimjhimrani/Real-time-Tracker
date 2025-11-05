import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, date
import time as time_sleep
import os
import hashlib
import pytz
from transformers import pipeline
import calendar

# --- App Configuration ---
st.set_page_config(page_title="AI-Powered Timesheet & Payroll Tool", layout="wide")

# --- Timezone Configuration ---
IST = pytz.timezone('Asia/KKolkata')

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
    # --- NEW TABLE FOR ATTENDANCE LOGGING ---
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

# --- Core Logic Functions ---
def log_attendance(employee_id, attendance_date, status, reason=""):
    """Logs or updates the daily attendance status for an employee."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # INSERT OR REPLACE ensures that there is only one entry per employee per day
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
    
    # --- AUTOMATICALLY LOG ATTENDANCE ON TASK SUBMISSION ---
    now_time = now.time()
    status = "Present"
    if now_time >= pd.to_datetime("13:00").time():
        # Check if they were already present in the morning
        conn = get_db_connection()
        prev_entry = pd.read_sql_query("SELECT * FROM attendance_log WHERE employee_id = ? AND attendance_date = ?", conn, params=(employee_id, entry_date))
        conn.close()
        if prev_entry.empty:
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

    # --- SUMMARY CALCULATION ---
    summary = df.groupby('employee_id')['status'].value_counts().unstack(fill_value=0)
    summary = pd.merge(employees, summary, on='employee_id', how='left').fillna(0)

    # Calculate working days (Mon-Fri)
    _, num_days = calendar.monthrange(year, month)
    working_days = sum(1 for i in range(1, num_days + 1) if date(year, month, i).weekday() < 5)
    
    # Calculate Absences
    summary['Total Logged'] = summary.get('Present', 0) + summary.get('Half-day', 0) + summary.get('Leave', 0)
    summary['Absent'] = working_days - summary['Total Logged']
    summary['Absent'] = summary['Absent'].clip(lower=0) # Don't show negative absences

    # --- DETAILED DAY-BY-DAY REPORT ---
    # Create a calendar-like view
    dates = [date(year, month, i) for i in range(1, num_days + 1)]
    detailed_report = pd.DataFrame(index=employees['employee_id'], columns=dates).fillna('Absent')

    for _, row in df.iterrows():
        detailed_report.loc[row['employee_id'], pd.to_datetime(row['attendance_date']).date()] = row['status']
    
    detailed_report = pd.merge(employees, detailed_report, on='employee_id', how='left')

    return summary, detailed_report

# --- Streamlit UI Views ---
def employee_view():
    st.header(f"Employee Portal: {st.session_state['employee_id']}")
    page = st.sidebar.radio("Menu", ["Submit Task", "Mark Leave / Absence"])
    
    if page == "Submit Task":
        st.subheader("Timesheet Entry")
        now_time = datetime.now(IST).time()
        if not (pd.to_datetime("08:30").time() <= now_time <= pd.to_datetime("10:00").time() or now_time >= pd.to_datetime("13:00").time()):
            st.warning("You can only submit tasks between 8:30 AM - 10:00 AM or after 1:00 PM.")
            return

        with st.form("task_form"):
            # Form fields and logic for task submission
            entry_date = st.date_input("Date", value=datetime.now(IST).date())
            st.session_state.task_description = st.text_area("Task Description", value=st.session_state.get('task_description', ''))
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.session_state.project_name = st.text_input("Project Name", value=st.session_state.get('project_name', ''))
            with col2:
                st.write("")
                st.write("")
                if st.form_submit_button("💡 Suggest Project"):
                    # AI Suggestion Logic
                    pass # (Keeping this section brief for clarity)

            hours_worked = st.number_input("Hours Worked", min_value=0.5, step=0.5)
            
            if st.form_submit_button("Submit Task"):
                add_timesheet_entry(st.session_state['employee_id'], st.session_state.project_name, st.session_state.task_description, hours_worked, entry_date)
                st.success("Task submitted and attendance logged!")
                st.session_state.project_name, st.session_state.task_description = "", ""

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

def admin_view():
    page = st.sidebar.selectbox("Admin Menu", ["Dashboard", "Manage Employees", "Monthly Report"])

    if page == "Dashboard":
        # Live dashboard logic here (can be simplified now)
        st.header("Manager Dashboard (Today's Live Status)")
        # ...

    elif page == "Manage Employees":
        st.header("Manage Employees")
        with st.form("add_employee_form", clear_on_submit=True):
            # Form to add employees
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
        with col1:
            year = st.selectbox("Select Year", range(current_year - 5, current_year + 1), index=5)
        with col2:
            month = st.selectbox("Select Month", range(1, 13), index=datetime.now(IST).month - 1)
            
        if st.button("Generate Report"):
            summary_df, detailed_df = generate_monthly_report(year, month)
            
            if summary_df.empty:
                st.warning("No data found for the selected period or no employees in the system.")
            else:
                st.subheader("Monthly Summary")
                st.dataframe(summary_df.set_index('employee_id'), use_container_width=True)
                
                # --- DOWNLOAD BUTTON ---
                csv = summary_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Summary as CSV",
                    data=csv,
                    file_name=f'attendance_summary_{year}_{month:02d}.csv',
                    mime='text/csv',
                )

                st.subheader("Day-by-Day Detailed Report")
                st.dataframe(detailed_df.set_index('employee_id'), use_container_width=True)


# --- Main App Logic ---
def main():
    initialize_database()
    st.title("AI-Powered Timesheet & Payroll Tool")

    # Session state initialization
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
        if role == "Employee Login":
            # Login page logic
            pass
        else: # Admin/Manager
            password = st.sidebar.text_input("Enter Admin Password", type="password")
            if st.sidebar.button("Access Admin Panel"):
                if password == ADMIN_PASSWORD:
                    st.session_state.admin_logged_in = True
                    st.rerun()
                else:
                    st.sidebar.error("Incorrect password.")

if __name__ == "__main__":
    main()
