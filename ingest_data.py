import pandas as pd
import sqlite3
import os
import random
import numpy as np
from datetime import datetime, timedelta

DB_NAME = "career_intelligence.db"

def create_database():
    """
    Creates the SQLite database and tables if they don't exist.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    print(f"Connected to database: {DB_NAME}")

    # 1. Job Postings Fact Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS job_postings_fact (
        job_id INTEGER PRIMARY KEY,
        job_title_short TEXT,
        job_location TEXT,
        salary_year_avg REAL,
        job_work_from_home BOOLEAN,
        job_posted_date TIMESTAMP
    );
    """)

    # 2. Skills Dimension Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS skills_dim (
        skill_id INTEGER PRIMARY KEY,
        skills TEXT,
        category TEXT
    );
    """)

    # 3. Skills-Job Junction Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS skills_job_dim (
        job_id INTEGER,
        skill_id INTEGER,
        PRIMARY KEY (job_id, skill_id),
        FOREIGN KEY (job_id) REFERENCES job_postings_fact(job_id),
        FOREIGN KEY (skill_id) REFERENCES skills_dim(skill_id)
    );
    """)
    
    conn.commit()
    print("Database tables created/verified.")
    return conn

def generate_complex_trends():
    """
    Generates high-fidelity time-series data (6 months history).
    Simulates:
    - AI Boom (exponential growth)
    - Cloud Stability (linear growth)
    - Legacy Decline (slow decrease)
    """
    print("Generating High-Fidelity Time-Series Data (6 Months)...")
    
    # Time Horizon
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    days = (end_date - start_date).days
    
    # Skill Categories with Trend Profiles
    # (Skill Name, Base Salary, Volatility, Trend_Type)
    skill_profiles = [
        ('Python', 115000, 0.1, 'linear_growth'),
        ('SQL', 105000, 0.05, 'stable'),
        ('TensorFlow', 145000, 0.2, 'exponential'),
        ('React', 110000, 0.1, 'stable'),
        ('Node.js', 112000, 0.12, 'stable'),
        ('AWS', 130000, 0.08, 'linear_growth'),
        ('Azure', 128000, 0.08, 'linear_growth'),
        ('Kubernetes', 140000, 0.15, 'linear_growth'),
        ('Tableau', 95000, 0.05, 'stable'),
        ('Excel', 75000, 0.02, 'stable'),
        ('Java', 118000, 0.05, 'stable'),
        ('C++', 125000, 0.1, 'stable'),
        ('Rust', 150000, 0.25, 'volatile_growth'),
        ('Go', 142000, 0.15, 'linear_growth'),
        ('Snowflake', 135000, 0.1, 'linear_growth'),
        ('Databricks', 140000, 0.15, 'exponential'),
        ('LangChain', 155000, 0.3, 'exponential'),  # Hot skill
        ('OpenAI API', 160000, 0.3, 'exponential'), # Hot skill
        ('Power BI', 92000, 0.05, 'stable'),
        ('Linux', 105000, 0.05, 'stable')
    ]
    
    skills_data = [{'skill_id': i, 'skills': s[0], 'category': 'Tech'} for i, s in enumerate(skill_profiles)]
    
    # Generate Jobs day by day
    jobs_data = []
    job_skills_data = []
    job_id_counter = 1
    
    titles = ['Data Scientist', 'Data Engineer', 'Software Engineer', 'ML Engineer', 'Data Analyst', 'DevOps Engineer']
    locations = ['New York, NY', 'Remote', 'San Francisco, CA', 'Austin, TX', 'Seattle, WA', 'Chicago, IL', 'Boston, MA']

    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        # Weekday variability (more jobs on weekdays)
        is_weekend = current_date.weekday() >= 5
        base_jobs_per_day = 10 if is_weekend else 40
        
        # Add a global trend (overall market growing slightly)
        daily_volume = int(base_jobs_per_day * (1 + (day_offset / 365))) 
        
        for _ in range(daily_volume):
            job_id = job_id_counter
            job_id_counter += 1
            
            # Select primary skill based on its trend probability
            # Simplified: Use random choice weighted by "hotness" at this point in time
            chosen_profile_idx = random.randint(0, len(skill_profiles)-1)
            primary_skill = skill_profiles[chosen_profile_idx]
            skill_name, base_salary, volatility, trend = primary_skill
            
            # Adjust salary based on time and trend
            salary_noise = np.random.normal(0, base_salary * volatility)
            
            if trend == 'exponential':
                # Price surges in last 60 days
                surge_factor = 1.0 + (max(0, day_offset - 120) / 60) * 0.4 
                final_salary = (base_salary * surge_factor) + salary_noise
            elif trend == 'volatile_growth':
                 final_salary = (base_salary * (1 + (day_offset/days)*0.2)) + salary_noise + (random.choice([-1,1]) * base_salary * 0.1)
            else: # stable / linear
                 final_salary = (base_salary * (1 + (day_offset/days)*0.05)) + salary_noise
            
            # 10% null salaries
            if random.random() < 0.1:
                final_salary = None
            
            jobs_data.append({
                'job_id': job_id,
                'job_title_short': random.choice(titles),
                'job_location': random.choice(locations),
                'salary_year_avg': final_salary,
                'job_work_from_home': random.choice([True, False]),
                'job_posted_date': current_date
            })
            
            # Link Primary Skill
            job_skills_data.append({'job_id': job_id, 'skill_id': chosen_profile_idx})
            
            # Link Secondary Skills (Co-occurrence)
            # e.g., Python often goes with SQL or Pandas (mapped crudely here for randomness)
            num_secondary = random.randint(0, 3)
            secondary_indices = random.sample(range(len(skill_profiles)), num_secondary)
            for idx in secondary_indices:
                if idx != chosen_profile_idx:
                    job_skills_data.append({'job_id': job_id, 'skill_id': idx})

    return pd.DataFrame(jobs_data), pd.DataFrame(skills_data), pd.DataFrame(job_skills_data)

def sentinel_check(df_jobs):
    print("Running Sentinel Pro Data Integrity Check...")
    null_ratio = df_jobs['salary_year_avg'].isnull().mean()
    print(f"Sentinel: Null Salary Ratio = {null_ratio:.2%}")
    if null_ratio > 0.5:
        raise ValueError("Alert: Data Integrity Breach (Null Salaries > 50%)")
    return True

def ingest_data():
    conn = create_database()
    
    # Generate Data
    df_jobs, df_skills, df_job_skills = generate_complex_trends()
    
    try:
        sentinel_check(df_jobs)
        
        print(f"Ingesting {len(df_jobs)} jobs into SQLite...")
        
        df_jobs.to_sql('job_postings_fact', conn, if_exists='replace', index=False)
        df_skills.to_sql('skills_dim', conn, if_exists='replace', index=False)
        df_job_skills.to_sql('skills_job_dim', conn, if_exists='replace', index=False)
        
        print("Success: Sentinel Pro Ingestion Complete.")
        
    except Exception as e:
        print(f"Ingestion Failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    ingest_data()
