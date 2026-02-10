-- Create core tables for the Career Intelligence Engine

-- 1. Job Postings Fact Table
CREATE TABLE IF NOT EXISTS job_postings_fact (
    job_id INT PRIMARY KEY,
    job_title_short TEXT,
    job_location TEXT,
    salary_year_avg NUMERIC,
    job_work_from_home BOOLEAN,
    job_posted_date TIMESTAMP
);

-- 2. Skills Dimension Table
CREATE TABLE IF NOT EXISTS skills_dim (
    skill_id INT PRIMARY KEY,
    skills TEXT
);

-- 3. Skills-Job Junction Table (Many-to-Many Relationship)
CREATE TABLE IF NOT EXISTS skills_job_dim (
    job_id INT REFERENCES job_postings_fact(job_id),
    skill_id INT REFERENCES skills_dim(skill_id),
    PRIMARY KEY (job_id, skill_id)
);

-- Indexes for performance on commonly queried columns
CREATE INDEX IF NOT EXISTS idx_job_postings_fact_job_work_from_home ON job_postings_fact(job_work_from_home);
CREATE INDEX IF NOT EXISTS idx_job_postings_fact_salary_year_avg ON job_postings_fact(salary_year_avg);
