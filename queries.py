# queries.py

MOMENTUM_QUERY = """
WITH daily_counts AS (
    SELECT 
        s.skills,
        DATE(j.job_posted_date) as job_date,
        COUNT(j.job_id) as daily_jobs
    FROM job_postings_fact j
    JOIN skills_job_dim sj ON j.job_id = sj.job_id
    JOIN skills_dim s ON sj.skill_id = s.skill_id
    WHERE j.job_posted_date >= date('now', :days_filter)
    GROUP BY s.skills, job_date
),
moving_avg AS (
    SELECT
        skills,
        job_date,
        daily_jobs,
        AVG(daily_jobs) OVER (
            PARTITION BY skills 
            ORDER BY job_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as moving_avg_7d
    FROM daily_counts
)
SELECT * FROM moving_avg
ORDER BY job_date DESC
"""

VOLATILITY_QUERY = """
SELECT 
    s.skills,
    COUNT(j.job_id) as demand,
    PRINTF('$%,d', CAST(AVG(j.salary_year_avg) AS INT)) as avg_salary,
    AVG(j.salary_year_avg) as raw_avg_salary
FROM job_postings_fact j
JOIN skills_job_dim sj ON j.job_id = sj.job_id
JOIN skills_dim s ON sj.skill_id = s.skill_id
WHERE j.salary_year_avg IS NOT NULL
GROUP BY s.skills
ORDER BY demand DESC
LIMIT 10
"""

NETWORK_QUERY = """
SELECT 
    a.skills as source,
    b.skills as target,
    COUNT(*) as weight
FROM skills_job_dim sj1
JOIN skills_job_dim sj2 ON sj1.job_id = sj2.job_id
JOIN skills_dim a ON sj1.skill_id = a.skill_id
JOIN skills_dim b ON sj2.skill_id = b.skill_id
WHERE a.skills < b.skills -- Avoid duplicates (A-B vs B-A) and self-loops
GROUP BY a.skills, b.skills
ORDER BY weight DESC
LIMIT 50;
"""

HISTORY_QUERY = """
SELECT 
    j.job_posted_date,
    j.salary_year_avg
FROM job_postings_fact j
JOIN skills_job_dim sj ON j.job_id = sj.job_id
JOIN skills_dim s ON sj.skill_id = s.skill_id
WHERE s.skills = :skill
  AND j.salary_year_avg IS NOT NULL
ORDER BY j.job_posted_date ASC
"""

SKILL_VALUE_QUERY = """
SELECT s.skills, CAST(AVG(j.salary_year_avg) as INT) as avg_salary
FROM skills_dim s
JOIN skills_job_dim sj ON s.skill_id = sj.skill_id
JOIN job_postings_fact j ON sj.job_id = j.job_id
WHERE j.salary_year_avg IS NOT NULL
GROUP BY s.skills
"""

SIMULATOR_QUERY_ALL_SKILLS = """
SELECT skill_id, skills FROM skills_dim
"""

SIMULATOR_QUERY_AVG_ROI = """
-- Calculate avg salary for a specific set of skills
-- This is complex, so we might simplify logic in Python for the sake of the portfolio demo
-- Instead, we'll get the 'Premium' of adding a skill:
-- Avg Salary of jobs that have (Base Skills + New Skill)
SELECT AVG(j.salary_year_avg) as projected_salary
FROM job_postings_fact j
JOIN skills_job_dim sj ON j.job_id = sj.job_id
WHERE j.salary_year_avg IS NOT NULL
AND sj.skill_id IN :skill_ids
"""
