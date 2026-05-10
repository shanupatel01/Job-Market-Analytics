import pandas as pd
import numpy as np
from collections import Counter
import ast
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "cleaned_jobs.csv")

_df_cache = None


def load_data():
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    df = pd.read_csv(DATA_PATH)
    df['skills_extracted'] = df['skills_extracted'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) and x not in ['[]', ''] else []
    )
    df['salary_yearly'] = pd.to_numeric(df['salary_yearly'], errors='coerce')
    _df_cache = df
    return df


def filter_df(role=None):
    """Return full df or df filtered to rows matching role title."""
    df = load_data()
    if role:
        mask = df['job_title'].str.lower().str.contains(role.lower(), na=False)
        filtered = df[mask]
        return filtered if not filtered.empty else df
    return df


def get_overview_stats(role=None):
    df = filter_df(role)
    total_jobs    = len(df)
    avg_salary    = df['salary_yearly'].dropna().mean()
    median_salary = df['salary_yearly'].dropna().median()
    remote_count  = df['work_from_home'].astype(str).str.lower().eq('true').sum()
    remote_pct    = remote_count / total_jobs * 100 if total_jobs else 0
    jobs_w_salary = df['salary_yearly'].notna().sum()
    top_role      = df['role_category'].value_counts().idxmax() if total_jobs else 'N/A'
    return {
        'total_jobs':    total_jobs,
        'avg_salary':    avg_salary,
        'median_salary': median_salary,
        'remote_pct':    remote_pct,
        'jobs_w_salary': jobs_w_salary,
        'top_role':      top_role,
    }


def get_top_roles(n=8, role=None):
    df = filter_df(role)
    vc = df['role_category'].value_counts().head(n)
    return vc.index.tolist(), vc.values.tolist()


def get_top_skills(n=15, role=None):
    df = filter_df(role)
    all_skills = [s for lst in df['skills_extracted'] for s in lst]
    counts = Counter(all_skills).most_common(n)
    return [c[0].title() for c in counts], [c[1] for c in counts]


def get_top_cities(n=10, role=None):
    df = filter_df(role)
    vc = df[df['city'] != 'Unknown']['city'].value_counts().head(n)
    return vc.index.tolist(), vc.values.tolist()


def get_salary_by_role(role=None):
    df = filter_df(role)
    filtered = df[df['salary_yearly'].notna() & (df['salary_yearly'] > 0)]
    grouped  = filtered.groupby('role_category')['salary_yearly'].mean().sort_values(ascending=False).head(8)
    return grouped.index.tolist(), grouped.values.tolist()


def get_salary_distribution(role=None):
    df = filter_df(role)
    salaries = df['salary_yearly'].dropna()
    salaries = salaries[(salaries > 10000) & (salaries < 400000)]
    return salaries.tolist()


def get_remote_vs_onsite(role=None):
    df = filter_df(role)
    remote  = df['work_from_home'].astype(str).str.lower().eq('true').sum()
    on_site = len(df) - remote
    return ['Remote', 'On-site'], [int(remote), int(on_site)]


def get_posting_trend(role=None):
    df = filter_df(role).copy()
    df['date_posted'] = pd.to_datetime(df['date_posted'], errors='coerce')
    monthly = df.dropna(subset=['date_posted'])
    monthly = monthly.groupby(monthly['date_posted'].dt.to_period('M')).size()
    monthly = monthly.sort_index().tail(18)
    return [str(p) for p in monthly.index], monthly.values.tolist()


def get_skills_salary(role=None):
    df = filter_df(role)
    rows = []
    for _, row in df.iterrows():
        for skill in row['skills_extracted']:
            rows.append({'skill': skill, 'salary': row['salary_yearly']})
    sdf = pd.DataFrame(rows)
    if sdf.empty or sdf['salary'].dropna().empty:
        return [], []
    result = (sdf[sdf['salary'].notna()]
              .groupby('skill')['salary']
              .agg(['mean', 'count'])
              .query('count >= 10')
              .sort_values('mean', ascending=False)
              .head(12))
    return [s.title() for s in result.index.tolist()], result['mean'].tolist()


def analyze_job_market(job_role):
    df = load_data()
    mask     = df['job_title'].str.lower().str.contains(job_role.lower(), na=False)
    job_data = df[mask]
    if job_data.empty:
        return None
    total        = len(job_data)
    avg_salary   = job_data['salary_yearly'].dropna().mean()
    top_cities   = job_data[job_data['city'] != 'Unknown']['city'].value_counts().head(5).index.tolist()
    all_skills   = [s for lst in job_data['skills_extracted'] for s in lst]
    top_skills   = [s.title() for s, _ in Counter(all_skills).most_common(8)]
    remote_count = job_data['work_from_home'].astype(str).str.lower().eq('true').sum()
    remote_pct   = remote_count / total * 100 if total > 0 else 0
    top_companies = job_data['company'].value_counts().head(5).index.tolist()
    return {
        'role':          job_role,
        'total':         total,
        'avg_salary':    avg_salary,
        'top_cities':    top_cities,
        'top_skills':    top_skills,
        'remote_pct':    remote_pct,
        'top_companies': top_companies,
    }
