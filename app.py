import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from pypdf import PdfReader
import io

# Import SQL Queries
import queries

# -----------------------------------------------------------------------------
# 1. SENTINEL UI CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentinel Pro | Career Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #C0C0C0; }
    .stMetric { background-color: #1E1E1E; padding: 15px; border-radius: 5px; border: 1px solid #333; }
    h1, h2, h3 { color: #00FF99 !important; font-family: 'Consolas', 'Courier New', monospace; }
    .stDataFrame { border: 1px solid #444; }
    .pro-badge { background-color: #00FF99; color: black; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ SENTINEL PRO | AI-Enhanced")

# -----------------------------------------------------------------------------
# 2. DATABASE & CACHING (PERFORMANCE LAYER)
# -----------------------------------------------------------------------------
DB_NAME = "career_intelligence.db"

@st.cache_resource
def get_engine():
    return create_engine(f"sqlite:///{DB_NAME}")

engine = get_engine()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def run_query(query, params=None):
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)

@st.cache_data(ttl=3600)
def train_model(df):
    model = LinearRegression()
    X = df[['day_index']]
    y = df['salary_year_avg']
    model.fit(X, y)
    return model

# -----------------------------------------------------------------------------
# 3. NAVIGATION
# -----------------------------------------------------------------------------
page = st.sidebar.radio("SYSTEM MODULES", ["📊 Live Monitor", "🕸️ Skill Network", "🔮 AI Forecast", "📝 Resume Scanner", "🚀 Career Simulator"])

st.sidebar.markdown("---")
st.sidebar.caption("v3.0.0 | Production-Grade")

# -----------------------------------------------------------------------------
# MODULE 1: LIVE MONITOR
# -----------------------------------------------------------------------------
if page == "📊 Live Monitor":
    st.header("📈 MARKET MOMENTUM")
    selected_days = st.slider("Time Horizon", 30, 180, 90)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.spinner("Analyzing Momentum..."):
            df_momentum = run_query(queries.MOMENTUM_QUERY, params={"days_filter": f"-{selected_days} days"})
        
        if not df_momentum.empty:
            top_skills = df_momentum.groupby('skills')['daily_jobs'].sum().nlargest(5).index.tolist()
            df_chart = df_momentum[df_momentum['skills'].isin(top_skills)]
            fig = px.line(df_chart, x='job_date', y='moving_avg_7d', color='skills',
                          title=f"7-Day Moving Average Demand (Top 5 Skills)",
                          template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("High Volatility Assets")
        df_vol = run_query(queries.VOLATILITY_QUERY)
        st.dataframe(df_vol[['skills', 'demand', 'avg_salary']], use_container_width=True, height=400)

# -----------------------------------------------------------------------------
# MODULE 2: SKILL NETWORK
# -----------------------------------------------------------------------------
elif page == "🕸️ Skill Network":
    st.header("🕸️ SKILL NEURAL NETWORK")
    
    with st.spinner("Mapping Neural Network..."):
        df_network = run_query(queries.NETWORK_QUERY)
        
    if not df_network.empty:
        G = nx.from_pandas_edgelist(df_network, 'source', 'target', 'weight')
        pos = nx.spring_layout(G, k=0.5)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

        node_x, node_y, node_text, node_adj = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_adj.append(len(G.adj[node]))

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
            marker=dict(showscale=True, colorscale='Viridis', size=20, color=node_adj, colorbar=dict(title='Connections'), line_width=2))

        fig_net = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title=dict(text='Skill Cluster Topography', font=dict(size=16)),
                        showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40),
                        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        st.plotly_chart(fig_net, use_container_width=True)

# -----------------------------------------------------------------------------
# MODULE 3: AI FORECAST
# -----------------------------------------------------------------------------
elif page == "🔮 AI Forecast":
    st.header("🔮 SALARY PREDICTION ENGINE")
    skills_list = run_query("SELECT skills FROM skills_dim ORDER BY skills")['skills'].tolist()
    target_skill = st.selectbox("Select Target Asset", skills_list)
    
    df_history = run_query(queries.HISTORY_QUERY, params={"skill": target_skill})
    
    if len(df_history) > 10:
        df_history['date'] = pd.to_datetime(df_history['job_posted_date'])
        df_history['day_index'] = (df_history['date'] - df_history['date'].min()).dt.days
        
        model = train_model(df_history) # Uses caching now!
        
        future_days = np.array(range(df_history['day_index'].max() + 1, df_history['day_index'].max() + 31)).reshape(-1, 1)
        future_dates = [df_history['date'].max() + timedelta(days=i) for i in range(1, 31)]
        predictions = model.predict(future_days)
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=df_history['date'], y=df_history['salary_year_avg'], mode='markers', name='Historical', marker=dict(color='#444', size=5)))
        fig_forecast.add_trace(go.Scatter(x=df_history['date'], y=model.predict(df_history[['day_index']]), mode='lines', name='Trend', line=dict(color='yellow')))
        fig_forecast.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='Forecast', line=dict(color='#00FF99', dash='dash')))
        fig_forecast.update_layout(title="Linear Regression Forecast", template="plotly_dark", yaxis_tickprefix="$")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        current_val, future_val = model.predict(df_history[['day_index']])[-1], predictions[-1]
        st.metric(f"30-Day Price Target ({target_skill})", f"${future_val:,.0f}", f"{((future_val - current_val) / current_val) * 100:+.2f}%")

# -----------------------------------------------------------------------------
# MODULE 4: RESUME SCANNER 2.0 (PDF Support)
# -----------------------------------------------------------------------------
elif page == "📝 Resume Scanner":
    st.header("📝 RESUME SENTINEL SCANNER")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    resume_text = ""
    
    if uploaded_file:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            resume_text += page.extract_text() or ""
        st.info("PDF Content Extracted Successfully.")
    else:
        resume_text = st.text_area("Or Paste Text", height=150, placeholder="Paste job description...")
    
    if resume_text:
        skills_df = run_query(queries.SKILL_VALUE_QUERY)
        skill_value_map = dict(zip(skills_df['skills'].str.lower(), skills_df['avg_salary']))
        all_skills = set(skill_value_map.keys())
        
        found_skills = []
        # Check logic
        normalized_text = resume_text.lower()
        for skill in all_skills:
            if " " in skill and skill in normalized_text: # Multi-word
                found_skills.append(skill)
        
        text_words = set(''.join(e for e in w if e.isalnum()) for w in normalized_text.split())
        for skill in all_skills:
            if " " not in skill and skill in text_words:
                 found_skills.append(skill)
                
        found_skills = list(set(found_skills))
        
        if found_skills:
            total_value = sum([skill_value_map.get(s, 0) for s in found_skills])
            avg_value = total_value / len(found_skills)
            
            st.metric("Estimated Market Rate", f"${avg_value:,.0f}", help="Average salary of all detected skills.")
            st.caption(f"Based on {len(found_skills)} detected skills: {', '.join([s.title() for s in found_skills])}")
            
            st.subheader("Asset Breakdown")
            st.dataframe(pd.DataFrame([{"Skill": s.title(), "Value": f"${skill_value_map.get(s,0):,.0f}"} for s in found_skills]), use_container_width=True)

        else:
            st.warning("No assets detected.")

# -----------------------------------------------------------------------------
# MODULE 5: CAREER SIMULATOR (New!)
# -----------------------------------------------------------------------------
elif page == "🚀 Career Simulator":
    st.header("🚀 CAREER ROI SIMULATOR")
    st.info("Select your current stack. The AI will calculate the 'Marginal Value' of adding a new skill.")
    
    all_skills = run_query("SELECT skills, skill_id FROM skills_dim")
    skill_options = all_skills['skills'].tolist()
    
    current_stack = st.multiselect("Current Tech Stack", skill_options, default=['SQL'])
    
    if current_stack:
        # Smarter Logic: 
        # Score = (0.7 * Correlation) + (0.3 * Salary)
        # We prize "Natural Fit" slightly more than raw "Money" for career pathing
        
        salary_df = run_query(queries.VOLATILITY_QUERY)
        candidates = {}
        
        # 1. Broad Parameter Scan
        for skill in current_stack:
            neighbors = run_query(queries.NEIGHBORS_QUERY, params={"target_skill": skill})
            
            for index, row in neighbors.iterrows():
                neighbor = row['neighbor']
                weight = row['weight']
                
                if neighbor in current_stack:
                    continue
                    
                # Look up salary
                try:
                    salary_row = salary_df[salary_df['skills'] == neighbor].iloc[0]
                    raw_salary = salary_row['raw_avg_salary']
                    
                    if neighbor not in candidates:
                        candidates[neighbor] = {
                            "skill": neighbor,
                            "total_weight": 0,
                            "salary": raw_salary,
                            "sources": []
                        }
                    candidates[neighbor]['total_weight'] += weight
                    candidates[neighbor]['sources'].append(skill)
                except:
                    continue

        if candidates:
            df_candidates = pd.DataFrame(candidates.values())
            
            # Max scaling
            max_weight = df_candidates['total_weight'].max()
            max_salary = df_candidates['salary'].max()
            
            df_candidates['norm_weight'] = df_candidates['total_weight'] / max_weight
            df_candidates['norm_salary'] = df_candidates['salary'] / max_salary
            
            df_candidates['score'] = (df_candidates['norm_weight'] * 0.7) + (df_candidates['norm_salary'] * 0.3)
            
            df_candidates = df_candidates.sort_values('score', ascending=False).head(5)
            
            st.subheader("💡 Strategic Recommendations")
            
            col1, col2 = st.columns(2)
            
            # TOP PICK (Highest Score)
            top_pick = df_candidates.iloc[0]
            with col1:
                st.success(f"⭐ **Top Pick: {top_pick['skill']}**")
                st.caption(f"Strongest logic flow from {', '.join(top_pick['sources'])}.")
                st.metric("Projected Salary", f"${top_pick['salary']:,.0f}")
                
            # HIGH VALUE (Highest Salary in top 5)
            df_salary = df_candidates.sort_values('salary', ascending=False)
            high_value = df_salary.iloc[0]
            if high_value['skill'] == top_pick['skill'] and len(df_candidates) > 1:
                high_value = df_salary.iloc[1]
                
            with col2:
                st.info(f"💰 **High Value Alternative: {high_value['skill']}**")
                st.caption("Lower correlation, but higher market valuation.")
                st.metric("Projected Salary", f"${high_value['salary']:,.0f}")

            with st.expander("View Analysis Logic"):
                st.dataframe(df_candidates[['skill', 'score', 'salary', 'total_weight']].style.format({'salary': '${:,.0f}', 'score': '{:.2f}'}), use_container_width=True)
            
        else:
            st.warning("Not enough data to generate path. Try adding more skills.")
