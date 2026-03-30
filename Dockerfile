# ─────────────────────────────────────────
#  SPIKES CGM Dashboard — Docker Image
# ─────────────────────────────────────────
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY SPIKES_app.py .
COPY spike_analysis.py .
COPY pillar_analysis.py .
COPY cohort_level.py .

# Streamlit config — disable the "welcome" screen and telemetry
RUN mkdir -p /root/.streamlit && \
    echo '[general]\nemail = ""\n' > /root/.streamlit/credentials.toml && \
    echo '[server]\nheadless = true\nport = 8501\nenableCORS = false\n' > /root/.streamlit/config.toml

# Expose the Streamlit port
EXPOSE 8501

# Launch the app
CMD ["streamlit", "run", "SPIKES_app.py"]
