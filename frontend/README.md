## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Tobywooo/Agentic_AI_Cosmic_Mart.git
cd Agentic_AI_Cosmic_Mart
```

---

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

Copy the example file:

```powershell
copy .env.example .env
```

Update `.env` with the appropriate LLM configuration and API credentials.

---

### 5. Start the Backend

```bash
python run.py
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Verify the server is running:

```text
http://127.0.0.1:8000/health
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

### 6. Start the Frontend

Open a second terminal.

```bash
cd frontend
python -m http.server 5500
```

Open the dashboard:

```text
http://127.0.0.1:5500/support-dashboard.html
```

---

### 7. Verify End-to-End Functionality

1. Select a customer from the dropdown.
2. Send a message through the chat interface.
3. Confirm a request is sent to:

```text
POST /chat
```

4. View case status updates and agent responses.
5. Test escalation by triggering a human handoff.
6. Review escalated cases in the Specialist dashboard.

---

### Common Issues

#### HTTP 503 Service Unavailable

The backend is running but cannot reach the configured LLM provider.

Check:

```bash
http://127.0.0.1:8000/health
http://127.0.0.1:8000/health/llm
```

Verify:

- `.env` exists
- API key is configured
- Base URL is configured
- Model name is valid

#### Frontend Loads but Messages Fail

Verify:

```text
http://127.0.0.1:8000/health
```

returns a successful response and that the backend server is running.

---

### Git Workflow

Pull latest updates:

```bash
git checkout main
git pull origin main

git checkout project_dev_WD
git merge main
```

Commit changes:

```bash
git add .
git commit -m "Add frontend integration"
```

Push updates:

```bash
git push
```