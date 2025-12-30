# Getting Started with AiForge Dashboard

Welcome to the AiForge Dashboard! This guide will help you get up and running quickly.

## What You've Got

The dashboard is now **live and running** at http://localhost:5173!

### Current Features (v0.1 - MVP)

✅ **Experiment Dashboard**
- View all your ML experiments in a beautiful card layout
- Filter by status (running, completed, failed, pending)
- Search by name, ID, or tags
- Real-time progress bars for running experiments
- Quick actions (view charts, reports, config)

✅ **Modern Dark UI**
- Dark mode optimized for late-night research sessions
- Responsive layout (works on desktop, tablet)
- Clean, professional design
- Syntax highlighting for code/configs

✅ **REST API Backend**
- FastAPI server ready to connect to LIMS
- Mock data for immediate testing
- Auto-documentation at http://localhost:8000/docs

✅ **Component Library**
- Button, Card, Badge, ProgressBar components
- Consistent design system
- TailwindCSS for rapid styling

### What's Next (Coming Soon)

🚧 **Experiment Detail Page**
- Interactive training curves (Recharts)
- Resource usage visualization
- Per-epoch metrics table
- Configuration viewer with YAML syntax highlighting
- Timeline of experiment events

🚧 **Synthetic Data Monitor**
- Pipeline visualization (SP → AP → Gen → Filter → Format)
- Real-time generation progress
- Quality metrics breakdown
- Dataset lineage graphs

🚧 **Meta-Learning Interface**
- Prediction simulator
- Optimization tools
- Data gap analysis

🚧 **Reports Gallery**
- Browse all generated reports
- In-browser HTML viewer
- Download PDF/Markdown

🚧 **Real-time Updates**
- WebSocket connection for live metrics
- Live training curves that update every few seconds
- Status notifications

## Running the Dashboard

### Quick Start (Easiest)

```bash
cd /Volumes/T7/AiForge/dashboard
./start_dashboard.sh
```

This launches both frontend and backend automatically.

### Manual Start (For Development)

**Terminal 1 - Backend:**
```bash
cd /Volumes/T7/AiForge/dashboard/backend

# First time only:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Every time:
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd /Volumes/T7/AiForge/dashboard/frontend

# First time only:
npm install

# Every time:
npm run dev
```

### Accessing the Dashboard

Once running:
- **Dashboard UI**: http://localhost:5173
- **API Backend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI - try it out!)

## Current State

### What Works Right Now

1. **Browse Experiments**: The main dashboard shows mock experiment data
2. **Filter & Search**: Try filtering by status or searching for tags
3. **Responsive Design**: Resize your browser window - it adapts!
4. **Navigation**: Click through the sidebar (other pages show "Coming soon")

### Mock Data

The dashboard currently uses **mock data** for demonstration. You'll see:
- 1 running experiment (Qwen2.5-3B Code Fine-tune)
- 1 completed experiment (Llama-3.2-1B Math Reasoning)
- 1 failed experiment (Phi-3-Mini Chat Tuning)
- 1 pending experiment (Qwen2.5-7B Instruction)

### Connecting to Real LIMS Data

To connect to your actual AiForge LIMS experiments:

1. **Backend automatically detects LIMS**: The `backend/main.py` already imports from `src/aiforge/lims`

2. **Set workspace path** (if not default):
   ```bash
   export AIFORGE_WORKSPACE=/path/to/your/workspace
   ```

3. **Restart backend**: It will automatically load real experiments from your LIMS

## Customization

### Changing Colors

Edit `frontend/tailwind.config.js`:

```javascript
colors: {
  primary: {
    // Change these hex values to your preferred color
    500: '#0ea5e9',  // Main accent color
    600: '#0284c7',  // Hover states
    // ... etc
  }
}
```

### Adding New Pages

1. Create a new page component in `frontend/src/pages/`
2. Add a route case in `frontend/src/App.tsx`
3. Add a menu item in `frontend/src/components/layout/Sidebar.tsx`

### Adding API Endpoints

Edit `backend/main.py`:

```python
@app.get("/api/your-endpoint")
async def your_endpoint():
    return {"data": "your data"}
```

## Project Structure Quick Reference

```
dashboard/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/              # Reusable UI components
│   │   │   ├── layout/          # Sidebar, Header
│   │   │   └── experiments/     # Experiment-specific components
│   │   ├── pages/               # Full page components
│   │   ├── lib/utils.ts         # Helper functions
│   │   ├── types/index.ts       # TypeScript type definitions
│   │   └── App.tsx              # Main app component
│   └── package.json
│
└── backend/
    ├── main.py                  # FastAPI application
    └── requirements.txt
```

## Development Tips

### Hot Reload

Both frontend and backend support hot reload:
- **Frontend**: Save any `.tsx` file → browser auto-refreshes
- **Backend**: Use `uvicorn main:app --reload` → API auto-reloads

### Debugging

**Frontend (Chrome DevTools):**
- Open browser DevTools (F12)
- Check Console for errors
- Use React DevTools extension

**Backend:**
- Check terminal output for Python errors
- Visit http://localhost:8000/docs to test API endpoints
- Add `print()` statements in `main.py`

### Common Issues

**Port already in use:**
```bash
# Kill processes on ports
lsof -ti:5173 | xargs kill -9  # Frontend
lsof -ti:8000 | xargs kill -9  # Backend
```

**Dependencies not found:**
```bash
# Frontend
cd frontend && npm install

# Backend
cd backend && source venv/bin/activate && pip install -r requirements.txt
```

## Next Steps

### For Users

1. **Explore the UI**: Click around, try filtering and searching
2. **Check the API docs**: Visit http://localhost:8000/docs
3. **Connect your LIMS**: Follow "Connecting to Real LIMS Data" above

### For Developers

1. **Implement Experiment Detail Page**:
   - Create `frontend/src/pages/ExperimentDetailPage.tsx`
   - Add charts using Recharts
   - Wire up to `/api/experiments/{id}` endpoint

2. **Add Synthetic Data Monitor**:
   - Create components in `frontend/src/components/synthetic/`
   - Add backend endpoints for synthetic data jobs
   - Implement pipeline visualization

3. **Add WebSocket Support**:
   - Install `socket.io` on frontend
   - Add WebSocket endpoint to FastAPI
   - Stream real-time metrics

## Resources

- **React Docs**: https://react.dev
- **TailwindCSS**: https://tailwindcss.com/docs
- **FastAPI**: https://fastapi.tiangolo.com
- **Recharts**: https://recharts.org/en-US/

## Feedback

The dashboard is in early development. If you encounter issues or have suggestions:

1. Check existing issues in the AiForge repo
2. Open a new issue with the `dashboard` label
3. Include screenshots and error messages

---

**Happy Experimenting! 🚀**
