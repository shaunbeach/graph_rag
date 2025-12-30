# AiForge Dashboard

A modern web dashboard for the AiForge ML Laboratory System.

## Features

- **Experiment Dashboard**: View and manage all your ML experiments
- **Real-time Monitoring**: Live updates for running experiments
- **Metrics Visualization**: Interactive charts for training progress
- **Synthetic Data Monitor**: Track data generation pipelines
- **Meta-Learning Interface**: Predictions and optimization tools
- **Reports Gallery**: Browse and view experiment reports

## Architecture

- **Frontend**: React 18 + TypeScript + Vite + TailwindCSS
- **Backend**: FastAPI (Python)
- **Data Source**: AiForge LIMS (JSON-based storage)
- **Charts**: Recharts
- **Hosting**: Local-only (localhost)

## Quick Start

### Prerequisites

- Node.js 18+ (for frontend)
- Python 3.10+ (for backend)
- AiForge LIMS installed

### Installation

1. **Install Frontend Dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Install Backend Dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

### Running the Dashboard

#### Option 1: Run Both (Recommended)

```bash
# From the dashboard directory
./start_dashboard.sh
```

This will start both the frontend (port 5173) and backend (port 8000).

#### Option 2: Run Separately

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
# API will be available at http://localhost:8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
# Dashboard will be available at http://localhost:5173
```

### Accessing the Dashboard

Once both servers are running, open your browser to:
- **Dashboard**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs

## Project Structure

```
dashboard/
├── frontend/                 # React application
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── ui/          # Base UI components (Button, Card, etc.)
│   │   │   ├── layout/      # Layout components (Sidebar, Header)
│   │   │   ├── experiments/ # Experiment-specific components
│   │   │   ├── synthetic/   # Synthetic data components
│   │   │   └── charts/      # Chart components
│   │   ├── pages/           # Page components
│   │   ├── lib/             # Utility functions
│   │   ├── types/           # TypeScript types
│   │   ├── hooks/           # Custom React hooks
│   │   └── api/             # API client
│   ├── public/              # Static assets
│   └── package.json
│
└── backend/                 # FastAPI server
    ├── main.py             # API endpoints
    └── requirements.txt    # Python dependencies
```

## Development

### Frontend Development

```bash
cd frontend
npm run dev        # Start dev server with hot reload
npm run build      # Build for production
npm run preview    # Preview production build
```

### Backend Development

```bash
cd backend
python main.py     # Start FastAPI server
```

The backend will auto-reload on code changes if you run with:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /api/experiments` - List all experiments
- `GET /api/experiments/{id}` - Get experiment details
- `GET /api/system/status` - Get system status
- `GET /docs` - Interactive API documentation (Swagger UI)

## Integration with AiForge CLI

You can add this dashboard to the AiForge CLI:

```bash
aiforge dashboard start    # Start the dashboard
aiforge dashboard stop     # Stop the dashboard
aiforge dashboard status   # Check if dashboard is running
```

See `../src/aiforge/cli/dashboard_commands.py` for implementation.

## Features Roadmap

- [x] Experiment Dashboard
- [x] Basic UI Components
- [x] REST API Backend
- [ ] Experiment Detail Page with Charts
- [ ] Synthetic Data Monitor
- [ ] Meta-Learning Predictor Interface
- [ ] Reports Gallery
- [ ] WebSocket Real-time Updates
- [ ] Configuration Builder
- [ ] System Status Dashboard

## Technology Stack

**Frontend:**
- React 18
- TypeScript
- Vite (build tool)
- TailwindCSS (styling)
- Recharts (charts)
- Lucide React (icons)
- Axios (HTTP client)

**Backend:**
- FastAPI
- Uvicorn (ASGI server)
- Pydantic (data validation)

## Contributing

This dashboard is part of the AiForge project. For contribution guidelines, see the main AiForge documentation.

## License

Same as AiForge main project.
