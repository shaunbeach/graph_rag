# AiForge Dashboard - Development Roadmap

**Last Updated:** 2025-11-08
**Current Phase:** MVP Complete → Waiting for Real Data

---

## 🎯 Strategic Approach

**Philosophy:** Experiments first, then UI integration.

We're taking a **data-driven approach** to dashboard development:
1. Run real experiments using CLI to understand actual needs
2. Integrate dashboard with real data once we know what matters
3. Build additional features based on actual usage patterns

This avoids building features we don't need and ensures the UI serves real workflows.

---

## ✅ Phase 0: MVP (COMPLETE)

**Timeline:** Completed 2025-11-08
**Status:** ✅ Done

### What Was Built

**Frontend:**
- [x] React 18 + TypeScript + Vite setup
- [x] TailwindCSS dark theme
- [x] Component library (Button, Card, Badge, ProgressBar)
- [x] Sidebar navigation
- [x] Experiment Dashboard page
- [x] Experiment cards with all states (running/completed/failed/pending)
- [x] Filter and search functionality
- [x] Mock data for 4 experiments
- [x] Responsive layout

**Backend:**
- [x] FastAPI REST API
- [x] Auto-detects LIMS installation
- [x] Mock data fallback
- [x] CORS configuration
- [x] API documentation (Swagger UI)

**Infrastructure:**
- [x] Launch script
- [x] Documentation (README, GETTING_STARTED, SUMMARY)
- [x] Fixed all TypeScript/config issues

**Demo:** http://localhost:5173 (currently running with mock data)

---

## 🏃 Phase 1: Real Experiments (IN PROGRESS)

**Timeline:** 1-2 weeks
**Owner:** User
**Status:** 🟡 Waiting to Start

### Objectives

**Primary Goal:** Run 5-10 varied experiments to understand:
- Which metrics actually matter in practice
- Common experiment states and failure modes
- Where the CLI is clunky (= where UI helps most)
- What data LIMS captures automatically
- Real workflow patterns

### Tasks

#### User Will Execute:
```bash
# 1. Create and run experiments
aiforge lims create --name "experiment-1" --config config.yaml
# Train model
# Track metrics

# 2. Generate reports
aiforge report generate EXP_ID

# 3. Try synthetic data generation
aiforge synthetic generate prompts.txt

# 4. Use meta-learning features
aiforge predict train-predictor
aiforge predict simulate config.yaml
```

#### What to Track:
- [ ] Which CLI commands are used most frequently
- [ ] What experiment metadata is most important
- [ ] Which metrics are checked most often
- [ ] Common pain points or confusing workflows
- [ ] Missing features or information

### Success Criteria

- [ ] At least 5 completed experiments in LIMS
- [ ] User comfortable with CLI workflow
- [ ] User has identified 3-5 UI pain points or wishlist items
- [ ] LIMS workspace has real data to integrate

### Deliverables for Next Phase

1. List of most important metrics/views
2. Common experiment patterns observed
3. CLI pain points (what would be easier in UI)
4. Wishlist of features to prioritize

---

## 🔌 Phase 2: Real Data Integration (NEXT)

**Timeline:** 1 day
**Owner:** Claude
**Status:** ⏸ Blocked (waiting for Phase 1)

### Objectives

Connect the existing dashboard to real LIMS data without major changes.

### Tasks

#### Backend Integration:
- [ ] Verify LIMS import path
- [ ] Set `AIFORGE_WORKSPACE` environment variable
- [ ] Start backend with real data
- [ ] Test API endpoints with real experiments
- [ ] Fix any data format mismatches

#### Frontend Updates:
- [ ] Remove or hide mock data
- [ ] Add API client (Axios)
- [ ] Wire ExperimentsPage to backend API
- [ ] Add loading states
- [ ] Add error handling
- [ ] Test with real data

#### Validation:
- [ ] All experiment states display correctly
- [ ] Metrics show accurate values
- [ ] Filters work with real data
- [ ] Search works across all fields
- [ ] Performance is acceptable (< 1s load time)

### Expected Code Changes

**Minimal - backend already supports this!**

```bash
# backend/main.py - NO CHANGES NEEDED
# Already has LIMS detection and loading logic

# frontend/src/api/client.ts - NEW FILE
# Simple Axios wrapper

# frontend/src/pages/ExperimentsPage.tsx - MINOR CHANGES
# Replace mock data with API call
```

### Success Criteria

- [ ] Dashboard shows user's real experiments
- [ ] All experiment metadata visible
- [ ] Filtering and search work
- [ ] No performance issues
- [ ] User confirms data accuracy

---

## 🎨 Phase 3: Essential Features (NEXT AFTER PHASE 2)

**Timeline:** 3-5 days
**Owner:** Claude
**Status:** ⏸ Blocked (waiting for Phase 1 feedback)

### Features to Build (Priority TBD)

Will be prioritized based on Phase 1 learnings. Likely candidates:

#### Feature A: Experiment Detail Page
**Why:** "View Details" button currently does nothing
**Effort:** Medium (2-3 days)

Components:
- [ ] Create `ExperimentDetailPage.tsx`
- [ ] Add routing (React Router)
- [ ] Implement tabs: Overview, Metrics, Config, Dataset, Checkpoints, Logs
- [ ] Build training/validation charts (Recharts)
- [ ] Config viewer with syntax highlighting
- [ ] Timeline of experiment events
- [ ] Resource usage visualization

#### Feature B: Real-time Updates
**Why:** See live training progress without CLI
**Effort:** Medium (2-3 days)

Components:
- [ ] WebSocket connection to backend
- [ ] Live metric streaming during training
- [ ] Auto-updating progress bars
- [ ] Status change notifications
- [ ] Toast notifications for events

#### Feature C: Synthetic Data Monitor
**Why:** Visualize data generation pipeline
**Effort:** Medium (2-3 days)

Components:
- [ ] Create `SyntheticDataPage.tsx`
- [ ] Pipeline visualization (SP → AP → Gen → Filter → Format)
- [ ] Progress tracking per stage
- [ ] Quality metrics breakdown
- [ ] Dataset lineage graph
- [ ] Sample preview

#### Feature D: Charts & Metrics
**Why:** Better understand training dynamics
**Effort:** Small (1 day)

Components:
- [ ] Integrate Recharts properly
- [ ] Training/validation curves
- [ ] Loss curves
- [ ] Learning rate schedule visualization
- [ ] Gradient norm tracking
- [ ] Resource utilization over time

### Feature Prioritization Process

After Phase 1, we'll ask:
1. What did you use most in the CLI?
2. What was most frustrating about CLI-only workflow?
3. What metrics did you check repeatedly?
4. What would save you the most time?

Then build features in that order.

---

## 🚀 Phase 4: Advanced Features (FUTURE)

**Timeline:** TBD (1-2 weeks)
**Status:** 📋 Planning

### Feature Ideas (Not Committed)

#### Experiment Comparison
- Side-by-side metric comparison
- Configuration diff viewer
- Statistical significance testing
- Leaderboard by metric

#### Meta-Learning UI
- Prediction simulator interface
- Optimization parameter tuner
- Data gap analysis visualization
- Auto-seed generation with preview

#### Reports Gallery
- Browse all generated reports
- In-browser HTML viewer
- Download PDF/Markdown
- Report search and filtering

#### Configuration Builder
- Visual form for experiment configs
- RAG-powered suggestions
- Hardware awareness
- Validation and preview
- Export to YAML or CLI command

#### System Dashboard
- Real-time health monitoring
- Disk space tracking
- Ollama model management
- Recent activity timeline
- Quick actions

---

## 📊 Feature Matrix

| Feature | Priority | Effort | Blocked By | Status |
|---------|----------|--------|------------|--------|
| **MVP Dashboard** | P0 | High | - | ✅ Complete |
| **Real Data Integration** | P0 | Low | Phase 1 | ⏸ Blocked |
| **Experiment Detail Page** | P1 | Medium | Phase 1 | ⏸ Planning |
| **Real-time Updates** | P1 | Medium | Phase 1 | ⏸ Planning |
| **Charts Integration** | P1 | Low | Phase 2 | ⏸ Planning |
| **Synthetic Data Monitor** | P2 | Medium | Phase 1 | ⏸ Planning |
| **Meta-Learning UI** | P2 | High | Phase 1 | ⏸ Planning |
| **Reports Gallery** | P2 | Medium | - | 📋 Backlog |
| **Experiment Comparison** | P3 | Medium | Phase 2 | 📋 Backlog |
| **Configuration Builder** | P3 | High | - | 📋 Backlog |
| **System Dashboard** | P3 | Low | - | 📋 Backlog |
| **Mobile Support** | P4 | High | - | 📋 Backlog |
| **Authentication** | P4 | High | - | 📋 Backlog |

**Priority Levels:**
- **P0:** Critical (MVP, must-have)
- **P1:** High (core functionality)
- **P2:** Medium (nice-to-have)
- **P3:** Low (future enhancement)
- **P4:** Nice-to-have (maybe never)

---

## 🎯 Decision Points

### When to Build Feature X?

**Decision Framework:**
1. Did user request it based on real usage?
2. Does it solve a real pain point from Phase 1?
3. Can we build it without breaking existing features?
4. Is the ROI worth the effort?

**Default Answer:** Wait until user asks for it!

### When to Optimize Performance?

**Threshold:** Only if:
- Load time > 2 seconds
- User complains about slowness
- Backend struggles with > 100 experiments

Otherwise: Don't optimize prematurely.

### When to Add Authentication?

**Trigger:** User says they need to share dashboard with team/collaborators

Until then: Local-only is fine.

---

## 📝 Notes from Planning Session

### Key Insights

1. **Real data reveals real needs** - Better to design UX after seeing actual usage patterns
2. **LIMS already works** - No need to rush UI if CLI is functional
3. **Avoid over-engineering** - Only build what's actually needed
4. **Iterative approach** - Start simple, add complexity based on feedback

### User Preferences

- Wants to run experiments first before heavy UI development
- Comfortable with CLI for now
- Understands value of data-driven UI design
- Willing to provide detailed feedback after Phase 1

### Technical Constraints

- TailwindCSS must stay at v3.3.0 (v3.4+ has breaking changes)
- Config files must be `.cjs` (CommonJS) due to ES module setup
- Type imports must use `import type` syntax
- Backend already ready for real LIMS integration

---

## 🔄 Review Cadence

**After Phase 1 (User's Experiments):**
- Review experiment data
- Identify top 3-5 pain points
- Prioritize Phase 3 features
- Estimate timelines for high-priority items

**After Phase 2 (Real Data Integration):**
- Validate data accuracy
- Check performance
- Identify any data format issues
- Get user feedback on what's missing

**After Phase 3 (Essential Features):**
- Measure usage patterns
- Identify most-used features
- Plan Phase 4 based on actual usage
- Sunset unused features

---

## 📚 Related Documentation

- [Project Status](.claude/dashboard-project-status.md) - Current state, continuation guide
- [README](README.md) - Project overview
- [Getting Started](GETTING_STARTED.md) - User guide
- [Build Summary](SUMMARY.md) - Technical details

---

**Next Review:** After user completes Phase 1 (5-10 experiments)

**End of Roadmap**
