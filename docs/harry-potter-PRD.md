# Harry Potter Quiz Game — Product Requirements Document

**Version:** 0.1-draft  
**Date:** 2026-03-11  
**Author:** Peter Wazinski  
**Status:** Draft — awaiting stakeholder sign-off

---

## 1. Overview

A bilingual (English / German) web-based quiz game set in the Harry Potter universe.
Players progress through five proficiency tiers by correctly answering sets of
multiple-choice questions of increasing difficulty.  The question bank (~500 questions)
is bootstrapped via a structured Microsoft Copilot prompt and can be extended by
registered users (subject to admin approval).

---

## 2. Goals

| # | Goal |
|---|------|
| G1 | Fun, replayable trivia experience for Harry Potter fans of all ages |
| G2 | Bilingual from day 1 — every question and UI string available in EN and DE |
| G3 | Zero hosting cost — free-tier cloud services only |
| G4 | No Node.js dependency — pure Python stack |
| G5 | Automated question bank bootstrapping via a repeatable LLM prompt |

---

## 3. Target Audience

- Harry Potter fans aged 12+
- Both casual (Muggle) and hardcore (Headmaster) knowledge levels
- German and English speaking users

---

## 4. Game Flow

```
Start screen
  └─ Choose language (EN / DE)
  └─ Guest play  OR  Log in (Google / GitHub OAuth)

Round screen
  ├─ Tier badge shown (current level)
  ├─ 10 multiple-choice questions  (4 options each)
  │    ├─ Optional per-round countdown timer (configurable — default 300 s)
  │    └─ Image shown if available (character portrait, book cover, etc.)
  └─ Score summary
        ├─ Pass threshold ≥ 70 %  →  advance to next tier
        └─ Fail               →  replay same tier  OR  quit

Progression
  Muggle  →  Student  →  Prefect  →  Auror  →  Headmaster
  (tier 1)   (tier 2)   (tier 3)   (tier 4)    (tier 5)

End screen
  ├─ Achieved proficiency badge + certificate text
  └─ Leaderboard position (if logged in)
```

---

## 5. Proficiency Tiers

| Tier | Name | Questions | Pass threshold | Topics |
|------|------|-----------|----------------|--------|
| 1 | 🧙 Muggle | 10 | 7 / 10 | Very basic: main characters, books order, core spells |
| 2 | 🎓 Student | 10 | 7 / 10 | Houses, creatures, Hogwarts layout, supporting cast |
| 3 | 🏅 Prefect | 10 | 7 / 10 | History of Magic, potions, detailed plot events |
| 4 | ⚡ Auror | 10 | 8 / 10 | Obscure characters, magical law, dates, exact quotes |
| 5 | 🎓 Headmaster | 10 | 9 / 10 | Deep lore, J.K. Rowling interviews, Pottermore/Wizarding World canon |

A player who completes **all 5 tiers** in one session receives the
**"Order of Merlin, First Class"** achievement.

---

## 6. Scoring

- +10 points per correct answer
- −0 points for wrong answer (no penalty — encourages guessing)
- Time bonus: +1 point per second remaining in round (only if timer is enabled)
- Running total displayed on every question screen
- High scores saved per user account; top-10 global leaderboard per tier

---

## 7. Questions

### 7.1 Format

Each question is stored as a JSON record:

```jsonc
{
  "id": "uuid",
  "tier": 1,                        // 1–5
  "topic": "characters",            // free tag
  "question_en": "...",
  "question_de": "...",
  "options_en": ["A", "B", "C", "D"],
  "options_de": ["A", "B", "C", "D"],
  "correct_index": 2,               // 0-based
  "explanation_en": "...",
  "explanation_de": "...",
  "image_url": null,                // optional Wikimedia URL
  "source": "copilot-generated",   // or "user-submitted"
  "status": "approved"             // "pending" | "approved" | "rejected"
}
```

### 7.2 Initial bank — Microsoft Copilot generation

Use the following prompt in Microsoft Copilot (repeat for each tier, replacing `TIER`
and `DIFFICULTY_NOTES`).  Run 3–4 times per tier (varying the `topic` tag) to reach
~100 questions per tier, then deduplicate.

---

**Prompt template** (paste into Copilot chat, replace `TIER` = 1 … 5):

```
You are a Harry Potter trivia expert.  Generate 30 multiple-choice questions for
difficulty tier TIER (1 = easiest, 5 = hardest).

DIFFICULTY_NOTES:
- Tier 1: Very basic facts known to anyone who watched the films once.
- Tier 2: Knowledge of all 7 books at a casual level.
- Tier 3: Careful reader familiar with all plot details.
- Tier 4: Fan who has read companion books (Fantastic Beasts, Quidditch Through the Ages).
- Tier 5: Deep-lore fan who has read Pottermore/Wizarding World articles and
          J.K. Rowling interviews.

Rules:
1. Each question has exactly 4 answer options (A, B, C, D).
2. There is exactly one correct answer.
3. Distractors must be plausible, not obviously wrong.
4. Do NOT repeat questions from previous tiers.
5. Provide a 1-sentence explanation for the correct answer.
6. Output ONLY a JSON array — no markdown, no prose, just raw JSON.

Output format (for each question):
{
  "tier": TIER,
  "topic": "<characters|spells|creatures|places|history|objects|other>",
  "question_en": "<English question>",
  "question_de": "<German translation of question>",
  "options_en": ["<A>", "<B>", "<C>", "<D>"],
  "options_de": ["<German A>", "<German B>", "<German C>", "<German D>"],
  "correct_index": <0–3>,
  "explanation_en": "<English explanation>",
  "explanation_de": "<German explanation>",
  "image_url": null
}
```

---

A helper script `tools/import_questions.py` will validate the JSON output and bulk-load
it into the database.  Run it per tier file:

```bash
python tools/import_questions.py --file questions_tier1.json --approve
```

### 7.3 User-contributed questions

- Any logged-in user can submit a question via a form in the UI.
- Submitted questions receive `status = "pending"`.
- Admin reviews in the Admin Dashboard; approves → `status = "approved"`.
- Pending questions **never** appear in a live game round.

---

## 8. Multilingual Support

- Language selection on the start screen; persisted in browser local storage and
  (if logged in) in user profile.
- All UI strings stored in `translations/en.json` and `translations/de.json`.
- Questions stored with both `_en` and `_de` fields — no runtime translation API needed.
- Dates and numbers formatted with locale-aware formatting (`babel`).

---

## 9. User System

| Capability | Guest | Logged-in user |
|---|---|---|
| Play game | ✓ | ✓ |
| Progress saved between sessions | Local storage only (lost on clear) | ✓ Database |
| Submit questions | ✗ | ✓ (pending approval) |
| Appear on leaderboard | ✗ | ✓ |
| See own history | ✗ | ✓ |

**Authentication:** OAuth 2.0 via Google and/or GitHub.  
Implementation: `Authlib` library + FastAPI; tokens stored as HTTP-only cookies.

---

## 10. Leaderboard

- One leaderboard per tier (top 10 globally).
- Shows: rank, display name, score, date achieved.
- Refreshed on every game completion.
- Guest scores shown locally only ("Your best: X pts") — not persisted server-side.

---

## 11. Admin Dashboard

Accessed at `/admin` (protected by admin role flag in DB).

| Feature | Description |
|---|---|
| Question list | Filterable by tier / status / topic |
| Approve / reject | Review pending user submissions with inline edit |
| Edit question | Fix typos, update answer, change tier |
| Bulk import | Upload the Copilot-generated JSON files |
| User list | View registered users, grant / revoke admin role |
| Stats | Questions per tier, approval rate, active players |

---

## 12. Tech Stack

### Backend

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.10 | Matches existing toolchain |
| Web framework | **FastAPI** | Fast, async, OpenAPI docs built-in |
| UI framework | **NiceGUI** (≥ 1.4) | Pure Python, reactive, no Node.js |
| ORM | **SQLModel** (= SQLAlchemy + Pydantic) | Type-safe, integrates with FastAPI |
| Auth | **Authlib** + OAuth 2.0 | Google / GitHub login |
| DB (dev) | SQLite | Zero-config local development |
| DB (prod) | **Supabase** free tier (PostgreSQL) | 500 MB free, persistent |
| Image storage | Wikimedia Commons URLs | Free, no upload needed for most images |
| i18n | **Babel** | Locale-aware date/number formatting |

### Free hosting

| Service | Role | Free tier limits |
|---|---|---|
| **Render.com** | Python web service (app server) | 750 hrs/month; sleeps after 15 min idle |
| **Supabase** | PostgreSQL database | 500 MB storage, unlimited API calls |
| **Cloudinary** *(optional)* | User-uploaded question images | 25 credits/month free |
| **UptimeRobot** | Keep-warm ping every 5 min | Free for 50 monitors |

> **Cold-start note:** Render free tier sleeps after 15 min of inactivity.
> The first request after sleep takes ~30 s.  A free UptimeRobot monitor
> (5-min interval) keeps the instance warm at zero cost.

### Proposed repository layout

```
harry_potter_quiz/
├── main.py                   # FastAPI + NiceGUI entry point
├── models.py                 # SQLModel DB models
├── routers/
│   ├── game.py               # game session endpoints
│   ├── auth.py               # OAuth routes
│   ├── questions.py          # CRUD for questions
│   └── admin.py              # admin-only routes
├── ui/
│   ├── pages/
│   │   ├── start.py
│   │   ├── round.py
│   │   ├── result.py
│   │   ├── leaderboard.py
│   │   └── admin.py
│   └── components/           # shared NiceGUI components
├── translations/
│   ├── en.json
│   └── de.json
├── tools/
│   └── import_questions.py   # CLI: validate + load Copilot JSON
├── tests/
├── alembic/                  # DB migrations
└── docs/
    └── harry-potter-PRD.md   # this file
```

---

## 13. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | Page load < 2 s on a 10 Mbit connection (excluding cold start) |
| NFR-2 | Mobile-responsive layout (min width 360 px) |
| NFR-3 | WCAG 2.1 AA colour contrast |
| NFR-4 | All OAuth tokens stored as HTTP-only cookies; no passwords stored |
| NFR-5 | No PII stored beyond display name + hashed email for leaderboard |
| NFR-6 | DB migrations via Alembic; zero-downtime deploy on Render |
| NFR-7 | All API endpoints covered by automated tests (`pytest` + `httpx`) |

---

## 14. Out of Scope (v1)

- Real-time multiplayer / lobby rooms
- Mobile native app (iOS / Android)
- Question audio or video
- Paid tiers or subscriptions
- Automated web scraping of fan wikis (manual Copilot prompt workflow instead)

---

## 15. Open Questions

| # | Question | Owner | Due |
|---|----------|-------|-----|
| OQ-1 | OAuth providers: Google only, or also GitHub? | Peter | — |
| OQ-2 | Failed rounds: unlimited retries or cooldown before retry? | Peter | — |
| OQ-3 | Admin account creation: seeded from env var on first migration, or CLI command? | Peter | — |
| OQ-4 | Timer default 300 s per round — confirm or adjust? | Peter | — |
| OQ-5 | Pass thresholds (7/7/7/8/9 out of 10) — confirm? | Peter | — |
| OQ-6 | UptimeRobot keep-warm acceptable, or pay Render Starter ($7/mo) for always-on? | Peter | — |

---

## 16. Implementation Roadmap

| Sprint | Duration | Deliverable |
|--------|----------|-------------|
| S0 | 1 day | Repo scaffold, DB models (`Question`, `User`, `Score`), Alembic migrations, SQLite dev DB |
| S1 | 2 days | `tools/import_questions.py` + run Copilot prompt for all 5 tiers → load ~500 questions |
| S2 | 2 days | FastAPI game endpoints: start session, get question, submit answer, score summary |
| S3 | 2 days | NiceGUI UI: start page, round page (question + timer), result page (EN only first) |
| S4 | 1 day | DE translation strings + language switcher |
| S5 | 2 days | OAuth login (Google), guest fallback, local-storage session |
| S6 | 1 day | Leaderboard endpoint + UI page |
| S7 | 2 days | Admin dashboard: question approval queue, bulk import, user list |
| S8 | 1 day | Image support: Wikimedia URL field displayed in question card |
| S9 | 1 day | Supabase prod DB + Render deploy + UptimeRobot keep-warm |
| S10 | 1 day | End-to-end tests, accessibility pass, README |

**Estimated total:** ~16 developer-days for a complete v1.
