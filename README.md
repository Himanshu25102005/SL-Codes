# Spotter AI — Full Stack Developer Assessment: Execution Timeline

**Start:** 10:00 AM IST, Fri June 19, 2026
**Target submission:** by night, Mon June 22, 2026 (IST) — this gives you a ~30hr buffer before the actual deadline (~5:44 AM IST, June 23)
**Total runway:** ~3.5 working days, ~7–8 hrs/day

---

## 🎯 Project Goal (keep this pinned while building)

You're building a **trip-planning + HOS-compliance tool for truck drivers.**

**Input (one form):**
- Current location
- Pickup location
- Dropoff location
- Current Cycle Used (Hrs) — hours already used in the 70hr/8-day cycle

**Output (two things):**
1. **Route map** — route + marked stops (fuel, rest, pickup/dropoff) using a free map API
2. **ELD Daily Log Sheets** — actual *visual* grid sheets (like `blank-paper-log.png`), auto-filled with Off Duty / Sleeper Berth / Driving / On Duty lines. Multi-day trips → multiple sheets, auto-paginated.

**Hard-coded assumptions (don't ask the user for these):**
- Property-carrying driver, 70hr/8-day cycle (not 60/7)
- No adverse driving conditions exception
- Fuel stop required every ≤1,000 miles
- 1 hour fixed for pickup, 1 hour fixed for dropoff

**Core HOS rules your backend must encode:**
- 11-hour driving limit
- 14-hour on-duty/driving window
- 30-min break required after 8 cumulative driving hours
- 70-hour/8-day rolling limit (subtract "Current Cycle Used" from 70 to get hours available)
- 34-hour restart (only if relevant to your trip-length scenarios — likely lower priority, see Day 3)

**Grading signal (stated explicitly by them):**
- Backend HOS math accuracy is checked
- **UI/UX is weighted heavily** — good design can offset minor calculation slip-ups, so don't starve the frontend polish for backend perfectionism

**Stack:** Django (backend/API) + Next.js (frontend) — you're strong on Next already, so Django is the real unknown to de-risk early.

---

## Day 0 — Fri, June 19 (10:00 AM – ~7:00 PM, ~8 hrs)
**Theme: Learn Django fast + lock the architecture**

| Time | Block |
|---|---|
| 10:00–10:30 | Re-read the assessment doc + reference PDF once more. Write down your own plain-English spec: what inputs → what outputs, list every HOS rule you must implement. Pin the assumptions list above next to your desk. |
| 10:30–11:00 | Pick your map API now (don't leave this for later — it gates everything). Recommended: **OpenRouteService** (free API key, gives route + distance + duration) or **OSRM public demo server** (no key, but rate-limited/less reliable) or **Mapbox** (generous free tier, great docs, easy Next.js integration via `react-map-gl`). → **Pick Mapbox or OpenRouteService and get your API key right now.** |
| 11:00–1:30 PM | Django crash course, hands-on (don't watch passively — type along): project setup, apps, models, migrations, Django REST Framework (DRF) basics — serializers, viewsets, urls.py routing. Build one trivial "hello world" API endpoint and call it from a `fetch()` to confirm the loop works end to end. |
| 1:30–2:15 | Lunch / break |
| 2:15–4:30 | Continue Django: DRF serializers in depth, request validation, CORS setup (critical — Next.js frontend + Django backend = different origins, install & configure `django-cors-headers` now so it's not a last-day surprise). |
| 4:30–5:00 | Break |
| 5:00–7:00 | Sketch your data model and API contract on paper/Notion: <br>• `POST /api/plan-trip/` → takes current/pickup/dropoff/cycle-used → returns route geometry + stop list + log-sheet data (JSON, not images — you'll render the grid in the frontend with SVG/Canvas). <br>• Decide the **HOS calculation algorithm structure** in pseudocode before writing real code (see Day 1). |

**End of Day 0 checkpoint:** You understand Django+DRF enough to build CRUD/API endpoints confidently, you have a map API key working in a test call, and you have a written API contract.

---

## Day 1 — Sat, June 20 (full day, ~7–8 hrs)
**Theme: Backend — HOS algorithm + route logic**

| Block | Focus |
|---|---|
| Morning (3–4 hrs) | Build the **HOS calculation engine** as a standalone Python module first (pure functions, no Django yet — easier to test). Logic to encode: <br>1. Get total trip duration/distance from map API. <br>2. Subtract "Current Cycle Used" from 70 to get hours available before reset needed. <br>3. Walk through the trip hour-by-hour (or leg-by-leg): insert a 30-min break after every 8 cumulative driving hours; cap any single duty window at 14 hours / 11 hours driving; once 70-hr cycle limit is hit, force a 10-hr (or 34-hr restart) off-duty block before resuming. <br>4. Insert fuel stops every ≤1,000 miles. <br>5. Insert 1-hr pickup and 1-hr dropoff blocks at start/end. <br>**Output of this module:** an ordered list of "events" (drive segment, break, fuel stop, sleeper/off-duty block) each with start time, end time, duty status, and location label. |
| Midday | Break |
| Afternoon (3–4 hrs) | Wrap this module in Django: `POST /api/plan-trip/` view that calls the map API for geocoding + route, runs your HOS engine on the result, and returns: route geometry (for the map), stop markers, and the ordered duty-status event list (this event list is what the frontend will use to draw the log sheets). Test with Postman/curl/Thunder Client using a few different "Current Cycle Used" values (0 hrs, 40 hrs, 65 hrs) to make sure your rolling-limit logic actually triggers restarts/breaks correctly. |
| Evening (~1 hr) | Write 3–4 manual test cases by hand (short trip, no breaks needed; medium trip, one 30-min break; long multi-day trip needing multiple log sheets) and verify your API output against what you'd expect on paper. This is your accuracy insurance — do this *before* you start the frontend. |

**End of Day 1 checkpoint:** Backend API takes the 4 inputs and returns a correct, testable JSON of route + duty events. This is the riskiest, most gradable part — don't move on until it's solid.

---

## Day 2 — Sun, June 21 (full day, ~7–8 hrs)
**Theme: Frontend — map + form + scaffolding**

| Block | Focus |
|---|---|
| Morning (3 hrs) | Next.js project setup. Build the input form (current/pickup/dropoff/cycle-used) with clean validation and a loading state. Wire it to your Django API (remember: CORS + correct API base URL, use `.env.local` for the backend URL so deploy doesn't break later). |
| Midday | Break |
| Afternoon (3–4 hrs) | Integrate the map component (Mapbox GL JS / `react-map-gl`, or Leaflet if you went OSRM/OpenRouteService route). Plot the route line + markers for pickup, dropoff, fuel stops, and rest stops, with a small popup/label per stop. |
| Evening (1–2 hrs) | Build the **log sheet rendering component** — this is the visually distinctive deliverable, worth real UI/UX points. Use SVG (recommended) to draw the 4-row grid (Off Duty / Sleeper Berth / Driving / On Duty) matching `blank-paper-log.png`'s layout, then plot your duty-status events from Day 1's backend output as horizontal segments across the grid. Start with just rendering ONE day's log correctly before handling multi-day pagination. |

**End of Day 2 checkpoint:** You can submit the form, see a route on a map, and see at least one correctly-drawn log sheet rendered from real backend data.

---

## Day 3 — Mon, June 22 (full day — this is your deploy + polish + submit day)
**Theme: Multi-day logs, polish, deploy, record, submit**

| Block | Focus |
|---|---|
| Morning (2–3 hrs) | Handle **multi-day log sheet pagination** — split duty events across midnight boundaries, render one grid per 24-hr period, label each with date/total hours per duty status (matches the "Total Hours" column in the reference doc). |
| Midday (2 hrs) | **UI/UX polish pass** — this is explicitly weighted by graders. Clean typography, consistent spacing, a sensible color palette, mobile-reasonable layout, loading/error states, maybe a subtle animation on stop markers. Don't over-engineer; just make it look intentional and finished. |
| Afternoon (2 hrs) | **Deploy:** <br>• Frontend → Vercel (trivial with Next.js) <br>• Backend → Render or Railway (free tier; remember to set `ALLOWED_HOSTS`, env vars for the API key, and re-check CORS against your live Vercel domain, not just localhost). <br>• Smoke-test the *live* URL end-to-end — a broken hosted demo is an instant red flag for graders. |
| Evening (1–1.5 hrs) | **Record the Loom (3–5 min):** structure it as (a) quick demo of the live app working end-to-end with a real example trip, (b) brief code walkthrough — show the HOS engine logic and the log-sheet renderer, since those are your two hardest/most impressive pieces, (c) one sentence on assumptions/tradeoffs you made given time constraints. |
| Late evening | Push final commit, write a short README (setup steps, assumptions, tech choices, API key note), and **send the submission email** with GitHub link, hosted link, and Loom link. |

---

## Safety margin
You're targeting submission Monday night IST, leaving from Mon night → Tue ~5:44 AM IST (their deadline) as pure buffer for anything that breaks during deploy (CORS issues, env var mistakes, map API quota hits — all common last-mile gotchas). **Don't use this buffer as extra build time — treat it as insurance, not a 4th work day.**

## Things to NOT lose time on
- Don't try to perfectly implement every exotic HOS exception (sleeper berth splitting, adverse driving, short-haul exceptions) — the brief explicitly tells you to assume none of that applies. Stick to: 11hr driving / 14hr window / 30-min break / 70hr-8day / fuel-every-1000mi / 1hr pickup+dropoff.
- Don't hand-roll geocoding — use whatever your map API provides for address → lat/lng.
- Don't build user auth, accounts, or persistence beyond what's needed to render one trip's result — there's no requirement for saved trips/history.
