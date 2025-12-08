# Build–Measure–Learn (B–M–L) Planning Document — Dukhtar







**Project summary (short)**
- Dukhtar is an AI-powered pregnancy and women's health assistant built on Flask. The MVP offers user authentication, a personalized pregnancy guide generator (week-by-week), an AI chat assistant (text + voice), image analysis for medical documents, and doctor/consultation pages. Core AI pipeline uses web search (Tavily), web scraping, LangChain-style retrieval, OpenAI embeddings, a Chroma vector store, and a Chat LLM to synthesize personalized guidance.



**A. Measure Phase — Actionable Metrics**

Goal: Collect data from real users interacting with the deployed MVP to validate primary product/impact hypotheses and make concrete product decisions (retain, pivot, or iterate).

Top-level hypotheses (examples to be validated):
- H1 (Value): Users receive immediately useful, personalized weekly pregnancy guidance and return for follow-ups.
- H2 (Usability): The chat & pregnancy guide flows are discoverable and easy to complete for expecting mothers in our target markets (English/Urdu-speaking users).
- H3 (Safety/Trust): Retrieval-augmented LLM answers with source context are judged trustworthy by qualified reviewers at an acceptable rate.

Three Actionable Quantitative Metrics (embedded in MVP):

1) Metric: Guide Completion Rate (GCR)
- Definition: Percent of initiated pregnancy guide requests that result in the user viewing the generated `pregnancy_results` page and optionally saving/exporting the guide.
- Event instrumentation: Emit `guide_initiated` when user submits the tracker form; emit `guide_generated` when server successfully generates and returns a guide; emit `guide_saved` when user clicks save/export; include fields: `user_id` (nullable), `pregnancy_week`, `language`, `generation_time_ms`, `sources_count`, `guide_id`.
- Why actionable: Low GCR indicates friction in input form, API failures, or content irrelevance. Action: If GCR < 50% after 200 events, investigate server errors/latency, introduce intermediate progress UI, and run usability tests.
- Validation threshold: Success if GCR >= 65% and median `generation_time_ms` < 6s on the backend for 200 unique sessions.

2) Metric: First-Response Helpfulness (FRH) — short-term retention signal
- Definition: After the first chat or guide generation, the percentage of users who perform a meaningful follow-up action within 7 days (return to chat, request another guide, or book a consultation).
- Event instrumentation: `first_interaction_timestamp`, `followup_action` with values {`chat_return`, `new_guide`, `book_consultation`} and `delta_days`. Track unique user cookie/session id for anonymous users.
- Why actionable: Measures whether immediate outputs drive repeat use. Action: If FRH < 20% after N=150 new users, prioritize improving content relevance (prompting), adding “save & remind” features, and tailoring follow-up nudges.
- Validation threshold: FRH >= 25% indicates product-market signals worth scaling A/B experiments.

3) Metric: Clinical Content Fidelity (CCF)
- Definition: Fraction of generated medical recommendations (diet, warning signs, medication advice prompts) that pass expert review or automated heuristics. Tracked via two signals: (1) external clinical review sample scoring, (2) automatic safety checks triggered per generation.
- Instrumentation: Flag each generated guide with `contains_medical_advice` boolean and `safety_checks_passed` boolean. Send a daily sample (random 5–10 guides) to a panel of qualified clinicians for annotation. Store `review_id`, `review_score` (0-100), and `issues_detected` tags.
- Why actionable: Directly informs whether model outputs are clinically safe and acceptable. Action: If CCF < 85% on clinician reviews, restrict the feature (read-only with disclaimers), adjust prompts to be more conservative, add stronger retrieval grounding, or require clinician moderation.
- Validation threshold: CCF >= 90% on sampled reviews supports broader rollout; 85–90% requires targeted prompt/tooling improvements.

Instrumentation & telemetry implementation notes
- Use server-side event logging (structured JSON) and a lightweight event collector (e.g., Postgres audit table + periodic ETL to analytics or a small MDS like Metabase/Redash). Event schema examples available in the repo's analytics layer (create `analytics.events` table).
- For anonymous sessions, use signed session IDs to map events cross-request. Persist `session_id` in cookie and as part of each event.
- Collect performance metrics (generation latency, API error rates) for correlation with user metrics.
- Ensure privacy: do not store PII in analytics streams; use user_id hash or null for anonymous.

---

**B. Measure Phase — Qualitative Metrics: 10+ Customer Discovery Interviews**

Objective: Capture in-depth user perceptions about usefulness, trust, and pain points after direct interaction with the MVP. These interviews are aimed to contextualize quantitative signals and produce hypotheses for the next build sprint.

Plan summary (high level)
- Recruit 12–20 participants (goal: 10 completed interviews) representing the target user segments: expecting mothers across trimesters, Urdu/English speakers, users of varying digital literacy, and one or two clinicians.
- Timing: Interviews conducted within 1–3 weeks post-MVP deployment, within 2–7 days of the user's first interaction (to ensure freshness).
- Format: 30–45 minutes remote calls (recorded with consent), semi-structured script, and a short follow-up survey. Offer a small stipend (e.g., gift card) for participation.

Recruitment & screening
- Recruitment channels: in-app prompt (non-intrusive), social media posts in target communities, partnerships with local clinics, and snowball sampling.
- Screening criteria: Age 18–45, currently pregnant or recently pregnant (within 6 months), has used the 'pregnancy guide' feature at least once. Include at least one clinician or skilled nurse.
- Target mix: 6 users (Urdu primary), 6 users (English primary), 2 clinicians.

Interview script & topics (semi-structured)
- Warm-up: Verify usage context and device used.
- Walkthrough: Ask participant to open the guide or recall the chat and walk through what they saw.
- Core questions:
  - What was your primary goal when you used Dukhtar?
  - Which parts of the guidance were most useful? Least useful?
  - Did anything surprise you or seem incorrect?
  - Did you trust the information? Why or why not?
  - How likely are you to use Dukhtar again? Why?
  - Did you feel the language and tone were culturally appropriate and readable?
  - If this advice recommended any action (e.g., contact doctor), would you follow it?
  - What features would make this more useful for you?
- Closing: Confirm contact permission for follow-up and offer the stipend.

Data capture & analysis
- Capture audio + notes, transcribe using Whisper or similar, and anonymize transcripts.
- Thematic analysis: two-person coding using affinity mapping, produce a 1-page synthesis with top 5 pain points and 5 suggestions.
- Use qualitative insights to: prioritize next features, adjust prompts and UX, and refine the quantitative validation thresholds.

Deliverables post-interviews
- Interview transcripts (redacted), interview synthesis, prioritized feature backlog derived from user quotes, and modifications to the instrumentation plan.

---

**C. Technical Depth & Complexity Statement (Detailed — Focused Component)**

Component: Retrieval-Augmented Personalized Pregnancy Guide Generation Pipeline

Overview
The most technically complex part of the MVP is the end-to-end system that synthesizes personalized, medically-informed pregnancy guidance for a user-supplied pregnancy week and personal profile. This pipeline must: (1) gather and fuse evidence from multiple external sources (Tavily search results and web scraping), (2) perform robust document processing and chunking, (3) create and manage high-quality embeddings, (4) build a retrieval layer (Chroma vector store) on demand, (5) run retrieval-augmented LLM prompting to generate the final guide, and (6) ensure safety, provenance, and acceptable latency.

Technical building blocks used in the MVP (as implemented in `app.py`):
- Multi-source information retrieval: Tavily API (search) plus targeted web scraping using `WebBaseLoader`.
- Text splitting: RecursiveCharacterTextSplitter for chunking long documents with overlap to preserve context.
- Embeddings: OpenAI embedding model `text-embedding-3-small` to map content into semantic vectors.
- Vector store: Chroma local persistence under `pregnancy_db/` for per-request/ephemeral vector stores.
- Retriever: `vectorstore.as_retriever(search_kwargs={"k": 8})` with k tunable.
- LLM: ChatOpenAI (GPT-family) with chain prompt templates in a RetrievalQA pattern.
- Safety & grounding: return_source_documents=True and `sources_count` metadata; prompt engineering to instruct conservative, evidence-based responses.

Complexities and why they are hard
1) Real-time data fusion and noise control
- Search results and web scraping return heterogeneous, inconsistent documents (different quality, structure, and potential contradictions). Normalizing and prioritizing high-quality clinical sources in real time requires careful ranking heuristics, duplication detection, and filtering.

2) Hallucination mitigation & provenance
- LLMs can produce plausible but incorrect medical statements. To reduce hallucination we must (a) ground answers with retrieved source snippets, (b) attach provenance metadata and confidence signals, and (c) ensure conservative phrasing for recommendations.

3) Latency vs. depth tradeoff
- Producing a high-quality guide requires running many searches, embeddings, and an LLM call. Keeping end-to-end latency acceptable for web usage (target median < 6s) requires asynchronous prefetching, per-user caching, and incremental result streaming.

4) Cost and model-choice constraints
- Using high-quality embeddings and LLMs increases per-query costs. The system must tune the chunk size, k, and embedding model to achieve acceptable quality/cost tradeoffs.

5) Safety, compliance, and scalability
- Medical content requires auditing and clinician-in-the-loop sampling. Scaling vector stores across many users involves distribution considerations (moving from local Chroma to managed vector DBs), sharding, and index life-cycle management.

Engineering techniques used to address complexity
- Retrieval augmentation: combine dense (embeddings) and sparse (search result metadata) retrieval to prioritize high-quality sources.
- Chunk overlap + semantic similarity deduplication to avoid repeated context and reduce misleading evidence.
- Prompt templates with explicit system role and safety guardrails (e.g., include "when to contact a clinician" sections and avoid prescriptive medication instructions).
- Sampling-based clinician reviews (daily sample) to monitor clinical fidelity.
- Cache/persist embeddings for common queries (e.g., well-known pregnancy weeks) to reduce repeated compute.
- Instrumentation: capture metrics about sources_count, generation_time_ms, and safety flags for automated monitoring.

Open research / iterative items beyond MVP
- Integrating a calibration model or verifier (e.g., a dedicated fact-checker LM or chain-of-reasoning verifier) to score claims.
- Replacing single-host Chroma with a distributed vector DB (e.g., Milvus, Pinecone) for large-scale sharding and replication.
- Semantic role-based prompt adapters for cultural/localization tuning (Urdu vs English tone and idioms).

---

**D. CCP Justification (Why this complex approach is necessary)**

Standard practice alternatives:
- Static, human-curated FAQs or rule-based pregnancy calculators.
- Simple prompt-only LLM replies without retrieval or provenance.

Why those are insufficient for Dukhtar's complexity:
1) Personalization requirement: Dukhtar delivers per-user advice that depends on BMI, week, activity levels, dietary restrictions, and language. Static FAQs cannot serve nuanced, individualized guidance.
2) Medical safety & provenance: Unverified LLM-only responses are prone to hallucination. Retrieval-augmented generation with source provenance is necessary to show evidence and to enable downstream clinician review.
3) Multilingual, multimodal needs: The product supports Urdu/English voice I/O and image analysis for prescriptions. This requires integrating speech-to-text, TTS, OCR and image pre-processing with the text pipeline — a cross-modal integration beyond simple chatbots.
4) Regulatory & trust constraints: Medical recommendations require auditable provenance and conservative phrasing. Retrieval + sampling-based clinician review provides the audit trail and continuous monitoring.
5) Conflicting constraints (latency, cost, accuracy): Delivering high-quality, safe guidance requires balancing LLM costs, retrieval depth, and response time. The end-to-end architecture (embeddings + retriever + LLM + safety layer) is designed to manage those trade-offs in a principled way.

Therefore, the retrieval-augmented, instrumented approach is necessary to meet product goals (personalization, trust, safety) and to enable concrete decisions from the Measure phase (e.g., tune k, choose caching, or limit rollout until CCF meets thresholds).


