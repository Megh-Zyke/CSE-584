# TriGuard Semantic Cache - Final Presentation Outline

**Target Length:** 16-20 Slides  
**Project:** TriGuard Semantic Cache (A 4-Stage Semantic Cache for GenAI APIs)  

---

### **Section 1: Introduction & Problem Definition (Slides 1-3)**
*Goal: Hook the audience and clearly define what you are trying to solve and why it matters.*

* **Slide 1: Title Slide**
  * Project Title: *TriGuard Semantic Cache: Uncertainty-Aware Verification for GenAI APIs*
  * Team Members & Roles
  * Date & Course Number
* **Slide 2: The Problem Statement**
  * LLM APIs (like GPT-4, Gemini) are relatively slow and expensive.
  * Exact-match caching (like Redis) fails because users ask the *same* question using *different* words (e.g., "What is the capital of France?" vs. "Which city serves as France's capital?"). 
* **Slide 3: Why Investigate This Problem? (Motivation)**
  * **Cost Impact:** Redundant queries multiply billing costs at scale.
  * **Latency Bottleneck:** Repetitive external API calls degrade application speed.
  * **The Challenge of Semantic Caching:** Aggressive semantic caching creates a risk of serving inaccurate or hallucinated historic responses if the cache mistakenly pairs slightly contradicting questions. 

---

### **Section 2: Background & Prior Works (Slides 4-5)**
*Goal: Provide academic/industry context and expose the limitations of existing solutions.*

* **Slide 4: Prior Works & State-of-the-Art**
  * Overview of **GPTCache** & **MeanCache** (Static Cosine Similarity thresholding).
  * Overview of **SISO / LangCache** (Dynamic caching, basic semantic LFU).
* **Slide 5: Limitations of Prior Works**
  * **Lack of Temporal Awareness:** Existing caches store volatile queries (e.g., "What's the stock of AAPL today?") alongside static ones ("What is Python?"), leading to stale data.
  * **Threshold Vulnerability:** Relying purely on embedding vector *similarity* often fails to capture minor *contradictions* (e.g., "Increase dosage" vs "Decrease dosage" are vector-similar but contextually opposite).

---

### **Section 3: Our Solution & Innovation (Slides 6-11)**
*Goal: Deep dive into the TriGuard Architecture. This is the bulk of the "Expanded Solution" requirement.*

* **Slide 6: Proposed Solution: The TriGuard Architecture**
  * High-level pipeline diagram showing the complete end-to-end journey of a query.
  * Introduction of the 4 key stages: Context Normalization, Dynamic TTL, Multi-Gate Verification, and Dual SLM Fallback.
* **Slide 7: Stage 1 - Context Normalization (Gate 1)**
  * **Mechanism:** Explaining the FLAN-T5 model integration.
  * **Purpose:** Resolving ambiguous pronouns from conversation history (e.g., translating "how long did it last?" to "how long did the French Revolution last?").
* **Slide 8: Stage 2 - Dynamic TTL Classification**
  * **Mechanism:** Logistic Regression classifier based on BGE-Small embeddings.
  * **Purpose:** Classifying queries into `Static` (∞), `Slow-Moving` (30 days), and `Volatile` (5 mins) ensuring live data isn't permanently cached.
* **Slide 9: Stage 3 - Fast Path & Semantic Vector Search**
  * How Redis handles exact hits for immediate response.
  * How ChromaDB handles cosine dense retrieval for semantic candidates.
* **Slide 10: Stage 4 - PyFS Contradiction & Reranking (The Core Innovation)**
  * Introduction to **Pythagorean Fuzzy Sets (PyFS)**.
  * **Mechanism:** Why evaluating *Non-Membership (nu)* and *Hesitancy (pi)* using NLI models is better than classic similarity scores.
* **Slide 11: Stage 5 - Dual SLM Verification (Gate 3)**
  * **Mechanism:** Using an async local SLM (Qwen via Ollama).
  * **Purpose:** The final check verifying both **Faithfulness** and **Confidence** before a new Gemini API response is legally admitted into the Cache. 

---

### **Section 4: Experiments, Analysis & Results (Slides 12-18)**
*Goal: Rigorously prove your solution works better than the baseline through data and metrics.*

* **Slide 12: Experimental Setup & Datasets**
  * What datasets were used (training the TTL classifier, evaluating PyFS hits).
  * The synthetic testing environment configuration (Locally hosted vs Cloud Gemini API).
* **Slide 13: Experiment 1 - Cache Hit Rates vs. Accuracy**
  * **Metric:** TriGuard's accuracy compared to standard baseline Vector Databases.
  * Analysis of false-positive avoidance (how PyFS caught the contradictions that naive Cosine Similarity missed).
* **Slide 14: Experiment 2 - Latency Profiling**
  * **Metric:** Time saved per query. 
  * Comparison graphs: Redis Fast-Path (~5ms) vs Chroma+PyFS Full Path (~250ms) vs Gemini API Miss (>1.5s). 
* **Slide 15: Experiment 3 - The Value of Context Normalization (Gate 1)**
  * Comparing Cache Hit Rates with and without the FLAN-T5 normalization.
  * Example of a lost multi-turn context successfully retrieved. 
* **Slide 16: Analysis - Dynamic TTL Efficiency**
  * Evaluating the precision and recall of the Temporal category classifier.
  * Demonstrating how the fallback logic protected the system from serving stale "Volatile" data.
* **Slide 17: Analysis - Cost Reduction & SLM Overhead**
  * Calculating the theoretical API cost ($) saved per 10k queries modeled on your hit-rate.
  * Discussion on the computing overhead of the Qwen SLM and how doing it *asynchronously* preserved UX.
* **Slide 18: Demonstration / End-to-End Walkthrough**
  * Quick trace of a real query going through the system output logs (Missing cache → Getting Gemini Response → Passing Gate 3 → Hitting Cache perfectly on a rephrased retry).

---

### **Section 5: Conclusion (Slides 19-20)**
*Goal: Summarize takeaways clearly.*

* **Slide 19: Key Takeaways & Learnings**
  * Similarity does not equal correctness; contradiction detection is required for generative caches.
  * Multi-layer asynchronous caching allows aggressive optimization without affecting user-perceived load times.
* **Slide 20: Q&A**
  * Link to repository.
  * Floor opens for questions.
