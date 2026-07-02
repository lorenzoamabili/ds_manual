# NLP in Production Case Studies

## Gmail — Smart Reply and the latency vs. quality trade-off

**Problem:** Gmail Smart Reply suggests short email replies ("Sounds good!", "Thanks, I'll look into it."). The model must: (1) generate contextually appropriate suggestions, (2) run in <100ms on mobile devices, (3) not generate embarrassing or offensive suggestions at scale (billions of sends/day), (4) reflect the *recipient's* writing style, not a generic voice.

**Approach:** The production Smart Reply model is a two-stage system: (1) an **intent classifier** selects from a finite set of ~20,000 pre-defined response candidates using a lightweight model; (2) a **response generator** fills in stylistic variation. The original 2016 paper used an LSTM seq2seq; current versions use a distilled [transformer](../10-nlp-and-llms.md) with speculative decoding for latency.

**Key technical decisions:**
- **Latency constraint → finite candidate set**: a full generative model at billions of requests is too slow. The practical solution (still in production) is a retrieval + light generation pipeline, not pure generation. This is a *systems constraint* that overrides the academically "best" approach.
- **Semantic diversity**: if all three suggestions mean the same thing, users find them useless. Google enforces semantic diversity across the three displayed suggestions by [clustering](../06-unsupervised-learning.md) candidates by intent and sampling one from each cluster.
- **Toxicity and sensitive topic suppression**: a model trained on real email will learn to suggest phrases containing personal information, slurs, or confidential content from training examples. Google applies a post-processing filter (a toxicity classifier + PII detector) to all candidates before display.
- **Personalisation**: Smart Reply learned that people's communication styles differ. A formal business email should suggest "I'll review this shortly" not "sounds great!" — context is modelled from the email thread's register (formal/informal NLP classifier).

**What failed first:** Early versions would suggest "I love you" in a business context because training data contained many personal emails with this phrase. Filtering by email context (business domain vs. personal) was added. This is a **distribution mismatch** — the model optimises for all email but must be safe in all email, including the high-stakes business subset.

**Transferable lesson:** NLP at consumer scale requires explicit safety layers (toxicity, PII, sensitive topic classifiers) applied post-generation. The generative model cannot be trusted to self-[censor](../16-survival-analysis.md), regardless of [fine-tuning](../11-computer-vision.md). Defense-in-depth: train, fine-tune, AND post-filter.

---

## Grammarly — writing assistance and the style vs. correctness boundary

**Problem:** Grammarly must distinguish *incorrect* writing (grammar errors, spelling mistakes) from *intentional style* (short sentences for emphasis, passive voice for formal academic writing, comma splices in dialogue). A grammar assistant that "corrects" stylistic choices is a worse product than one that stays in its lane.

**Approach:** A **multi-task transformer** (fine-tuned on a combination of grammar error corpora, style guides, and domain-specific writing samples) that classifies suggestions into categories: correctness (high-confidence fix), clarity (rewrite suggestion), engagement (vocabulary enrichment), and delivery (tone adjustment). Each category has a different confidence threshold and is presented differently in the UI.

**Key technical decisions:**
- **Domain-specific fine-tuning**: academic writing, legal documents, casual social media, and business email have different style norms. Grammarly fine-tunes per-domain and gates suggestions by detected document domain. A passive voice suggestion is suppressed for academic writing (where it's standard) but surfaced for business writing (where active voice is preferred).
- **Confidence [calibration](../04-evaluation-and-validation.md)**: a suggestion shown to a user must be right. Grammarly reports that their precision target on shown suggestions is very high (suppressing more to avoid false suggestions). This means recall (catching all errors) is sacrificed to protect precision. See [doc 04](../04-evaluation-and-validation.md).
- **User feedback loop**: users can accept, reject, or ignore suggestions. Rejected suggestions are a training signal (the model was wrong, or the user preferred their original). This creates an **active learning pipeline** where high-rejection suggestions are reviewed and used to fine-tune the model.

**What failed first:** The first transformer model was trained on general web text and performed well on grammatical correctness but poorly on contextual appropriateness — it would suggest "correcting" dialogue in fiction writing to "standard" grammar, ruining the character's voice. Domain detection was added as a prerequisite to suggestion generation.

**Transferable lesson:** Writing assistance is two problems: grammatical correctness (objective) and stylistic guidance (subjective, domain-dependent). Mixing them without domain conditioning produces a model that confidently gives wrong advice. The triage between "error" and "choice" requires domain-aware classification.

---

## Duolingo — generative AI for language exercise creation

**Problem:** Duolingo's courses require thousands of hand-crafted exercises per language. Creating exercises for lower-resource languages (Swahili, Navajo, Welsh) is bottlenecked by the availability of curriculum writers with the target language. GPT-4 can generate grammatically correct, culturally appropriate exercises at scale — but hallucination and cultural insensitivity must be caught before reaching learners.

**Approach:** A **human-in-the-loop generation pipeline**: GPT-4 generates exercise candidates; a validation model (fine-tuned on Duolingo's quality rubric) scores each candidate; high-scoring candidates go to a queue for human review; human reviewers approve, edit, or reject. The human reviewer's action is a training signal for the validation model.

**Key technical decisions:**
- **Rubric-based validation**: Duolingo defines a structured quality rubric (grammatical accuracy, cultural appropriateness, pedagogical value, difficulty calibration). The validation model is fine-tuned to score each dimension, not just an overall quality score. This makes failures diagnosable — a candidate that fails grammatical accuracy is a different problem from one that fails cultural appropriateness.
- **Low-resource language handling**: GPT-4's quality degrades for low-resource languages (less training data). For these, the validation model's threshold is more conservative (more human review), and the generation prompt includes explicit examples in the target language.
- **Hallucination detection**: LLMs hallucinate facts. An exercise that claims "the capital of Australia is Sydney" teaches learners wrong information. A factual validation step (entity-linked to Wikidata) checks claims in generated exercises.

**Transferable lesson:** [LLM](../10-nlp-and-llms.md)-generated content at consumer scale requires a **validation layer** (a separate model or human review queue) between generation and display. LLMs cannot self-certify their output quality. The cost of a human reviewer is justified by the cost of learners learning incorrect content at scale.

---

## Cross-cutting lessons

1. **Latency constraints often determine architecture**: the academically best NLP model is not always the production model. Retrieval + light generation, distillation, and speculative decoding are standard production patterns.
2. **Safety layers are not optional**: post-generation filtering (toxicity, PII, factual validation) must be applied to all generative NLP outputs at consumer scale.
3. **Domain conditioning**: NLP models trained on general text perform poorly on domain-specific corpora. Fine-tune per domain; gate suggestions by detected domain.
4. **Human-in-the-loop** pipelines (active learning, generation + review) are the standard pattern for high-precision NLP in production, not fully automated generation.
