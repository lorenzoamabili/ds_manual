# Real-World Case Studies

How the techniques in this manual are applied at scale, in production, by companies whose results are public. Each case study extracts the **transferable lesson** — not just "Netflix does recommenders" but *what specifically they did, what broke, and what they learned*.

Organised by domain and by cross-cutting function:

## By domain

| Domain | Companies | Signature lesson |
|--------|-----------|-----------------|
| [Product Analytics](cs-product-analytics.md) | Netflix, Spotify, Duolingo | North Star metric ≠ engagement metric |
| [FinTech](cs-fintech.md) | PayPal, Stripe, Monzo | Precision matters more than recall in fraud |
| [Retail](cs-retail.md) | Amazon, Walmart, Instacart | Demand forecasting at item × store × day grain |
| [HealthTech](cs-healthtech.md) | Google, NHS, sepsis alerts | Deployment ≠ outcome; silent model failures |
| [Manufacturing](cs-manufacturing.md) | Siemens, GE, BMW | Sensor lag + multivariate failure signatures |
| [MarTech](cs-martech.md) | Airbnb, Booking.com, LinkedIn | Experimentation culture beats model sophistication |
| [Energy](cs-energy.md) | DeepMind, Ørsted, Tesla | Safety constraints in ML-controlled systems |
| [Cybersecurity](cs-cybersecurity.md) | Darktrace, Cloudflare, Okta | Adversarial drift makes static models obsolete |
| [HRTech](cs-hrtech.md) | Google, IBM, LinkedIn | Predictions that change outcomes (Goodhart's law) |

## By function

| Function | Case study |
|----------|-----------|
| [Causal inference at scale](cs-causal.md) | Uber, Microsoft, Spotify |
| [NLP in production](cs-nlp.md) | Gmail, Grammarly, Duolingo |
| [Forecasting at scale](cs-forecasting.md) | Uber, Meta, DoorDash |

---

> Each case study follows the same structure: **Problem → Approach → Key technical decisions → What failed first → Transferable lesson.** The lesson is the point — the company is just the evidence.
