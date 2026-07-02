# Cybersecurity Case Studies

## Darktrace — unsupervised anomaly detection for enterprise threats

**Problem:** Signature-based security (antivirus, IDS rules) detects known threats. Novel threats — zero-days, insider threats, living-off-the-land attacks using legitimate tools — have no signature. Darktrace's core claim: detect threats from *behavioural deviation*, not signatures.

**Approach:** **Bayesian, unsupervised anomaly detection** on network traffic logs. For each device and user, a baseline of "normal" behaviour is built: typical connection patterns, volume, port usage, timing, peer relationships. An anomaly score measures how surprising the current behaviour is relative to the personalised baseline.

**Key technical decisions:**
- **Per-entity baselines** (not population baselines): the fact that an executive's laptop connects to an unusual IP at 3am is anomalous *for that device* even if 3am connections are common across the fleet. Personalised baselines eliminate the base rate problem that afflicts population-level anomaly detection. See [doc 13](../13-anomaly-detection.md).
- **Bayesian updating**: the baseline is not a static snapshot — it updates as the device's usage pattern changes legitimately (a developer who starts using a new cloud service). The model distinguishes *adaptation* (gradual normal change) from *intrusion* (sudden abnormal change) by the rate and context of the deviation.
- **Alert suppression / correlation**: raw anomaly scores are noisy. Darktrace correlates anomalies across devices, time, and network paths to identify *chains of suspicious activity* — the pattern of behaviour that constitutes an attack, not an isolated anomalous event.

**What failed first:** Early deployments produced alert volumes that overwhelmed security teams (the same alert fatigue problem as clinical early-warning systems). The raw anomaly score threshold that catches true positives also fires on normal but unusual events (new SaaS tools, team travel, system updates). They introduced a *context-aware suppression* layer: anomalies in time windows around known software deployments or scheduled maintenance are suppressed.

**Transferable lesson:** In adversarial environments, anomaly detection must be *contextual* — the same behaviour is suspicious in one context and normal in another. A threshold on a raw anomaly score without context is a false alarm machine.

---

## Cloudflare — bot detection at 55 million requests/second

**Problem:** Cloudflare processes ~55 million HTTP requests per second. A significant fraction are bots — some benign (search crawlers), some malicious (credential stuffing, scraping, DDoS). Blocking legitimate users by mistake (false positive) is immediately visible; allowing bad bots through harms Cloudflare's customers.

**Approach:** A **real-time ML scoring pipeline** at the CDN edge. Features are computed from the HTTP request itself (headers, TLS fingerprint, timing, JavaScript challenge result) and from device/IP reputation history. A gradient boosting model scores each request in <5ms.

**Key technical decisions:**
- **TLS and HTTP/2 fingerprinting** (JA3, HTTP/2 fingerprint): legitimate browsers have characteristic patterns in their TLS handshake and HTTP/2 settings that are hard to fake. Bots using automation frameworks (Selenium, Puppeteer) have distinct fingerprints even when they try to mimic browsers.
- **Adversarial adaptation**: bot operators actively probe Cloudflare's detection and update their tooling to evade it. This creates a cat-and-mouse game where the model's feature importances must be kept partially opaque and rotated. Publishing feature importance would hand the adversary a bypass guide.
- **Threshold at the request level**: Cloudflare doesn't block IPs (too coarse, shared IPs hit legitimate users). Instead, it issues challenges (CAPTCHA, JavaScript proof-of-work) to suspicious-scoring requests and observes the response — this is a **sequential decision** with the challenge response as a second-stage signal.

**What failed first:** Feature-based detection failed against "headless Chrome with human-mimicking mouse movements" bots that pass JavaScript challenges. Cloudflare moved to *behavioural sequences*: a human's browsing session has characteristic inter-request timing, navigation patterns, and mouse movement physics. A bot navigating a shopping cart in 0.3 seconds fails the behavioural sequence check even if it passes individual request checks.

**Transferable lesson:** Adversarial ML requires treating the adversary as an active agent who reads your output signal and adapts. Static models deployed against adaptive adversaries decay rapidly. Rotate features, use opaque signals, and add sequential/behavioural features that are harder to mimic.

---

## Okta — identity threat detection (UEBA)

**Problem:** Okta provides identity and access management for 17,000+ organisations. **User and Entity Behaviour Analytics (UEBA)**: detect when a legitimate user's account is compromised — the attacker authenticates with valid credentials, so signature-based security doesn't fire. The signal is *behavioural deviation* from the user's normal patterns.

**Approach:** A per-user **risk score** updated on every authentication event. Features: login time (is this the user's typical hour?), location (is this country/city normal for this user?), device (is this a new device?), access pattern (accessing resources this user never touches?), velocity (three logins from three countries in an hour — impossible travel).

**Key technical decisions:**
- **Impossible travel detection**: if a user authenticates from London at 09:00 and from São Paulo at 09:15, that's physically impossible. This heuristic is a near-zero false-positive signal that can immediately trigger a step-up authentication challenge or block.
- **Contextual risk scoring**: a single anomalous signal (new device) is lower risk than multiple simultaneous signals (new device + new country + 3am + sensitive resource access). The risk model combines signals multiplicatively (Bayesian-style) rather than additively.
- **Adaptive authentication**: rather than blocking outright (high false positive cost), Okta triggers **step-up authentication** (push notification, TOTP) when the risk score is elevated. The step-up challenge is itself a signal — a compromised account where the attacker doesn't have the second factor will fail the step-up.

**What failed first:** The location-based anomaly model initially used IP-to-country geolocation, which has high error rates for VPN users (a user in Germany using a US VPN appears to be in the US). Corporate VPN usage was flagged as anomalous travel. They added corporate VPN IP ranges as a "trusted location" override and improved the IP enrichment vendor.

**Transferable lesson:** Geolocation from IP is unreliable. Any system that uses IP-based location features must explicitly handle VPN, proxy, and corporate network cases — they are common enough to materially degrade precision.

---

## Cross-cutting lessons

1. **Per-entity baselines** outperform population baselines in security contexts where individual behaviour is the signal.
2. **Alert fatigue** is the primary deployment failure mode in security ML, identical to clinical early-warning systems.
3. **Adversarial adaptation** makes security ML a moving target. Features that are opaque and hard to mimic (TLS fingerprints, behavioural sequences) are more durable than inspectable features.
4. **Step-up authentication** (challenge, don't block) is a dominant strategy for identity risk: lowers false positive costs, adds a second-stage signal, and is safer than outright blocking on noisy features.
