where can we find reliable source of APT's

10:07 AM

Curated accessible APT sources for dual project applications
Curated accessible APT sources for dual project applications





Here are the best sources, organized by how useful they'd be for CTOC integration:

Tier 1 — Free, API-accessible, perfect for your pipeline

MITRE ATT&CK Groups (attack.mitre.org/groups/) — the gold standard. Groups are mapped to publicly reported technique use with original references, associated software, and attributed campaigns. 
MITRE
 You're already pulling the STIX bundles for techniques — the same data includes full group profiles with TTPs mapped per actor. This is what the War Games simulator should be pulling from.

MISP Threat Actor Intelligence Server — has a free public API at https://www.misp-project.org/tai/ that you can query directly. For example: curl -d '{"name":"APT34"}' -H "Content-Type: application/json" -X POST https://www.misp-project.org/tai/query 
GitHub
 returns full MISP galaxy format with descriptions, synonyms, suspected state sponsors, target sectors, and CFR references. No API key needed.

MISP Galaxy threat-actor.json (GitHub) — the raw JSON database behind MISP's threat actor galaxy, downloadable directly from github.com/MISP/misp-galaxy/blob/main/clusters/threat-actor.json. 
GitHub
 This is the motherlode for a local database — structured JSON with descriptions, country attribution, synonyms, references, and UUIDs you can cross-reference with MITRE ATT&CK.

ETDA Threat Group Cards — 504 groups listed (416 APT, 54 other, 34 unknown) with the entire actor database downloadable in JSON or MISP format. 
Etda
 This is at apt.etda.or.th and is one of the most complete open catalogs available.

Tier 2 — Excellent reference, some API access

Palo Alto Unit 42 Threat Actor Tracker — tracks groups by country attribution with extensive alias cross-referencing. For example, they map APT29 to at least 20 aliases: Midnight Blizzard, Cozy Bear, Dark Halo, Nobelium, UNC2452, and more. 
Palo Alto Networks
 Not a direct API, but the page is structured and scrapeable.

Google/Mandiant APT Groups (cloud.google.com/security/resources/insights/apt-groups) — deep profiles from Mandiant's incident response work, including target sectors, attack vectors, and associated malware for each numbered APT group. 
Google Cloud

Tier 3 — Aggregators and visual tools

APTMAP — an interactive map sourcing from MISP, MITRE, ETDA, and VX-Underground. It lets you filter APT groups by name, geography, targeting, TTP, sponsor, and motivation. 
Andreacristaldi
 Good for visual reference but as they note themselves, attribution from a single incident analysis can be unreliable.

SOCRadar Threat Actor Profiles — detailed dark web profiles of active groups including ransomware operators like Scattered Spider, Beast, and Lynx. 
SOCRadar

For CTOC integration specifically, I'd recommend this approach: pull the MISP threat-actor.json from GitHub monthly (same schedule as your other feeds), store it in MongoDB as a threat_actors collection, and use the MISP TAI API for real-time lookups when enriching incoming OTX/NVD data. That gives you actor names, aliases, country attribution, target sectors, and MITRE technique mappings — all the data the War Games simulator is currently hardcoding. Want me to spec out that feed connector?

which ones would be good for training data?

10:08 AM

Evaluated optimal APT data sources for LLM training pipeline
Evaluated optimal APT data sources for LLM training pipeline


Great question — training data for the MDR fine-tune has different requirements than live feeds. You need structured, detailed, and diverse content that can be transformed into instruction-tuning pairs. Let me look at what's actually downloadable and rich enough.


Synthesized practical APT sources for MDR model training
Synthesized practical APT sources for MDR model training




Good question — for the MDR fine-tune on the Spark you need sources that are structured, downloadable in bulk, and rich enough to generate diverse instruction-response pairs. Here's how I'd tier them:

Tier 1 — Direct training data (already in instruction/QA format)

CyberLLMInstruct — 54,928 instruction-response pairs spanning malware analysis, phishing simulations, and zero-day vulnerabilities, sourced from CTF challenges, academic papers, industry reports, and CVE databases. 
arXiv
 This is ready-to-go JSONL. The catch is fine-tuning on it can compromise safety resilience — Llama 3.1 8B's security score against prompt injection dropped from 0.95 to 0.15 after training 
arXiv
, so you'd want to mix it with safety-aligned data.

Primus — a comprehensive suite covering all major training stages including pretraining, instruction fine-tuning, and reasoning distillation with cybersecurity-specific self-reflection data. Continual pre-training on their dataset yielded a 15.9% improvement on aggregate cyber benchmarks, and reasoning distillation gave a 15.8% gain on CISSP certification tasks. 
arXiv
 This is probably the most complete open-source cyber training package available right now.

SecKnowledge / CyberPal.AI — a domain-knowledge-driven instruction dataset built from years of accumulated expert knowledge. They pulled from SIEM rules (400 QRadar rules with TTP mappings), detection rules, MITRE ATT&CK relationships, and structured chain-of-thought paths. Their fine-tuned models showed up to 24% improvement over baselines. 
arXiv
 The CoT approach is particularly valuable for MDR — it teaches the model to reason through detection-to-attribution chains, not just answer trivia.

Tier 2 — Rich structured data you'd transform into training pairs

MITRE ATT&CK STIX Data (github.com/mitre-attack/attack-stix-data) — the full knowledge base in STIX 2.1 JSON including groups, techniques, software, campaigns, and all the relationships between them. 
GitHub
 The October 2025 v18 release added Detection Strategies and Analytics as new object types 
MITRE
, which is gold for MDR training. You'd write a pipeline to generate pairs like:

"What techniques does APT28 use for initial access?" → structured answer from group→technique relationships
"Given this log pattern, which MITRE technique is this?" → detection strategy → technique mapping
"What mitigations apply to T1566 Phishing?" → technique → mitigation relationships
The mitreattack-python library makes this trivial — MitreAttackData("enterprise-attack.json") gives you methods like get_software_used_by_groups(), get_techniques_used_by_group(), and full relationship traversal. 
Readthedocs

MISP Galaxy threat-actor.json — the raw JSON database with descriptions, country attribution, synonyms, suspected state sponsors, target sectors, and CFR references for every tracked group. 
GitHub
 Cross-reference this with ATT&CK to generate attribution-style training pairs: "An attack uses PlugX malware targeting Mongolian NGOs via spear-phishing. Which threat actor is most likely responsible?"

ETDA Threat Group Cards — 504 groups (416 APT) downloadable in JSON or MISP format. 
Etda
 More breadth than ATT&CK alone, especially for less-documented groups.

Tier 3 — Supplementary/enrichment

HackMentor — a cybersecurity LLM fine-tuning project with data construction tools for building domain-specific instruction and conversation datasets, plus released LoRA weights you could study for architecture decisions. 
GitHub

Fine-Tuning-LLMs-for-Cyber-Threat-Intelligence (GitHub) — includes structured threat intelligence reports with attack characteristics, techniques used, attack patterns, and IOCs, plus CAPEC attack pattern data with IDs, abstraction levels, and status fields. 
GitHub
 Good for the incident-report-to-structured-analysis training pairs.

My recommended pipeline for your DGX Spark monthly fine-tune:

Base dataset: Primus pretraining corpus + CyberLLMInstruct instruction pairs (~55K)
ATT&CK synthetic pairs: Script that pulls enterprise-attack-18.1.json, traverses all group→technique→detection→mitigation relationships, generates ~20-30K QA pairs monthly
Live threat feed pairs: Your existing OTX/NVD/CISA KEV ingest generates real-world examples — transform each month's alerts into "analyze this IOC" / "map this CVE to ATT&CK" instruction pairs
MISP actor enrichment: Cross-reference MISP galaxy actors with ATT&CK groups for attribution training pairs
Safety mix: Blend in 10-15% general instruction-following data to prevent the safety degradation CyberLLMInstruct authors documented
That gives you a mix of static expert knowledge, structured relationship reasoning, and fresh monthly threat context — exactly what an MDR analyst model needs. Want me to scaffold the dataset generation pipeline?