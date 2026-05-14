"""Generate a high-coverage synthetic dataset for social bias detection.

Design goals:
- cover explicit + implicit + coded + contextual bias framings
- include social-media tones (LinkedIn and X/Twitter)
- avoid overfitting neutral to polished corporate grammar
- preserve clear training labels while storing richer subtype metadata
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

SEED = 42
RNG = random.Random(SEED)

# ---- Style controls ----
LEADS = [
    "",
    "hot take: ",
    "honestly ",
    "not saying this is always true, but ",
    "just being practical: ",
    "in my experience, ",
    "interesting how ",
    "let's be real, ",
]
TAILS = [
    "",
    " just saying.",
    " that is the pattern i keep seeing.",
    " curious if others noticed this.",
    " not trying to offend anyone.",
    " feels obvious at this point.",
]

# ---- Value banks ----
GENDER_A = ["women", "female candidates", "women in tech", "female professionals"]
GENDER_B = ["men", "male candidates", "men in tech", "male professionals"]
NAT_A = ["foreign workers", "immigrants", "international hires", "expat candidates", "overseas talent"]
NAT_B = ["locals", "domestic hires", "native-born workers", "local candidates"]
PROF_A = ["engineers", "designers", "product managers", "HR staff", "blue-collar workers", "junior developers"]
PROF_B = ["managers", "senior engineers", "executives", "consultants", "office workers"]
TASKS = ["admin tasks", "support coordination", "incident handling", "systems architecture", "client communication"]
ROLES = ["technical leadership", "architecture ownership", "critical production roles", "strategy decisions"]

FILLERS = {
    "g1": GENDER_A,
    "g2": GENDER_B,
    "n1": NAT_A,
    "n2": NAT_B,
    "p1": PROF_A,
    "p2": PROF_B,
    "task": TASKS,
    "role": ROLES,
}

# ---- Template bank by external label + internal subtype ----
TEMPLATE_BANK: Dict[str, Dict[str, List[str]]] = {
    "gender_bias": {
        "explicit_bias": [
            "{g1} are naturally better at {task} than {g2}.",
            "{g2} are more suited for {role} than {g1}.",
            "{g1} are too emotional for critical infrastructure decisions.",
            "{g2} handle technical pressure better than {g1}.",
            "{g2} outperform {g1} in leadership because they are more rational.",
        ],
        "implicit_bias": [
            "For {role}, i usually trust {g2} more than {g1}.",
            "I prefer hiring {g2} for high-stakes backend teams.",
            "When incidents escalate, {g2} tend to stay calmer than {g1}.",
            "For crunch-time ownership, {g2} are usually safer bets.",
            "I am not saying {g1} cannot do it, but {g2} seem more dependable here.",
        ],
        "subtle_linkedin": [
            "I look for a leadership profile that is steady under pressure, which in practice has been easier to find among {g2}.",
            "When the board asks for a 'safe pair of hands' for the transformation, the shortlist leans {g2}-heavy, fairly or not.",
            "Our talent partner suggested we be mindful of 'executive presence' in client rooms—code for some frustrating patterns in how we read {g1} and {g2} differently.",
        ],
        "coded_bias": [
            "not saying {g1} cannot lead, but some roles need a certain mindset.",
            "interesting how {g2} end up carrying the toughest technical decisions.",
            "just asking questions: why do {g2} dominate architecture ownership?",
            "culture fit for {role} often looks more like {g2}.",
            "if we're optimizing for stability, {g2} usually feel like the practical choice.",
        ],
        "contextual_bias": [
            "In high-pressure teams, {g2} should lead while {g1} focus on support work.",
            "For production outages, assign command to {g2} and coordination to {g1}.",
            "{g1} are great for communication, but {g2} should own deep systems work.",
            "Split responsibilities realistically: {g2} for core engineering, {g1} for people tasks.",
            "{g1} add value, just not usually in {role}.",
        ],
    },
    "nationality_bias": {
        "explicit_bias": [
            "{n1} work harder than {n2}.",
            "{n1} are better than {n2} at execution-heavy projects.",
            "Foreign workers are cheaper than locals and easier to manage.",
            "{n2} are less reliable than {n1} during tight deadlines.",
            "Candidates from {n2} underperform compared with {n1}.",
        ],
        "implicit_bias": [
            "We should prioritize local talent first before looking abroad.",
            "Global experience is nice, but local context matters more in hiring.",
            "I usually shortlist candidates with domestic work history first.",
            "Teams move faster when everyone already understands how things work here.",
            "For core roles, local candidates are often a lower-risk hire.",
        ],
        "subtle_linkedin": [
            "For this P&L remit, we need someone who 'gets' our regulatory culture out of the gate, which nudges the slate toward {n2}.",
            "Our clients expect crisp communication and zero friction—leadership is asking us to be thoughtful about {n1} vs {n2} trade-offs, even on paper.",
        ],
        "coded_bias": [
            "not saying international hires are bad, but culture fit is everything.",
            "interesting how imported talent often needs extra management overhead.",
            "just asking questions: why force global hiring when local talent exists?",
            "communication style mismatches can quietly hurt teams.",
            "we need people who 'get our way of working' from day one.",
        ],
        "contextual_bias": [
            "In customer-facing roles, {n2} can struggle more than {n1}.",
            "For fast execution, prioritize {n1} over {n2}.",
            "In this market, {n1} tend to be more practical hires than {n2}.",
            "When budgets are tight, {n1} are usually the smarter bet than {n2}.",
            "For leadership tracks, {n1} integrate better than {n2}.",
        ],
    },
    "profession_bias": {
        "explicit_bias": [
            "Engineers are socially awkward and poor communicators.",
            "Managers are incompetent and mostly create overhead.",
            "HR staff are not analytical enough for strategic decisions.",
            "Blue-collar workers are less intelligent than office workers.",
            "Junior developers usually slow teams down compared to senior developers.",
        ],
        "implicit_bias": [
            "I avoid putting engineers in client meetings when possible.",
            "Most managers add process noise instead of real value.",
            "For complex decisions, i trust engineering over HR every time.",
            "Junior hires are usually not worth the ramp-up cost.",
            "Office teams generally handle strategic thinking better than field teams.",
        ],
        "subtle_linkedin": [
            "For this initiative we need a commercial spine, not another deck—so the sponsor stack should skew toward {p2}, not {p1}.",
            "Nothing against {p1}, but in crisis weeks the exec team has learned to put {p2} in the room for the hard calls.",
        ],
        "coded_bias": [
            "not saying product managers are useless, but execution usually comes from engineers.",
            "interesting how leadership meetings get clearer when fewer non-technical voices join.",
            "just asking questions: do we really need this many managers?",
            "some roles are more optics than impact.",
            "certain departments are great at talking, not building.",
        ],
        "contextual_bias": [
            "{p1} are better than {p2} at practical decision-making.",
            "For high-impact work, lean on {p2} and keep {p1} in support lanes.",
            "{p1} can contribute, but {p2} should own final calls.",
            "{p2} outperform {p1} when accountability is strict.",
            "In crisis mode, trust {p2} over {p1}.",
        ],
    },
    "neutral": {
        "corporate_linkedin": [
            "Proud to share that our global hiring strategy focuses on role fit and long-term team outcomes.",
            "In our leadership forum we discussed how to make promotion criteria fairer across regions.",
            "We're expanding mentorship so earlier-career staff get structured sponsorship, not just ad hoc help.",
            "The operating review highlighted execution risks in Q3, not individual blame.",
            "Our employee resource groups are hosting a panel on pay transparency and career pathways.",
            "We refreshed interview panels to reduce groupthink while keeping bar high on craft.",
        ],
        "workplace_criticism_not_stereotype": [
            "The servant leadership model is over-discussed; managers are asked to be therapists, coaches, and therapists again.",
            "Transformational leadership theory makes great slides, but the day-to-day is still meeting-heavy bureaucracy.",
            "Middle management is not evil by default, but the org chart adds real drag when every decision needs a 'narrative'.",
            "We talk about red tape in government; our internal process bloat is the same class of problem.",
            "Leadership models borrowed from the 1990s do not address modern delivery pressure; the critique is of the *model*, not a demographic.",
            "Stakeholder storytelling is a skill; complaining about the narrative is not the same as stereotyping a team.",
        ],
        "clean_neutral": [
            "Teams perform better with psychological safety and clear goals.",
            "Engineers collaborate effectively across teams during release planning.",
            "Leadership requires clear communication and consistent feedback.",
            "Managers and engineers aligned on delivery milestones this quarter.",
            "Cross-functional collaboration helped resolve scope and quality trade-offs.",
            "The team improved reliability by tightening test coverage.",
        ],
        "messy_neutral": [
            "ngl the sprint was chaotic but we shipped and learned a lot.",
            "we had disagreements in planning, still landed on a workable roadmap.",
            "kinda messy week, but the team communicated better by friday.",
            "not perfect execution, but ownership was clearer this cycle.",
            "mixed feelings on velocity, still better handoffs than last month.",
            "team looked tired but still delivered core milestones.",
        ],
        "ambiguity_neutral": [
            "I tend to prefer candidates with direct domain experience for this role.",
            "This team might struggle with international communication styles if onboarding is weak.",
            "Sometimes local context matters more than broad global exposure.",
            "Not every strong coder is ready for client-facing work, and that's fine.",
            "Some roles need stronger structure, others need flexibility.",
            "Hiring is always trade-offs, not just one metric.",
        ],
        "inequality_discussion": [
            "Women are often assigned administrative tasks unfairly in many workplaces.",
            "Immigrants face barriers in hiring despite strong qualifications.",
            "People from minority nationalities report slower promotion cycles.",
            "Bias in feedback language can distort promotion outcomes.",
            "Local-experience filters can unfairly exclude qualified candidates.",
            "Underrepresented staff are often judged more harshly for the same behavior.",
        ],
    },
}


def _render(template: str) -> str:
    rendered = template.format(**{k: RNG.choice(v) for k, v in FILLERS.items()})
    lead = RNG.choice(LEADS)
    tail = RNG.choice(TAILS) if RNG.random() < 0.35 else ""
    text = f"{lead}{rendered}{tail}".strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _is_invalid(text: str) -> bool:
    lowered = text.lower()
    # Avoid simple same-side comparisons from random fills.
    disallowed = [
        "women are better than women",
        "men are better than men",
        "locals are better than locals",
        "immigrants are better than immigrants",
    ]
    return any(x in lowered for x in disallowed)


def _expand_bucket(
    public_label: str,
    subtype: str,
    templates: List[str],
    n: int,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen = set()
    attempts = 0
    max_attempts = n * 40

    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        text = _render(RNG.choice(templates))
        if _is_invalid(text):
            continue
        key = (text.lower(), public_label)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "text": text,
                "label": public_label,
                "subtype": subtype,  # Internal signal for analysis/debugging.
                "style": "x_like" if RNG.random() < 0.45 else "linkedin_like",
            }
        )

    if len(rows) < n:
        raise RuntimeError(
            f"Could not generate enough unique samples for {public_label}/{subtype}: {len(rows)}/{n}"
        )
    return rows


def _split_counts(total: int, buckets: int) -> List[int]:
    base = total // buckets
    rem = total % buckets
    return [base + (1 if i < rem else 0) for i in range(buckets)]


def generate_data(
    samples_per_class: int = 300,
    output_file: str = "data/training/generated_social_bias_data.csv",
) -> Tuple[Path, int]:
    rows: List[Dict[str, str]] = []

    for label, subtype_map in TEMPLATE_BANK.items():
        subtype_names = list(subtype_map.keys())
        counts = _split_counts(samples_per_class, len(subtype_names))
        for subtype, n in zip(subtype_names, counts):
            rows.extend(_expand_bucket(label, subtype, subtype_map[subtype], n))

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out, len(df)


if __name__ == "__main__":
    output_path, total = generate_data(samples_per_class=320)
    print(f"Generated {total} training samples at {output_path}")
