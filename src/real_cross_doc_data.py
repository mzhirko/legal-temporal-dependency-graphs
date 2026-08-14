"""
Real cross-document test data for the TDG pipeline.

Document 1: Employment Rights Act 1996, Section 111
  Source: legislation.gov.uk/ukpga/1996/18/section/111
  Type: legislation

Document 2: Ahmed v Newcastle City Council [2021] (Case 2501592/2020)
  Source: assets.publishing.service.gov.uk (employment tribunal judgment PDF)
  Type: court_decision

Document 3: Zegay v Boylan / Sportline (Case 2202972/2019)
  Source: assets.publishing.service.gov.uk (employment tribunal judgment PDF)
  Type: court_decision

Cross-doc relationships:
  - Both judgments cite s.111 ERA 1996 (coreference with statute)
  - Both apply the "3 months from EDT" rule (temporal entailment)
  - Both compute the early conciliation extension (same temporal pattern)
  - The two judgments have the same temporal skeleton (structural analogy)

Usage:
    from real_cross_doc_data import REAL_DOCS
    # Each entry has: id, domain, text, expected_facts, expected_cross_links
"""

# --- Document 1: ERA 1996 s.111 (the statute) ----------------------------
# Extracted from legislation.gov.uk -- the actual current text.
# This defines the temporal RULES that judgments apply.

ERA_S111_TEXT = (
    "111 Complaints to employment tribunal. "
    "(1) A complaint may be presented to an employment tribunal against an "
    "employer by any person that he was unfairly dismissed by the employer. "
    "(2) Subject to the following provisions of this section, an employment "
    "tribunal shall not consider a complaint under this section unless it is "
    "presented to the tribunal -- "
    "(a) before the end of the period of three months beginning with the "
    "effective date of termination, or "
    "(b) within such further period as the tribunal considers reasonable in a "
    "case where it is satisfied that it was not reasonably practicable for the "
    "complaint to be presented before the end of that period of three months. "
    "(2A) Section 207B (extension of time limits to facilitate conciliation "
    "before institution of proceedings) applies for the purposes of subsection (2)(a). "
    "(3) Where a dismissal is with notice, an employment tribunal shall consider "
    "a complaint under this section if it is presented after the notice is given "
    "but before the effective date of termination."
)

# --- Document 2: Ahmed v Newcastle City Council --------------------------
# Case Number: 2501592/2020, heard 25 February 2021.
# Claimant dismissed for gross misconduct. Filed ET1 out of time.
# Full temporal chain with specific dates.

AHMED_TEXT = (
    "The claimant was employed as Regional Transport Team Specialist "
    "Transport Planner by the respondent from 1 June 1994 to 16 March 2020, "
    "which was the effective date of termination of his employment following "
    "his dismissal for the stated reason of gross misconduct. "
    "The claimant started early conciliation with ACAS on 8 June 2020 and "
    "obtained a conciliation certificate dated 26 June 2020. "
    "The claimant's ET1 was presented on 24 August 2020. "
    "The time limit for an unfair dismissal complaint appears in section 111(2) "
    "of the Employment Rights Act 1996. "
    "I find that the effective date of termination of the claimant's employment "
    "was on 16 March 2020. The claimant began early conciliation on 8 June 2020 "
    "and obtained a conciliation certificate on 26 June 2020. "
    "I find that time for the presentation of the claim form expired at midnight "
    "on 26 July 2020. "
    "On 16 July 2020, the claimant emailed his ET1 to the Newcastle Employment "
    "Tribunal. This was rejected because the Tribunal may not accept an ET1 "
    "submitted by email. "
    "On 22 July 2020, the Tribunal received the claimant's ET1 from Kings Court. "
    "This was rejected because it had not been handed in to the designated office. "
    "The claimant's ET1 was finally presented on 24 August 2020. "
    "The claimant did not present his claim of unfair dismissal before the end "
    "of the period of three months beginning with the effective date of "
    "termination, as required by section 111(2)(a) of the Employment Rights "
    "Act 1996. The claim is struck out."
)

# --- Document 3: Zegay v Boylan / Sportline ------------------------------
# Case Number: 2202972/2019, heard at London Central.
# Different dates, same temporal structure. Includes detailed day-counting
# for the early conciliation extension.

ZEGAY_TEXT = (
    "My decision is that the effective date of termination was 15 March 2019. "
    "Day A was 7 May 2019 and Day B was 20 June 2019. Therefore, the period "
    "not to be counted is 8 May to 20 June 2019. This is 44 days. "
    "The Claimant did not present his claim within one month of Day B. "
    "The time limit (but for the early conciliation provisions) would have "
    "expired on 14 June 2019. Ignoring 44 days when performing the calculation, "
    "the time limit (as affected by the early conciliation provisions) expired "
    "on 28 July 2019. "
    "The Claimant's ET1 was presented on 4 September 2019. "
    "The time limit for an unfair dismissal complaint is set out in section "
    "111(2) of the Employment Rights Act 1996: an employment tribunal shall "
    "not consider a complaint unless it is presented before the end of the "
    "period of three months beginning with the effective date of termination. "
    "Section 207B provides for an extension of time limits to facilitate "
    "early conciliation. "
    "I therefore have to consider if it was reasonably practicable for the "
    "Claimant to submit his complaint by 28 July 2019."
)

# --- Combined test corpus ------------------------------------------------

REAL_DOCS = [
    {
        "id": "era_1996_s111",
        "domain": "legislation",
        "label": "Employment Rights Act 1996, s.111",
        "source": "legislation.gov.uk/ukpga/1996/18/section/111",
        "text": ERA_S111_TEXT,
        "expected_facts": {
            "time_limit": "P3M (3 months from effective date of termination)",
            "early_conciliation": "s.207B extension applies",
        },
    },
    {
        "id": "ahmed_v_newcastle",
        "domain": "court_decision",
        "label": "Ahmed v Newcastle City Council [2021] Case 2501592/2020",
        "source": "assets.publishing.service.gov.uk (tribunal PDF)",
        "text": AHMED_TEXT,
        "expected_facts": {
            "employment_start": "1994-06-01",
            "edt": "2020-03-16",  # effective date of termination
            "early_conciliation_start": "2020-06-08",  # Day A
            "conciliation_certificate": "2020-06-26",  # Day B
            "primary_limit": "2020-06-15",  # EDT + 3 months - 1 day
            "extended_limit": "2020-07-26",  # with early conciliation
            "et1_email_rejected": "2020-07-16",
            "et1_hand_rejected": "2020-07-22",
            "et1_presented": "2020-08-24",  # OUT OF TIME
        },
        "expected_chain": [
            "edt -> +3 months -> primary_limit",
            "primary_limit + early_conciliation_extension -> extended_limit",
            "et1_presented > extended_limit -> OUT OF TIME",
        ],
    },
    {
        "id": "zegay_v_boylan",
        "domain": "court_decision",
        "label": "Zegay v Boylan / Sportline [2020] Case 2202972/2019",
        "source": "assets.publishing.service.gov.uk (tribunal PDF)",
        "text": ZEGAY_TEXT,
        "expected_facts": {
            "edt": "2019-03-15",
            "day_a": "2019-05-07",
            "day_b": "2019-06-20",
            "conciliation_pause": "44 days (8 May to 20 June)",
            "primary_limit": "2019-06-14",  # EDT + 3 months - 1 day
            "extended_limit": "2019-07-28",  # + 44 days
            "et1_presented": "2019-09-04",  # OUT OF TIME
        },
        "expected_chain": [
            "edt -> +3 months -> primary_limit",
            "primary_limit + 44 days pause -> extended_limit",
            "et1_presented > extended_limit -> OUT OF TIME",
        ],
    },
]

# --- Expected cross-doc links --------------------------------------------

EXPECTED_CROSS_LINKS = [
    {
        "type": "coreference",
        "desc": "Both judgments reference s.111(2) ERA 1996 time limit rule",
        "docs": ["ahmed_v_newcastle", "zegay_v_boylan"],
        "concept": "3-month time limit from effective date of termination",
    },
    {
        "type": "coreference",
        "desc": "Both judgments apply s.207B early conciliation extension",
        "docs": ["ahmed_v_newcastle", "zegay_v_boylan"],
        "concept": "early conciliation time extension",
    },
    {
        "type": "entailment",
        "desc": "Statute s.111(2)(a) defines the rule; Ahmed applies it with EDT=16 March 2020",
        "docs": ["era_1996_s111", "ahmed_v_newcastle"],
        "concept": "P3M time limit -> computed as 26 July 2020 (with conciliation)",
    },
    {
        "type": "entailment",
        "desc": "Statute s.111(2)(a) defines the rule; Zegay applies it with EDT=15 March 2019",
        "docs": ["era_1996_s111", "zegay_v_boylan"],
        "concept": "P3M time limit -> computed as 28 July 2019 (with conciliation)",
    },
    {
        "type": "structural_analogy",
        "desc": "Both judgments share the same temporal skeleton: EDT -> +3M -> primary limit -> +conciliation -> extended limit -> ET1 presented -> OUT OF TIME",
        "docs": ["ahmed_v_newcastle", "zegay_v_boylan"],
    },
    {
        "type": "coreference",
        "desc": "Both judgments use the concept 'effective date of termination' from s.97 ERA",
        "docs": ["ahmed_v_newcastle", "zegay_v_boylan"],
        "concept": "effective date of termination",
    },
]


if __name__ == "__main__":
    print("Real cross-doc test corpus")
    print(f"  Documents: {len(REAL_DOCS)}")
    for d in REAL_DOCS:
        print(f"    {d['id']:25s} {d['domain']:18s} {len(d['text']):5d} chars  {d['label']}")
    print(f"\n  Expected cross-doc links: {len(EXPECTED_CROSS_LINKS)}")
    for link in EXPECTED_CROSS_LINKS:
        print(f"    [{link['type']:20s}] {link['docs'][0]} ↔ {link['docs'][1]}")
        print(f"      {link['desc']}")
