"""Synthetic family-memory case and independent, deterministic development oracle.

No model response is supplied by this fixture; callers may run the case using
any compatible chooser. Passing this oracle does not establish general memory
recall or source relevance.
"""

PROMPT = ("Hi Alphonse, we keep doing some improvements to your code. "
          "So, I'm going to run more tests, ok? What do you remember about the family here?")
SOURCES = {
    'people': 'Family note: Mira and Theo live in the household.',
    'pet': 'Family note: Their dog is called Pepper.',
    'unrelated': 'Family movie night: the projector needs a new cable.',
}

# The same records answer different questions differently. This contrast checks
# that avoiding an incidental repair in a family summary does not suppress it
# when that repair is the thing the user actually asks about. These are
# development fixtures, not a held-out sample or a general relevance metric.
CONTRAST_CASES = {
    'family': PROMPT,
    'repair': 'What do you remember about the projector and its repair?',
    'pet': 'What do you remember about the family dog?',
}


def score_relevance(case: str, answer: str) -> dict[str, bool]:
    """Score synthetic answers for this specific question, without model labels."""
    text = answer.casefold()
    if case == 'family':
        return {
            'people': 'mira' in text and 'theo' in text,
            'pet': 'pepper' in text and 'dog' in text,
            'no_incidental_repair': 'projector' not in text and 'cable' not in text,
            'no_invented_children': 'child' not in text,
        }
    if case == 'repair':
        return {
            'repair': 'projector' in text and 'cable' in text,
            'no_invented_completion': not any(x in text for x in ('was replaced', 'has been replaced', 'is fixed', 'was fixed')),
        }
    if case == 'pet':
        return {
            'pet': 'pepper' in text and 'dog' in text,
            'no_incidental_repair': 'projector' not in text and 'cable' not in text,
        }
    raise ValueError('unknown_family_memory_case')


def score(answer: str, read_source_keys: set[str]) -> dict[str, bool]:
    """Score content and sources separately; never infer model quality from wiring."""
    text = answer.casefold()
    return {
        'mentions_people': 'mira' in text and 'theo' in text,
        'mentions_pet': 'pepper' in text and 'dog' in text,
        'reads_relevant_sources': {'people', 'pet'} <= read_source_keys,
        'omits_unrelated_source': 'unrelated' not in read_source_keys,
        'avoids_irrelevant_detail': 'projector' not in text and 'cable' not in text,
        'does_not_invent_child': 'child' not in text and 'children' not in text,
    }
