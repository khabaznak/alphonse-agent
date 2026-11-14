"""Development contrast with a more discoverable, neutral source catalog.

The records, goals and post-run oracles are the original variant fixtures.
Only the source descriptions presented to the chooser differ. This is public
development material, not a held-out or clinical evaluation.
"""
from .selective_sources_variants import ALLOWED_SOURCES, CASE_IDS, GOAL, evaluate, run, sources

# General descriptions of source categories, identical across all cases. They
# do not identify which sources are required or reveal values from any record.
SOURCE_DESCRIPTION = (
    'Read a synthetic source: index lists available source names; '
    'conversation contains context and references from earlier interactions; '
    'prescription contains prescription records and their identifiers; '
    'treatment-status contains treatment or other household status notes.'
)
