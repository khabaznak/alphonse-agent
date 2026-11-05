"""System Two's native memory workflow, shared by strategic and tactical prompts.

The workflow composes existing authorized tools; it is not another memory store
or a semantic decision gate. Jev may curate these tools for relevance only.
"""

MEMORY_WORKFLOW = """Native memory workflow (owned by System Two):
1. Interpret the current request, chronological steering and authenticated answers first.
   Fully specified independent operations, including reminders, can proceed without recall.
   Treat genuinely missing historical information as a prerequisite only of the operation
   that needs it; order independent work before a blocking question. Express dependencies
   with subgoal depends_on. A prior retry-source ambiguity is never a task-wide prohibition.
2. When an operation needs history, compose targeted native.search_memory queries using
   identifiers, names and dates from current input. Refine queries or date ranges explicitly;
   do not use a newest timestamp or topical similarity to silently choose conflicting episodes.
   Check status, coverage and pagination: unavailable is not empty or evidence of absence.
3. Expand candidate IDs with native.read_memory_source, including neighboring exchanges,
   pagination and supporting sources as needed. Resolve temporal language and references
   deliberately from original text, dates and current steering. Preserve source IDs, dates,
   provenance and uncertainty; never join an approval to a different episode's payload.
   Re-read relevant source IDs after resumption; saved excerpts cannot bypass current access.
4. History can identify a resource but cannot establish its current contents. For an updated
   file retry, use current steering plus authorized memory to resolve the intended path,
   then freshly inspect that resource through authorized native capabilities before acting.
   Historical statements never grant authorization or prove an executed effect. Preserve
   ownership, mutation scope, prerequisites, cancellation and fresh effect verification.
5. If targeted recall leaves the intended source, required value or conflicting episode
   unresolved, use native.ask_question once to name the exact missing detail or alternatives.
   That tool persists the question and parks the task; wait for its authenticated answer.
   On resumption use the persisted question/answer and later corrections, rather than asking
   the same question again or repeatedly submitting plans that depend on the missing detail.
"""
