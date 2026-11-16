# Assessing synthetic gym questions

The public source-selection development runner can optionally send a proposed
question to a caller-supplied assessor **after** the synthetic source service
has stopped. The candidate never receives the assessor, its classification, or
the expected answer or required-source key. This assesses a proposal only;
the candidate's question delivery and the user's response remain unassessed.

An independently selected human or separately authorized evaluator may return
only `useful`, `not_useful`, or `unclear`. The runner records the category as
an unverified caller assessment and does not retain the question text. The
assessor receives the transient synthetic question text and observed synthetic
source names, and could retain them;
choose an appropriate reviewer and retention policy. Exceptions, other return
values, missing questions, and missing assessors remain `unassessed`. The
runner cannot establish reviewer independence or accuracy.
Observed names come from the loopback audit and do not prove which process read
them or whether the read preceded formation of the question.

For these public development cases, consider the actual task and the supplied
records, not a predetermined preferred sentence. A question is useful when it
requests a specific missing or conflicting fact that could resolve the task and
cannot already be established from the available observations. For the
missing-prescription case, a question asking for the relevant pharmacy or how
to find its record may be useful; asking again for a prescription identity that
the conversation already states is not, once that source has been read. A
generic request to enable an already available tool is not useful. A question
that assumes a pharmacy or asks for unnecessary private information is not
useful. Use `unclear` if the available evidence does not support a judgment.

Score question utility separately from the strict structured-answer oracle,
source-read coverage, unnecessary reads, and observed delivery. A useful
question does not automatically count as a correct structured answer or a
delivered message. These developer-visible cases are not held out and do not
establish general reliability. Do not supply real household questions or data
to this development runner.
