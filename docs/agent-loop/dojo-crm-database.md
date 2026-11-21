# Prospect CRM Dojo

The synthetic owner asks Alphonse to create a small SQLite CRM from a
disposable prospect CSV, retain its IDs, names, companies and statuses, and
keep the input unchanged. The request specifies the desired database file but
not a tool, command, program or database table design.

The independent observer opens the result read-only, checks the input's final
hash, and inspects bounded SQLite table rows. For a positive effect grade it
requires a conventional table with the requested columns, exactly the four
source records, and a successful update of one status on an in-memory database
copy. It accepts any table name, column order and additional columns. This
verifies the import and one basic update; it does not establish full CRM
functionality. Normalized schemas, additional records and failed or unavailable
probes require manual review. The observer shares the disposable container with
the candidate, so this is independent of the model's completion claim but not
an adversarially protected oracle. Final file hashes do not prove that no
temporary write occurred.

In the [real-model attempt](evidence/dojo-crm-database-20261120-01.jsonl),
Alphonse composed a Python program to create and populate the database. He
then tried the `sqlite3` command-line program to inspect it; that program was
not installed, so he used Python's SQLite library to inspect the data. The
observer independently found all four expected lead records and an unchanged
CSV. The owner client acknowledged the reply. The response content was
reviewed manually; the automated evaluator does not grade its wording.

The initial evaluator incorrectly marked the overall effect as passed based
only on record presence. It had not checked whether a user could update the
database. The [separate corrected regrade](evidence/dojo-crm-database-20261120-01.regrade.json)
preserves that initial journal, records its hash, and reports the overall
effect as **unassessed**: the fixture had already been destroyed before the
update probe was added. It still records the exact four rows and final CSV
hash as observed evidence. Future runs use the improved readback.

In the [follow-up run](evidence/dojo-crm-database-20261120-02.jsonl), the
model again created a database from the CSV. The revised observer found the
four exact mapped source rows, an unchanged CSV and a successful status update
on an in-memory copy of the database. The client acknowledged the reply. This
passes the specific import-and-update check; it does not establish complete
CRM functionality, production reliability, or host-Bash availability.

The attempt used three Bash actions, one of which failed, and four model
decisions; the provider reported 103,808 input tokens. This single synthetic
run does not establish a reliability rate, full CRM functionality, host Bash
availability in normal owner conversations, or production authorization. The
owner worker used an injected disposable-container Bash service, and the
provider has its own environment. The source-tree digest is an initial snapshot;
other work was being performed concurrently, so it is not proof of every code
object loaded during the run.
