# Agent Skills packages in Alphonse

Alphonse follows the [Agent Skills specification](https://agentskills.io/specification)
for installed and bundled packages. A package is a directory with `SKILL.md`;
additional directories and files are preserved, including nested scripts,
references, binary assets and license files. Scripts are not independently
registered tools and installing a package does not execute them.

## Format validation

`name` and `description` are required strings. Names have 1–64 characters,
normalized with Unicode NFKC, using lowercase alphanumeric characters and single
hyphens, without leading/trailing hyphens. The normalized directory name must
match. This follows the Unicode naming behavior of the official reference
validator. Descriptions must be nonblank and no longer than 1,024 characters;
that limit applies to installation, authoring, description edits, the native
schema and Desktop. Markdown bodies have no format length or nonempty rule.

Supported optional frontmatter is retained in `SkillRecord` and the admin Skills
workspace response, as well as in the package files:

- `license`: a string, such as `MIT` or `LICENSE.txt`.
- `compatibility`: a string of 1–500 characters.
- `metadata`: a mapping of string keys to string values. Quote numeric versions.
- `allowed-tools`: an experimental, space-separated string.

Unknown top-level fields are rejected; put extensions inside `metadata`.
Wrong YAML types are rejected rather than converted to strings.

`compatibility` is descriptive: Alphonse does not automatically install packages,
resolve dependencies or select a runtime from it. `allowed-tools` is preserved
as descriptive metadata only. Its preapproval/filtering execution semantics are
unsupported: it never grants authority, bypasses authorization, changes native
registry exposure or registers scripts. System Two uses skill instructions and
authorized native capabilities to compose commands; Jev provides relevance
judgments, never permission.

## Alphonse operational limits

These are local product constraints, not Agent Skills specification rules:

- The authoring form/tool requires nonempty instructions and limits them to
  12,000 characters. Imported packages may have an empty or longer body.
- `SKILL.md` and individual files edited/imported through the text editor have a
  256,000-byte limit. Whole-package copies preserve larger supporting files.
- Package install, replace and export reject symlinks and special files rather
  than following links outside the package. Generic Bash remains a trusted host
  capability; these package-copy rules are not an OS sandbox.
- Only administrators manage skills and export packages through IPC.

Operational rejections use `skill_operational_*` diagnostics. Listing skips
unusable packages for runtime selection but logs each failure and returns
`diagnostics` with the directory, category (`specification`, `operational` or
`filesystem`) and reason. Desktop displays these failures in the Skills
workspace. Invalid packages are never silently lost from the management report.

Install and replacement stage a complete package before activation. Export
copies the complete package into a new `<destination>/<skill-name>` directory,
refuses existing destinations, and preserves source files and executable modes.
It is available via the admin `export_skill` IPC operation and **Export package**
in Desktop. Package export does not collect external artifact storage or data.
Bundled resources use recursive setuptools package data, including nested
scripts, references and assets in both source and wheel distributions.

## Verification

`tests/fixtures/skills/validation.json` covers valid/invalid frontmatter and
boundaries. `sample/` is a minimal disposable package, not a household artifact.
`tests/test_agent_v2_skills.py` checks conformance, metadata retention, lifecycle,
authorization and diagnostics. `tests/test_skill_distribution.py` builds an
sdist, builds its wheel, installs into a temporary directory, then checks
resources and bundled discovery from that installed location.

Reference comparisons use the official
[skills-ref validator](https://github.com/agentskills/agentskills/tree/main/skills-ref),
reviewed at commit `69ef37e9424c0a7ea9dd2293b559e43ec8176379`.
With its `strictyaml` dependency installed and `skills-ref/src` on `PYTHONPATH`,
the comparison tests run; without it they explicitly skip. Compare shared typed
metadata rules through `validate_metadata`, and the complete package through
`validate`. The reference parser uses schema-free StrictYAML, which treats YAML
scalars as strings; Alphonse uses typed YAML and rejects non-string fields.
The reference metadata validator also omits license, allowed-tools and metadata
type checks and accepts empty compatibility strings. These deliberate stricter
checks follow the published field types and 1–500 compatibility constraint and
are marked separately in the fixture matrix.
