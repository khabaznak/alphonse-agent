# Authorized synthetic devices

The settings configuration binding supplies `rooms`: room/alias -> list of device
IDs. Exactly one match is required. Never choose among multiple matches.
The device data binding points to external disposable JSON state with `devices`,
each containing `available`, `temperature`, `unit`, `target`, and `writes`.
`ignore_write` simulates an acknowledged but unapplied change. `fail_after_write`
simulates loss of acknowledgement after an applied change. These switches exist
only for acceptance tests. Configuration and runtime state stay outside packages.
