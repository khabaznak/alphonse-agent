# Scheduled capabilities: Dojo observation

The real-model run in `evidence/dojo-telegram-scheduled-bash-20261122-01.jsonl` asked Alphonse to read a disposable inspection record after one minute and report its value and unit. The owner loop created the scheduled task; after it was due, the scheduled worker selected `native.bash`, read the file, and delivered the correct reading through a fake Telegram client. The fixture was unchanged. This is one exploratory run, not an estimate of reliability or verification of real Telegram delivery. Bash in trusted-local mode has host access; the disposable fixture does not make it a sandbox.

The scheduled owner-path regression tests cover Bash for Desktop and Telegram, and the existing Gmail unread-count capability, using synthetic credentials and data. They exercise authorization and revocation checks; they do not establish full Gmail functionality or test a real mailbox.
