# Synthetic scoped Bash boundary probes

These are fixed-command diagnostics, not an enabled owner capability or a
complete security review. On the local macOS host, Docker was supplied a pinned
local `python:3.11-slim` image by immutable ID
`sha256:9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7`.
Each probe created only a temporary synthetic directory and removed it afterward.
No household files, credentials or model-generated commands were used.

In the first probe, Docker used `--network=none`, `--read-only`,
`--cap-drop=ALL`, `--security-opt=no-new-privileges`, process/memory/CPU limits,
an unprivileged user and one bind mount of a temporary workspace. The child
read and wrote that workspace. It did not see a synthetic marker in a sibling
host directory or `/var/run/docker.sock`, and a connection to an external test
address returned failure. The parent independently observed the workspace write.

In the second probe, the container ran under the host caller's numeric UID and
GID and the temporary workspace had mode `0700`. It read and wrote the mounted
workspace. A preexisting workspace symlink pointing to a synthetic sibling host
file did not resolve inside the container, and the unmounted host parent was
not visible. The parent independently observed the new workspace file.

These observations establish only the named fixture behaviors on this host and
image. They do not prove complete container isolation, user-specific authority,
safe external effects, restart recovery or a supported installed configuration.
Notably, a link created in the mounted workspace can still point to a sensitive
host location when later interpreted by a *host* process; any host-side
publication and readback must reject traversal and symlink escape. A bind-mounted
workspace is writable by the child and is not itself a task-scoped permission.

An independent macOS `sandbox-exec` check using default-deny rules, including
broad system path and process/Mach/IPC exceptions, returned exit 134 before a
fixed Bash command could run on this host. An allow-default profile with selected
path/network denials started Bash, but did not cover unknown private resources
or establish non-impersonation. Neither profile was installed. The probes do not
show that a properly reviewed macOS policy is impossible.

Before this could become a normal owner capability, a trusted issuer must bind
an active actor, task and explicit resource/effect scope to a particular worker;
the worker must not receive access to the issuer, daemon socket, grants DB or
credentials. Dispatch and publication must recheck authorization, cancellation
and claim identity, preserve uncertain effects without replay, and independently
verify requested effects. Generic unrestricted `native.bash` remains trusted host
execution and must not be presented as confined by a cwd or this fixture.
