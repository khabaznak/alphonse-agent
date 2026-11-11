# Coding sandbox assets

`seccomp.json` is derived from Moby's default seccomp profile at revision
`4b8214fc4971e06172938a1f1554ccff6647ee80`, licensed under the adjacent
`MOBY-LICENSE`. The only addition allows `clone`, `unshare`, `mount`, `umount2`,
and `pivot_root` so an unprivileged Codex process can construct its nested
filesystem sandbox. These additional syscalls widen the default Docker policy;
the container must also run as a non-root user with all capabilities dropped,
no-new-privileges, a read-only root, bounded processes, and no Docker socket.
The nested Codex sandbox must enforce filesystem and network restrictions.
This profile is an experimental prototype, not production qualification.
