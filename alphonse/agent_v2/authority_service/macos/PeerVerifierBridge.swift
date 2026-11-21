// C ABI bridge for the existing kernel-audit-token verifier. The separately
// protected service supplies signing identities and endpoint role; caller
// messages must never choose them. Code-signature validity alone does not
// establish a human request or protect an untrusted interpreter's scripts.
import Foundation
import Darwin

// SO_ACCEPTCONN is not readable via getsockopt on Darwin Unix sockets.
// Inspect this process's fd metadata without accepting a client or changing
// the socket backlog; unknown/short kernel results fail closed.
@_cdecl("alphonse_authority_is_unix_listener")
public func authorityIsUnixListener(_ fd: Int32) -> Int32 {
    var info = socket_fdinfo()
    let expected = MemoryLayout<socket_fdinfo>.size
    let received = proc_pidfdinfo(getpid(), fd, PROC_PIDFDSOCKETINFO, &info, Int32(expected))
    guard received == expected,
          info.psi.soi_family == AF_UNIX,
          info.psi.soi_type == SOCK_STREAM,
          (Int32(info.psi.soi_options) & SO_ACCEPTCONN) != 0 else { return 0 }
    return 1
}

@_cdecl("alphonse_authority_verify_socket_peer")
public func authorityVerifySocketPeer(_ fd: Int32,
                                      _ rolePtr: UnsafePointer<CChar>?,
                                      _ desktopTeamPtr: UnsafePointer<CChar>?,
                                      _ desktopIDPtr: UnsafePointer<CChar>?,
                                      _ workerTeamPtr: UnsafePointer<CChar>?,
                                      _ workerIDPtr: UnsafePointer<CChar>?) -> Int32 {
    guard let rolePtr, let desktopTeamPtr, let desktopIDPtr, let workerTeamPtr, let workerIDPtr,
          let role = AuthorityPeerRole(rawValue: String(cString: rolePtr)) else { return 0 }
    do {
        let desktop = try AuthoritySigningIdentity(team: String(cString: desktopTeamPtr),
                                                   identifier: String(cString: desktopIDPtr))
        let worker = try AuthoritySigningIdentity(team: String(cString: workerTeamPtr),
                                                  identifier: String(cString: workerIDPtr))
        let policy = try AuthorityPeerPolicy(desktop: desktop, worker: worker)
        try verifyLocalSocketPeer(fd, role: role, policy: policy)
        return 1
    } catch {
        return 0
    }
}

// Separate ABI keeps code-hash pinning explicit and prevents mixed role policy.
@_cdecl("alphonse_authority_verify_socket_peer_cdhash")
public func authorityVerifySocketPeerCodeHash(_ fd: Int32,
                                              _ rolePtr: UnsafePointer<CChar>?,
                                              _ desktopHashPtr: UnsafePointer<CChar>?,
                                              _ workerHashPtr: UnsafePointer<CChar>?) -> Int32 {
    guard let rolePtr, let desktopHashPtr, let workerHashPtr,
          let role = AuthorityPeerRole(rawValue: String(cString: rolePtr)) else { return 0 }
    do {
        let desktop = try AuthorityCodeHash(String(cString: desktopHashPtr))
        let worker = try AuthorityCodeHash(String(cString: workerHashPtr))
        let policy = try AuthorityPeerPolicy(desktopHash: desktop, workerHash: worker)
        try verifyLocalSocketPeer(fd, role: role, policy: policy)
        return 1
    } catch {
        return 0
    }
}
