import Darwin
import Foundation

@main
struct LocalPeerVerificationChecks {
    static func main() throws {
        let desktop = try AuthoritySigningIdentity(team: "ABCDEFGHIJ", identifier: "com.example.desktop")
        let worker = try AuthoritySigningIdentity(team: "ABCDEFGHIJ", identifier: "com.example.worker")
        let policy = try AuthorityPeerPolicy(desktop: desktop, worker: worker)

        do {
            try verifyLocalSocketPeer(-1, role: .desktop, policy: policy)
            fatalError("invalid descriptor must fail")
        } catch LocalPeerVerificationError.peerTokenUnavailable {
            // Correct: no caller can invent a peer token for an invalid fd.
        }

        var pair: [Int32] = [0, 0]
        guard socketpair(AF_UNIX, SOCK_STREAM, 0, &pair) == 0 else {
            fatalError("socketpair unavailable")
        }
        defer { close(pair[0]); close(pair[1]) }
        do {
            try verifyLocalSocketPeer(pair[0], role: .desktop, policy: policy)
            fatalError("untrusted local process must not match Apple signing identity")
        } catch LocalPeerVerificationError.signatureRejected {
            // A real peer token resolves, but this test executable is not
            // signed by the configured Apple developer identity.
        } catch LocalPeerVerificationError.codeUnavailable {
            // A platform unable to resolve the code must also fail closed.
        }
        print("unix peer verifier rejected invalid and untrusted peers")
    }
}
