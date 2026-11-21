import Foundation
import Security

@main
struct PeerPolicyCheck {
    static func main() throws {
        let desktop = try AuthoritySigningIdentity(team: "ABCD123456", identifier: "local.alphonse.desktop")
        let worker = try AuthoritySigningIdentity(team: "ABCD123456", identifier: "local.alphonse.worker")
        let policy = try AuthorityPeerPolicy(desktop: desktop, worker: worker)
        precondition(policy.requirement(for: .desktop) != policy.requirement(for: .worker))
        precondition(policy.requirement(for: .desktop).contains("identifier \"local.alphonse.desktop\""))
        precondition(policy.requirement(for: .worker).contains("certificate leaf[subject.OU] = \"ABCD123456\""))
        do {
            _ = try AuthorityPeerPolicy(desktop: desktop, worker: desktop)
            fatalError("same signing identity accepted for both roles")
        } catch AuthorityPeerPolicyError.indistinctIdentities { }
        for bad in ["", "anything\" or true", "local.alphonse\nworker"] {
            do {
                _ = try AuthoritySigningIdentity(team: "ABCD123456", identifier: bad)
                fatalError("malformed identifier accepted")
            } catch AuthorityPeerPolicyError.invalidIdentifier { }
        }
        for bad in ["", "ABCDEFGHI", "ABCDEFGHIJK", "abcd123456", "ABCD\"23456"] {
            do {
                _ = try AuthoritySigningIdentity(team: bad, identifier: "local.alphonse.worker")
                fatalError("malformed team accepted")
            } catch AuthorityPeerPolicyError.invalidTeam { }
        }
        let desktopHash = try AuthorityCodeHash(String(repeating: "a", count: 40))
        let workerHash = try AuthorityCodeHash(String(repeating: "b", count: 40))
        let pinned = try AuthorityPeerPolicy(desktopHash: desktopHash, workerHash: workerHash)
        precondition(policy.requiresHardenedRuntime && pinned.requiresHardenedRuntime)
        if #available(macOS 13.0, *) {
            do {
                _ = try RoleGuardedAuthorityListener(role: .desktop, policy: pinned) { _, _ in true }
                fatalError("XPC accepted a pinned policy without hardening verification")
            } catch AuthorityPeerPolicyError.hardenedRuntimeNotEnforcedByXPC { }
        }
        precondition(pinned.requirement(for: .desktop) == "cdhash H\"" + String(repeating: "a", count: 40) + "\"")
        precondition(pinned.requirement(for: .desktop) != pinned.requirement(for: .worker))
        do {
            _ = try AuthorityPeerPolicy(desktopHash: desktopHash, workerHash: desktopHash)
            fatalError("same CDHash accepted for both roles")
        } catch AuthorityPeerPolicyError.indistinctIdentities { }
        for bad in ["", String(repeating: "a", count: 39), String(repeating: "A", count: 40),
                    String(repeating: "g", count: 40), String(repeating: "a", count: 41)] {
            do {
                _ = try AuthorityCodeHash(bad)
                fatalError("malformed CDHash accepted")
            } catch AuthorityPeerPolicyError.invalidCodeHash { }
        }
        print("macOS XPC peer policy checks passed")
    }
}
