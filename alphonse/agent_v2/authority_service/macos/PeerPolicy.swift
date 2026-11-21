// macOS XPC peer policy for a future separately protected authority service.
// This file does not install a service or turn on host Bash. The policy and
// service binary must themselves be protected from the Bash user's writes.
import Foundation
import Security

public enum AuthorityPeerRole: String {
    case desktop
    case worker
}

public enum AuthorityPeerPolicyError: Error {
    case invalidTeam
    case invalidIdentifier
    case invalidCodeHash
    case indistinctIdentities
    case invalidRequirement
    case hardenedRuntimeNotEnforcedByXPC
}

/// Exact code-directory hash from protected provisioning. Ad-hoc signatures
/// carry no signer identity: this pins code bytes, not human intent, a path,
/// protected resources, an interpreter's scripts, or a complete trust boundary.
public struct AuthorityCodeHash: Equatable {
    public let value: String
    public init(_ value: String) throws {
        let hex = CharacterSet(charactersIn: "0123456789abcdef")
        guard value.utf8.count == 40,
              value.unicodeScalars.allSatisfy({ hex.contains($0) }) else {
            throw AuthorityPeerPolicyError.invalidCodeHash
        }
        self.value = value
    }
    public var requirement: String { "cdhash H\"\(value)\"" }
}

/// A signed executable identity provisioned by the service, never by a caller.
public struct AuthoritySigningIdentity: Equatable {
    public let team: String
    public let identifier: String

    public init(team: String, identifier: String) throws {
        let teamChars = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        guard team.utf8.count == 10,
              team.unicodeScalars.allSatisfy({ teamChars.contains($0) }) else {
            throw AuthorityPeerPolicyError.invalidTeam
        }
        let idChars = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-_")
        guard !identifier.isEmpty, identifier.utf8.count <= 255,
              identifier.unicodeScalars.allSatisfy({ idChars.contains($0) }) else {
            throw AuthorityPeerPolicyError.invalidIdentifier
        }
        self.team = team
        self.identifier = identifier
    }

    public var requirement: String {
        "anchor apple generic and identifier \"\(identifier)\" and certificate leaf[subject.OU] = \"\(team)\""
    }
}

/// Distinct roles must arrive on separately provisioned listeners; a role
/// asserted inside a client's message has no authority.
public struct AuthorityPeerPolicy {
    private let requirements: [AuthorityPeerRole: String]
    public let requiresHardenedRuntime: Bool
    public let usesPinnedCodeHash: Bool

    public init(desktop: AuthoritySigningIdentity, worker: AuthoritySigningIdentity) throws {
        guard desktop != worker else { throw AuthorityPeerPolicyError.indistinctIdentities }
        try self.init(requirements: [.desktop: desktop.requirement, .worker: worker.requirement], usesPinnedCodeHash: false)
    }

    /// Both roles use independently pinned hashes, never caller-provided data.
    public init(desktopHash: AuthorityCodeHash, workerHash: AuthorityCodeHash) throws {
        guard desktopHash != workerHash else { throw AuthorityPeerPolicyError.indistinctIdentities }
        try self.init(requirements: [.desktop: desktopHash.requirement, .worker: workerHash.requirement], usesPinnedCodeHash: true)
    }

    private init(requirements: [AuthorityPeerRole: String], usesPinnedCodeHash: Bool) throws {
        self.requirements = requirements
        self.requiresHardenedRuntime = true
        self.usesPinnedCodeHash = usesPinnedCodeHash
        for requirementString in requirements.values {
            var requirement: SecRequirement?
            guard SecRequirementCreateWithString(requirementString as CFString,
                    SecCSFlags(), &requirement) == errSecSuccess, requirement != nil else {
                throw AuthorityPeerPolicyError.invalidRequirement
            }
        }
    }

    public func requirement(for role: AuthorityPeerRole) -> String {
        // The initializer always supplies both known roles.
        requirements[role]!
    }
}

/// Bind a role to its own XPC listener. XPC checks signatures against the
/// configured role requirement. No peer identity comes from a message body.
/// A matching code signature is not evidence of a human-authorized request.
/// This legacy XPC probe does not enforce runtime hardening; it must not be
/// used as the production authority ingress. Pinned hashes are refused here.
@available(macOS 13.0, *)
public final class RoleGuardedAuthorityListener: NSObject, NSXPCListenerDelegate {
    public let role: AuthorityPeerRole
    private let requirement: String
    private let accept: (AuthorityPeerRole, NSXPCConnection) -> Bool

    public init(role: AuthorityPeerRole, policy: AuthorityPeerPolicy,
                accept: @escaping (AuthorityPeerRole, NSXPCConnection) -> Bool) throws {
        guard !policy.usesPinnedCodeHash else { throw AuthorityPeerPolicyError.hardenedRuntimeNotEnforcedByXPC }
        self.role = role
        self.requirement = policy.requirement(for: role)
        self.accept = accept
    }

    public func configure(_ listener: NSXPCListener) {
        listener.setConnectionCodeSigningRequirement(requirement)
        listener.delegate = self
    }

    public func listener(_ listener: NSXPCListener,
                         shouldAcceptNewConnection connection: NSXPCConnection) -> Bool {
        connection.setCodeSigningRequirement(requirement)
        // Configure only: do not perform authority transitions merely because
        // the callback was invoked. Expose only this role's protocol, handle
        // XPC errors, and resume accepted peers. The service must retain this
        // delegate for at least as long as the listener.
        return accept(role, connection)
    }
}
