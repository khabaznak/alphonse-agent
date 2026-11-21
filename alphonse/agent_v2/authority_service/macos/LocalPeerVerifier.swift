// Server-side identity check for an accepted macOS Unix-domain connection.
// This is a building block, not an installed authority service. In particular,
// code signature alone does not prove that a human initiated an owner request.
import Darwin
import Foundation
import Security

public enum LocalPeerVerificationError: Error {
    case peerTokenUnavailable
    case codeUnavailable
    case requirementUnavailable
    case signatureRejected
    case hardenedRuntimeUnavailable
}

/// Verify the signature of the process represented by the kernel-provided
/// socket peer audit token. `role` and `policy` are service-owned connection
/// configuration, never fields supplied in an incoming request. This is
/// deliberately bound to the accepted socket rather than a caller-supplied
/// PID, which can be recycled or forged. The service still needs to bind the
/// verified process to an authenticated household actor and an exact request.
public func verifyLocalSocketPeer(_ fd: Int32, role: AuthorityPeerRole,
                                  policy: AuthorityPeerPolicy) throws {
    try verifyLocalSocketPeerRequirement(fd, requirementString: policy.requirement(for: role),
                                         requireHardened: policy.requiresHardenedRuntime)
}

// Internal primitive shared with the disposable Apple-system-client test.
// Production callers select role-specific, provisioned policy via the public
// function above; never take the requirement from a connected peer.
func verifyLocalSocketPeerRequirement(_ fd: Int32, requirementString: String,
                                      requireHardened: Bool = false) throws {
    var token = audit_token_t()
    var tokenLength = socklen_t(MemoryLayout<audit_token_t>.size)
    guard getsockopt(fd, SOL_LOCAL, LOCAL_PEERTOKEN, &token, &tokenLength) == 0,
          tokenLength == socklen_t(MemoryLayout<audit_token_t>.size) else {
        throw LocalPeerVerificationError.peerTokenUnavailable
    }
    let tokenData = withUnsafeBytes(of: &token) { Data($0) }
    let attributes = [kSecGuestAttributeAudit as String: tokenData] as CFDictionary
    var peer: SecCode?
    guard SecCodeCopyGuestWithAttributes(nil, attributes, SecCSFlags(), &peer) == errSecSuccess,
          let peer else {
        throw LocalPeerVerificationError.codeUnavailable
    }
    var requirement: SecRequirement?
    guard SecRequirementCreateWithString(requirementString as CFString,
                                          SecCSFlags(), &requirement) == errSecSuccess,
          let requirement else {
        throw LocalPeerVerificationError.requirementUnavailable
    }
    guard SecCodeCheckValidity(peer, SecCSFlags(), requirement) == errSecSuccess else {
        throw LocalPeerVerificationError.signatureRejected
    }
    if requireHardened {
        var info: CFDictionary?
        // Security's C API accepts either a dynamic SecCodeRef or a static
        // SecStaticCodeRef for this call, although Swift's imported signature
        // spells only the static type. Preserve the dynamic object so the
        // documented dynamic status key is available.
        let dynamicCode = unsafeBitCast(peer, to: SecStaticCode.self)
        let informationFlags = SecCSFlags(rawValue: kSecCSSigningInformation | kSecCSDynamicInformation)
        guard SecCodeCopySigningInformation(dynamicCode, informationFlags, &info) == errSecSuccess,
              let values = info as? [String: Any],
              let staticFlags = values[kSecCodeInfoFlags as String] as? NSNumber,
              let dynamicFlags = values[kSecCodeInfoStatus as String] as? NSNumber,
              staticFlags.uint32Value & 0x10000 != 0,  // kSecCodeSignatureRuntime
              dynamicFlags.uint32Value & 0x10000 != 0,
              dynamicFlags.uint32Value & 0x10000000 == 0 else { // kSecCodeStatusDebugged
            throw LocalPeerVerificationError.hardenedRuntimeUnavailable
        }
        let entitlementValue = values[kSecCodeInfoEntitlementsDict as String]
        guard entitlementValue == nil || entitlementValue is [String: Any] else {
            throw LocalPeerVerificationError.hardenedRuntimeUnavailable
        }
        // Unknown/non-dictionary entitlements cannot be safely interpreted.
        if entitlementValue == nil, values[kSecCodeInfoEntitlements as String] != nil {
            throw LocalPeerVerificationError.hardenedRuntimeUnavailable
        }
        let entitlements = entitlementValue as? [String: Any] ?? [:]
        let prohibited = ["com.apple.security.get-task-allow", "com.apple.security.cs.debugger",
                          "com.apple.security.cs.allow-jit", "com.apple.security.cs.allow-unsigned-executable-memory",
                          "com.apple.security.cs.disable-executable-page-protection",
                          "com.apple.security.cs.disable-library-validation",
                          "com.apple.security.cs.allow-dyld-environment-variables"]
        if prohibited.contains(where: { entitlements[$0] != nil }) {
            throw LocalPeerVerificationError.hardenedRuntimeUnavailable
        }
    }
}
