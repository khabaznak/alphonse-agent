// Executed with an accepted socket inherited from a disposable fixture server.
// The requirement is fixed by the test executable, never received over IPC.
import Foundation

@main
struct AppleSystemClientVerifier {
    static func main() throws {
        guard CommandLine.arguments.count == 3, let fd = Int32(CommandLine.arguments[1]) else {
            throw LocalPeerVerificationError.peerTokenUnavailable
        }
        let correct = CommandLine.arguments[2] == "correct"
        let identifier = correct ? "com.apple.curl" : "com.apple.nonexistent-fixture"
        do {
            try verifyLocalSocketPeerRequirement(fd,
                requirementString: "anchor apple and identifier \"\(identifier)\"")
            guard correct else { fatalError("wrong Apple identity unexpectedly accepted") }
            print("matched Apple system client")
        } catch LocalPeerVerificationError.signatureRejected {
            guard !correct else { throw LocalPeerVerificationError.signatureRejected }
            print("rejected wrong signing identity")
        }
    }
}
