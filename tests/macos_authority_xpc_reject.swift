// A signed authority must never accept an unsigned/improperly signed client.
// This isolated local test checks rejection only, not a valid signed client.
import Foundation

@objc protocol AuthorityFixtureProtocol {
    func ping(_ reply: @escaping (String) -> Void)
}

final class AuthorityFixture: NSObject, AuthorityFixtureProtocol {
    func ping(_ reply: @escaping (String) -> Void) { reply("unexpected") }
}

final class OpenFixtureListener: NSObject, NSXPCListenerDelegate {
    let service = AuthorityFixture()
    func listener(_ listener: NSXPCListener, shouldAcceptNewConnection connection: NSXPCConnection) -> Bool {
        connection.exportedInterface = NSXPCInterface(with: AuthorityFixtureProtocol.self)
        connection.exportedObject = service
        connection.resume()
        return true
    }
}

@main
struct XPCRejectionCheck {
    static func run(_ listener: NSXPCListener) throws -> Bool {
        listener.resume()
        let connection = NSXPCConnection(listenerEndpoint: listener.endpoint)
        connection.remoteObjectInterface = NSXPCInterface(with: AuthorityFixtureProtocol.self)
        let done = DispatchSemaphore(value: 0)
        let lock = NSLock()
        var delivered = false
        connection.invalidationHandler = { done.signal() }
        connection.resume()
        let proxy = connection.remoteObjectProxyWithErrorHandler { _ in done.signal() } as! AuthorityFixtureProtocol
        proxy.ping { _ in lock.lock(); delivered = true; lock.unlock(); done.signal() }
        let timedOut = done.wait(timeout: .now() + 5) == .timedOut
        connection.invalidate()
        listener.invalidate()
        lock.lock()
        let result = delivered
        lock.unlock()
        guard !timedOut else { throw NSError(domain: "AuthorityXPCTimeout", code: 1) }
        return result
    }

    static func main() throws {
        let desktop = try AuthoritySigningIdentity(team: "ABCD123456", identifier: "local.alphonse.desktop")
        let worker = try AuthoritySigningIdentity(team: "ABCD123456", identifier: "local.alphonse.worker")
        let policy = try AuthorityPeerPolicy(desktop: desktop, worker: worker)
        let service = AuthorityFixture()
        let guardDelegate = try RoleGuardedAuthorityListener(role: .desktop, policy: policy) { _, connection in
            connection.exportedInterface = NSXPCInterface(with: AuthorityFixtureProtocol.self)
            connection.exportedObject = service
            connection.resume()
            return true
        }
        // First prove this environment can actually deliver an XPC call.
        let openDelegate = OpenFixtureListener()
        let openListener = NSXPCListener.anonymous()
        openListener.delegate = openDelegate
        guard try run(openListener) else {
            throw NSError(domain: "AuthorityXPCBaselineUnavailable", code: 1)
        }
        let listener = NSXPCListener.anonymous()
        guardDelegate.configure(listener)
        guard try !run(listener) else {
            throw NSError(domain: "AuthorityXPCFixture", code: 1)
        }
        print("unsigned or wrong-signed XPC peer rejected")
    }
}
