// Harmless cross-process test service; run only via the disposable fixture.
import Foundation

@objc protocol AuthorityCrossProcessFixture {
    func ping(_ reply: @escaping (String) -> Void)
}

final class FixtureService: NSObject, AuthorityCrossProcessFixture {
    func ping(_ reply: @escaping (String) -> Void) {
        // The independent observer checks whether a request reached this
        // service, even if the client later rejects its reply.
        FileHandle.standardOutput.write(Data("synthetic-ping\n".utf8))
        reply("synthetic-pong")
    }
}

final class OpenFixture: NSObject, NSXPCListenerDelegate {
    let service: FixtureService
    init(_ service: FixtureService) { self.service = service }
    func listener(_ listener: NSXPCListener, shouldAcceptNewConnection connection: NSXPCConnection) -> Bool {
        connection.exportedInterface = NSXPCInterface(with: AuthorityCrossProcessFixture.self)
        connection.exportedObject = service
        connection.resume()
        return true
    }
}

@main
struct FixtureMain {
    static func main() throws {
        guard CommandLine.arguments.count == 2 else { throw NSError(domain: "fixtureArguments", code: 1) }
        let prefix = CommandLine.arguments[1]
        let service = FixtureService()
        let openDelegate = OpenFixture(service)
        let desktop = try AuthoritySigningIdentity(team: "ABCD123456", identifier: "local.alphonse.desktop")
        let worker = try AuthoritySigningIdentity(team: "ABCD123456", identifier: "local.alphonse.worker")
        let policy = try AuthorityPeerPolicy(desktop: desktop, worker: worker)
        let bind: (AuthorityPeerRole, NSXPCConnection) -> Bool = { _, connection in
            connection.exportedInterface = NSXPCInterface(with: AuthorityCrossProcessFixture.self)
            connection.exportedObject = service
            connection.resume()
            return true
        }
        let desktopDelegate = try RoleGuardedAuthorityListener(role: .desktop, policy: policy, accept: bind)
        let workerDelegate = try RoleGuardedAuthorityListener(role: .worker, policy: policy, accept: bind)
        let open = NSXPCListener(machServiceName: prefix + ".open")
        let desktopListener = NSXPCListener(machServiceName: prefix + ".desktop")
        let workerListener = NSXPCListener(machServiceName: prefix + ".worker")
        // Fixture-only weak requirements prove role routing with ad-hoc
        // binaries; identifiers alone are not an authority boundary.
        let mockDesktop = NSXPCListener(machServiceName: prefix + ".mock-desktop")
        let mockWorker = NSXPCListener(machServiceName: prefix + ".mock-worker")
        open.delegate = openDelegate
        desktopDelegate.configure(desktopListener)
        workerDelegate.configure(workerListener)
        mockDesktop.setConnectionCodeSigningRequirement("identifier \"local.alphonse.fixture.desktop\"")
        mockWorker.setConnectionCodeSigningRequirement("identifier \"local.alphonse.fixture.worker\"")
        mockDesktop.delegate = openDelegate
        mockWorker.delegate = openDelegate
        open.resume()
        desktopListener.resume()
        workerListener.resume()
        mockDesktop.resume()
        mockWorker.resume()
        // All listeners and delegates are retained for the entire fixture lifetime.
        withExtendedLifetime((service, openDelegate, desktopDelegate, workerDelegate,
                              open, desktopListener, workerListener, mockDesktop, mockWorker)) { dispatchMain() }
    }
}
