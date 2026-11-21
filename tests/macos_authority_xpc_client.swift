// Sends only an invented ping; never sends authority requests or secrets.
import Foundation

@objc protocol AuthorityCrossProcessFixture {
    func ping(_ reply: @escaping (String) -> Void)
}

@main
struct ClientMain {
    static func main() {
        guard CommandLine.arguments.count == 2 || CommandLine.arguments.count == 3 else { exit(2) }
        let done = DispatchSemaphore(value: 0)
        let lock = NSLock()
        var result = "timeout"
        let connection = NSXPCConnection(machServiceName: CommandLine.arguments[1], options: [])
        if CommandLine.arguments.count == 3 {
            connection.setCodeSigningRequirement(CommandLine.arguments[2])
        }
        connection.remoteObjectInterface = NSXPCInterface(with: AuthorityCrossProcessFixture.self)
        connection.invalidationHandler = { lock.lock(); if result == "timeout" { result = "invalidated" }; lock.unlock(); done.signal() }
        connection.resume()
        let proxy = connection.remoteObjectProxyWithErrorHandler { _ in
            lock.lock(); if result == "timeout" { result = "rejected" }; lock.unlock(); done.signal()
        } as! AuthorityCrossProcessFixture
        proxy.ping { value in
            lock.lock(); result = value == "synthetic-pong" ? "delivered" : "unexpected"; lock.unlock(); done.signal()
        }
        _ = done.wait(timeout: .now() + 8)
        lock.lock(); let final = result; lock.unlock()
        connection.invalidate()
        print(final)
        if final == "timeout" || final == "unexpected" { exit(1) }
    }
}
