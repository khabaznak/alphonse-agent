import AppKit
import Darwin
import Foundation
import WebKit

final class Probe: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKScriptMessageHandler {
    let url: URL
    let mode: String
    let output: URL
    var window: NSWindow!
    var view: WKWebView!
    var navStarts = 0
    var navFinishes = 0
    var terminated = 0
    var completed = false
    var before: [String: Any]?
    var beforeProcess: Int32?
    var events: [[String: Any]] = []
    let began = ProcessInfo.processInfo.systemUptime

    init(url: URL, mode: String, output: URL) {
        self.url = url
        self.mode = mode
        self.output = output
    }

    func note(_ name: String, fields: [String: Any] = [:]) {
        var record = fields
        record["event"] = name
        record["ms"] = Int((ProcessInfo.processInfo.systemUptime - began) * 1000)
        events.append(record)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.userContentController.add(self, name: "blinkLifecycle")
        configuration.userContentController.addUserScript(WKUserScript(source: """
          window.webkit.messageHandlers.blinkLifecycle.postMessage({
            event: 'document_start', sidebar: Boolean(document.querySelector('button[title="Conversation"]'))
          });
          let frame = 0;
          function sample() {
            const mounted = Boolean(document.querySelector('button[title="Conversation"]'));
            if (frame === 0 || mounted) window.webkit.messageHandlers.blinkLifecycle.postMessage({
              event: mounted ? 'first_shell_frame' : 'first_frame_without_shell', sidebar: mounted
            });
            if (!mounted && frame++ < 120) requestAnimationFrame(sample);
          }
          requestAnimationFrame(sample);
          """, injectionTime: .atDocumentStart, forMainFrameOnly: true))
        view = WKWebView(frame: NSRect(x: 0, y: 0, width: 1120, height: 760), configuration: configuration)
        view.navigationDelegate = self
        window = NSWindow(contentRect: view.frame, styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.title = "Disposable Alphonse window lifecycle probe"
        window.contentView = view
        window.center()
        window.orderFront(nil)
        note("window_opened", fields: ["pid": getpid(), "window": window.windowNumber])
        view.load(URLRequest(url: url))
        DispatchQueue.main.asyncAfter(deadline: .now() + 35) {
            if !self.completed { self.fail("Window fixture timed out.") }
        }
    }

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        if let fields = message.body as? [String: Any], let name = fields["event"] as? String {
            note(name, fields: ["sidebar": fields["sidebar"] ?? false])
        }
    }

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        navStarts += 1
        note("native_navigation_start")
    }

    func webView(_ webView: WKWebView, didCommit navigation: WKNavigation!) {
        note("native_navigation_commit")
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        navFinishes += 1
        note("native_navigation_finish")
        let initial = navFinishes == 1
        Task { @MainActor in
            do {
                if initial {
                    before = try await call("prepare")
                    guard before?["working"] as? Bool == true,
                          before?["cards"] as? [Int] == [1, 2] else {
                        fail("The fixture was not busy with two earlier private cards.")
                        return
                    }
                    beforeProcess = rendererPid()
                    note("busy_before_injection", fields: ["renderer_pid": beforeProcess ?? 0])
                    if mode == "control" {
                        finish(after: ["after": try await call("finishNormally")])
                    } else if mode == "reload" {
                        note("explicit_native_reload")
                        view.reload()
                    } else {
                        guard let pid = beforeProcess, pid > 100, pid != getpid(), isWebContent(pid) else {
                            fail("The isolated WKWebView renderer could not be identified safely.")
                            return
                        }
                        note("isolated_renderer_stop", fields: ["renderer_pid": pid])
                        guard kill(pid, SIGKILL) == 0 else {
                            fail("The isolated renderer could not be stopped.")
                            return
                        }
                    }
                } else {
                    let after = try await call("settleAfterReload")
                    finish(after: after)
                }
            } catch {
                fail(error.localizedDescription)
            }
        }
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        terminated += 1
        note("native_web_content_terminated")
        if mode == "terminate" && terminated == 1 {
            // Match the default Tauri runtime's WebKit termination handler.
            note("reload_from_termination_handler")
            view.reload()
        } else {
            fail("Unexpected WebKit content termination.")
        }
    }

    private func rendererPid() -> Int32? {
        let selector = NSSelectorFromString("_webProcessIdentifier")
        guard view.responds(to: selector), let implementation = view.method(for: selector) else { return nil }
        typealias Identifier = @convention(c) (NSObject, Selector) -> Int32
        return unsafeBitCast(implementation, to: Identifier.self)(view, selector)
    }

    private func isWebContent(_ pid: Int32) -> Bool {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-p", String(pid), "-o", "comm="]
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do { try process.run() } catch { return false }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return process.terminationStatus == 0 && String(decoding: data, as: UTF8.self).lowercased().contains("webkit.webcontent")
    }

    private func call(_ method: String) async throws -> [String: Any] {
        for _ in 0..<150 {
            if try await view.evaluateJavaScript("Boolean(window.reloadProbe)") as? Bool == true { break }
            try await Task.sleep(nanoseconds: 40_000_000)
        }
        _ = try await view.evaluateJavaScript("window.reloadProbe.start('" + method + "')")
        for _ in 0..<220 {
            let value = try await view.evaluateJavaScript("JSON.stringify({result: window.reloadProbe.result, error: window.reloadProbe.error})")
            if let json = value as? String, let row = try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any] {
                if let error = row["error"] as? String { throw NSError(domain: "Fixture", code: 1, userInfo: [NSLocalizedDescriptionKey:error]) }
                if let result = row["result"] as? [String: Any] { return result }
            }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        throw NSError(domain: "Fixture", code: 2, userInfo: [NSLocalizedDescriptionKey:"No result from " + method])
    }

    private func finish(after: [String: Any]) {
        let receipt: [String: Any] = ["mode": mode, "host_pid": getpid(), "window": window.windowNumber,
            "renderer_before": beforeProcess ?? 0, "renderer_after": rendererPid() ?? 0,
            "navigation_starts": navStarts, "navigation_finishes": navFinishes, "content_terminations": terminated,
            "before": before ?? [:], "result": after, "events": events]
        do { try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys]).write(to: output) }
        catch { fail(error.localizedDescription); return }
        view.takeSnapshot(with: nil) { image, _ in
            if let tiff = image?.tiffRepresentation, let bitmap = NSBitmapImageRep(data: tiff),
               let png = bitmap.representation(using: .png, properties: [:]) {
                try? png.write(to: self.output.deletingPathExtension().appendingPathExtension("png"))
            }
            self.completed = true
            NSApp.terminate(nil)
        }
    }

    private func fail(_ message: String) {
        let receipt: [String: Any] = ["mode": mode, "error": message, "events": events,
                                      "before": before ?? [:], "content_terminations": terminated]
        if let data = try? JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted]) { try? data.write(to: output) }
        fputs(message + "\n", stderr)
        exit(1)
    }
}

let args = CommandLine.arguments
guard args.count == 4, let url = URL(string: args[1]), url.scheme == "http", url.host == "127.0.0.1",
      url.port == 4180, url.path == "/desktop/tests/window-blink/fixture.html",
      ["control", "reload", "terminate"].contains(args[2]) else { exit(2) }
let delegate = Probe(url: url, mode: args[2], output: URL(fileURLWithPath: args[3]))
let app = NSApplication.shared
app.setActivationPolicy(.accessory)
app.delegate = delegate
app.run()
