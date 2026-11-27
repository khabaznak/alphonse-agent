# Alphonse Desktop window blink investigation

## November 27 incident

All times are America/Mexico_City. The retained request was submitted at 09:12:39.706. The answer receipt was acknowledged at 09:12:57.270.

macOS retained unified logs throughout that interval. With info and debug records included, Alphonse Desktop PID 48163 was the only app PID found in the 09:11:50–09:14:00 window. The app reported the same WebKit content PID 48201 both before and after the reply.

| Time | WebKit content PID | Reported memory footprint |
| --- | --- | --- |
| 09:12:11.777 | 48201 | 4,177 MB |
| 09:12:41.776 | 48201 | 4,212 MB |
| 09:13:11.777 | 48201 | 4,198 MB |
| 09:13:41.777 | 48201 | 4,189 MB |

The narrower 09:12:20–09:13:12 check found 748 ResourceLoading records and no main-frame navigation start, commit, or completion marker. The same WebKit installation emitted explicit main-frame start and commit records during the controlled fixture reloads. A separate system-process query found no related WindowServer, WebKit GPU, or process-manager termination or reset marker. These records provide evidence against an app or renderer restart and against a logged main-frame reload during the reply. Actual window pixels and the renderer's allocation history were outside the historical logging. The high footprint is a lead for further diagnosis. It does not establish a leak or the cause of the blink.

## Controlled reproduction

An isolated native macOS WKWebView loaded the real Desktop frontend with synthetic IPC and nonpersistent WebKit storage. The fixture submitted and acknowledged two private Gmail cards, then held a third reply pending. The termination test targeted only the renderer PID obtained from the isolated view and invoked the same reload action as the pinned Tauri runtime's default WebKit termination handler. The installed app and household daemon were outside the fixture. The endpoint was loopback-only.

- [Normal reply](native-control.json). One page load. Both earlier cards stayed visible in history and the third arrived automatically.
- [Forced same-process reload](native-reload.json). The app window and renderer PID stayed the same. The second document had an animation-frame callback with the Desktop shell absent, followed by the shell mounting 66 ms later. Receipt-only history left two receipts. The third answer remained ready until the test clicked **Show answer**. After that click it was the only rendered card.
- [Stopped renderer and reloaded](native-terminate.json). The native termination callback fired, the renderer PID changed, the host window stayed open, and the second document registered a frame without the shell. With receipt-only history the recovery was the same as the explicit reload. This action intentionally stopped the disposable renderer.
- [Reload with card restoration](native-restored.json). Both acknowledged cards came back directly from authorized mock history. The in-flight answer still required manual recovery from **Needs your attention**.
- [Chromium comparison](chromium-run.txt). The browser control repeated the same delivery and card-loss distinctions.

The frame callbacks measure the state of the page shell, not the exact screen pixels or user-perceived flash duration. Early attempts failed before the target phase while preparing the simulated composer. Those traces are retained. The succeeding attempts waited for owner draft status before submitting.

## Relevance to the report

A WebKit reload can produce a whole-page interruption and explain earlier cards turning into receipts under the previous history behavior. The reproduction only showed the pending card after explicit recovery. In the reported turn, Desktop recorded acknowledgment 18 seconds after submission, and the screenshots alone do not establish a recovery interaction. The controlled mechanism therefore does not establish that the original blink was a reload.
