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

## Renderer memory and the later reload

The original renderer was launched in a development Desktop process on
November 26. The macOS log first measured 191 MiB at 12:24:44 and recorded
3,596 MiB by 14:23:44. At 09:12 on November 27 it was around 4,200 MiB.
At **09:34:23** it reached 4,771 MiB. WebKit ran memory relief, measured
4,734 MiB, and logged that the process exceeded its inactive memory limit.
The existing Alphonse host terminated WebContent PID 48201 and loaded the
main frame in PID 51747. The replacement renderer reported 108 MiB in its
first periodic sample. This reload occurred about 22 minutes after the
reported 09:12 reply.

WebKit's termination statistics recorded 4,632 MiB of compressed memory,
119 MiB resident, 6 MiB of JavaScript GC heap, one page and one document, and
no back-forward cache pages. The VM tag breakdown attributed 4,687 MiB of dirty
memory to bmalloc in 3,729 regions. The postmortem identifies the allocator
and memory accounting. It lacks allocation call sites and object ownership.

## Disposable native memory comparison

In two simultaneous five-minute trials, an ad-hoc signed copy of the Desktop
bundle with private-history restoration used the real Tauri bridge and WebKit.
An isolated Unix socket supplied synthetic data. The idle control ran 354 polls,
ending at 62.2 MiB. The reply trial ran 353 polls and 117 history reads, advancing
to 120 private cards and 240 total conversation rows. It ended at 153.7 MiB and
peaked at 177.5 MiB. Live WebKit malloc bytes in the reply trial climbed from
63.5 MiB at 21 cards to 95.8 MiB at 120 cards. The footprint stayed around 152–154
MiB for the last 44 seconds after the history stopped growing. Both trials kept
one renderer and one committed main-frame load. The measured trial began at the
first periodic WebKit process report about 36 seconds after launch.

These trials reproduce the ordinary memory cost of a growing timeline. They
did not reproduce the gigabyte-scale historical growth or an automatic reload.
They used the packaged debug app, a synthetic IPC server and a short observation
window. The original process used the development app and real household runtime.

On macOS, run each trial from the repository root with an existing Desktop bundle:

```sh
python3 desktop/tests/window-blink/native_tauri_probe.py \
  "/path/to/Alphonse Desktop.app" idle /tmp/alphonse-memory-samples
python3 desktop/tests/window-blink/native_tauri_probe.py \
  "/path/to/Alphonse Desktop.app" replies /tmp/alphonse-memory-samples
```

The probe creates uniquely identified fixture copies with dedicated Unix sockets,
reads their renderer's allocation totals using `vmmap`, and collects navigation
and renderer termination markers. [Idle samples](memory-idle.json) and
[reply samples](memory-replies.json) include IPC counts and memory categories.
[Idle lifecycle](memory-idle-lifecycle.txt) and
[reply lifecycle](memory-replies-lifecycle.txt) preserve the native markers.
