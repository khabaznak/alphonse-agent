# Desktop design system — #64

Implements the first foundation in #63, Alphonse Design Study 01. The shell,
Settings workspace, flowing responses, widget drawer and later workflow changes
remain in their respective implementation issues.

## Appearance

`desktop/src/styles.css` owns semantic light/dark tokens for surfaces, text,
borders, inputs, accents, focus, navigation, user messages, code, overlays and
success/warning/error states. Existing questions, schedules, forms and management
workspaces consume these tokens. State text and existing labels remain visible;
color is supplementary. Shared base controls, card/badge primitives, spacing and
radii are available for the following workspace work.

Appearance defaults to the OS setting and follows live changes. Settings →
Appearance offers a native System/Light/Dark selector. Overrides are local to
this Mac and keyed by authenticated household user ID. Before onboarding the
skin follows the system; anonymous choices last only for the current session.
The former Classic/Modern preference is preserved but does not select a new
color skin. Storage failures leave session appearance usable.

Manrope headings and DM Sans body/controls are bundled, with their SIL OFL
licenses and system fallbacks, in `desktop/public/fonts`. The Tauri CSP now
allows only local fonts and removes the former Google Fonts exceptions.

## Motion decision

Use CSS transitions/keyframes; do not add Anime.js. Current requirements are
short control feedback, mascot entrance/bobbing, and activity indicators. CSS
covers those needs without a new animation runtime. Retain the mascot state
images. The shared reduced-motion media query disables all animations,
transitions and smooth scrolling, including progress and verification spinners;
status text still conveys the activity. Reassess Web Animations or a library only
when a concrete later interaction needs coordinated sequencing.

Use 140 ms control transitions and 220 ms entrance motion for new surfaces.
Use the 4/8/12/16/24 px spacing scale, 7 px controls and 10 px cards; retain
existing compact chart labels and layout until the shell/workspace issues.

## Validation

- `cd desktop && npm test`: all 63 tests pass, including per-user persistence,
  legacy/default values, unavailable storage, system resolution, live media
  updates, explicit overrides and listener cleanup.
- `cd desktop && npm run build`: TypeScript and Vite production build pass.
  Vite reports the existing main-bundle size advisory (over 500 kB).
- `cd desktop && npm run tauri -- build --bundles app`: macOS ARM Tauri package
  builds successfully. This uses the existing development daemon sidecar;
  it is a UI verification package, not a release-ready standalone daemon.
- Opened the packaged `.app` with `ALPHONSE_DESKTOP_DIAGNOSTIC_MODE=static`:
  verified system-dark, explicit Light and Dark, native selector keyboard
  operation, visible focus, readable typography, overlay and panel surfaces.

- Local browser checks confirmed both font families load, zero Google Fonts
  requests, live system dark→light changes, zero active CSS animations under
  reduced motion, and themed error/disabled states. The browser preview reports
  the expected missing-Tauri bridge error. Error/disabled states still need a
  dedicated native workflow pass before closing #64.

Real household workflow acceptance remains #78. This foundation does not close
#63 or imply reliable daily-use acceptance.
