# Phase 3 design comparison

Reference: the user-supplied `design.pptx`, one slide with four phone states. The mockup's 70% progress, 92.3% confidence and unsupported classes are design content, not application data or instructions to fabricate predictions.

The slide was rendered and inspected directly. Its phone UI uses Arial, white surfaces, dark green controls, light neutral cards, pink detection cards and pale green recommendations. Notable reference colors include `#071323`, `#093C30`, `#3D536E`, `#35A364` and `#710F09`. The implementation follows the same hierarchy, component order, rounded image/cards, outlined camera button, leaf motif and bottom decoration. The application shows one state at a time, with a centered phone-like frame on desktop.

## Screenshot review

Screenshots are captured with Chromium at height 844 and widths 375, 390 and 430. Filename pattern: `{width}-01-upload.png`, `-02-preview.png`, `-03-analyzing.png`, `-04-result.png`.

| Screen | Comparison and deliberate functional changes |
| --- | --- |
| Upload | Leaf mark/header, two-line title, three-line description, central leaf illustration, solid Upload, outlined Camera, support card and corner leaves follow the reference. The initial bottom illustration was visibly misplaced; it was replaced with two light corner leaves after screenshot review. |
| Preview | Same full-width rounded leaf photo, neutral quality card and two stacked actions. The checklist explicitly says “Before analyzing, please check” so it does not claim automated quality assessment. The photo/card heights are adjusted to fit the requested 844px viewport. |
| Analyzing | Same centered leaf, title, scanned user image, scan line, processing text and lower status section. Uses indeterminate waiting rather than a fake 70% or unsupported completed backend stages. A real request error stops the animation and exposes retry. |
| Result | Same pink thumbnail/disease card, prediction bars, green recommendations and two bottom buttons. Contains exactly four supported classes; values come from the API. Recommendations are tentative. A short model-guidance note and readable native dialog are added. Share appears only where the Web Share API is available. |

No horizontal overflow occurred at any tested width. Upload → Preview → Analyzing → Result, reset, image replacement, retry, invalid response rejection and offline error behavior are covered in Playwright. Camera input's capture attribute and photo selection are covered; actual mobile hardware capture remains a manual device check.

The reference's vector leaf artwork was recreated as SVG and is close in shape/palette, not a pixel-identical licensed illustration asset. Icons use Lucide, so individual icon contours differ slightly from the slide. No screenshot is used as the interface. The sample leaf extracted from the design is only a test fixture, not a production default image. Screenshot prediction values (70/12/10/8%) are test-only intercepted responses; no model accuracy or confidence is inferred from them.

## Remaining verification

- Actual trained-checkpoint responses are blocked in Codex by HF HTTP 403.
- Memory/latency under Render Free constraints have not passed a real-artifact service test.
- Real-phone camera permissions, iOS/Android installation and cold-start behavior need device/hosting checks after review.
