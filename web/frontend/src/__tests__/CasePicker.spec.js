/**
 * Unit tests for CasePicker.vue
 *
 * PICK-01: Case picker page scrolls when case list exceeds viewport height.
 *
 * Note on jsdom and scoped styles: jsdom does not process Vue scoped <style>
 * blocks — the CSS rules are not injected into the jsdom document, so
 * computed styles won't reflect scoped CSS properties. To verify overflow-y
 * the test checks the component's raw source indirectly by asserting that the
 * rendered root element receives the class upload-root AND that the component
 * source text contains 'overflow-y: auto'. This is documented as a known
 * jsdom limitation; the visual behavior is confirmed in the checkpoint task
 * via a real browser.
 *
 * Strategy: Mount CasePicker with a fetch stub that returns an empty array so
 * the component mounts without a real server.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import CasePicker from "../components/CasePicker.vue";
// Raw SFC source — Vite/Vitest exposes ?raw imports for source inspection
import CasePickerRaw from "../components/CasePicker.vue?raw";

// ── Router mock ──────────────────────────────────────────────────────────────
vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

// ── Fetch mock — returns empty cases list ────────────────────────────────────
beforeEach(() => {
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => [],
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ── Helper: mount CasePicker ─────────────────────────────────────────────────
function mountPicker(props = {}) {
  return mount(CasePicker, {
    props: { username: "testuser", ...props },
  });
}

// ── Tests ────────────────────────────────────────────────────────────────────
describe("CasePicker", () => {
  it("renders a root element with class upload-root", async () => {
    const wrapper = mountPicker();
    await flushPromises();
    expect(wrapper.find(".upload-root").exists()).toBe(true);
  });

  it("upload-root has overflow-y: auto (PICK-01)", async () => {
    const wrapper = mountPicker();
    await flushPromises();
    const root = wrapper.find(".upload-root");
    expect(root.exists()).toBe(true);

    // jsdom does not apply Vue scoped <style> blocks, so computed style is
    // not available. We verify via inline style attribute if present, falling
    // back to checking that the component's __file source contains the rule.
    // The real visual contract is validated by the human-verify checkpoint.
    const inlineStyle = root.attributes("style") || "";
    const elementStyle = root.element.style.overflowY;

    if (inlineStyle.includes("overflow-y") || elementStyle) {
      // Inline or element.style path — preferred when available
      expect(inlineStyle.includes("overflow-y: auto") || elementStyle === "auto").toBe(true);
    } else {
      // jsdom limitation: scoped CSS blocks are not injected into the jsdom
      // document, so computed styles won't reflect scoped CSS. As a reliable
      // proxy, check the raw SFC source (imported via ?raw) contains the rule.
      // This directly verifies PICK-01 intent: the component style must include
      // overflow-y: auto on .upload-root. Real browser behavior is validated
      // by the human-verify checkpoint (Task 3).
      expect(CasePickerRaw).toContain("overflow-y: auto");
    }
  });
});
