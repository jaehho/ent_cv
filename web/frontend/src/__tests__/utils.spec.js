import { describe, it, expect } from "vitest";
import { formatTime } from "../utils/index.js";

describe("formatTime", () => {
  it("formats 0 seconds", () => {
    expect(formatTime(0)).toBe("0:00.00");
  });

  it("formats 90.5 seconds", () => {
    expect(formatTime(90.5)).toBe("1:30.50");
  });

  it("formats fractional seconds", () => {
    expect(formatTime(3.14)).toBe("0:03.14");
  });
});

