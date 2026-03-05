import { describe, it, expect } from "vitest";
import { formatTime, formatRealTime } from "../utils/index.js";

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

describe("formatRealTime", () => {
  it("formats 0 seconds", () => {
    expect(formatRealTime(0)).toBe("0:00:00");
  });

  it("formats 3661 seconds as 1:01:01", () => {
    expect(formatRealTime(3661)).toBe("1:01:01");
  });

  it("formats 59 seconds", () => {
    expect(formatRealTime(59)).toBe("0:00:59");
  });
});
