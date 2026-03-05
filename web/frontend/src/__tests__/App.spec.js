import { describe, it, expect } from "vitest";
import { mount } from "@vue/test-utils";
import App from "../App.vue";

describe("App.vue", () => {
  it("renders LoginForm when not authenticated", () => {
    const wrapper = mount(App, {
      global: {
        stubs: {
          LoginForm: { template: '<div class="login-stub" />' },
          "router-view": { template: '<div class="router-stub" />' },
        },
      },
    });
    expect(wrapper.find(".login-stub").exists()).toBe(true);
    expect(wrapper.find(".router-stub").exists()).toBe(false);
  });
});
