<template>
  <div class="header">
    <div class="header-left">
      <button class="hdr-btn" @click="emit('back')">
        <ArrowLeft :size="14" :stroke-width="2" />
        <span>Cases</span>
      </button>
    </div>
    <div class="header-right">
      <span v-if="username" class="username">{{ username }}</span>
      <button class="hdr-btn hdr-btn--small" @click="emit('logout')">Sign Out</button>
      <a
        :href="`/api/cases/${caseId}/csv/`"
        class="hdr-btn"
        download
        title="Download raw detections as CSV"
      >
        <Download :size="14" :stroke-width="2" />
        <span>CSV</span>
      </a>
      <!-- Mode toggle only renders when both options are actually available — if
           the case has no pre-rendered prediction frames there's nothing to toggle
           against and a single button would just look like a dead control. -->
      <div v-if="hasPredictionFrames" class="mode-toggle">
        <button
          class="mode-btn"
          :class="{ 'mode-btn--active': videoMode === 'raw' }"
          @click="emit('update:videoMode', 'raw')"
        >Raw + Overlay</button>
        <button
          class="mode-btn"
          :class="{ 'mode-btn--active': videoMode === 'prediction' }"
          @click="emit('update:videoMode', 'prediction')"
        >Prediction Video</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ArrowLeft, Download } from "lucide-vue-next";

defineProps({
  caseId:              { type: String, required: true },
  username:            { type: String, default: "" },
  videoMode:           { type: String, default: "raw" },
  hasPredictionFrames: { type: Boolean, default: false },
});
const emit = defineEmits(["back", "logout", "update:videoMode"]);
</script>

<style scoped>
.header {
  padding: 10px 20px; display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid var(--border);
  background: var(--bg-1);
  flex-shrink: 0;
}
.header-left  { display: flex; align-items: center; gap: 16px; }
.header-right { display: flex; align-items: center; gap: 10px; }

.username { font-size: 12px; color: var(--text-faint); }

.hdr-btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 14px;
  border: 1px solid var(--border-strong); border-radius: 5px;
  color: var(--text-dim); font-size: 13px;
  background: transparent; text-decoration: none;
  font-family: var(--font-mono); cursor: pointer;
  transition: color .15s, border-color .15s;
}
.hdr-btn:hover { color: var(--text); border-color: var(--accent-border); }
.hdr-btn--small { font-size: 11px; }

.mode-toggle {
  display: flex; overflow: hidden;
  border: 1px solid var(--border-strong); border-radius: 5px;
  font-family: var(--font-mono);
}
.mode-btn {
  padding: 6px 14px; border: none;
  background: transparent;
  color: var(--text-faint); font-size: 12px;
  font-family: inherit; cursor: pointer;
  transition: all .15s;
}
.mode-btn--active { background: var(--accent-soft); color: var(--accent); }
</style>
