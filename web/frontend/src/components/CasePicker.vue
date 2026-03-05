<template>
  <div class="upload-root">
    <div class="upload-center">
      <div class="upload-header">
        <div></div>
        <div class="sign-out-area">
          <span v-if="username" class="username">{{ username }}</span>
          <button class="sign-out-btn" @click="emit('logout')">Sign Out</button>
        </div>
      </div>
      <div class="upload-label">Mount Sinai &amp; The Cooper Union</div>
      <h1 class="upload-title">ENT Computer Vision</h1>
      <p class="upload-subtitle">Select a prediction case to examine</p>

      <div v-if="loadingCases" class="picker-status">Loading cases…</div>
      <div v-else-if="cases.length === 0" class="picker-status">No cases found in <code>/mnt/data/ent_cv/predictions/</code></div>
      <div v-else>
        <div v-for="group in groupedCases" :key="group.label" style="margin-bottom:28px">
          <div class="date-group-label">{{ group.label }}</div>
          <div class="cases-grid">
            <button
              v-for="c in group.cases"
              :key="c.name"
              class="case-card"
              @click="goToCase(c.name)"
            >
              <div class="case-icon">📁</div>
              <div class="case-name">{{ c.display }}</div>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, shallowRef, onMounted } from "vue";
import { useRouter } from "vue-router";

defineProps({ username: { type: String, default: "" } });
const emit = defineEmits(["logout"]);

const router = useRouter();
const cases = shallowRef([]);
const loadingCases = ref(false);

const DATE_CASE_RE = /^(\d{4})(\d{2})(\d{2})_(\d+)$/;

const groupedCases = computed(() => {
  if (!cases.value || cases.value.length === 0) return [];
  const dateMap = new Map();
  const others = [];
  for (const c of cases.value) {
    const m = DATE_CASE_RE.exec(c);
    if (m) {
      const label = `${m[1]}/${m[2]}/${m[3]}`;
      if (!dateMap.has(label)) dateMap.set(label, []);
      dateMap.get(label).push({
        name: c,
        display: `Case ${parseInt(m[4], 10).toString().padStart(2, "0")}`,
        sortKey: parseInt(m[4], 10),
      });
    } else {
      others.push({ name: c, display: c, sortKey: c });
    }
  }
  for (const arr of dateMap.values()) arr.sort((a, b) => a.sortKey - b.sortKey);
  const groups = [...dateMap.entries()]
    .sort((a, b) => b[0].localeCompare(a[0]))
    .map(([label, cs]) => ({ label, cases: cs }));
  if (others.length > 0) groups.push({ label: "Others", cases: others });
  return groups;
});

async function loadCases() {
  loadingCases.value = true;
  try {
    const res = await fetch("/api/cases/");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    cases.value = await res.json();
  } catch (err) {
    console.error("Failed to load cases:", err);
  } finally {
    loadingCases.value = false;
  }
}

function goToCase(name) {
  router.push({ name: "case", params: { id: name } });
}

onMounted(() => {
  loadCases();
});
</script>

<style scoped>
* { box-sizing: border-box; }
.upload-root {
  min-height: 100vh; background: #06060a;
  display: flex; align-items: center; justify-content: center;
  font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
  color: #e0e0e6;
}
.upload-center { text-align: center; max-width: 860px; width: 100%; padding: 40px; }
.upload-header {
  display: flex; justify-content: flex-end; align-items: center;
  margin-bottom: 24px;
}
.sign-out-area { display: flex; gap: 10px; align-items: center; }
.username { font-size: 12px; color: #666; }
.sign-out-btn {
  font-size: 11px; font-family: inherit; cursor: pointer;
  background: none; border: 1px solid #333; color: #888;
  padding: 4px 12px; border-radius: 4px; transition: border-color 0.2s, color 0.2s;
}
.sign-out-btn:hover { border-color: #4ecdc4; color: #4ecdc4; }
.upload-label { font-size: 13px; letter-spacing: 6px; color: #555; margin-bottom: 12px; text-transform: uppercase; }
.upload-title {
  font-size: 42px; font-weight: 700; margin: 0 0 8px;
  background: linear-gradient(135deg, #ff6b6b, #4ecdc4, #45b7d1);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.upload-subtitle { color: #666; font-size: 14px; margin-bottom: 48px; line-height: 1.6; }
.date-group-label {
  font-size: 13px; font-weight: 600; color: #4ecdc4; letter-spacing: 1px;
  text-transform: none; margin-bottom: 10px; padding-bottom: 4px;
  border-bottom: 1px solid #1a1a25;
}
.cases-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px; width: 100%; margin-bottom: 8px;
}
.case-card {
  padding: 20px 16px; border: 1px dashed #2a2a35;
  border-radius: 10px; cursor: pointer; background: #0c0c14;
  transition: border-color 0.2s, background 0.2s;
  text-align: center; font-family: inherit; color: #ccc;
}
.case-card:hover { border-color: #4ecdc4; background: #0e0e18; }
.case-icon { font-size: 22px; margin-bottom: 8px; }
.case-name { font-size: 11px; font-weight: 600; word-break: break-all; }
.picker-status { font-size: 13px; color: #555; padding: 32px 0; }
.picker-status code { color: #4ecdc4; }
</style>
