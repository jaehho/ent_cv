import { createRouter, createWebHistory } from "vue-router";
import CasePicker from "./components/CasePicker.vue";
import YOLOVisualizer from "./components/YOLOVisualizer.vue";

const routes = [
  { path: "/", redirect: "/cases/" },
  { path: "/cases/", name: "cases", component: CasePicker },
  { path: "/cases/:id", name: "case", component: YOLOVisualizer, props: true },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
