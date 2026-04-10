import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import enAlerts from "./locales/en/alerts.json";
import enCommon from "./locales/en/common.json";
import enComponents from "./locales/en/components.json";
import enFlow from "./locales/en/flow.json";
import enHeader from "./locales/en/header.json";
import enLogin from "./locales/en/login.json";
import enModals from "./locales/en/modals.json";
import enSettings from "./locales/en/settings.json";
import enSidebar from "./locales/en/sidebar.json";
import zhAlerts from "./locales/zh-CN/alerts.json";
import zhCommon from "./locales/zh-CN/common.json";
import zhComponents from "./locales/zh-CN/components.json";
import zhFlow from "./locales/zh-CN/flow.json";
import zhHeader from "./locales/zh-CN/header.json";
import zhLogin from "./locales/zh-CN/login.json";
import zhModals from "./locales/zh-CN/modals.json";
import zhSettings from "./locales/zh-CN/settings.json";
import zhSidebar from "./locales/zh-CN/sidebar.json";

const savedLanguage = (() => {
  const stored = window.localStorage.getItem("languagePreference");
  return stored ?? "en";
})();

i18n.use(initReactI18next).init({
  resources: {
    en: {
      common: enCommon,
      modals: enModals,
      alerts: enAlerts,
      header: enHeader,
      login: enLogin,
      sidebar: enSidebar,
      settings: enSettings,
      flow: enFlow,
      components: enComponents,
    },
    "zh-CN": {
      common: zhCommon,
      modals: zhModals,
      alerts: zhAlerts,
      header: zhHeader,
      login: zhLogin,
      sidebar: zhSidebar,
      settings: zhSettings,
      flow: zhFlow,
      components: zhComponents,
    },
  },
  lng: savedLanguage,
  fallbackLng: "en",
  ns: [
    "common",
    "modals",
    "alerts",
    "header",
    "login",
    "sidebar",
    "settings",
    "flow",
    "components",
  ],
  defaultNS: "common",
  interpolation: {
    escapeValue: false,
  },
});

export default i18n;
