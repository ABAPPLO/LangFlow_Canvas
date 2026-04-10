import { create } from "zustand";

type LanguageStoreType = {
  language: string;
  setLanguage: (language: string) => void;
};

export const useLanguageStore = create<LanguageStoreType>((set) => ({
  language: (() => {
    const stored = window.localStorage.getItem("languagePreference");
    return stored ?? "en";
  })(),
  setLanguage: (language: string) => {
    set(() => ({ language }));
    window.localStorage.setItem("languagePreference", language);
  },
}));
