import { useTranslation } from "react-i18next";
import { useLanguageStore } from "@/stores/languageStore";
import { cn } from "@/utils/utils";

export const LanguageSwitcher = () => {
  const { i18n } = useTranslation();
  const { language, setLanguage } = useLanguageStore();

  const handleLanguageChange = (lang: string) => {
    setLanguage(lang);
    i18n.changeLanguage(lang);
  };

  const languages = [
    { code: "en", label: "EN" },
    { code: "zh-CN", label: "中文" },
  ];

  return (
    <div className="relative ml-auto inline-flex rounded-full border border-border">
      {languages.map(({ code, label }, index) => (
        <button
          key={code}
          type="button"
          className={cn(
            "relative z-10 inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors",
            language === code
              ? "bg-foreground text-background"
              : "text-foreground hover:bg-foreground hover:text-background",
            index === 0 && "mr-0.5",
          )}
          onClick={() => handleLanguageChange(code)}
          data-testid={`menu_language_${code}_button`}
        >
          {label}
        </button>
      ))}
    </div>
  );
};

export default LanguageSwitcher;
