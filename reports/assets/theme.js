(() => {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch {}
  if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
  document.addEventListener("DOMContentLoaded", () => {
    const buttons = document.querySelectorAll("[data-theme-choice]");
    const sync = () => buttons.forEach((b) => b.setAttribute("aria-pressed", String((document.documentElement.dataset.theme || "auto") === b.dataset.themeChoice)));
    buttons.forEach((b) => b.addEventListener("click", () => {
      const choice = b.dataset.themeChoice;
      if (choice === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = choice;
      try { choice === "auto" ? localStorage.removeItem("theme") : localStorage.setItem("theme", choice); } catch {}
      sync();
    }));
    sync();
  });
})();
