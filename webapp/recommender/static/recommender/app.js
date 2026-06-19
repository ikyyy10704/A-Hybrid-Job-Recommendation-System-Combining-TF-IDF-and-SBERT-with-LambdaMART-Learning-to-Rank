(function () {
  function apply(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("jm-theme", theme);
    document.querySelectorAll(".theme-toggle i").forEach(function (ic) {
      ic.className = theme === "light" ? "ph ph-moon" : "ph ph-sun";
    });
  }
  document.addEventListener("DOMContentLoaded", function () {
    apply(localStorage.getItem("jm-theme") || "dark");
    document.querySelectorAll(".theme-toggle").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var cur = document.documentElement.getAttribute("data-theme");
        apply(cur === "dark" ? "light" : "dark");
      });
    });

    // Tombol pendukung "Akurasi model" — buka/tutup panel ringkasan evaluasi.
    var evalToggle = document.getElementById("eval-toggle");
    var evalInfo = document.getElementById("eval-info");
    if (evalToggle && evalInfo) {
      evalToggle.addEventListener("click", function () {
        var open = evalInfo.hasAttribute("hidden");
        if (open) {
          evalInfo.removeAttribute("hidden");
        } else {
          evalInfo.setAttribute("hidden", "");
        }
        evalToggle.setAttribute("aria-expanded", open ? "true" : "false");
      });
    }

    // Tombol "Tampilkan lainnya" — buka sebagian baris tersembunyi (tanpa reload).
    var more = document.getElementById("load-more");
    if (more) {
      var step = parseInt(more.getAttribute("data-step"), 10) || 10;
      more.addEventListener("click", function () {
        var hidden = document.querySelectorAll(".more-row:not(.shown)");
        for (var i = 0; i < hidden.length && i < step; i++) hidden[i].classList.add("shown");
        var left = document.querySelectorAll(".more-row:not(.shown)").length;
        if (left === 0) {
          more.style.display = "none";
        } else {
          var rem = more.querySelector(".rem");
          if (rem) rem.textContent = left;
        }
      });
    }
  });
})();
