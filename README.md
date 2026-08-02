# Allain A. Kumar — Personal Portfolio

A personal, single-page portfolio website for **Allain A. Kumar**, built with plain HTML, CSS, and vanilla JavaScript. No frameworks, no build step — just open the file in a browser.

![Portfolio preview](photo.jpeg)

## 🚀 Features

- **Responsive, single-page design** — works on desktop and mobile with a collapsible hamburger menu
- **Animated hero section** — typing-effect subtitle cycling through roles
- **Sections:** About, Skills, Work History (timeline), Education, Certificates, and Contact
- **Clickable certificate cards** — each opens the real PDF certificate in a new tab
- **Scroll-based animations** — fade-in on scroll, animated stat counters, sticky navbar
- **Pure frontend** — no backend or installation required

## 📁 Project Structure

```
PORTFOLIO/
├── index.html          # Main page (all content/sections)
├── style.css           # All styling (dark theme, responsive rules, animations)
├── script.js           # Interactions (typing effect, counters, scroll reveal, menu)
├── photo.jpeg          # Profile photo
├── certificates/       # PDF certificates, linked from the Certificates section
│   ├── Introduction-to-Modern-AI.pdf
│   ├── Introduction-to-Data-Science.pdf
│   └── Create-Digital-Content-Communicate-Collaborate-Online.pdf
├── LICENSE             # MIT License
└── README.md           # This file
```

## 💻 How to Use

**To view locally:**
```
1. Make sure index.html, style.css, script.js, and photo.jpeg are in the same folder.
2. Double-click index.html → it opens in your default browser.
```

**To host online (GitHub Pages):**
```
1. Create a new public repository on GitHub.
2. Upload all files (keep the folder structure).
3. Go to Settings → Pages → set Source to the main branch, root.
4. Your site goes live at https://<username>.github.io/<repo-name>/
```

It's also compatible with any static host — Netlify, Vercel, Firebase Hosting, etc. — just drag-and-drop the folder.

## ✏️ Customizing

| To change... | Edit |
|---|---|
| Text, sections, info, contact details | `index.html` |
| Colors, fonts, layout, animations | `style.css` |
| Typing phrases, counters, menu behavior | `script.js` |
| Profile picture | replace `photo.jpeg` (keep the same filename) |
| Certificates | drop new PDFs into `certificates/`, then add a `.cert-card` link in `index.html` |

### Theme colors
Defined at the top of `style.css`. The accent color is `#38bdf8` (sky blue) against a dark `#0f172a` background.

## ⚠️ Note

An internet connection is used only to load the Google Fonts (Inter / Poppins). Offline, the site falls back to system fonts and remains fully functional.

## 📄 License

This project is open source under the **MIT License** — see [LICENSE](LICENSE).

## 👤 Contact

- **Email:** allainkumar9@gmail.com
- **Phone:** +63 929 518 0501
- **Location:** Santa Cruz, Laguna, Philippines
