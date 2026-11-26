// ===== Navbar scroll effect =====
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 40);
});

// ===== Mobile menu =====
const hamburger = document.getElementById('hamburger');
const navLinks = document.getElementById('navLinks');
hamburger.addEventListener('click', () => navLinks.classList.toggle('open'));
navLinks.querySelectorAll('a').forEach(a =>
  a.addEventListener('click', () => navLinks.classList.remove('open'))
);

// ===== Projects dropdown (click/tap toggle; hover handled in CSS) =====
const dropdown = document.getElementById('projectsDropdown');
const dropdownToggle = document.getElementById('dropdownToggle');
dropdownToggle.addEventListener('click', e => {
  e.stopPropagation();
  const open = dropdown.classList.toggle('open');
  dropdownToggle.setAttribute('aria-expanded', open);
});
document.addEventListener('click', e => {
  if (!dropdown.contains(e.target)) {
    dropdown.classList.remove('open');
    dropdownToggle.setAttribute('aria-expanded', false);
  }
});

// ===== Typed effect =====
const roles = [
  'Computer Science Student',
  'Volunteer',
  'Former Store Owner'
];
const typedEl = document.getElementById('typed');
let roleIndex = 0, charIndex = 0, deleting = false;

function type() {
  const current = roles[roleIndex];
  typedEl.textContent = deleting
    ? current.substring(0, charIndex--)
    : current.substring(0, charIndex++);

  let speed = deleting ? 40 : 90;
  if (!deleting && charIndex > current.length) {
    speed = 1600;
    deleting = true;
  } else if (deleting && charIndex < 0) {
    deleting = false;
    roleIndex = (roleIndex + 1) % roles.length;
    speed = 400;
  }
  setTimeout(type, speed);
}
type();

// ===== Counter animation =====
function animateCount(el) {
  const target = +el.dataset.target;
  const duration = 1200;
  const start = performance.now();
  (function step(now) {
    const p = Math.min((now - start) / duration, 1);
    el.textContent = Math.floor(p * target) + (p === 1 ? '+' : '');
    if (p < 1) requestAnimationFrame(step);
  })(performance.now());
}

// ===== Scroll reveal + counter trigger =====
document.querySelectorAll(
  '.about-grid, .stats, .skills-grid, .timeline-item, .edu-card, .project-card, .cert-card, .contact-card, .section-title'
).forEach(el => el.classList.add('reveal'));

const observer = new IntersectionObserver(entries => {
  entries.filter(e => e.isIntersecting).forEach((entry, i) => {
    // stagger items that come into view together
    entry.target.style.transitionDelay = i * 90 + 'ms';
    setTimeout(() => { entry.target.style.transitionDelay = ''; }, 1200 + i * 90);
    entry.target.classList.add('visible');
    if (entry.target.classList.contains('stats')) {
      entry.target.querySelectorAll('.count').forEach(animateCount);
    }
    observer.unobserve(entry.target);
  });
}, { threshold: 0.15 });

document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

// ===== Active nav link highlight =====
document.querySelectorAll('section[id], header[id]').forEach(sec => {
  new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      navLinks.querySelectorAll('a').forEach(a =>
        a.classList.toggle('active', a.getAttribute('href') === '#' + sec.id)
      );
    });
  }, { rootMargin: '-40% -49% -55% -49%' }).observe(sec);
});

// ===== Horizontal scroll (GSAP ScrollTrigger, desktop only) =====
const hscroll = document.getElementById('hscroll');
const track = document.getElementById('track');
const panels = [...track.children];

// shrink a section's content if it's taller than the screen
function fitPanels() {
  panels.forEach(panel => {
    const box = panel.querySelector('.container');
    box.style.zoom = '';
    const cs = getComputedStyle(panel);
    const avail = panel.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
    if (box.offsetHeight > avail) box.style.zoom = (avail / box.offsetHeight).toFixed(3);
  });
}

if (window.gsap && window.ScrollTrigger) {
  gsap.registerPlugin(ScrollTrigger);
  const distance = () => track.scrollWidth - window.innerWidth;
  const starLayer = document.getElementById('starLayer');
  const star = document.getElementById('star');
  const TRAIL = 10;
  const dots = document.getElementById('dots');
  let pinTrigger = null;

  // one progress dot per section
  const dotBtns = panels.map(panel => {
    const btn = document.createElement('button');
    const name = panel.id === 'home' ? 'Home' : panel.id[0].toUpperCase() + panel.id.slice(1);
    btn.type = 'button';
    btn.title = name;
    btn.setAttribute('aria-label', 'Go to ' + name);
    btn.addEventListener('click', () => scrollToPanel('#' + panel.id));
    dots.appendChild(btn);
    return btn;
  });
  function setActiveDot(i) {
    dotBtns.forEach((b, j) => b.classList.toggle('active', i === j));
  }

  gsap.matchMedia().add('(min-width: 801px)', () => {
    document.body.classList.add('hs');
    fitPanels();
    ScrollTrigger.addEventListener('refreshInit', fitPanels);
    setActiveDot(0);

    // pin the page and turn vertical scroll into sideways travel, snapping to each section
    const tween = gsap.to(track, {
      x: () => -distance(),
      ease: 'none',
      scrollTrigger: {
        trigger: hscroll,
        pin: true,
        scrub: 1,
        end: () => '+=' + distance(),
        invalidateOnRefresh: true,
        snap: { snapTo: 1 / (panels.length - 1), inertia: false, duration: { min: .25, max: .7 }, delay: .05, ease: 'power2.inOut' },
        onUpdate: self => {
          setActiveDot(Math.round(self.progress * (panels.length - 1)));
          document.body.classList.toggle('hs-moved', self.progress > .02);
        },
        onLeave: () => document.body.classList.add('hs-past'),
        onEnterBack: () => document.body.classList.remove('hs-past')
      }
    });
    pinTrigger = tween.scrollTrigger;

    return () => {
      ScrollTrigger.removeEventListener('refreshInit', fitPanels);
      document.body.classList.remove('hs', 'hs-moved', 'hs-past');
      panels.forEach(p => { p.querySelector('.container').style.zoom = ''; });
      pinTrigger = null;
    };
  });

  // mobile keeps the normal vertical page; the dots follow whichever section is mid-screen
  function updateMobileDot() {
    if (pinTrigger) return;
    let i = 0;
    panels.forEach((p, j) => { if (p.getBoundingClientRect().top <= innerHeight / 2) i = j; });
    setActiveDot(i);
  }
  window.addEventListener('scroll', updateMobileDot, { passive: true });
  updateMobileDot();

  // hero entrance
  gsap.from('.hero-text > *', { y: 40, autoAlpha: 0, duration: .9, stagger: .08, ease: 'power3.out' });
  gsap.from('.img-frame', { scale: .85, autoAlpha: 0, duration: 1.1, ease: 'power3.out', delay: .2 });

  // the star keeps flying around the screen leaving a trail of fading copies;
  // scrolling pushes it along its path, spins it faster and makes it grow
  const copyStar = cls => {
    const el = star.cloneNode(true);
    el.removeAttribute('id');
    el.setAttribute('class', cls);
    starLayer.insertBefore(el, star);
    return el;
  };
  const ghosts = Array.from({ length: TRAIL }, () => copyStar('ghost'));
  // two tiny stars that circle and chase the main one
  const minis = [0, 1].map(i => ({
    el: copyStar('mini'), x: innerWidth / 2, y: innerHeight / 2,
    follow: i ? .05 : .08, orbit: i ? 95 : 70, phase: i * Math.PI, dir: i ? -1 : 1
  }));
  const s = { x: innerWidth / 2, y: innerHeight / 2, angle: 0, scale: 0, speed: 0, lastY: scrollY, intro: 0 };
  const trail = [];
  gsap.to(s, { intro: 1, duration: 1.4, ease: 'back.out(1.6)', delay: .3 });
  const place = (el, x, y, angle, scale) => {
    el.style.transform = `translate(${x}px, ${y}px) rotate(${angle}deg) scale(${scale})`;
  };
  function flyStar(time, dt) {
    const w = innerWidth, h = innerHeight;
    const small = w <= 800; // the star is drawn smaller on phones (see style.css)
    const max = document.documentElement.scrollHeight - h;
    const p = pinTrigger ? pinTrigger.progress : (max > 0 ? scrollY / max : 0);
    const f = Math.min(dt / 16.7, 3); // keep speeds the same at any frame rate
    s.speed += ((scrollY - s.lastY) / Math.max(dt, 1) * 1000 - s.speed) * .15;
    s.lastY = scrollY;
    const tx = w * .5 + w * (small ? .34 : .4) * Math.sin(time * .8 + p * 9);
    const ty = h * .55 + h * .3 * Math.sin(time * 1.15 + p * 13) * Math.cos(time * .37 + p * 4);
    s.x += (tx - s.x) * .07 * f;
    s.y += (ty - s.y) * .07 * f;
    s.angle += (1.2 + Math.abs(s.speed) * .004) * f;
    s.scale += (s.intro * (1 + Math.min(Math.abs(s.speed) / 2500, .9)) - s.scale) * .12;
    place(star, s.x, s.y, s.angle, s.scale);

    trail.unshift([s.x, s.y, s.angle, s.scale]);
    trail.length = Math.min(trail.length, TRAIL * 4 + 1);
    ghosts.forEach((g, i) => {
      const [x, y, angle, scale] = trail[Math.min((i + 1) * 4, trail.length - 1)];
      const k = 1 - (i + 1) / (TRAIL + 1);
      place(g, x, y, angle, scale * (.35 + .55 * k));
      g.style.opacity = (k * .5).toFixed(3);
    });

    minis.forEach(m => {
      const a = time * 2.4 * m.dir + m.phase;
      const r = m.orbit * s.scale * (small ? .65 : 1);
      m.x += (s.x + Math.cos(a) * r - m.x) * m.follow * f;
      m.y += (s.y + Math.sin(a) * r - m.y) * m.follow * f;
      place(m.el, m.x, m.y, -s.angle * 2, s.scale * .22);
    });
  }
  gsap.ticker.add(flyStar);

  // in-page links scroll to the matching point along the track
  function scrollToPanel(hash, behavior = 'smooth') {
    const panel = hash && document.querySelector(hash);
    if (!panel || panel.parentElement !== track) return false;
    if (!pinTrigger) { panel.scrollIntoView({ behavior }); return true; }
    const offset = panel.offsetLeft - (window.innerWidth - panel.offsetWidth) / 2;
    window.scrollTo({ top: pinTrigger.start + Math.min(Math.max(offset, 0), distance()), behavior });
    return true;
  }
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const hash = a.getAttribute('href');
      if (scrollToPanel(hash)) {
        e.preventDefault();
        history.replaceState(null, '', hash);
      }
    });
  });

  document.fonts.ready.then(() => {
    ScrollTrigger.refresh();
    if (location.hash) scrollToPanel(location.hash, 'instant');
  });
}
