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
  let panelAt = i => i / (panels.length - 1); // scroll progress where section i is in view
  let rebuild = null;
  let activeIndex = 0;

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
    activeIndex = i;
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
    panelAt = i => i / (panels.length - 1);

    return () => {
      ScrollTrigger.removeEventListener('refreshInit', fitPanels);
      document.body.classList.remove('hs', 'hs-moved', 'hs-past');
      panels.forEach(p => { p.querySelector('.container').style.zoom = ''; });
      pinTrigger = null;
    };
  });

  // phones get the same sideways travel, but a section taller than the screen
  // first scrolls up inside its panel before the next one slides in
  gsap.matchMedia().add('(max-width: 800px)', () => {
    document.body.classList.add('hs');
    let tl = null, moves = [], lastWidth = innerWidth;
    const boxes = panels.map(p => p.querySelector('.container'));

    function build() {
      const current = tl ? activeIndex : 0;
      if (tl) { tl.scrollTrigger.kill(true); tl.kill(); }
      gsap.set(track, { x: 0 });
      gsap.set(boxes, { y: 0 });
      tl = gsap.timeline({ defaults: { ease: 'none' } });
      moves = [];
      const marks = [];
      // timeline durations are in pixels, so scroll length matches the distance travelled
      panels.forEach((panel, i) => {
        const cs = getComputedStyle(panel);
        const over = boxes[i].offsetHeight - (panel.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom));
        panel.classList.toggle('tall', over > 0);
        marks.push(tl.duration());
        if (over > 0) tl.to(boxes[i], { y: -over, duration: over });
        const next = panels[i + 1];
        if (next) {
          moves.push([tl.duration(), tl.duration() + next.offsetWidth]);
          tl.to(track, { x: -next.offsetLeft, duration: next.offsetWidth });
        }
      });
      const total = tl.duration();
      pinTrigger = ScrollTrigger.create({
        animation: tl,
        trigger: hscroll,
        pin: true,
        scrub: 1,
        end: '+=' + total,
        // only snap while a section is part-way through sliding in
        snap: {
          snapTo: v => {
            const t = v * total;
            const m = moves.find(([a, b]) => t > a && t < b);
            return m ? (t - m[0] < m[1] - t ? m[0] : m[1]) / total : v;
          },
          inertia: false, duration: { min: .25, max: .7 }, delay: .05, ease: 'power2.inOut'
        },
        onUpdate: self => {
          const t = self.progress * total;
          setActiveDot(moves.filter(([a, b]) => t >= (a + b) / 2).length);
          document.body.classList.toggle('hs-moved', self.progress > .02);
        },
        onLeave: () => document.body.classList.add('hs-past'),
        onEnterBack: () => document.body.classList.remove('hs-past')
      });
      panelAt = i => marks[i] / total;
      if (current) { pinTrigger.refresh(); scrollToPanel('#' + panels[current].id, 'instant'); }
    }
    build();
    setActiveDot(0);
    rebuild = build;

    // rebuild when the width changes (rotation), not when the address bar hides
    const onResize = () => {
      if (innerWidth === lastWidth) return;
      lastWidth = innerWidth;
      build();
    };
    window.addEventListener('resize', onResize);

    return () => {
      window.removeEventListener('resize', onResize);
      tl.scrollTrigger.kill(true);
      tl.kill();
      gsap.set(track, { clearProps: 'transform' });
      gsap.set(boxes, { clearProps: 'transform' });
      panels.forEach(p => p.classList.remove('tall'));
      document.body.classList.remove('hs', 'hs-moved', 'hs-past');
      rebuild = null;
      pinTrigger = null;
    };
  });

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
  const s = { x: innerWidth / 2, y: innerHeight / 2, angle: 0, scale: 0, speed: 0, lastY: scrollY, intro: 0, clock: 0, energy: 0, touching: false };
  const trail = [];
  gsap.to(s, { intro: 1, duration: 1.4, ease: 'back.out(1.6)', delay: .3 });
  // the star only flies while the screen is touched (or mouse held) or the page is scrolling
  ['touchstart', 'pointerdown'].forEach(e => addEventListener(e, () => { s.touching = true; }, { passive: true }));
  ['touchend', 'touchcancel', 'pointerup'].forEach(e => addEventListener(e, () => { s.touching = false; }, { passive: true }));
  const place = (el, x, y, angle, scale) => {
    el.style.transform = `translate(${x}px, ${y}px) rotate(${angle}deg) scale(${scale})`;
  };
  function flyStar(time, dt) {
    const w = innerWidth, h = innerHeight;
    const small = w <= 800; // the star is drawn smaller on phones (see style.css)
    const max = document.documentElement.scrollHeight - h;
    const p = pinTrigger ? pinTrigger.progress : (max > 0 ? scrollY / max : 0);
    const f = Math.min(dt / 16.7, 3); // keep speeds the same at any frame rate
    const active = s.touching || scrollY !== s.lastY;
    s.speed += ((scrollY - s.lastY) / Math.max(dt, 1) * 1000 - s.speed) * .15;
    s.lastY = scrollY;
    // energy ramps up quickly when active and fades out gently, so the star glides to a stop
    s.energy += ((active ? 1 : 0) - s.energy) * (active ? .2 : .05) * f;
    s.clock += dt / 1000 * s.energy;
    const c = s.clock;
    const tx = w * .5 + w * (small ? .34 : .4) * Math.sin(c * .8 + p * 9);
    const ty = h * .55 + h * .3 * Math.sin(c * 1.15 + p * 13) * Math.cos(c * .37 + p * 4);
    s.x += (tx - s.x) * .07 * f;
    s.y += (ty - s.y) * .07 * f;
    s.angle += (.3 + .9 * s.energy + Math.abs(s.speed) * .004) * f; // always spins slowly, faster when active
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
      const a = c * 2.4 * m.dir + m.phase;
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
    if (!pinTrigger || !panel || panel.parentElement !== track) return false;
    const at = panelAt(panels.indexOf(panel));
    window.scrollTo({ top: pinTrigger.start + at * (pinTrigger.end - pinTrigger.start), behavior });
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

  // section heights change once the web fonts and images arrive
  window.addEventListener('load', () => { if (rebuild) rebuild(); ScrollTrigger.refresh(); });
  document.fonts.ready.then(() => {
    if (rebuild) rebuild();
    ScrollTrigger.refresh();
    if (location.hash) scrollToPanel(location.hash, 'instant');
  });
}
