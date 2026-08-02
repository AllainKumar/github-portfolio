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
  '.about-grid, .stats, .skills-grid, .timeline-item, .edu-card, .cert-card, .contact-card, .section-title'
).forEach(el => el.classList.add('reveal'));

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
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
  }, { rootMargin: '-40% 0px -55% 0px' }).observe(sec);
});
