/**
 * Fun Guitar - Main Application Script
 * Handles navigation, animations, and UI interactions
 */

document.addEventListener('DOMContentLoaded', () => {
    // ========== Mobile Navigation Toggle ==========
    const navToggle = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');

    if (navToggle) {
        navToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            navToggle.classList.toggle('active');
        });

        // Close mobile nav on link click
        navLinks.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('active');
                navToggle.classList.remove('active');
            });
        });
    }

    // ========== Smooth Scroll for Anchor Links ==========
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const navHeight = document.querySelector('.navbar').offsetHeight;
                const targetPos = target.getBoundingClientRect().top + window.pageYOffset - navHeight - 20;
                window.scrollTo({ top: targetPos, behavior: 'smooth' });
            }
        });
    });

    // ========== Navbar Scroll Effect ==========
    let lastScroll = 0;
    const navbar = document.querySelector('.navbar');

    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;
        
        if (currentScroll > 100) {
            navbar.style.borderBottomColor = 'rgba(42, 42, 58, 0.8)';
        } else {
            navbar.style.borderBottomColor = 'rgba(42, 42, 58, 0.3)';
        }

        lastScroll = currentScroll;
    });

    // ========== Intersection Observer for Animations ==========
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Animate feature cards, tech cards, and pipeline steps
    const animatedElements = document.querySelectorAll(
        '.feature-card, .tech-card, .pipeline-step, .tab-demo, .desktop-card'
    );
    
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });

    // Stagger animation for grid items
    document.querySelectorAll('.features-grid, .tech-grid').forEach(grid => {
        const children = grid.children;
        Array.from(children).forEach((child, index) => {
            child.style.transitionDelay = `${index * 0.08}s`;
        });
    });

    // ========== Animated Counter for Stats ==========
    const statValues = document.querySelectorAll('.stat-value');
    let statsAnimated = false;

    const statsObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !statsAnimated) {
                statsAnimated = true;
                statValues.forEach(stat => {
                    const target = parseInt(stat.textContent);
                    animateCounter(stat, 0, target, 1500);
                });
            }
        });
    }, { threshold: 0.5 });

    const heroStats = document.querySelector('.hero-stats');
    if (heroStats) {
        statsObserver.observe(heroStats);
    }

    function animateCounter(element, start, end, duration) {
        const startTime = performance.now();
        
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(start + (end - start) * eased);
            
            element.textContent = current;
            
            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }
        
        requestAnimationFrame(update);
    }

    // ========== Tab Animation ==========
    const tabOutput = document.getElementById('sample-tab');
    if (tabOutput) {
        const originalText = tabOutput.textContent;
        let tabAnimated = false;

        const tabObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && !tabAnimated) {
                    tabAnimated = true;
                    typewriterEffect(tabOutput, originalText);
                }
            });
        }, { threshold: 0.3 });

        tabObserver.observe(tabOutput);
    }

    function typewriterEffect(element, text) {
        element.textContent = '';
        let i = 0;
        const speed = 8; // ms per character
        
        function type() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(type, speed);
            }
        }
        
        type();
    }

    // ========== Keyboard Shortcuts ==========
    document.addEventListener('keydown', (e) => {
        // Press 'T' to jump to tuner section
        if (e.key === 't' && !e.ctrlKey && !e.metaKey && 
            document.activeElement.tagName !== 'INPUT' && 
            document.activeElement.tagName !== 'TEXTAREA') {
            const tuner = document.getElementById('tuner');
            if (tuner) {
                const navHeight = document.querySelector('.navbar').offsetHeight;
                const targetPos = tuner.getBoundingClientRect().top + window.pageYOffset - navHeight - 20;
                window.scrollTo({ top: targetPos, behavior: 'smooth' });
            }
        }
    });
});
