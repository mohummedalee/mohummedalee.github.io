/**
 * Mobile scroll behavior - scrolls to content on page load
 * Only active on mobile viewports (< 48em / 768px)
 */
(function() {
    // Check if we're on mobile (same breakpoint as theme: 48em = 768px)
    function isMobile() {
        return window.innerWidth < 768;
    }

    // Scroll to content on page load
    function scrollToContent() {
        if (!isMobile()) return;

        // Don't scroll on homepage
        if (window.location.pathname === '/') return;

        // Don't scroll if there's a hash in URL
        if (window.location.hash) return;

        // Wait for layout to complete
        setTimeout(function() {
            // Target the main content area
            const content = document.querySelector('.content.container');
            if (content) {
                const contentTop = content.getBoundingClientRect().top + window.scrollY;
                if (contentTop > 0) {
                    window.scrollTo(0, contentTop);
                }
            }
        }, 50);
    }

    // Run on window load (after all assets are loaded)
    window.addEventListener('load', scrollToContent);
})();
