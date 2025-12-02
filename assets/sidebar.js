// Sidebar Navigation Component
// This script automatically injects itself into any page in the bench tools directory
// No manual integration needed - just include the script tag

(function() {
    'use strict';
    
    // Detect the current page path to generate correct relative links
    function getBasePath() {
        const path = window.location.pathname;
        // Check if we're in a tool subdirectory (tools/*/*.html)
        if (path.includes('/tools/') && path.split('/').length > 3) {
            return '../../';
        }
        // Otherwise assume we're at root
        return './';
    }
    
    // Fetch and populate tools dynamically from manifest
    async function loadToolsList() {
        try {
            const basePath = getBasePath();
            const response = await fetch(basePath + 'tools/tools-manifest.json');
            const tools = await response.json();
            return tools;
        } catch (error) {
            console.warn('Could not load tools manifest:', error);
            return [];
        }
    }
    
    // Create and inject sidebar HTML
    async function createSidebar() {
        const basePath = getBasePath();
        const tools = await loadToolsList();
        
        // Inject Font Awesome if not already present
        if (!document.getElementById('bench-fontawesome')) {
            const link = document.createElement('link');
            link.id = 'bench-fontawesome';
            link.rel = 'stylesheet';
            link.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css';
            document.head.appendChild(link);
        }
        
        // Inject CSS if not already present
        if (!document.getElementById('bench-sidebar-styles')) {
            const link = document.createElement('link');
            link.id = 'bench-sidebar-styles';
            link.rel = 'stylesheet';
            link.href = basePath + 'assets/sidebar.css';
            document.head.appendChild(link);
        }
        
        const sidebar = document.createElement('div');
        sidebar.id = 'bench-sidebar';
        sidebar.className = 'bench-sidebar';
        
        // Generate tools submenu dynamically
        let toolsSubmenuHTML = '';
        tools.forEach(tool => {
            const toolPath = basePath + 'tools/' + tool.id + '/';
            toolsSubmenuHTML += `
                <a href="${toolPath}" class="sidebar-link sidebar-sublink">
                    <i class="fas fa-file-pdf"></i>
                    ${tool.name}
                </a>
            `;
        });
        
        sidebar.innerHTML = `
            <div class="sidebar-header">
                <i class="fa-solid fa-wrench sidebar-wrench-icon"></i>
                <h3>Bench</h3>
                <button class="sidebar-close" aria-label="Close sidebar">×</button>
            </div>
            <nav class="sidebar-nav">
                <a href="${basePath}home.html" class="sidebar-link">
                    <i class="fa-solid fa-house sidebar-icon"></i>
                    Home
                </a>
                <a href="#tools" class="sidebar-link sidebar-tools-toggle">
                    <i class="fa-solid fa-wrench sidebar-icon"></i>
                    Tools
                    <i class="fa-solid fa-chevron-down sidebar-chevron"></i>
                </a>
                <div class="sidebar-submenu" id="tools-submenu">
                    ${toolsSubmenuHTML}
                </div>
            </nav>
        `;
        
        document.body.appendChild(sidebar);
        
        // Create overlay
        const overlay = document.createElement('div');
        overlay.id = 'bench-sidebar-overlay';
        overlay.className = 'bench-sidebar-overlay';
        document.body.appendChild(overlay);
        
        // Create menu button
        const menuButton = document.createElement('button');
        menuButton.id = 'bench-menu-btn';
        menuButton.className = 'bench-menu-btn';
        menuButton.setAttribute('aria-label', 'Open menu');
        menuButton.innerHTML = '<i class="fa-solid fa-bars"></i>';
        document.body.appendChild(menuButton);
        
        // Add event listeners
        attachEventListeners();
    }
    
    function attachEventListeners() {
        const menuBtn = document.getElementById('bench-menu-btn');
        const sidebar = document.getElementById('bench-sidebar');
        const overlay = document.getElementById('bench-sidebar-overlay');
        const closeBtn = sidebar.querySelector('.sidebar-close');
        const toolsToggle = sidebar.querySelector('.sidebar-tools-toggle');
        const toolsSubmenu = document.getElementById('tools-submenu');
        
        function openSidebar() {
            sidebar.classList.add('open');
            overlay.classList.add('visible');
            document.body.style.overflow = 'hidden';
        }
        
        function closeSidebar() {
            sidebar.classList.remove('open');
            overlay.classList.remove('visible');
            document.body.style.overflow = '';
        }
        
        menuBtn.addEventListener('click', openSidebar);
        closeBtn.addEventListener('click', closeSidebar);
        overlay.addEventListener('click', closeSidebar);
        
        // Tools submenu toggle
        if (toolsToggle && toolsSubmenu) {
            toolsToggle.addEventListener('click', (e) => {
                e.preventDefault();
                toolsSubmenu.classList.toggle('open');
                toolsToggle.classList.toggle('active');
            });
        }
        
        // Close sidebar when clicking a link (except tools toggle)
        const links = sidebar.querySelectorAll('.sidebar-link:not(.sidebar-tools-toggle)');
        links.forEach(link => {
            link.addEventListener('click', () => {
                closeSidebar();
            });
        });
        
        // Keyboard accessibility
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && sidebar.classList.contains('open')) {
                closeSidebar();
            }
        });
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createSidebar);
    } else {
        createSidebar();
    }
})();
