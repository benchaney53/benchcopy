let uploadedFiles = [];

// Default configuration (folders are required and patterns must live inside)
const DEFAULT_STRUCTURE = {
    "Transport": {
        folderPattern: "*Transport*",
        patterns: [
            "*Base* *Inspection*",
            "*Base* *Transport*",
            "*M1* *Inspection*",
            "*M1* *Transport*",
            "*M2* *Inspection*",
            "*M2* *Transport*",
            "*M3* *Inspection*",
            "*M3* *Transport*",
            "*Top* *Inspection*",
            "*Top* *Transport*",
            "*DriveTrain* *Inspection*",
            "*DriveTrain* *Transport*",
            "*Nacelle* *Inspection*",
            "*Nacelle* *Transport*",
            "*Hub* *Inspection*",
            "*Hub* *Transport*",
            "*Blade* *Inspection* x3",
            "*Blade* *Transport* x3"
        ]
    },
    "Mechanical": {
        folderPattern: "*Mechanical*",
        patterns: [
            "*10 Percent Checklists*",
            "*Bolt Certificates*",
            "*Flange Reports*",
            "*Aviation Light Manual*",
            "*Rescue Kit Inspection*",
            "*Safety Cable Checklist*",
            "*Foundation Tensioning Concrete and Grout Documents*",
            "*Quality Control of Foundation Earthing*",
            "*Quality Control of Earthing Between Turbines*",
            "*Hardware Lubrication Checklist*",
            "*Generator Alignment Test Report*",
            "*High Voltage Cable Test Report*",
            "*Recording of Main Components*",
            "*Service Inspection Form*",
            "*Mechanical Completion Checklist*",
            "*Mechanical Completion Certificate*",
            "*Punchlist*",
            "*Service Lift Installation Checklist*"
        ]
    },
    "Commissioning": {
        folderPattern: "*Commissioning*",
        patterns: [
            "*CMS Commissioning Procedure*",
            "*Pre-Commissioning Certificate*",
            "*Commissioning Completion Certificate*",
            "*Start-Up Procedure*",
            "*SCADA Functionality Checklist*",
            "*Birth Certificate*",
            "*Final Punchlist*"
        ]
    }
};

// Load configuration from localStorage or use defaults
function loadConfig() {
    const saved = localStorage.getItem('jobBookCheckerConfig');
    if (saved) {
        try {
            const parsed = JSON.parse(saved);
            // Backward compatibility: if stored as simple arrays, convert
            const converted = {};
            let needsConvert = false;
            for (const [key, value] of Object.entries(parsed)) {
                if (Array.isArray(value)) {
                    needsConvert = true;
                    converted[key] = {
                        folderPattern: `*${key}*`,
                        patterns: value
                    };
                } else {
                    converted[key] = value;
                }
            }
            return needsConvert ? converted : parsed;
        } catch (e) {
            console.error('Failed to parse saved config:', e);
        }
    }
    return JSON.parse(JSON.stringify(DEFAULT_STRUCTURE));
}

// Save configuration to localStorage
function saveConfig() {
    localStorage.setItem('jobBookCheckerConfig', JSON.stringify(REQUIRED_STRUCTURE));
}

// Current configuration (mutable)
let REQUIRED_STRUCTURE = loadConfig();

// Extract Pad/WTG info from path parts
function extractLocation(pathParts) {
    let pad = null;
    let wtg = null;
    for (const part of pathParts) {
        const padMatch = part.match(/Pad\s+([A-Za-z0-9]+)/i);
        const wtgMatch = part.match(/WTG\s+([A-Za-z0-9]+)/i);
        if (padMatch) pad = padMatch[1];
        if (wtgMatch) wtg = wtgMatch[1];
        if (pad && wtg) break;
    }
    return { pad, wtg };
}

// Strip "Pad X WTG Y" (with optional dashes/spaces) from start of a filename
function stripLocationPrefix(filename, pad, wtg) {
    if (!pad || !wtg) return { matches: true, remainder: filename };
    const regex = new RegExp(
        `^\\s*Pad\\s*${pad}\\s*-?\\s*WTG\\s*${wtg}\\s*-?\\s*`,
        'i'
    );
    const match = filename.match(regex);
    if (!match) return { matches: false, remainder: filename };
    const remainder = filename.slice(match[0].length).trim();
    return { matches: true, remainder };
}

// Configuration panel elements
const configToggle = document.getElementById('configToggle');
const configBody = document.getElementById('configBody');
const toggleIcon = document.getElementById('toggleIcon');
const categoriesContainer = document.getElementById('categoriesContainer');
const addCategoryBtn = document.getElementById('addCategoryBtn');
const resetConfigBtn = document.getElementById('resetConfigBtn');

// Toggle configuration panel
configToggle.addEventListener('click', () => {
    configBody.classList.toggle('open');
    toggleIcon.classList.toggle('open');
});

// Track which categories are expanded
let expandedCategories = {};

// Render categories and patterns
function renderConfig() {
    categoriesContainer.innerHTML = '';
    
    Object.entries(REQUIRED_STRUCTURE).forEach(([folderName, folderConfig], catIndex) => {
        const patterns = folderConfig.patterns || [];
        const isExpanded = expandedCategories[folderName] || false;
        const categorySection = document.createElement('div');
        categorySection.className = 'category-section';
        categorySection.dataset.category = folderName;
        
        categorySection.innerHTML = `
            <div class="category-header" data-category="${folderName}">
                <div class="category-header-left">
                    <svg class="category-toggle-icon ${isExpanded ? 'open' : ''}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="6 9 12 15 18 9"/>
                    </svg>
                    <input type="text" value="${folderName}" class="category-name-input" data-original="${folderName}" onclick="event.stopPropagation()" aria-label="Folder display name">
                    <span class="category-count">${patterns.length} pattern${patterns.length !== 1 ? 's' : ''}</span>
                </div>
                <button class="btn-small danger delete-category-btn" onclick="event.stopPropagation()">Delete</button>
            </div>
            <div class="category-body ${isExpanded ? 'open' : ''}" data-category="${folderName}">
                <div class="category-content">
                    <div class="pattern-list" data-category="${folderName}">
                        <div class="pattern-item" style="background: transparent; padding-left: 0; padding-right: 0;">
                            <input type="text" value="${folderConfig.folderPattern || `*${folderName}*`}" class="folder-pattern-input" data-category="${folderName}" aria-label="Folder match pattern" title="Folder must match this pattern (wildcards allowed)">
                        </div>
                        ${patterns.map((pattern, patIndex) => `
                            <div class="pattern-item">
                                <input type="text" value="${pattern}" class="pattern-input" data-category="${folderName}" data-index="${patIndex}">
                                <button class="btn-icon delete-pattern-btn" title="Remove pattern">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <line x1="18" y1="6" x2="6" y2="18"/>
                                        <line x1="6" y1="6" x2="18" y2="18"/>
                                    </svg>
                                </button>
                            </div>
                        `).join('')}
                    </div>
                    <button class="btn-small add-pattern-btn" style="margin-top: 10px;">+ Add Pattern</button>
                </div>
            </div>
        `;
        
        categoriesContainer.appendChild(categorySection);
    });
    
    attachConfigListeners();
}

// Attach event listeners for config controls
function attachConfigListeners() {
    // Category header toggle (expand/collapse)
    document.querySelectorAll('.category-header').forEach(header => {
        header.addEventListener('click', (e) => {
            // Don't toggle if clicking on input or button
            if (e.target.closest('input') || e.target.closest('button')) return;
            
            const categoryName = header.dataset.category;
            const categoryBody = header.nextElementSibling;
            const toggleIcon = header.querySelector('.category-toggle-icon');
            
            expandedCategories[categoryName] = !expandedCategories[categoryName];
            categoryBody.classList.toggle('open');
            toggleIcon.classList.toggle('open');
        });
    });
    
    // Folder pattern changes
    document.querySelectorAll('.folder-pattern-input').forEach(input => {
        input.addEventListener('change', (e) => {
            const category = e.target.dataset.category;
            const newValue = e.target.value.trim() || `*${category}*`;
            if (!REQUIRED_STRUCTURE[category]) return;
            REQUIRED_STRUCTURE[category].folderPattern = newValue;
            saveConfig();
        });
    });
    
    // Category name changes
    document.querySelectorAll('.category-name-input').forEach(input => {
        input.addEventListener('change', (e) => {
            const original = e.target.dataset.original;
            const newName = e.target.value.trim();
            
            if (newName && newName !== original && !REQUIRED_STRUCTURE[newName]) {
                // Transfer expanded state to new name
                expandedCategories[newName] = expandedCategories[original];
                delete expandedCategories[original];
                
                REQUIRED_STRUCTURE[newName] = REQUIRED_STRUCTURE[original];
                delete REQUIRED_STRUCTURE[original];
                saveConfig();
                renderConfig();
            } else if (newName !== original) {
                e.target.value = original;
            }
        });
    });
    
    // Pattern changes
    document.querySelectorAll('.pattern-input').forEach(input => {
        input.addEventListener('change', (e) => {
            const category = e.target.dataset.category;
            const index = parseInt(e.target.dataset.index);
            const newValue = e.target.value.trim();
            
            if (newValue) {
                REQUIRED_STRUCTURE[category].patterns[index] = newValue;
                saveConfig();
            }
        });
    });
    
    // Delete pattern buttons
    document.querySelectorAll('.delete-pattern-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const patternItem = e.target.closest('.pattern-item');
            const input = patternItem.querySelector('.pattern-input');
            const category = input.dataset.category;
            const index = parseInt(input.dataset.index);
            
            REQUIRED_STRUCTURE[category].patterns.splice(index, 1);
            saveConfig();
            renderConfig();
        });
    });
    
    // Add pattern buttons
    document.querySelectorAll('.add-pattern-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const categorySection = e.target.closest('.category-section');
            const categoryName = categorySection.dataset.category;
            
            REQUIRED_STRUCTURE[categoryName].patterns.push('*New Pattern*');
            expandedCategories[categoryName] = true; // Ensure category stays open
            saveConfig();
            renderConfig();
            
            // Focus the new input
            const inputs = document.querySelector(`.category-section[data-category="${categoryName}"] .pattern-input:last-of-type`);
            if (inputs) {
                inputs.focus();
                inputs.select();
            }
        });
    });
    
    // Delete category buttons
    document.querySelectorAll('.delete-category-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const categorySection = e.target.closest('.category-section');
            const categoryName = categorySection.dataset.category;
            
            if (confirm(`Delete folder "${categoryName}" and all its patterns?`)) {
                delete REQUIRED_STRUCTURE[categoryName];
                delete expandedCategories[categoryName];
                saveConfig();
                renderConfig();
            }
        });
    });
}

// Add new category
addCategoryBtn.addEventListener('click', () => {
    let newName = 'New Category';
    let counter = 1;
    while (REQUIRED_STRUCTURE[newName]) {
        newName = `New Category ${counter++}`;
    }
    
    REQUIRED_STRUCTURE[newName] = {
        folderPattern: `*${newName}*`,
        patterns: ['*Pattern*']
    };
    expandedCategories[newName] = true; // Auto-expand new category
    saveConfig();
    renderConfig();
    
    // Focus the new category name input
    const inputs = categoriesContainer.querySelectorAll('.category-name-input');
    if (inputs.length > 0) {
        inputs[inputs.length - 1].focus();
        inputs[inputs.length - 1].select();
    }
});

// Reset to defaults
resetConfigBtn.addEventListener('click', () => {
    if (confirm('Reset all file patterns to defaults? This cannot be undone.')) {
        REQUIRED_STRUCTURE = JSON.parse(JSON.stringify(DEFAULT_STRUCTURE));
        saveConfig();
        renderConfig();
    }
});

// Initialize config panel
renderConfig();

const uploadArea = document.getElementById('uploadArea');
const folderInput = document.getElementById('folderInput');
const checkBtn = document.getElementById('checkBtn');
const resultsDiv = document.getElementById('results');

// Click to upload
uploadArea.addEventListener('click', () => {
    folderInput.click();
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const items = e.dataTransfer.items;
    if (items) {
        handleDroppedItems(items);
    }
});

// File input change
folderInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    uploadedFiles = files.map(file => ({
        name: file.name,
        path: file.webkitRelativePath || file.name,
        file: file
    }));
    if (uploadedFiles.length > 0) {
        uploadArea.innerHTML = `<p>✓ Folder selected: ${uploadedFiles.length} files</p>`;
        checkBtn.disabled = false;
    }
});

async function handleDroppedItems(items) {
    uploadedFiles = [];
    
    for (let i = 0; i < items.length; i++) {
        const item = items[i].webkitGetAsEntry();
        if (item) {
            await traverseFileTree(item);
        }
    }
    
    if (uploadedFiles.length > 0) {
        uploadArea.innerHTML = `<p>✓ Folder selected: ${uploadedFiles.length} files</p>`;
        checkBtn.disabled = false;
    }
}

function traverseFileTree(item, path = '') {
    return new Promise((resolve) => {
        if (item.isFile) {
            item.file((file) => {
                uploadedFiles.push({
                    name: file.name,
                    path: path + file.name,
                    file: file
                });
                resolve();
            });
        } else if (item.isDirectory) {
            const dirReader = item.createReader();
            dirReader.readEntries(async (entries) => {
                for (const entry of entries) {
                    await traverseFileTree(entry, path + item.name + '/');
                }
                resolve();
            });
        }
    });
}

checkBtn.addEventListener('click', () => {
    checkFiles();
});

function checkFiles() {
    resultsDiv.innerHTML = '';
    
    // Helper function to convert wildcard pattern to regex
    function wildcardToRegex(pattern) {
        const regexPattern = pattern
            .replace(/[.+?^${}()|[\]\\]/g, '\\$&')
            .replace(/\*/g, '.*');
        return new RegExp(`^${regexPattern}$`, 'i');
    }
    
    // Helper function to check if a file matches a pattern
    function matchesPattern(filename, pattern) {
        const cleanPattern = pattern.replace(/ x\d+$/, '');
        const regex = wildcardToRegex(cleanPattern);
        return regex.test(filename);
    }

    // Group files by tower (top-level folder)
    const towerGroups = {};
    uploadedFiles.forEach(file => {
        const pathParts = file.path.split(/[\/\\]/);
        // The tower name is typically the first or second folder in the path
        // We'll use the folder that appears before Transport/Mechanical/Commissioning
        let towerName = 'Unknown';
        
        for (let i = 0; i < pathParts.length; i++) {
            const part = pathParts[i];
            // Check if this or next part contains one of our category folders
            const isCategory = ['transport', 'mechanical', 'commissioning'].some(cat => 
                part.toLowerCase().includes(cat)
            );
            
            if (isCategory && i > 0) {
                // The tower name is the parent folder
                towerName = pathParts[i - 1];
                break;
            }
        }
        
        if (!towerGroups[towerName]) {
            towerGroups[towerName] = [];
        }
        towerGroups[towerName].push(file);
    });
    
    // If only one tower group and it's "Unknown", treat all files as one tower
    const towerNames = Object.keys(towerGroups);
    let html = '';
    const summary = {
        missingFolders: [],   // {tower, folder}
        missingPatterns: [],  // {tower, folder, pattern}
        unmatchedFiles: [],   // {tower, path}
        extraFolders: []      // {tower, folder}
    };
    
    if (towerNames.length === 1 && towerNames[0] === 'Unknown') {
        // Single tower mode
        const res = checkSingleTower('Job Book', uploadedFiles);
        html = res.html;
        summary.missingFolders.push(...res.stats.missingFolders);
        summary.missingPatterns.push(...res.stats.missingPatterns);
        summary.unmatchedFiles.push(...res.stats.unmatchedFiles);
        summary.extraFolders.push(...res.stats.extraFolders);
    } else {
        // Multiple towers mode
        html = '<div class="towers-summary">';
        
        let totalTowers = towerNames.filter(n => n !== 'Unknown').length;
        let completeTowers = 0;
        
        // Check each tower
        for (const [towerName, files] of Object.entries(towerGroups)) {
            if (towerName === 'Unknown' && files.length === 0) continue;
            
            const result = checkSingleTower(towerName, files);
            const isComplete = result.stats.missingCount === 0;
            if (isComplete && towerName !== 'Unknown') completeTowers++;
            
            html += result.html;

            summary.missingFolders.push(...result.stats.missingFolders);
            summary.missingPatterns.push(...result.stats.missingPatterns);
            summary.unmatchedFiles.push(...result.stats.unmatchedFiles);
            summary.extraFolders.push(...result.stats.extraFolders);
        }
        
        html = `<div class="result-section info">
            <div class="summary">
                ${completeTowers} of ${totalTowers} towers complete
            </div>
        </div>` + html;
        
        html += '</div>';
    }

    // Global summary section
    const totalMissing = summary.missingFolders.length + summary.missingPatterns.length;
    const hasUnmatched = summary.unmatchedFiles.length > 0;
    const hasExtraFolders = summary.extraFolders.length > 0;
    const hasSummary = totalMissing > 0 || hasUnmatched || hasExtraFolders;

    if (hasSummary) {
        let summaryHtml = `<div class="result-section warning"><h3>Summary</h3>`;
        if (totalMissing > 0) {
            summaryHtml += `<div class="summary">${totalMissing} missing item(s) across all towers</div>`;
            if (summary.missingFolders.length > 0) {
                summaryHtml += `<h4>Missing folders</h4><ul class="file-list missing">`;
                summary.missingFolders.forEach(f => summaryHtml += `<li>${f.tower}: ${f.folder}</li>`);
                summaryHtml += `</ul>`;
            }
            if (summary.missingPatterns.length > 0) {
                summaryHtml += `<h4>Missing files</h4><ul class="file-list missing">`;
                summary.missingPatterns.forEach(p => summaryHtml += `<li>${p.tower}: ${p.folder}: ${p.pattern}</li>`);
                summaryHtml += `</ul>`;
            }
        }
        if (hasUnmatched) {
            summaryHtml += `<h4>Unmatched files</h4><ul class="file-list missing">`;
            summary.unmatchedFiles.forEach(u => summaryHtml += `<li>${u.tower}: ${u.path}</li>`);
            summaryHtml += `</ul>`;
        }
        if (hasExtraFolders) {
            summaryHtml += `<h4>Extra folders</h4><ul class="file-list">`;
            summary.extraFolders.forEach(f => summaryHtml += `<li>${f.tower}: ${f.folder}</li>`);
            summaryHtml += `</ul>`;
        }
        summaryHtml += `</div>`;
        html = summaryHtml + html;
    }
    
    resultsDiv.innerHTML = html;
}

function checkSingleTower(towerName, towerFiles) {
    let totalRequired = 0;
    let totalFound = 0;
    let missingFiles = [];
    let missingFolders = [];
    const topLevelFolders = new Set();
    let pad = null;
    let wtg = null;
    const matchedFilePaths = new Set();
    
    // Helper function to convert wildcard pattern to regex
    function wildcardToRegex(pattern) {
        const regexPattern = pattern
            .replace(/[.+?^${}()|[\]\\]/g, '\\$&')
            .replace(/\*/g, '.*');
        return new RegExp(`^${regexPattern}$`, 'i');
    }
    
    // Helper function to check if a file matches a pattern
    function matchesPattern(filename, pattern) {
        const cleanPattern = pattern.replace(/ x\d+$/, '');
        const regex = wildcardToRegex(cleanPattern);
        return regex.test(filename);
    }
    
    // Collect top-level folders within this tower
    towerFiles.forEach(file => {
        const pathParts = file.path.split(/[\/\\]/).filter(Boolean);
        const loc = extractLocation(pathParts);
        if (loc.pad) pad = loc.pad;
        if (loc.wtg) wtg = loc.wtg;
        const idxTower = pathParts.findIndex(p => p === towerName);
        if (idxTower >= 0 && idxTower + 1 < pathParts.length) {
            topLevelFolders.add(pathParts[idxTower + 1]);
        } else if (pathParts.length > 1) {
            topLevelFolders.add(pathParts[0]);
        }
    });

    // Check folders and files
    for (const [folderName, folderConfig] of Object.entries(REQUIRED_STRUCTURE)) {
        const patterns = folderConfig.patterns || [];
        const folderPattern = folderConfig.folderPattern || `*${folderName}*`;

        // Determine if folder exists in tower
        const folderExists = towerFiles.some(file => {
            const pathParts = file.path.split(/[\/\\]/);
            return pathParts.some(part => matchesPattern(part, folderPattern));
        });

        if (!folderExists) {
            // Mark folder missing once
            missingFolders.push({ tower: towerName, folder: folderName, pattern: folderPattern });
        }

        patterns.forEach(pattern => {
            const multiplierMatch = pattern.match(/ x(\d+)$/);
            const requiredCount = multiplierMatch ? parseInt(multiplierMatch[1]) : 1;
            const cleanPattern = pattern.replace(/ x\d+$/, '');
            
            totalRequired += requiredCount;
            
            const matchingFiles = folderExists ? towerFiles.filter(file => {
                const pathParts = file.path.split(/[\/\\]/);
                const inCorrectFolder = pathParts.some(part => matchesPattern(part, folderPattern));
                if (!inCorrectFolder) return false;

                // Enforce location prefix (Pad/WTG) matching if available
                const locationCheck = stripLocationPrefix(file.name.replace(/\.[^/.]+$/, ''), pad, wtg);
                if (!locationCheck.matches) return false;

                const matches = matchesPattern(locationCheck.remainder || file.name, cleanPattern);
                if (matches) {
                    matchedFilePaths.add(file.path);
                }
                return matches;
            }) : [];
            
            const foundCount = matchingFiles.length;
            
            if (foundCount >= requiredCount) {
                totalFound += requiredCount;
            } else {
                totalFound += foundCount;
                for (let i = foundCount; i < requiredCount; i++) {
                    missingFiles.push({
                        tower: towerName,
                        folder: folderName,
                        pattern: pattern
                    });
                }
            }
        });
    }
    
    // Extra folders: top-level folders that don't match required folder patterns
    const requiredFolderPatterns = Object.values(REQUIRED_STRUCTURE).map(cfg => cfg.folderPattern || '*');
    const extraFolders = Array.from(topLevelFolders).filter(folderName => {
        return !requiredFolderPatterns.some(pat => matchesPattern(folderName, pat));
    });
    const unmatchedFiles = towerFiles.filter(file => !matchedFilePaths.has(file.path));
    const stats = {
        missingCount: missingFiles.length + missingFolders.length,
        missingFolders: missingFolders.map(f => ({ tower: towerName, folder: f })),
        missingPatterns: missingFiles.map(m => ({ tower: towerName, folder: m.folder, pattern: m.pattern })),
        unmatchedFiles: unmatchedFiles.map(f => ({ tower: towerName, path: f.path })),
        extraFolders: extraFolders.map(f => ({ tower: towerName, folder: f }))
    };
    
    const summaryClass = stats.missingCount === 0 ? 'success' : 'warning';
    
    let html = `
        <div class="result-section ${summaryClass}">
            <h3>${towerName}</h3>
            ${pad || wtg ? `<div class="tower-meta">${pad ? `Pad ${pad}` : ''}${pad && wtg ? ' · ' : ''}${wtg ? `WTG ${wtg}` : ''}</div>` : ''}
            <div class="summary">
                ${totalFound} of ${totalRequired} required files found
            </div>
    `;
    
    if (missingFiles.length === 0 && missingFolders.length === 0) {
        html += `<p>✓ All required files are present!</p>`;
    } else {
        const totalMissing = missingFiles.length + missingFolders.length;
        html += `<p>Missing ${totalMissing} item(s):</p>`;

        if (missingFolders.length > 0) {
            html += `<h4>Required folders</h4><ul class="file-list missing">`;
            missingFolders.forEach(fp => {
                html += `<li>${fp}</li>`;
            });
            html += `</ul>`;
        }
        
        const groupedMissing = {};
        missingFiles.forEach(item => {
            if (!groupedMissing[item.folder]) {
                groupedMissing[item.folder] = [];
            }
            groupedMissing[item.folder].push(item.pattern);
        });
        
        for (const [folder, patterns] of Object.entries(groupedMissing)) {
            html += `<h4>${folder}/</h4><ul class="file-list missing">`;
            patterns.forEach(pattern => {
                html += `<li>${pattern}</li>`;
            });
            html += `</ul>`;
        }
    }

    if (unmatchedFiles.length > 0) {
        html += `<h4>Unmatched files</h4><ul class="file-list missing">`;
        unmatchedFiles.forEach(f => {
            html += `<li>${f.path}</li>`;
        });
        html += `</ul>`;
    }

    // Always show extra folders if present (even when all required items exist)
    if (extraFolders.length > 0) {
        html += `<h4>Extra folders</h4><ul class="file-list">`;
        extraFolders.forEach(folder => {
            html += `<li>${folder}</li>`;
        });
        html += `</ul>`;
    }
    
    html += `</div>`;
    
    return { html, stats };
}
