// PDF.js setup
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

let uploadedPdfBytes = null;
let basePdfBytes = null;
let pdfDoc = null;
let pdfJsDoc = null;
let formFields = [];
let originalFormFields = [];

// Debug helper
const DEBUG_FILL = true;
function logFill(...args) {
    if (DEBUG_FILL) console.log('[fill]', ...args);
}

function isProbablyPdf(bytes) {
    if (!bytes || bytes.length < 5) return false;
    const header = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3], bytes[4]);
    return header === '%PDF-';
}

function normalizeFieldName(name) {
    return (name || '').toString().trim().toLowerCase();
}

function findCaseInsensitiveOption(options, value) {
    const target = (value ?? '').toString().trim().toLowerCase();
    if (!target) return null;
    for (const opt of options || []) {
        if ((opt || '').toString().trim().toLowerCase() === target) {
            return opt;
        }
    }
    return null;
}

async function ensureValidPdfBytes() {
    const bytes = requirePdfLoaded();
    if (!isProbablyPdf(bytes)) {
        throw new Error('The loaded file is not a valid PDF (missing %PDF header). Please re-upload a PDF.');
    }
    try {
        // Quick sanity check that pdf-lib can parse the bytes
        await PDFLib.PDFDocument.load(bytes);
    } catch (err) {
        throw new Error(`The loaded PDF could not be parsed. Please re-upload the PDF. (${err.message})`);
    }
    return bytes;
}

function requirePdfLoaded() {
    const bytes = basePdfBytes || uploadedPdfBytes;
    if (!bytes || !(bytes instanceof Uint8Array) || bytes.length === 0) {
        throw new Error('No PDF is loaded. Please upload a PDF file before importing CSV.');
    }
    if (!isProbablyPdf(bytes)) {
        throw new Error('The loaded file is not a valid PDF (missing %PDF header). Please re-upload a PDF.');
    }
    return bytes;
}

// Upload handling
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const loading = document.getElementById('loading');
const formSection = document.getElementById('form-section');
const formFieldsContainer = document.getElementById('form-fields');
const fieldCount = document.getElementById('field-count');
const exportCsvBtn = document.getElementById('export-csv-btn');
const importCsvBtn = document.getElementById('import-csv-btn');
const csvImportInput = document.getElementById('csv-import-input');
const clearBtn = document.getElementById('clear-btn');
const resetBtn = document.getElementById('reset-btn');
const downloadBtn = document.getElementById('download-btn');

function setActionButtonsEnabled(enabled) {
    [exportCsvBtn, importCsvBtn, clearBtn, resetBtn, downloadBtn].forEach(btn => {
        if (btn) {
            btn.disabled = !enabled;
        }
    });
}

// Disable action buttons until a PDF is loaded
setActionButtonsEnabled(false);

uploadArea.addEventListener('click', () => fileInput.click());

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
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        if (files[0].type === 'application/pdf') {
            handleFile(files[0]);
        } else {
            alert('Please select a PDF file.');
        }
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        if (file.type === 'application/pdf') {
            handleFile(file);
        } else {
            alert('Please select a PDF file.');
            e.target.value = ''; // Clear the invalid selection
        }
    }
});

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' bytes';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

async function handleFile(file) {
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.classList.add('visible');

    loading.classList.add('visible');
    formSection.classList.remove('visible');
    setActionButtonsEnabled(false);

    try {
        const arrayBuffer = await file.arrayBuffer();
        uploadedPdfBytes = new Uint8Array(arrayBuffer);

        if (!isProbablyPdf(uploadedPdfBytes)) {
            throw new Error('Selected file is not a valid PDF (missing %PDF header).');
        }
        
        // Load PDF with pdf-lib
        pdfDoc = await PDFLib.PDFDocument.load(uploadedPdfBytes);
        basePdfBytes = uploadedPdfBytes.slice(); // keep pristine copy for downstream fills
        
        // Load PDF with PDF.js for rendering
        pdfJsDoc = await pdfjsLib.getDocument({ data: uploadedPdfBytes }).promise;
        
        // Extract form fields
        await extractFormFields();

        loading.classList.remove('visible');
        formSection.classList.add('visible');
        setActionButtonsEnabled(true);

    } catch (error) {
        alert('Error loading PDF: ' + error.message);
        console.error(error);
        loading.classList.remove('visible');
        setActionButtonsEnabled(false);
    }
}

async function extractFormFields() {
    formFields = [];
    formFieldsContainer.innerHTML = '';
    
    const form = pdfDoc.getForm();
    const fields = form.getFields();
    
    fieldCount.textContent = fields.length;
    
    console.log(`Found ${fields.length} fields`);
    
    for (const field of fields) {
        const fieldName = field.getName();
        const fieldType = field.constructor.name;
        
        console.log(`Processing field: ${fieldName}, Type: ${fieldType}`);
        
        let fieldInfo = {
            name: fieldName,
            type: fieldType,
            field: field,
            value: null,
            pageNumber: null,
            rect: null
        };
        
        // Try to get the page number and position for this field
        try {
            const acroField = field.acroField;
            if (acroField && acroField.dict) {
                const pageRef = acroField.dict.get(PDFLib.PDFName.of('P'));
                if (pageRef) {
                    const pages = pdfDoc.getPages();
                    const pageIndex = pages.findIndex(p => p.ref === pageRef);
                    if (pageIndex !== -1) {
                        fieldInfo.pageNumber = pageIndex + 1; // 1-based
                    }
                }
            }
            // Fallback: check widget annotations and get rectangle
            if (!fieldInfo.pageNumber) {
                const pages = pdfDoc.getPages();
                for (let i = 0; i < pages.length; i++) {
                    const annotations = pages[i].node.Annots();
                    if (annotations) {
                        const annots = annotations.asArray();
                        for (const annot of annots) {
                            const annotDict = annot instanceof PDFLib.PDFDict ? annot : null;
                            if (annotDict) {
                                const t = annotDict.get(PDFLib.PDFName.of('T'));
                                if (t && t.toString().includes(fieldName)) {
                                    fieldInfo.pageNumber = i + 1;
                                    // Get field rectangle
                                    const rect = annotDict.get(PDFLib.PDFName.of('Rect'));
                                    if (rect) {
                                        fieldInfo.rect = rect.asRectangle();
                                    }
                                    break;
                                }
                            }
                        }
                        if (fieldInfo.pageNumber) break;
                    }
                }
            }
            
            // Try to get rectangle from widgets if not found
            if (!fieldInfo.rect && field.acroField) {
                try {
                    const widgets = field.acroField.getWidgets();
                    if (widgets && widgets.length > 0) {
                        const widget = widgets[0];
                        const rect = widget.dict.get(PDFLib.PDFName.of('Rect'));
                        if (rect) {
                            fieldInfo.rect = rect.asRectangle();
                        }
                    }
                } catch (e) {
                    // Ignore
                }
            }
        } catch (err) {
            console.warn(`Could not determine page for field ${fieldName}:`, err);
        }
        
        // Get current value based on field type
        try {
            // Check if it's a text field (type 'r' or 'PDFTextField')
            if (fieldType === 'PDFTextField' || fieldType === 'r') {
                fieldInfo.type = 'PDFTextField';
                
                // Try multiple methods to extract the value
                if (typeof field.getText === 'function') {
                    fieldInfo.value = field.getText() || '';
                } else if (typeof field.getValue === 'function') {
                    fieldInfo.value = field.getValue() || '';
                } else if (field.value !== undefined) {
                    fieldInfo.value = field.value || '';
                } else {
                    fieldInfo.value = '';
                }
                
                // Check if multiline
                if (typeof field.isMultiline === 'function') {
                    fieldInfo.isMultiline = field.isMultiline();
                } else {
                    fieldInfo.isMultiline = false;
                }
                
                console.log(`Field ${fieldName} (type ${fieldType}) extracted value:`, fieldInfo.value);
                
            } else if (fieldType === 'PDFCheckBox' || fieldType === 'e') {
                // Type 'e' is checkbox
                fieldInfo.type = 'PDFCheckBox';
                
                if (typeof field.isChecked === 'function') {
                    fieldInfo.value = field.isChecked();
                } else if (typeof field.getValue === 'function') {
                    fieldInfo.value = !!field.getValue();
                } else if (field.value !== undefined) {
                    fieldInfo.value = !!field.value;
                } else {
                    fieldInfo.value = false;
                }
                
                console.log(`Field ${fieldName} (type ${fieldType}) checked:`, fieldInfo.value);
                
            } else if (fieldType === 'PDFDropdown') {
                fieldInfo.value = field.getSelected() || [];
                fieldInfo.options = field.getOptions();
            } else if (fieldType === 'PDFRadioGroup') {
                fieldInfo.value = field.getSelected();
                fieldInfo.options = field.getOptions();
            } else {
                console.warn(`Unknown field type: ${fieldType} for field: ${fieldName}`);
                // Treat unknown types as text fields and try to extract value
                fieldInfo.type = 'PDFTextField';
                if (typeof field.getValue === 'function') {
                    fieldInfo.value = field.getValue() || '';
                } else if (field.value !== undefined) {
                    fieldInfo.value = field.value || '';
                } else {
                    fieldInfo.value = '';
                }
            }
        } catch (error) {
            console.error(`Error reading field ${fieldName}:`, error);
            fieldInfo.type = 'PDFTextField';
            fieldInfo.value = '';
        }
        
        formFields.push(fieldInfo);
        
        try {
            createFormField(fieldInfo);
        } catch (error) {
            console.error(`Error creating UI for field ${fieldName}:`, error);
        }
    }
    
    // Sort fields by page number, then Y position (top to bottom), then X position (left to right)
    formFields.sort((a, b) => {
        // First sort by page number
        if (a.pageNumber !== b.pageNumber) {
            return (a.pageNumber || 999) - (b.pageNumber || 999);
        }
        
        // Then by Y position (PDF coordinates are bottom-up, so we reverse for top-to-bottom reading order)
        if (a.rect && b.rect) {
            // Higher Y values in PDF = higher on page, so reverse for reading order (top to bottom)
            const yDiff = (b.rect.y || 0) - (a.rect.y || 0);
            if (Math.abs(yDiff) > 5) { // 5 point tolerance for "same line"
                return yDiff;
            }
            // If on same horizontal line, sort by X position (left to right)
            return (a.rect.x || 0) - (b.rect.x || 0);
        }
        
        return 0;
    });
    
    // Clear and re-render fields in sorted order
    formFieldsContainer.innerHTML = '';
    formFields.forEach(fieldInfo => {
        try {
            createFormField(fieldInfo);
        } catch (error) {
            console.error(`Error creating UI for field ${fieldInfo.name}:`, error);
        }
    });
    
    // Save original values for reset functionality
    originalFormFields = JSON.parse(JSON.stringify(formFields.map(f => ({
        name: f.name,
        value: f.value,
        type: f.type
    }))));
    
    console.log(`Created ${formFieldsContainer.children.length} form field elements`);
}

// Show page reference modal
async function showPageReference(pageNumber, fieldName, rect) {
    // Get current theme colors
    const isLightMode = document.body.classList.contains('light-mode');
    const cardBg = isLightMode ? 'white' : '#3a3a3a';
    const textColor = isLightMode ? '#333' : '#e0e0e0';
    const borderColor = isLightMode ? '#ddd' : '#555';
    
    // Create modal
    const modal = document.createElement('div');
    modal.style.cssText = 'position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.8); display: flex; align-items: center; justify-content: center; z-index: 1000; overflow: hidden;';
    
    const modalContent = document.createElement('div');
    modalContent.style.cssText = `background: ${cardBg}; padding: 20px; border-radius: 8px; max-width: 90vw; max-height: 90vh; display: flex; flex-direction: column; position: relative;`;
    
    const header = document.createElement('div');
    header.style.cssText = 'display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-shrink: 0;';
    
    const title = document.createElement('h3');
    title.textContent = `Page ${pageNumber} - ${fieldName}`;
    title.style.cssText = `margin: 0; color: ${textColor};`;
    
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '✕';
    closeBtn.style.cssText = `background: none; border: none; font-size: 24px; cursor: pointer; color: ${textColor}; padding: 0; width: 32px; height: 32px;`;
    closeBtn.addEventListener('click', () => document.body.removeChild(modal));
    
    header.appendChild(title);
    header.appendChild(closeBtn);
    modalContent.appendChild(header);
    
    const canvasContainer = document.createElement('div');
    canvasContainer.style.cssText = 'position: relative; overflow: auto; flex: 1; min-height: 0;';
    
    const canvas = document.createElement('canvas');
    canvas.style.cssText = `max-width: 100%; height: auto; border: 1px solid ${borderColor}; display: block;`;
    canvasContainer.appendChild(canvas);
    
    modalContent.appendChild(canvasContainer);
    modal.appendChild(modalContent);
    document.body.appendChild(modal);
    
    // Prevent body scroll when modal is open
    document.body.style.overflow = 'hidden';
    
    // Close on background click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            document.body.style.overflow = '';
            document.body.removeChild(modal);
        }
    });
    
    // Restore body scroll on close button
    closeBtn.addEventListener('click', () => {
        document.body.style.overflow = '';
    });
    
    // Render the page
    try {
        const page = await pdfJsDoc.getPage(pageNumber);
        const scale = 1.5;
        const viewport = page.getViewport({ scale });
        
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        
        const context = canvas.getContext('2d');
        await page.render({
            canvasContext: context,
            viewport: viewport
        }).promise;
        
        // Draw red rectangle around the field if we have coordinates
        if (rect) {
            const pageHeight = viewport.height / scale;
            
            // PDF coordinates are from bottom-left, canvas is from top-left
            const x = rect.x * scale;
            const y = (pageHeight - rect.y - rect.height) * scale;
            const width = rect.width * scale;
            const height = rect.height * scale;
            
            context.strokeStyle = 'red';
            context.lineWidth = 3;
            context.strokeRect(x, y, width, height);
            
            // Add a semi-transparent red overlay
            context.fillStyle = 'rgba(255, 0, 0, 0.1)';
            context.fillRect(x, y, width, height);
        }
    } catch (error) {
        console.error('Error rendering page:', error);
        alert('Error loading page preview');
        document.body.removeChild(modal);
    }
}

function createFormField(fieldInfo) {
    const fieldDiv = document.createElement('div');
    fieldDiv.className = 'form-field';
    
    const labelRow = document.createElement('div');
    labelRow.style.display = 'flex';
    labelRow.style.justifyContent = 'space-between';
    labelRow.style.alignItems = 'center';
    labelRow.style.marginBottom = '4px';
    
    // Field name label
    const fieldLabel = document.createElement('label');
    fieldLabel.textContent = fieldInfo.name;
    fieldLabel.style.fontWeight = '600';
    fieldLabel.style.fontSize = '14px';
    labelRow.appendChild(fieldLabel);
    
    // Action buttons container
    const actionsDiv = document.createElement('div');
    actionsDiv.className = 'field-actions';
    
    // Add reference button if we have a page number
    if (fieldInfo.pageNumber) {
        const refBtn = document.createElement('button');
        refBtn.textContent = `Reference: Page ${fieldInfo.pageNumber}`;
        refBtn.className = 'ref-btn';
        refBtn.addEventListener('click', (e) => {
            e.preventDefault();
            showPageReference(fieldInfo.pageNumber, fieldInfo.name, fieldInfo.rect);
        });
        actionsDiv.appendChild(refBtn);
    }
    
    // Add delete button
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = 'Delete';
    deleteBtn.className = 'delete-field-btn';
    deleteBtn.dataset.fieldName = fieldInfo.name;
    deleteBtn.addEventListener('click', (e) => {
        e.preventDefault();
        if (confirm(`Are you sure you want to delete the field "${fieldInfo.name}"?`)) {
            // Mark field as deleted
            const field = formFields.find(f => f.name === fieldInfo.name);
            if (field) {
                field.deleted = true;
            }
            // Remove the field div from DOM
            fieldDiv.remove();
            // Update field count
            const activeFields = formFields.filter(f => !f.deleted).length;
            fieldCount.textContent = activeFields;
        }
    });
    actionsDiv.appendChild(deleteBtn);
    
    labelRow.appendChild(actionsDiv);
    
    fieldDiv.appendChild(labelRow);
    
    let input;
    
    if (fieldInfo.type === 'PDFTextField') {
        if (fieldInfo.isMultiline) {
            input = document.createElement('textarea');
            input.value = fieldInfo.value;
        } else {
            input = document.createElement('input');
            input.type = 'text';
            input.value = fieldInfo.value;
        }
        input.dataset.fieldName = fieldInfo.name;
        input.addEventListener('input', (e) => {
            const field = formFields.find(f => f.name === e.target.dataset.fieldName);
            if (field) field.value = e.target.value;
        });
        fieldDiv.appendChild(input);
        formFieldsContainer.appendChild(fieldDiv);
        return;
        
    } else if (fieldInfo.type === 'PDFCheckBox') {
        const checkboxWrapper = document.createElement('div');
        checkboxWrapper.className = 'checkbox-field';
        
        input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = fieldInfo.value;
        input.dataset.fieldName = fieldInfo.name;
        input.addEventListener('change', (e) => {
            const field = formFields.find(f => f.name === e.target.dataset.fieldName);
            if (field) field.value = e.target.checked;
        });
        
        const checkboxLabel = document.createElement('label');
        checkboxLabel.textContent = 'Checked';
        checkboxLabel.style.fontWeight = 'normal';
        
        checkboxWrapper.appendChild(input);
        checkboxWrapper.appendChild(checkboxLabel);
        fieldDiv.appendChild(checkboxWrapper);
        formFieldsContainer.appendChild(fieldDiv);
        return;
        
    } else if (fieldInfo.type === 'PDFDropdown') {
        input = document.createElement('select');
        input.dataset.fieldName = fieldInfo.name;
        
        const emptyOption = document.createElement('option');
        emptyOption.value = '';
        emptyOption.textContent = '-- Select --';
        input.appendChild(emptyOption);
        
        fieldInfo.options.forEach(option => {
            const optionElement = document.createElement('option');
            optionElement.value = option;
            optionElement.textContent = option;
            if (fieldInfo.value.includes(option)) {
                optionElement.selected = true;
            }
            input.appendChild(optionElement);
        });
        
        input.addEventListener('change', (e) => {
            const field = formFields.find(f => f.name === e.target.dataset.fieldName);
            if (field) field.value = [e.target.value];
        });
        fieldDiv.appendChild(input);
        formFieldsContainer.appendChild(fieldDiv);
        return;
        
    } else if (fieldInfo.type === 'PDFRadioGroup') {
        const radioWrapper = document.createElement('div');
        radioWrapper.style.display = 'flex';
        radioWrapper.style.flexDirection = 'column';
        radioWrapper.style.gap = '8px';
        
        fieldInfo.options.forEach(option => {
            const radioDiv = document.createElement('div');
            radioDiv.style.display = 'flex';
            radioDiv.style.alignItems = 'center';
            radioDiv.style.gap = '8px';
            
            const radioInput = document.createElement('input');
            radioInput.type = 'radio';
            radioInput.name = fieldInfo.name;
            radioInput.value = option;
            radioInput.checked = fieldInfo.value === option;
            radioInput.dataset.fieldName = fieldInfo.name;
            
            radioInput.addEventListener('change', (e) => {
                if (e.target.checked) {
                    const field = formFields.find(f => f.name === e.target.dataset.fieldName);
                    if (field) field.value = e.target.value;
                }
            });
            
            const radioLabel = document.createElement('label');
            radioLabel.textContent = option;
            radioLabel.style.fontWeight = 'normal';
            radioLabel.style.cursor = 'pointer';
            radioLabel.addEventListener('click', () => radioInput.click());
            
            radioDiv.appendChild(radioInput);
            radioDiv.appendChild(radioLabel);
            radioWrapper.appendChild(radioDiv);
        });
        
        fieldDiv.appendChild(radioWrapper);
        formFieldsContainer.appendChild(fieldDiv);
        return;
    }
    
    // Fallback for unknown field types - treat as text input
    console.log(`Creating fallback text input for field: ${fieldInfo.name}, type: ${fieldInfo.type}`);
    input = document.createElement('input');
    input.type = 'text';
    input.value = fieldInfo.value || '';
    input.dataset.fieldName = fieldInfo.name;
    input.addEventListener('input', (e) => {
        const field = formFields.find(f => f.name === e.target.dataset.fieldName);
        if (field) field.value = e.target.value;
    });
    fieldDiv.appendChild(input);
    formFieldsContainer.appendChild(fieldDiv);
}

// Clear form
document.getElementById('clear-btn').addEventListener('click', () => {
    if (confirm('Are you sure you want to clear all field values?')) {
        formFields.forEach(fieldInfo => {
            if (fieldInfo.type === 'PDFTextField') {
                fieldInfo.value = '';
            } else if (fieldInfo.type === 'PDFCheckBox') {
                fieldInfo.value = false;
            } else if (fieldInfo.type === 'PDFDropdown') {
                fieldInfo.value = [];
            } else if (fieldInfo.type === 'PDFRadioGroup') {
                fieldInfo.value = '';
            }
        });
        
        // Update UI - only clear value inputs, not field name inputs
        document.querySelectorAll('.form-field input[data-field-name], .form-field textarea[data-field-name]').forEach(input => {
            input.value = '';
        });
        document.querySelectorAll('.form-field input[type="checkbox"]').forEach(checkbox => {
            checkbox.checked = false;
        });
        document.querySelectorAll('.form-field select').forEach(select => {
            select.selectedIndex = 0;
        });
        document.querySelectorAll('.form-field input[type="radio"]').forEach(radio => {
            radio.checked = false;
        });
    }
});

// Reset form
document.getElementById('reset-btn').addEventListener('click', () => {
    if (confirm('Are you sure you want to reset all fields to their original values?')) {
        // Restore original values
        formFields.forEach(fieldInfo => {
            const original = originalFormFields.find(f => f.name === fieldInfo.name);
            if (original) {
                fieldInfo.value = original.value;
            }
        });
        
        // Update UI
        formFields.forEach(fieldInfo => {
            if (fieldInfo.type === 'PDFTextField') {
                const input = document.querySelector(`.form-field input[data-field-name="${fieldInfo.name}"], .form-field textarea[data-field-name="${fieldInfo.name}"]`);
                if (input) input.value = fieldInfo.value;
            } else if (fieldInfo.type === 'PDFCheckBox') {
                const checkbox = document.querySelector(`.form-field input[type="checkbox"][data-field-name="${fieldInfo.name}"]`);
                if (checkbox) checkbox.checked = fieldInfo.value;
            } else if (fieldInfo.type === 'PDFDropdown') {
                const select = document.querySelector(`.form-field select[data-field-name="${fieldInfo.name}"]`);
                if (select && fieldInfo.value.length > 0) {
                    select.value = fieldInfo.value[0];
                } else if (select) {
                    select.selectedIndex = 0;
                }
            } else if (fieldInfo.type === 'PDFRadioGroup') {
                const radios = document.querySelectorAll(`.form-field input[type="radio"][data-field-name="${fieldInfo.name}"]`);
                radios.forEach(radio => {
                    radio.checked = radio.value === fieldInfo.value;
                });
            }
        });
    }
});

function csvEscape(value) {
    const stringValue = (value ?? '').toString();
    if (/[,"\n]/.test(stringValue)) {
        return '"' + stringValue.replace(/"/g, '""') + '"';
    }
    return stringValue;
}

function exportFieldsToCsv() {
    if (!formFields.length) {
        alert('Please upload a PDF with fields before exporting CSV.');
        return;
    }

    const activeFields = formFields.filter(field => !field.deleted);
    const headerRow = activeFields.map(field => csvEscape(field.name)).join(',');
    const valueRow = activeFields.map(field => {
        let value = field.value;
        if (Array.isArray(value)) {
            value = value.join('; ');
        } else if (typeof value === 'boolean') {
            value = value ? 'true' : 'false';
        }
        return csvEscape(value ?? '');
    }).join(',');

    const csvContent = [headerRow, valueRow].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${fileName.textContent || 'form-fields'}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function splitCsvLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
        const char = line[i];

        if (inQuotes) {
            if (char === '"') {
                if (line[i + 1] === '"') {
                    current += '"';
                    i++;
                } else {
                    inQuotes = false;
                }
            } else {
                current += char;
            }
        } else {
            if (char === '"') {
                inQuotes = true;
            } else if (char === ',') {
                result.push(current);
                current = '';
            } else {
                current += char;
            }
        }
    }

    result.push(current);
    return result;
}

function parseCsvContent(content) {
    const lines = content.split(/\r?\n/).filter(line => line.trim().length > 0);
    if (!lines.length) {
        return { headers: [], rows: [] };
    }

    const headers = splitCsvLine(lines[0]).map(h => h.trim());
    const rows = lines.slice(1).map(line => splitCsvLine(line));
    return { headers, rows };
}

function csvRowsToDataMap(parsedCsv) {
    const headers = parsedCsv.headers.map(h => h.trim());
    const normalizedHeaders = headers.map(normalizeFieldName);
    const dataMaps = [];

    parsedCsv.rows.forEach((values, rowIdx) => {
        const map = {};
        const normalizedMap = {};

        normalizedHeaders.forEach((nHeader, idx) => {
            if (!headers[idx]) return;
            const val = values[idx] ?? '';
            map[headers[idx]] = val;
            normalizedMap[nHeader] = val;
        });

        const hasData = Object.values(map).some(v => (v ?? '').toString().trim().length > 0);
        if (hasData) {
            map.__normalized = normalizedMap;
            dataMaps.push(map);
        } else {
            console.warn(`Row ${rowIdx + 2} is empty and was skipped.`);
        }
    });

    return dataMaps;
}

async function generateFilledPdf(dataMap, baseBytes) {
    const bytes = baseBytes || basePdfBytes || requirePdfLoaded();
    const tempPdf = await PDFLib.PDFDocument.load(bytes);
    const helvetica = await tempPdf.embedFont(PDFLib.StandardFonts.Helvetica);
    const form = tempPdf.getForm();
    const pdfFields = form.getFields();
    const pdfFieldMap = {};
    pdfFields.forEach(field => {
        pdfFieldMap[field.getName()] = field;
    });

    logFill('Generating PDF with data keys:', Object.keys(dataMap));

    for (const fieldInfo of formFields) {
        if (fieldInfo.deleted) continue;
        const pdfField = pdfFieldMap[fieldInfo.name];
        if (!pdfField) continue;

        const normalized = dataMap.__normalized || {};
        const rawValue = dataMap[fieldInfo.name];
        const normValue = normalized[normalizeFieldName(fieldInfo.name)];
        const value = (rawValue !== undefined ? rawValue : normValue);
        if (value === undefined) {
            logFill('No value for field', fieldInfo.name, '- skipping');
            continue;
        }

        try {
            if (fieldInfo.type === 'PDFTextField') {
                const textVal = (value ?? '').toString();
                pdfField.setText(textVal);
                logFill('Set text', fieldInfo.name, '=>', textVal);
            } else if (fieldInfo.type === 'PDFCheckBox') {
                const isChecked = typeof value === 'string' ? value.trim().toLowerCase() === 'true' : !!value;
                if (isChecked) {
                    pdfField.check();
                } else {
                    pdfField.uncheck();
                }
                logFill('Set checkbox', fieldInfo.name, '=>', isChecked);
            } else if (fieldInfo.type === 'PDFDropdown') {
                if (value !== undefined && value !== null && value !== '') {
                    const candidate = Array.isArray(value) ? value[0] : value;
                    const match = findCaseInsensitiveOption(pdfField.getOptions ? pdfField.getOptions() : fieldInfo.options, candidate) || candidate;
                    pdfField.select(match);
                    logFill('Set dropdown', fieldInfo.name, '=>', match);
                }
            } else if (fieldInfo.type === 'PDFRadioGroup') {
                if (value !== undefined && value !== null && value !== '') {
                    const candidate = Array.isArray(value) ? value[0] : value;
                    const match = findCaseInsensitiveOption(pdfField.getOptions ? pdfField.getOptions() : fieldInfo.options, candidate) || candidate;
                    pdfField.select(match);
                    logFill('Set radio', fieldInfo.name, '=>', match);
                }
            } else {
                const textVal = (value ?? '').toString();
                pdfField.setText(textVal);
                logFill('Set fallback text', fieldInfo.name, '=>', textVal);
            }
        } catch (error) {
            console.error(`Error setting value for field ${fieldInfo.name}:`, error);
        }
    }

    // Refresh appearances so values render in output
    form.updateFieldAppearances(helvetica);

    return await tempPdf.save();
}

async function processCsvImports(files) {
    if (!files.length) return;

    let validatedBytes;
    try {
        validatedBytes = await ensureValidPdfBytes();
    } catch (err) {
        alert(err.message);
        return;
    }

    logFill('Starting CSV import for', files.length, 'file(s)');

    loading.classList.add('visible');

    try {
        const zip = new JSZip();
        let pdfCount = 0;

        for (const file of files) {
            const content = await file.text();
            const parsed = parseCsvContent(content);
            const dataMaps = csvRowsToDataMap(parsed);

            if (!dataMaps.length) {
                console.warn(`No data found in CSV ${file.name}`);
                continue;
            }

            const fileBaseName = file.name.replace(/\.csv$/i, '') || 'filled-form';
            let rowNumber = 1;

            for (const dataMap of dataMaps) {
                const pdfBytes = await generateFilledPdf(dataMap, validatedBytes);
                zip.file(`filled_${fileBaseName}_row${rowNumber}.pdf`, pdfBytes);
                pdfCount += 1;
                rowNumber += 1;
            }
        }

        if (pdfCount === 0) {
            alert('No filled PDFs were generated from the provided CSV data.');
            return;
        }

        const zipBlob = await zip.generateAsync({ type: 'blob' });
        const url = URL.createObjectURL(zipBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${fileName.textContent || 'filled-forms'}.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (error) {
        alert('Error processing CSV imports: ' + error.message);
        console.error(error);
    } finally {
        loading.classList.remove('visible');
    }
}

exportCsvBtn.addEventListener('click', exportFieldsToCsv);
importCsvBtn.addEventListener('click', () => csvImportInput.click());
csvImportInput.addEventListener('change', (event) => {
    processCsvImports(Array.from(event.target.files));
    event.target.value = '';
});
// Download filled PDF
document.getElementById('download-btn').addEventListener('click', async () => {
    try {
        loading.classList.add('visible');
        
        const form = pdfDoc.getForm();
        
        // First, remove deleted fields from the PDF
        for (const fieldInfo of formFields) {
            if (fieldInfo.deleted) {
                try {
                    form.removeField(fieldInfo.field);
                    console.log(`Removed field: ${fieldInfo.name}`);
                } catch (error) {
                    console.error(`Error removing field ${fieldInfo.name}:`, error);
                }
            }
        }
        
        // Apply all field values for non-deleted fields
        for (const fieldInfo of formFields) {
            if (fieldInfo.deleted) continue; // Skip deleted fields
            
            try {
                if (fieldInfo.type === 'PDFTextField') {
                    fieldInfo.field.setText(fieldInfo.value);
                } else if (fieldInfo.type === 'PDFCheckBox') {
                    if (fieldInfo.value) {
                        fieldInfo.field.check();
                    } else {
                        fieldInfo.field.uncheck();
                    }
                } else if (fieldInfo.type === 'PDFDropdown') {
                    if (fieldInfo.value.length > 0) {
                        fieldInfo.field.select(fieldInfo.value[0]);
                    }
                } else if (fieldInfo.type === 'PDFRadioGroup') {
                    if (fieldInfo.value) {
                        fieldInfo.field.select(fieldInfo.value);
                    }
                }
            } catch (error) {
                console.error(`Error setting field ${fieldInfo.name}:`, error);
            }
        }
        
        // Save PDF
        const pdfBytes = await pdfDoc.save();
        
        // Download
        const blob = new Blob([pdfBytes], { type: 'application/pdf' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'filled_' + fileName.textContent;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        loading.classList.remove('visible');
        
    } catch (error) {
        alert('Error generating PDF: ' + error.message);
        console.error(error);
        loading.classList.remove('visible');
    }
});
