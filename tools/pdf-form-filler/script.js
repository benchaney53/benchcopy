// PDF.js setup
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

let uploadedPdfBytes = null;
let pdfDoc = null;
let formFields = [];

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
    if (files.length > 0 && files[0].type === 'application/pdf') {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
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
    
    try {
        const arrayBuffer = await file.arrayBuffer();
        uploadedPdfBytes = new Uint8Array(arrayBuffer);
        
        // Load PDF with pdf-lib
        pdfDoc = await PDFLib.PDFDocument.load(uploadedPdfBytes);
        
        // Extract form fields
        await extractFormFields();
        
        loading.classList.remove('visible');
        formSection.classList.add('visible');
        
    } catch (error) {
        alert('Error loading PDF: ' + error.message);
        console.error(error);
        loading.classList.remove('visible');
    }
}

async function extractFormFields() {
    formFields = [];
    formFieldsContainer.innerHTML = '';
    
    const form = pdfDoc.getForm();
    const fields = form.getFields();
    
    fieldCount.textContent = fields.length;
    
    for (const field of fields) {
        const fieldName = field.getName();
        const fieldType = field.constructor.name;
        
        let fieldInfo = {
            name: fieldName,
            type: fieldType,
            field: field,
            value: null
        };
        
        // Get current value based on field type
        if (fieldType === 'PDFTextField') {
            fieldInfo.value = field.getText() || '';
            fieldInfo.isMultiline = field.isMultiline();
        } else if (fieldType === 'PDFCheckBox') {
            fieldInfo.value = field.isChecked();
        } else if (fieldType === 'PDFDropdown') {
            fieldInfo.value = field.getSelected() || [];
            fieldInfo.options = field.getOptions();
        } else if (fieldType === 'PDFRadioGroup') {
            fieldInfo.value = field.getSelected();
            fieldInfo.options = field.getOptions();
        }
        
        formFields.push(fieldInfo);
        createFormField(fieldInfo);
    }
}

function createFormField(fieldInfo) {
    const fieldDiv = document.createElement('div');
    fieldDiv.className = 'form-field';
    
    const label = document.createElement('label');
    label.textContent = fieldInfo.name;
    fieldDiv.appendChild(label);
    
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
        fieldDiv.appendChild(radioWrapper);
        formFieldsContainer.appendChild(fieldDiv);
        return;
    }
    
    // This should never be reached now, but keep as fallback
    if (input) {
        fieldDiv.appendChild(input);
    }
    
    formFieldsContainer.appendChild(fieldDiv);
}   }
    
    formFieldsContainer.appendChild(fieldDiv);
}

// Reset form
document.getElementById('reset-btn').addEventListener('click', async () => {
    if (confirm('Are you sure you want to reset all fields to their original values?')) {
        await extractFormFields();
    }
});

// Download filled PDF
document.getElementById('download-btn').addEventListener('click', async () => {
    try {
        loading.classList.add('visible');
        
        const form = pdfDoc.getForm();
        
        // Apply all field values
        for (const fieldInfo of formFields) {
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
