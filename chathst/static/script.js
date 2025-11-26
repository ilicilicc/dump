document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatBox = document.querySelector('.chat-box');
    const runTrialsBtn = document.getElementById('run-trials-btn');
    const trialsResult = document.getElementById('trials-result');
    const injectContextBtn = document.getElementById('inject-context-btn');
    const modelVersionSelect = document.getElementById('model-version');

    const modelCapabilities = {
        'v3_ultra': ['speculative-decoding'],
        'v4_unified': ['speculative-decoding', 'mode'],
        'v5_2_unified': ['speculative-decoding', 'mode'],
        'v6_giga': ['speculative-decoding', 'mode', 'context-injection'],
        'v6_1': ['speculative-decoding', 'mode', 'context-injection', 'multimodality'],
        'v7_0_agile': ['speculative-decoding', 'mode'],
        'v7_1_ultimate': ['speculative-decoding', 'mode', 'context-injection', 'multimodality'],
        'chaos_logic': ['chaos-logic'],
        'error_networks': ['chaos-logic', 'error-networks']
    };

    function updateVisibleControls() {
        const selectedModel = modelVersionSelect.value;
        const capabilities = modelCapabilities[selectedModel] || [];

        document.getElementById('mode-controls').classList.toggle('hidden', !capabilities.includes('mode'));
        document.getElementById('chaos-logic-controls').classList.toggle('hidden', !capabilities.includes('chaos-logic'));
        document.getElementById('error-networks-controls').classList.toggle('hidden', !capabilities.includes('error-networks'));
        document.getElementById('speculative-decoding-controls').classList.toggle('hidden', !capabilities.includes('speculative-decoding'));
        document.getElementById('context-injection-controls').classList.toggle('hidden', !capabilities.includes('context-injection'));

        // Hide/show multimodal buttons
        const multimodalButtons = document.querySelectorAll('.multimodal-btn');
        multimodalButtons.forEach(btn => {
            btn.classList.toggle('hidden', !capabilities.includes('multimodality'));
        });
    }

    modelVersionSelect.addEventListener('change', updateVisibleControls);

    // Initial setup
    updateVisibleControls();

    sendBtn.addEventListener('click', () => {
        const message = userInput.value;
        if (message.trim() !== '') {
            addMessage('user', message);
            userInput.value = '';
            // Simulate bot response
            setTimeout(() => {
                addMessage('bot', 'This is a simulated response.');
            }, 1000);
        }
    });

    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendBtn.click();
        }
    });

    runTrialsBtn.addEventListener('click', () => {
        // Simulate running trials
        trialsResult.textContent = 'Running trials...';
        setTimeout(() => {
            const successes = Math.floor(Math.random() * 12);
            trialsResult.textContent = `Result: ${successes}/11 successes`;
        }, 1000);
    });

    injectContextBtn.addEventListener('click', () => {
        const context = document.getElementById('context-injection').value;
        if (context.trim() !== '') {
            addMessage('system', `Context injected: ${context.substring(0, 50)}...`);
            document.getElementById('context-injection').value = '';
        }
    });

    function addMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        messageElement.textContent = message;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});
