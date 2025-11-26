document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatBox = document.querySelector('.chat-box');
    const runTrialsBtn = document.getElementById('run-trials-btn');
    const trialsResult = document.getElementById('trials-result');
    const injectContextBtn = document.getElementById('inject-context-btn');

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
