document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatBox = document.querySelector('.chat-box');
    const injectContextBtn = document.getElementById('inject-context-btn');
    const fileUploadBtn = document.getElementById('file-upload-btn');
    const fileUpload = document.getElementById('file-upload');
    const fileStatus = document.getElementById('file-status');

    sendBtn.addEventListener('click', () => {
        const message = userInput.value;
        if (message.trim() !== '') {
            addMessage('user', message);
            userInput.value = '';
            // Simulate bot response
            setTimeout(() => {
                addMessage('bot', 'This is a simulated response based on the provided context.');
            }, 1000);
        }
    });

    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendBtn.click();
        }
    });

    fileUploadBtn.addEventListener('click', () => {
        fileUpload.click();
    });

    fileUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            fileStatus.textContent = `Loaded: ${file.name}`;
            const reader = new FileReader();
            reader.onload = (event) => {
                document.getElementById('context-injection').value = event.target.result;
            };
            reader.readAsText(file);
        }
    });

    injectContextBtn.addEventListener('click', () => {
        const context = document.getElementById('context-injection').value;
        if (context.trim() !== '') {
            addMessage('system', `Context injected: ${context.substring(0, 100)}...`);
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
