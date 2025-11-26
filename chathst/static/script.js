document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatBox = document.querySelector('.chat-box');

    sendBtn.addEventListener('click', () => {
        const message = userInput.value;
        if (message.trim() !== '') {
            addMessage('user', message);
            userInput.value = '';
            // In a real app, you'd send the message to the server
            // and receive a response. For now, we'll simulate a bot response.
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

    function addMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        messageElement.textContent = message;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});
