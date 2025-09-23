document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');

    // Function to handle login form submission
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const loginBtn = loginForm.querySelector('button');
            loginBtn.disabled = true;
            loginBtn.textContent = 'Logging in...';

            const formData = new FormData(loginForm);
            
            try {
                const response = await fetch('/login', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.success) {
                    window.location.href = '/dashboard';
                } else {
                    alert(data.message);
                }
            } catch (error) {
                console.error('Login failed:', error);
                alert('An error occurred during login. Please try again.');
            } finally {
                loginBtn.disabled = false;
                loginBtn.textContent = 'Login';
            }
        });
    }

    // Function to handle registration form submission
    if (registerForm) {
        registerForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const registerBtn = registerForm.querySelector('button');
            registerBtn.disabled = true;
            registerBtn.textContent = 'Registering...';

            const formData = new FormData(registerForm);

            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                alert(data.message);
                
                if (data.success) {
                    window.location.href = '/login';
                }
            } catch (error) {
                console.error('Registration failed:', error);
                alert('An error occurred during registration. Please try again.');
            } finally {
                registerBtn.disabled = false;
                registerBtn.textContent = 'Register';
            }
        });
    }
});