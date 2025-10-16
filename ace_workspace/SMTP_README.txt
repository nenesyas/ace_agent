SMTP Rehberi

1) ace_workspace/secrets.json dosyası oluşturun ve içine aşağıdaki yapıyı koyun:

{
  "smtp": {
    "host": "smtp.example.com",
    "port": 587,
    "username": "you@example.com",
    "password": "app-or-password",
    "use_tls": true
  }
}

2) Gmail kullanıyorsanız uygulama şifresi veya OAuth kullanın.
3) SMTP bilgilerini girdikten sonra 'email to:someone@example.com subject:Test body:Merhaba' komutunu kullanın.
