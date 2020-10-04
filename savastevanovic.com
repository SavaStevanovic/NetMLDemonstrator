server {
    listen 443 ssl http2;

    root /var/www/savastevanovic.com;

    index index.html;

    server_name savastevanovic.com www.savastevanovic.com;

    location / {
        try_files $uri $uri/ =404;
    }

    location /menager {
        proxy_pass http://127.0.0.1:4321;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    ssl on;
    ssl_certificate /etc/letsencrypt/live/savastevanovic.com/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/savastevanovic.com/privkey.pem;
 # managed by Certbot

}
server {
    if ($host = www.savastevanovic.com) {
        return 301 https://$host$request_uri;
    } # managed by Certbot


    if ($host = savastevanovic.com) {
        return 301 https://$host$request_uri;
    } # managed by Certbot


    listen 80;
    server_name savastevanovic.com www.savastevanovic.com;
    return 301 https://savastevanovic.com$request_uri;




}