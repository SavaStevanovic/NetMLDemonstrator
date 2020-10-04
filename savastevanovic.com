# Expires map
map $sent_http_content_type $expires {
    default                    off;
    text/html                  epoch;
    text/css                   max;
    application/javascript     max;
    ~image/                    max;
}

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

    gzip on;
    # gzip_static on;
    gzip_min_length 10240;
    gzip_comp_level 1;
    gzip_vary on;
    gzip_disable msie6;
    gzip_proxied expired no-cache no-store private auth;
    gzip_types
        # text/html is always compressed by HttpGzipModule
        text/css
        text/javascript
        text/xml
        text/plain
        text/x-component
        application/javascript
        application/x-javascript
        application/json
        application/xml
        application/rss+xml
        application/atom+xml
        font/truetype
        font/opentype
        application/vnd.ms-fontobject
        image/svg+xml;

    expires $expires;
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
