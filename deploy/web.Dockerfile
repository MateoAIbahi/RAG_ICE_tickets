FROM nginx:alpine
COPY web/nginx.conf /etc/nginx/conf.d/default.conf
COPY web/index.html /usr/share/nginx/html/index.html