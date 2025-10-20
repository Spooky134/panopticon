FROM nginx:latest

RUN rm /etc/nginx/conf.d/default.conf

COPY nginx_gateway/nginx.conf /etc/nginx/conf.d/

RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# RUN mkdir -p /var/www/static

EXPOSE 80
EXPOSE 443

# CMD ["nginx", "-g", "daemon off;"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/ || exit 1


