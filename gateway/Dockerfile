FROM nginx:1.29.5-alpine

RUN rm /etc/nginx/conf.d/default.conf

COPY gateway/nginx.conf /etc/nginx/conf.d/

RUN apk add --no-cache curl

# RUN mkdir -p /var/www/static

EXPOSE 80
EXPOSE 443

# CMD ["nginx", "-g", "daemon off;"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/ || exit 1


