FROM node:20
WORKDIR /app
RUN npm config set legacy-peer-deps true
RUN npm install -g @ionic/cli
COPY package*.json ./
RUN npm install
EXPOSE 80
ENTRYPOINT ["ionic"]    
CMD ["serve", "--port", "80", "--host", "0.0.0.0", "--disable-host-check"]
