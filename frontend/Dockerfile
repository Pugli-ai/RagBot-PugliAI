FROM node:20

WORKDIR /app


# RUN apk add --update python3 py3-pip

RUN npm config set legacy-peer-deps true

# RUN npm cache clean --force

# RUN npm install -g npm@9.7.1

RUN npm install -g @ionic/cli

COPY package*.json ./

RUN npm install

COPY . .

EXPOSE 8200

ENTRYPOINT ["ionic"]    

CMD ["serve", "--port", "8200", "--address", "0.0.0.0"]
