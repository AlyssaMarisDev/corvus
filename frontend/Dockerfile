FROM node:22

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

EXPOSE 8081

# A mobile app can't run inside Docker, so this container runs the Expo dev
# server. Connect with Expo Go on a device in the same network, or run
# `npx expo start --web` locally to use the app in a browser.
CMD ["npx", "expo", "start"]
