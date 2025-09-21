#
# Dockerfile for the IMDB Frontend Service (React + Vite)
#

# ==============================================================================
# Build Stage
#
# This stage uses a Node.js image to build the static React application.
# It installs dependencies and runs the Vite build script. The output of this
# stage is the 'dist' directory containing optimized static assets.
# ==============================================================================
FROM node:20 AS build

# Set the working directory for all subsequent commands.
WORKDIR /app

# Copy the package.json and package-lock.json files.
# This is done first to leverage Docker's layer caching. The next step will
# only re-run if these specific files have changed, speeding up builds.
COPY package*.json ./

# Install dependencies using 'npm ci' which is faster and more reliable for
# CI/CD environments as it uses the package-lock.json file exclusively.
RUN npm ci

# Copy the rest of the application's source code into the container.
COPY . .

# Execute the build script defined in package.json to compile the
# React application into static HTML, CSS, and JavaScript files.
RUN npm run build

# ==============================================================================
# Runtime/Serve Stage
#
# This stage uses a lightweight Nginx web server to serve the static files
# created in the 'build' stage. This results in a very small and secure final image.
# ==============================================================================
FROM nginx:alpine AS runtime

# Copy only the built static assets from the 'build' stage into the
# Nginx public HTML directory where they can be served.
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port 80, which is the default port Nginx listens on.
EXPOSE 80

# The command to start the Nginx server.
# The '-g "daemon off;"' flag ensures Nginx runs in the foreground, which is
# required for Docker containers to remain active.
CMD ["nginx", "-g", "daemon off;"]