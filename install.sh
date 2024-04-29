#!/bin/bash

#sudo -u ec2-user -i <<'EOF'
cd /usr/local/bin
sudo mkdir ffmpeg && cd ffmpeg

# check https://johnvansickle.com/ffmpeg/ for latest build
sudo wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
sudo tar -xf ffmpeg-release-amd64-static.tar.xz

# Verify the lastest version is ffmpeg-5.0.1 otherwise update the following commands
sudo ln -s /usr/local/bin/ffmpeg/ffmpeg-6.1-amd64-static/ffmpeg /usr/bin/ffmpeg
sudo ln -s /usr/local/bin/ffmpeg/ffmpeg-6.1-amd64-static/ffprobe /usr/bin/ffprobe
