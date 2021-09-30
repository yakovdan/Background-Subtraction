sudo apt install -y cloud-guest-utils   # Debian stretch, Ubuntu
sudo growpart /dev/sda 1
sudo resize2fs /dev/sda1