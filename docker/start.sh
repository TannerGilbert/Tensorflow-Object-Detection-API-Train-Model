if type nvidia-smi > /dev/null || wmic path win32_VideoController get name | grep -q "NVIDIA"; then
 cd gpu
else
 cd cpu
fi

docker-compose up