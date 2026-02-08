curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update -y
apt-get install -y vim
apt-get install -y   libxrender1 libxext6 libx11-6 libxi6 libxrandr2 libxcursor1 libxinerama1   libgl1-mesa-glx libglib2.0-0
apt-get install -y libegl1 libegl1-mesa libgl1 libgl1-mesa-dri libglvnd0 mesa-utils-extra
apt-get install -y libvulkan1 vulkan-tools
apt-get install -y software-properties-common
apt-get install -y g++-13 gcc-13
apt-get install -y cmake ninja-build build-essential


uv venv -p 3.13 --seed --clear
source .venv/bin/activate
uv pip install -r requirements.txt
