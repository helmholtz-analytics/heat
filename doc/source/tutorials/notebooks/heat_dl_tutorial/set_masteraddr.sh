MASTER_ADDR=$(scontrol show hostnames | head -n 1)
MASTER_ADDR="${MASTER_ADDR}i"
export MASTER_ADDR=$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')
export MASTER_PORT=6000
