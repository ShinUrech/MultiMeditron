USER_ID=${USER_ID:-1000}
GROUP_ID=${GROUP_ID:-1000}
USERNAME=${USERNAME:root}

groupadd -f -g $GROUP_ID $USERNAME || true
id -u $USERNAME &>/dev/null || useradd -f -m -u $USER_ID -g $GROUP_ID $USERNAME || true

exec gosu $USER_ID:$GROUP_ID "$@"
