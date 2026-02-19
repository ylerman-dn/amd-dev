# AMD Development Environment

This repository contains scripts and references to other code repositories that aim to be a unified place to help set up and test RCCL (ROCm Communication Collectives Library) and other components related to the AMD stack as it runs LLMs (Large Language Models).

## Overview

The repository provides tools, scripts, and documentation to:
- Set up development environments for AMD GPU clusters
- Test and profile RCCL performance
- Configure networking components for multi-GPU and multi-node setups
- Run inference workloads on AMD hardware

## User Setup

The following commands set up a personal user on a machine:

### Create user with password:
```bash
USERNAME=your_username_here # Set your username
```
```bash
adduser $USERNAME
# Add to needed groups
adduser $USERNAME sudo # For sudo
adduser $USERNAME render # For amd-smi
adduser $USERNAME video # For amd-smi
pkill -u $USERNAME -f "cursor-server" # restart Cursor's termainal server in order to be aware to the new added groups
```

### For each host:

#### Enable SSH between hosts
```bash
USERNAME=your_username_here # Set your username
```
```bash
ssh-keygen -t ed25519
ssh-copy-id $USERNAME@172.30.160.150

ssh $USERNAME@172.30.160.150
ssh-keygen -t ed25519
ssh-copy-id $USERNAME@172.30.160.145
```

#### Fix access to GitHub
```bash
cat > ~/.ssh/config <<EOF
# GitHub over HTTPS port (443) instead of SSH port (22)
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
EOF
```

#### Add SSH keys to GitHub
```bash
cat ~/.ssh/id_ed25519.pub
```

#### Clone and initialize the repo
```bash
git clone git@github.com:Ahalperin/amd-dev.git
cd amd-dev/dn
./create-dev-env.sh
```
