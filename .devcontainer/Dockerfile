#-------------------------------------------------------------------------------------------------------------
# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0
# Contributions to this project, i.e., https://github.com/ag2ai/ag2, are licensed under the Apache License, Version 2.0 (Apache-2.0).

# Portions derived from  https://github.com/microsoft/autogen under the MIT License.
# SPDX-License-Identifier: MIT
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for license information.
#-------------------------------------------------------------------------------------------------------------

FROM mcr.microsoft.com/vscode/devcontainers/python:3.10

#
# Update the OS and maybe install packages
#
ENV DEBIAN_FRONTEND=noninteractive

# add git lhs to apt
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

RUN apt-get update \
   && apt-get upgrade -y  \
   && apt-get -y install --no-install-recommends build-essential npm git-lfs \
   && apt-get autoremove -y \
   && apt-get clean -y \
   && arch=$(arch | sed s/aarch64/arm64/ | sed s/x86_64/amd64/) \
   && wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.5.23/quarto-1.5.23-linux-${arch}.deb \
   && dpkg -i quarto-1.5.23-linux-${arch}.deb \
   && rm -rf /var/lib/apt/lists/* quarto-1.5.23-linux-${arch}.deb
ENV DEBIAN_FRONTEND=dialog

# For docs
RUN npm install --global yarn
RUN pip install --upgrade pip
RUN pip install pydoc-markdown
RUN pip install pyyaml
RUN pip install colored
