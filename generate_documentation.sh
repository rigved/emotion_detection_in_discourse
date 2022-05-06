#!/bin/bash

########################################################################################################################
# What triggered that emotion? Emotion cause extraction in conversational discourse.
# Copyright (C) 2022  Rigved Rakshit
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
########################################################################################################################

# Create a temporary directory to store the modified code
mkdir -p "src_doc"

# Loop through every Python file
for file in src/*.py; do
  # If the file is empty or doesn't exist, move on to the next one
  if [[ ! -s "${file}" ]]; then
    continue
  fi

  # Strip the path and get the filename
  file=$(basename "${file}")

  echo "Preparing ${file} for doxygen..."

  # Convert the Python code into doxygen supported Python code
  doxypy "src/${file}" > "src_doc/${file}"
done

# Generate the documentation
doxygen "emotion_discourse_doc.conf"

# Remove the temporary code directory
rm -r "src_doc"
