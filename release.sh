#!/bin/sh
set -e

repository="$1"
version="$2"

cwd=$(pwd)
cd `dirname $0`

if [ -z "$repository" ]; then
    echo "Usage: $0 <repository> <version>"
    exit 1
fi

if [ -z "$version" ]; then
    echo "Usage: $0 <repository> <version>"
    exit 1
fi

echo "Running poetry lock and install ..."
poetry lock --no-update
poetry install

echo "Setting package version ..."
poetry version $2
echo "__version__ = \"$2\"" > graph_diffusers/_version.py

echo "Running tests ..."
poetry run pytest tests/

echo "Running tests without torch_sparse ..."
poetry run pip uninstall -y torch_sparse
poetry run pytest tests/

echo "Running tests without torch ..."
poetry run pip uninstall -y torch
poetry run pytest tests/

echo "Restoring environment ..."
poetry install

echo "Cleaning up dist/ ..."
rm -rf dist/

echo "Publishing to $repository ..."
poetry publish \
    --repository $repository \
    --build \
    --no-interaction

git tag -m "Release $version" $version
git push origin $version

echo "Done!"

cd $cwd