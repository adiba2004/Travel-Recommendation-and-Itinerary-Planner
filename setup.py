from setuptools import setup

with open("README.md", "r", encoding = "utf-8") as f:
    long_description = f.read()

## edit below variables as per your requriments -
REPO_NAME = "Travel-Recommendation-System"
AUTHOR_USER_NAME = "Sania"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = ['streamlit', 'numpy']


setup(
    name = SRC_REPO,
    version = "0.0.1",
    author = AUTHOR_USER_NAME,
    description = "A small package for Travel Recommendation System",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    author_email = "saniawaseem2003@gmail.com",
    packages = [SRC_REPO],
    license = "MIT",
    python_requries = ">=3.7",
    install_requires = LIST_OF_REQUIREMENTS
)