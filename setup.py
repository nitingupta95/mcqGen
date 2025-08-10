from setuptools import find_packages,setup # pyright: ignore[reportMissingModuleSource]

setup(
    name='mcqgenrator',
    version='0.0.1',
    author="nitinGupta",
    author_email="ng61315@gmail.com",
    install_requires=["openai","langchain","streamlit","python-dotenv", "PyPDF2"],
    packages=find_packages()
)