from git import Repo
import os
import shutil

def download_gabc2volpiano():
    """
    Function to download extra github python files of gabc2volpiano project.
    The project is meant to convert gabc format into volpiano format.
    It is not implemented as python library .. so we have to install it this way.
    """
    Repo.clone_from("https://github.com/bacor/gabc2volpiano.git", "./temp")

    if not os.path.exists("./gabc2volpiano"):
        os.makedirs("./gabc2volpiano")
    os.rename("./temp/gabc2volpiano/converter.py", "./gabc2volpiano/converter.py")
    os.rename("./temp/gabc2volpiano/parser.py", "./gabc2volpiano/parser.py")
    os.rename("./temp/gabc2volpiano/__init__.py", "./gabc2volpiano/__init__.py")
    os.rename("./temp/gabc2volpiano/gabc.peg", "./gabc2volpiano/gabc.peg")

    shutil.rmtree('./temp')