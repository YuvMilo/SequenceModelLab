import nox

locations = ["src"]


@nox.session(python=["3.8"])
def tests(session):
    args = session.posargs or []
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", "pytest", *args, external=True)


@nox.session(python=["3.8"])
def lint(session):
    args = session.posargs or locations
    session.install(
        "flake8",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-import-order",
    )
    session.run("flake8", *args)


