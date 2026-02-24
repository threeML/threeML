import argparse
import logging
import pathlib
import re
import subprocess

from packaging.version import Version

log = logging.getLogger(__name__)


def get_next_version():
    latest_v = Version(
        subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"])
        .decode("utf8")
        .strip("\n")
    )
    assert (
        latest_v.is_prerelease
    ), "Latest tag ist not a prerelease - please manually specify next version"

    return f"v{latest_v.major}.{latest_v.minor}.{latest_v.micro}"


def generate_base_changelog(future_release, output_path):

    subprocess.run(
        [
            "github_changelog_generator",
            "-u",
            "threeML",
            "-p",
            "threeML",
            "--exclude-tags-regex",
            "(.+)(dev)([0-9]?)",
            "--future-release",
            future_release,
            "-o",
            output_path,
        ]
    )
    log.info("Successfully created changelog")


def refomat_base_changelog(file):
    with open(file, "r") as f:
        lines = f.readlines()
    full_text = []
    current_minor = None
    for i, li in enumerate(lines):
        if (
            re.match("^(## \[v\d\.\d\.\d\]).*", li)  # normal vX.X.X versioning
            or re.match("^(## \[v\d\.\d\]).*", li)  # catch vX.X (v1.0)
            or re.match("^(## \[\d\.\d\.\d\]).*", li)  # catch missing "v" (0.5.1)
        ):
            if not re.match("^(## \[v\d\.\d\]).*", li):
                version = (
                    re.match(".*(\d\.\d\.\d).*", li).group(1).strip("[").strip("]")
                )
            else:
                version = re.match(".*(\d\.\d).*", li).group(1).strip("[").strip("]")

            li = "#" + li
            version = Version(version)
            if current_minor != version.minor:
                full_text.append(f"\n## v{version.major}.{version.minor}\n")
                current_minor = version.minor
            full_text.append(li)
        else:
            full_text.append(li)

    with open(file, "w") as f:
        f.writelines(full_text)
    log.info("Successfully reformatted changelog")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        add_help=True,
        epilog=(
            "Please checkout ",
            "https://github.com/github-changelog-generator/github-changelog-generator",
            " beforehand - especially the part about the token authentication",
            "\n\n",
            "Currently you need the master-branch from github-changelog-generator or",
            "Verion>v1.16.4! - clone the branch, run `gem build <FILENAME>.gemspec` and"
            " `gem install <FILENAME>.gem`",
        ),
    )
    parser.add_argument(
        "-f",
        "--future_version",
        help="Specify the version of the next release, e.g. v.2.5.2",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args([])
    print(args)
    if args.verbose:
        log.setLevel("INFO")
    if args.future_version is None:
        args.future_version = get_next_version()

    current_path = pathlib.Path(__file__).parent.resolve()

    generate_base_changelog(args.future_version, current_path / "CHANGELOG.md")
    refomat_base_changelog(current_path / "CHANGELOG.md")
    subprocess.run(
        [
            "mv",
            current_path / "CHANGELOG.md",
            current_path.parent.resolve() / "CHANGELOG.md",
        ]
    )
    log.info(
        "Moved changelog to " + str(current_path.parent.resolve() / "CHANGELOG.md"),
    )
