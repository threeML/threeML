import os
import sys
import time

import versioneer

VERSION = versioneer.get_version()
RELEASE_NOTES = 'docs/release_notes.rst'
BUILD_DATE = time.strftime("%a, %d %b %Y %H:%M:%S + 0000", time.gmtime())
TAG_MODES = ['major', 'minor', 'patch']


def cmd(command, dry_run=False):
    print('About to execute "%s"...' % command)
    if dry_run:
        return 0
    os.system(command)


def check_branch():
    cmd = 'git rev-parse --abbrev-ref HEAD'
    branch_name = os.popen(cmd).read().split('\n')[0]
    print('Current branch is: %s' % branch_name)
    if branch_name != 'master':
        print("You can't tag a branch different form master. Abort.")
        sys.exit(1)


def update_version(mode):
    """ Return the new tag version.
    """
    prev_tag = VERSION.split('+')[0]
    print('Previous tag was %s...' % prev_tag)
    version, release, patch = [int(item) for item in prev_tag.split('.')]
    if mode == 'major':
        version += 1
        release = 0
        patch = 0
    elif mode == 'minor':
        release += 1
        patch = 0
    elif mode == 'patch':
        patch += 1
    else:
        RuntimeError('Unknown release mode %s.' % mode)
    return '%s.%s.%s' % (version, release, patch)


def update_release_notes(mode, tag, dry_run=False):
    """ Write the new tag and build date on top of the release notes.
    """
    print('Updating %s...' % RELEASE_NOTES)
    title = 'Release Notes\n=============\n\n'
    version = '\nVersion %s\n-----------\n\n' % tag[:-2]
    if mode == 'patch':
        title += version
        subtitle = ''
    else:
        subtitle = version
    notes = open(RELEASE_NOTES).read().strip('\n').strip(title)
    subtitle += '\nv%s\n^^^^^^^^\n' % tag
    if not dry_run:
        output_file = open(RELEASE_NOTES, 'w')
        output_file.writelines(title)
        output_file.writelines(subtitle)
        output_file.writelines('*%s*\n\n' % BUILD_DATE)
        output_file.writelines(notes)
        output_file.close()


def tag_package(mode, dry_run=False):
    """ Tag the package with git.
    """
    cmd('git pull', dry_run)
    cmd('git status', dry_run)
    check_branch()
    tag = update_version(mode)
    update_release_notes(mode, tag, dry_run)
    msg = 'Prepare for tag %s' % tag
    cmd('git commit -m "%s" %s' % (msg, RELEASE_NOTES), dry_run)
    cmd('git push', dry_run)
    msg = 'New tag %s' % tag
    cmd('git tag -a v%s -m "%s"' % (tag, msg), dry_run)
    cmd('git push --tags', dry_run)


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-t', dest='tagmode', type=str, default=None,
                      help='The release tag mode %s.' % TAG_MODES)
    parser.add_option('-n', action='store_true', dest='dryrun',
                      help='Dry run (i.e. do not actually do anything).')
    (opts, args) = parser.parse_args()
    if not opts.tagmode and not (opts.src):
        parser.print_help()
        parser.error('Please specify at least one valid option.')
    tag = None
    if opts.tagmode is not None:
        if opts.tagmode not in TAG_MODES:
            parser.error('Invalid tag mode %s (allowed: %s)' %
                         (opts.tagmode, TAG_MODES))
        tag_package(opts.tagmode, opts.dryrun)
