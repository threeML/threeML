import time
import os
import versioneer

VERSION = versioneer.get_version()
RELEASE_NOTES = 'docs/release_notes.rst'
BUILD_DATE = time.strftime('%a, %d %b %Y %H:%M:%S %z')
TAG_MODES = ['major', 'minor', 'patch']

def cmd(command, dry_run=False):
    print('About to execute "%s"...' % command)
    if dry_run:
        return 0  
    #os.system(command)

def update_version(mode):
    """ Return the new tag version.
    """
    prevTag = VERSION.split('+')[0]
    print('Previous tag was %s...' % prevTag)
    version, release, patch = [int(item) for item in prevTag.split('.')]
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
        abort('Unknown release mode %s.' % mode)
    return '%s.%s.%s' % (version, release, patch)

def update_release_notes(tag, dry_run=False):
    """ Write the new tag and build date on top of the release notes.
    """
    title = 'Release notes\n=============\n\n'
    print('Reading in %s...' % RELEASE_NOTES)
    notes = open(RELEASE_NOTES).read().strip('\n').strip(title)
    subtitle = '\nv%s\n--------\n' % tag
    #if not dry_run:
    outputFile = open(RELEASE_NOTES, 'w')
    outputFile.writelines(title)
    outputFile.writelines(subtitle)
    outputFile.writelines('*%s*\n\n' % BUILD_DATE)
    outputFile.writelines(notes)
    outputFile.close()

def tag_package(mode, dry_run=False):
    """ Tag the package with git.
    """
    cmd('git pull', dry_run)
    #cmd('git status', dry_run)
    tag = update_version(mode)
    update_release_notes(tag, dry_run)
    msg = 'Prepare for tag %s [ci skip]' % tag
    cmd('git commit -m "%s" %s' % (msg, RELEASE_NOTES), dry_run)
    cmd('git push', dry_run)
    msg = 'New tag %s' % tag
    cmd('git tag -a v%s -m "%s"' % (tag, msg), dry_run)
    cmd('git push --tags', dry_run)


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-t', dest = 'tagmode', type = str, default = None,
                      help = 'The release tag mode %s.' % TAG_MODES)
    parser.add_option('-n', action = 'store_true', dest = 'dryrun',
                      help = 'Dry run (i.e. do not actually do anything).')
    (opts, args) = parser.parse_args()
    if not opts.tagmode and not (opts.src):
        parser.print_help()
        parser.error('Please specify at least one valid option.')
    tag = None
    if opts.tagmode is not None:
        if opts.tagmode not in TAG_MODES:
            parser.error('Invalid tag mode %s (allowed: %s)' %\
                             (opts.tagmode, TAG_MODES))
        tag_package(opts.tagmode, opts.dryrun)
