alias ll='ls -laG'
alias l='ls -G'

export PYTHONPATH=/usr/local/lib/python:$PYTHONPATH
export PATH=/usr/local/share/python:$PATH
export PATH=/usr/local/bin:$PATH

# MacPorts Installer addition on 2011-12-24_at_10:13:14: adding an appropriate PATH variable for use with MacPorts.
export PATH=/opt/local/bin:/opt/local/sbin:$PATH

source /usr/local/share/python/virtualenvwrapper.sh
