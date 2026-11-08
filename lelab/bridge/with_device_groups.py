"""Refresh already-authorized device groups inherited from an old WSL shell."""
import grp
import json
import os
import pwd
import shlex
import sys

required=('dialout','video')
effective=set(os.getgroups()) | {os.getgid()}
missing=[name for name in required if grp.getgrnam(name).gr_gid not in effective]
if missing:
    username=pwd.getpwuid(os.getuid()).pw_name
    name=missing[0]
    group=grp.getgrnam(name)
    if username not in group.gr_mem and pwd.getpwuid(os.getuid()).pw_gid != group.gr_gid:
        raise SystemExit(f'{username} is not authorized for group {name}; update account membership first.')
    refreshed=os.environ.get('LELAB_REFRESHED_GROUPS','').split(',')
    if name in refreshed:
        raise SystemExit('Could not refresh device groups. Start LeLab from a fresh WSL login.')
    os.environ['LELAB_REFRESHED_GROUPS']=','.join([g for g in refreshed if g]+[name])
    command=shlex.join([sys.executable,str(__file__),*sys.argv[1:]])
    print(f'Refreshing existing {name} membership for this LeLab process...',flush=True)
    os.execv('/usr/bin/sg',['sg',name,'-c',command])

if sys.argv[1:]==['--probe']:
    print(json.dumps({'uid':os.getuid(),'gid':os.getgid(),
        'effective_groups':[grp.getgrgid(g).gr_name for g in sorted(effective)],
        'device_groups_ready':True},indent=2))
elif len(sys.argv)>1:
    os.execvp(sys.argv[1],sys.argv[1:])
else:
    raise SystemExit('Provide a command or --probe')
