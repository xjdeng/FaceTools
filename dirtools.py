from path import Path as path

def all_subdirs(tgt):
    p = path(tgt)
    dirs = p.dirs()
    if len(dirs) == 0:
        return []
    result = dirs
    for d in dirs:
        result += all_subdirs(d)
    return result