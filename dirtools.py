from path import Path as path

def all_subdirs(tgt):
    """
Get a list of all subdirectories under and including tgt.
    """
    p = path(tgt)
    dirs = p.dirs()
    result = dirs + [tgt]
    if len(dirs) <= 1:
        return result    
    for d in dirs:
        result += all_subdirs(d)
    return list(set(result))