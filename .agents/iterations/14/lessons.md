# Lessons

Do not assume `rsync --link-dest` can hardlink between SpaceHPC project and scratch paths. Preserve symlinks with `rsync -a`, exclude known large auxdata paths, and recreate those as symlinks in each seed/runtime userdir.
