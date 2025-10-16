# ace_docker_perms.py
def ensure_path_perms(container, dest_path, users=("www-data","nginx","root")):
    """
    Try common webserver users to chown the deployed file and chmod 644.
    Returns True if any chown succeeded, else False.
    """
    try:
        for user in users:
            rc = container.exec_run(["chown", f"{user}:{user}", dest_path])
            if getattr(rc, "exit_code", 0) == 0:
                container.exec_run(["chmod", "644", dest_path])
                return True
    except Exception:
        pass
    return False
