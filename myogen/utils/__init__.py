from myogen.utils.nmodl import load_nmodl_files


def setup_myogen(verbose: bool = True, force_nmodl_reload: bool = False):
    """
    Set up MyoGen with automatic configuration.

    This function handles all necessary setup including NMODL compilation
    and loading. Users should call this once at the start of their script
    for best results.

    Args:
        verbose: If True, print setup progress messages
        force_nmodl_reload: If True, force recompilation of NMODL files

    Returns:
        bool: True if setup was successful, False otherwise

    Example:
        >>> from myogen.utils import setup_myogen
        >>> setup_myogen()
        >>> from myogen import simulator  # Now safe to import
    """
    if verbose:
        print("Setting up MyoGen...")

    success = load_nmodl_files(force_reload=force_nmodl_reload, quiet=not verbose)

    if success and verbose:
        print("✓ MyoGen setup complete!")
    elif not success and verbose:
        print("⚠ MyoGen setup encountered issues, but may still work")

    return success


__all__ = ["load_nmodl_files", "setup_myogen"]
