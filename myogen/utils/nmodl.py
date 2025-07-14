"""
Initialize and set up NMODL (NEURON MODeling Language) files for the model.

This module handles the compilation and loading of NMODL files, which are used to define
custom mechanisms and models in NEURON simulations. It performs the following steps:
1. Locates and copies NMODL files to the appropriate directory
2. Compiles the NMODL files (platform-specific approach)
3. Loads the compiled files into NEURON

The module is automatically executed when the package is imported.
"""

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from neuron import h


def find_nmodl_directory() -> Path:
    """Find the NMODL directory in the pyNN/neuron installation."""
    src_path = Path(__file__).parent.parent / "simulator"
    try:
        nmodl_path = next(src_path.parent.parent.rglob("*pyNN/neuron/nmodl"))
        return nmodl_path
    except StopIteration:
        print("Error: Could not find pyNN/neuron/nmodl directory")
        raise FileNotFoundError("Could not find pyNN/neuron/nmodl directory")


def _copy_mod_files(src_path: Path, nmodl_path: Path) -> List[Path]:
    """Copy .mod files from source to NMODL directory."""
    mod_files = list(src_path.glob("*mod"))
    if not mod_files:
        print("Warning: No .mod files found in source directory")
        return []

    for mod_file in mod_files:
        try:
            shutil.copy(mod_file, nmodl_path / mod_file.name)
            print(f"Copied {mod_file.stem}")
        except (shutil.Error, IOError) as e:
            print(f"Error: Failed to copy {mod_file.stem}: {str(e)}")
            raise

    return mod_files


def _find_mknrndll() -> Optional[Path]:
    """Find the mknrndll executable on Windows systems."""
    # Common locations for mknrndll
    possible_locations = [
        Path(os.environ.get("NEURONHOME", "")) / "bin",
        Path(os.environ.get("NEURONHOME", "")) / "mingw",
        Path("C:/nrn/bin"),
        Path("C:/Program Files/NEURON/bin"),
        Path("C:/Program Files (x86)/NEURON/bin"),
    ]

    print("Searching for mknrndll.bat in common locations...")
    for location in possible_locations:
        if location and location.parent.exists():  # Check if parent directory exists
            mknrndll_path = location / "mknrndll.bat"
            print(f"  Checking: {mknrndll_path}")
            if mknrndll_path.exists():
                print(f"  ✓ Found: {mknrndll_path}")
                return mknrndll_path
            else:
                print(f"  ✗ Not found")

    # Try to find it in PATH
    print("Searching for mknrndll.bat in PATH...")
    try:
        result = subprocess.run(
            ["where", "mknrndll.bat"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            found_path = Path(result.stdout.strip())
            print(f"  ✓ Found in PATH: {found_path}")
            return found_path
        else:
            print("  ✗ Not found in PATH")
    except Exception as e:
        print(f"  ✗ Error searching PATH: {e}")

    print("mknrndll.bat not found. Please ensure NEURON is properly installed.")
    return None


def _compile_mod_files_windows(nmodl_path: Path) -> None:
    """Compile NMODL files on Windows using mknrndll."""
    mknrndll_path = _find_mknrndll()

    if mknrndll_path is None:
        raise FileNotFoundError(
            "Could not find mknrndll.bat. Please make sure NEURON is properly installed "
            "and NEURONHOME environment variable is set correctly."
        )

    print(f"Using mknrndll: {mknrndll_path}")

    # Change to the directory containing the mod files and run mknrndll.bat
    original_dir = os.getcwd()
    try:
        os.chdir(nmodl_path)

        # Remove any existing DLL files to avoid conflicts
        for dll_file in nmodl_path.glob("*nrnmech.dll"):
            try:
                dll_file.unlink()
                print(f"Removed existing DLL: {dll_file.name}")
            except Exception as e:
                print(f"Warning: Could not remove {dll_file.name}: {e}")

        # On Windows, we need to use cmd.exe to run batch files
        cmd = ["cmd", "/c", str(mknrndll_path)]
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)

        # Check if stderr has any warnings (not necessarily errors)
        if result.stderr:
            print(f"Compilation warnings/info: {result.stderr}")

    except subprocess.CalledProcessError as e:
        print(f"Error during compilation: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        raise
    finally:
        os.chdir(original_dir)


def _compile_mod_files_unix(nmodl_path: Path) -> None:
    """Compile NMODL files on Unix-like systems using pyNN's utility."""
    from pyNN.utility.build import compile_nmodl

    try:
        print(f"Compiling NMODL files from {nmodl_path}")
        compile_nmodl(nmodl_path)
    except Exception as e:
        print(f"Error: Failed to compile NMODL files: {str(e)}")
        raise


def _compile_and_load_mod_files(nmodl_path: Path, mod_files: List[Path]) -> None:
    """Compile and load NMODL files into NEURON based on platform."""
    if not mod_files:
        print("No mod files to compile")
        return

    # Platform-specific compilation
    if platform.system() == "Windows":
        _compile_mod_files_windows(nmodl_path)
    else:
        _compile_mod_files_unix(nmodl_path)

    # Load the compiled mechanisms
    # The location and naming of compiled files differs between platforms
    if platform.system() == "Windows":
        # On Windows, NEURON creates a DLL file, but the name might have a prefix
        # Look for both 'nrnmech.dll' and '*.nrnmech.dll' patterns
        dll_files = list(nmodl_path.glob("*nrnmech.dll"))

        if dll_files:
            # Use the first DLL found (usually there should be only one)
            dll_path = dll_files[0]
            try:
                h.nrn_load_dll(str(dll_path))
                print(f"Successfully loaded {dll_path.name}")
            except Exception as e:
                print(f"Warning: Error loading {dll_path.name}: {str(e)}")
                print(
                    "This may be because some mechanisms are already loaded, which is usually not a problem."
                )
                # Continue execution - don't re-raise the exception
        else:
            print(
                f"Warning: No nrnmech.dll file was found after compilation in {nmodl_path}"
            )
            print("Available files:")
            for item in nmodl_path.iterdir():
                if item.is_file():
                    print(f"  {item.name}")
    else:
        # On Unix, load individual .o files
        for mod_file in mod_files:
            o_file_path = str(nmodl_path / f"{mod_file.stem}.o")
            try:
                print(f"Loading {o_file_path}")
                h.nrn_load_dll(o_file_path)
                print(f"Successfully loaded {mod_file.stem}")
            except Exception as e:
                print(f"Warning: Failed to load {mod_file.stem}: {str(e)}")
                print(
                    "This may be because the mechanism is already loaded, which is usually not a problem."
                )


def load_nmodl_files(force_reload: bool = False, quiet: bool = False):
    """
    Main function to handle NMODL file setup.

    Args:
        force_reload: If True, force recompilation even if mechanisms seem loaded
        quiet: If True, suppress most output messages
    """

    def log(message: str):
        if not quiet:
            print(message)

    # Check if mechanisms are already loaded by trying to import neuron and check for our mechanisms
    if not force_reload:
        try:
            from neuron import h

            # Try to access one of our custom mechanisms to see if it's already loaded
            h.AdExpIF  # This will fail if mechanisms aren't loaded, which is what we want
            log("NMODL mechanisms appear to already be loaded, skipping reload")
            return True
        except (AttributeError, NameError):
            # Mechanisms not loaded, proceed with loading
            pass
        except ImportError:
            log("Warning: NEURON not available, skipping NMODL loading")
            return False
        except Exception as e:
            log(f"Warning: Error checking mechanism status: {e}")
            # Continue with loading attempt

    try:
        src_path = Path(__file__).parent.parent / "simulator"
        log(f"Loading NMODL files from {src_path}")

        nmodl_path = find_nmodl_directory()
        mod_files = _copy_mod_files(src_path / "nmodl_files", nmodl_path)

        if mod_files:
            _compile_and_load_mod_files(nmodl_path, mod_files)
            log("NMODL files processing complete!")
            return True
        else:
            log("Warning: No NMODL files were processed")
            return False

    except Exception as e:
        log(f"Error during NMODL setup: {str(e)}")
        if not quiet:
            # Log the error but don't crash the program
            import traceback

            traceback.print_exc()
        return False


# load_nmodl_files()
