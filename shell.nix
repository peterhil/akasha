# shell.nix

with import (
    fetchTarball "https://github.com/NixOS/nixpkgs/archive/refs/tags/21.05.tar.gz"
) {};

pkgs.mkShell {
    nativeBuildInputs = with pkgs; [
        SDL2
        SDL2_image
        SDL2_mixer
        SDL2_ttf
        # blas
        # expat
        freetype
        # gfortran
        # lapack
        libogg
        libpng
        libsamplerate
        libsndfile
        portaudio
        # readline
    ];

    # shellHook = ''
    #     export LD_LIBRARY_PATH="${pkgs.libsamplerate}/lib:${pkgs.libsndfile}/lib:${pkgs.portaudio}/lib:$LD_LIBRARY_PATH"

    #     export DYLD_FALLBACK_LIBRARY_PATH="${pkgs.libsamplerate}/lib:${pkgs.libsndfile}/lib:${pkgs.portaudio}/lib:$DYLD_FALLBACK_LIBRARY_PATH"

    #     # Change to 10.04 when Nix switches to newer SDK:
    #     export MACOSX_DEPLOYMENT_TARGET="10.12"
    # '';
}
