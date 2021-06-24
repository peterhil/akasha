{
  pkgs ?
  import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/refs/tags/21.05.tar.gz") {}
}:

pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    SDL2
    SDL2_image
    SDL2_mixer
    SDL2_ttf
    freetype
    libogg
    libpng
    libsamplerate
    libsndfile
    portaudio
  ];
}
