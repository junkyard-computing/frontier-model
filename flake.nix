{
  description = "frontier-model: run large MoE LLMs on the Pixel Fold (CPU + GPU + TPU)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.cmake pkgs.ninja pkgs.gcc pkgs.pkg-config
          pkgs.curl pkgs.unzip pkgs.liburing pkgs.fio
          pkgs.vulkan-headers pkgs.vulkan-loader pkgs.shaderc pkgs.glslang pkgs.spirv-headers pkgs.vulkan-tools
          (pkgs.python3.withPackages (ps: [ ps.numpy ps.pyyaml ps.torch ]))
        ];
        # llama.cpp builds with -march=native (GGML_NATIVE); the cc wrapper strips it otherwise
        NIX_ENFORCE_NO_NATIVE = "0";
      };
    };
}
