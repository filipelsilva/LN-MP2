{ pkgs ? import <nixpkgs> {} }:
let
	packages = ps: with ps; [
		pandas
		numpy
		matplotlib
		scipy
		scikit-learn
	];
	my-python = pkgs.python3.withPackages packages;
in my-python.env

