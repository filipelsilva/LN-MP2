.PHONY: pdf
pdf:
	pandoc -s -V papersize:a4  -V geometry:margin=1in -V colorlinks=true report.md -o 45.pdf

.PHONY: clean
clean:
	rm -rf 45.pdf
