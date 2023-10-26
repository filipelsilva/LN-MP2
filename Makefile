.PHONY: pdf
pdf:
	pandoc -s -V papersize:a4  -V geometry:margin=0.5in -V colorlinks=true report.md -o 45.pdf

.PHONY: zip
zip: pdf
	zip -r 45.zip 45.pdf reviews.py results.txt

.PHONY: clean
clean:
	rm -rf 45.pdf 45.zip
