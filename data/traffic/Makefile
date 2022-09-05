.PHONY: all


all: GTSRB
	@echo Done!

clean:
	rm -f *.zip

clear: clean
	rm *.csv
	rm -r GTSRB


GTSRB: GTSRB_Final_Test_GT.zip GTSRB_Final_Test_Images.zip GTSRB_Final_Training_Images.zip
	unzip GTSRB_Final_Test_GT.zip
	unzip GTSRB_Final_Test_Images.zip
	unzip GTSRB_Final_Training_Images.zip	



GTSRB_Final_Test_GT.zip:
	curl -L -O https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip

GTSRB_Final_Test_Images.zip:
	curl -L -O https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip

GTSRB_Final_Training_Images.zip:
	curl -L -O https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip

