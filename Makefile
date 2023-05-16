all:	models env

models:
	unzip -d ./FC_Model/ FC_Model/other_FC_model.zip
	unzip -d ./MD_Model/ MD_Model/other_MD_model.zip

env:
	conda create -y -n GraphTheoryDetector
	./env.sh