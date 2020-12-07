Setup (only complete these stesps once):
Create your QuoTable folder (i.e. C:\QuoTable) and run the commands below from that folder:
git clone https://github.com/google-research/tapas.git
pip install ./tapas
gsutil cp gs://tapas_models/2020_04_21/tapas_sqa_base.zip . && unzip tapas_sqa_base.zip
** in my case the unzip portion of this command errored so I manually unziped the file
copy all 6 files from the QuoTable Local Install folder to your QuoTable folder

To Run QuoTable:
python QuoTable_UI.py

Once in QuoTable the basic UI should explain itself.  There are only 3 commands (including exit).
When entering the table name include the extension of the file (i.e. NBATop200.csv)
Current QuoTable is very verbose as it runs (due to TAPAS) hopefully this is not an issue once we go to web.
Currently QuoTable is not user friendly as I was not expecting this UI to be relevant once we go to web.
If we decide to do a local demo for presentation please let me with some lead time so I can clean it up.


AMIN EDIT:
Setup (only complete these stesps once):
Create your QuoTable folder (i.e. C:\QuoTable) and run the commands below from that folder:
### Need to use another version instead of below:
#### git clone https://github.com/google-research/tapas.git
##Manually download this version:
https://github.com/google-research/tapas/tree/0d86b11d7129f1db01757fa7ebead78e80dd8ba0

pip install tapas

        IMPORTANT: if you get the following error:
        ERROR: Cannot uninstall 'PyYAML'. It is a distutils installed project and thus w e cannot accurately determine which files belong to it which would lead to only a partial uninstall.

        Instead do:
        pip install tapas --ignore-installed PyYAML

pip install gsutil

gsutil cp gs://tapas_models/2020_04_21/tapas_sqa_base.zip . && unzip tapas_sqa_base.zip
** in my case the unzip portion of this command errored so I manually unziped the file
copy all 6 files from the QuoTable Local Install folder to your QuoTable folder

To Run QuoTable:
python QuoTable_UI.py

Once in QuoTable the basic UI should explain itself.  There are only 3 commands (including exit).
When entering the table name include the extension of the file (i.e. NBATop200.csv)
Current QuoTable is very verbose as it runs (due to TAPAS) hopefully this is not an issue once we go to web.
Currently QuoTable is not user friendly as I was not expecting this UI to be relevant once we go to web.
If we decide to do a local demo for presentation please let me with some lead time so I can clean it up.
