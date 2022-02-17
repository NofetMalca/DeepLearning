# DeepLearning
In the project we were required to create a system for deep learning of recognizing 7 fonts for English letters and signs. I chose for this purpose to create my own model, and not to take an existing model in order to get to know the layers of the model better, and not to take something finished. I chose to work in Google Colab because I do not have a powerful video card in my computer.

#For running the TEST:
Download the model and weights from the link below (final_model.json), (final_model.h5):
final_model.h5-
https://drive.google.com/file/d/1-2XLtBSAZwQmzJh9oaAJA0TxI7PU1Hm-/view?usp=sharing
final_model.json-
https://drive.google.com/file/d/1pD_YyBVhc1W09XoMrn3qNhYgsvd9ZIcX/view?usp=sharing


Put these two files, as well as in addition the file of the training set we received SynthText.h5 
and the test set we received SynthText_test.h5 in the TEST folder.
(Update projectTest.py with file names if necessary)

The files that should be in the TEST folder eventually:
• Final_model.h5 (my model)
• Final_model.json (my model)
• SynthText.h5 (we received)
• SynthText_test.h5 (we received)
• projectTest.py (my test)

Then you can run the projectTest.py file and it will export 
the prediction file to predictionsFonts.csv.
