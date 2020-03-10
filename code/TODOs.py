# *) in the __getitem__ we search across the list of relations.. isn't it better to do that in advance?
# i.e. building a search tree one time or something like that?

# *) also in __getitem__, some MID are missing in the trainset under train.
# for example: F0129/MID1/ is in the relations but not in the train-folder!!

# *) normalize data. notice that we have both black&white images and color ones! pre-process?

# *) change optimizer to ADAM