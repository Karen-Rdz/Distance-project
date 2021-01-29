file = open("Test.txt", "r")
# open(file, mode)
# Types of modes:
#   "r" = Read
#   "w" = Write
#   "a" = Write at the end of the file
#   "r+" = Read and Write

file.write ("Hello  World\n")
# write = (Text)

file.close()

# read = Text in one line
# read(Num) = Number of letter to read
# readline(Num) = Just one line:
    # When you use ir one time it returns the first one, when you use it again it returns the next one

print file.readlines()

for line in file:
    print line,
# Another way to show lines