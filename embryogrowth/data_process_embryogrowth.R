#####
# CITATION:
# embryogrowth: tools to analyze the thermal reaction 
# norm of embryo growth. 7.7 ed: the comprehensive R archive 
# network
######
# install.packages("embryogrowth")
# install.packages("energy")
# library(embryogrowth)
# library(energy)

data(DatabaseTSD)
DatabaseTSD.version() # "2022-03-29"


species.ori <- DatabaseTSD$Species
species.set <- unique(species.ori)
species.matrix <- array(0, c(length(species.ori), length(species.set)))
species.names <- c()
ind <- 1
for (i in species.set) {
  species.matrix[species.ori == i, ind] <- 1
  species.names <-c(species.names, i)
  ind <- ind + 1
}
species.matrix
species.names



subspecies.ori <- DatabaseTSD$Subspecies
subspecies.set <- unique(subspecies.ori)
subspecies.matrix <- array(0, c(length(subspecies.ori), length(subspecies.set)))
ind <- 1
for (i in subspecies.set) {
  if (is.na(i)) {
    subspecies.matrix[is.na(subspecies.ori), ind] <- 1
  } else {
    subspecies.matrix[subspecies.ori == i, ind] <- 1
  }
  ind<- ind + 1
}
colSums(subspecies.matrix)
rowSums(subspecies.matrix)




area.ori <- DatabaseTSD$Area
area.set <- unique(area.ori)
area.matrix <- array(0, c(length(area.ori), length(area.set)))
ind <- 1
for (i in area.set) {
  area.matrix[area.ori == i, ind] <- 1
  ind <- ind + 1
}
rowSums(area.matrix)
colSums(area.matrix)

# For marine turtles, name of the RMU for this population
# Regional Management Units
# An organization for collecting these data.
RMU.ori <- DatabaseTSD$RMU
RMU.ori <- as.character(RMU.ori)
RMU.ori[is.na(RMU.ori)] <- rep('0', length(RMU.ori[is.na(RMU.ori)]))
RMU.set <- unique(RMU.ori)
RMU.matrix <- array(0, c(length(RMU.ori), length(RMU.set)))
ind <- 1
for (i in RMU.set) {
  RMU.matrix[RMU.ori == i, ind] <- 1
  ind<- ind + 1
}
RMU.matrix
colSums(RMU.matrix)
rowSums(RMU.matrix)


amp.ori <- DatabaseTSD$Incubation.temperature.Amplitude

ipmean.ori <- DatabaseTSD$IP.mean

temperature.ori <- DatabaseTSD$Incubation.temperature



dcor(species.matrix, area.matrix)

ind_ip.not.na <- !is.na(ipmean.ori)

dcor(species.matrix[ind_ip.not.na,], ipmean.ori[ind_ip.not.na])

X <- cbind(species.matrix, subspecies.matrix, area.matrix, RMU.matrix, amp.ori, ipmean.ori, temperature.ori)
y <- DatabaseTSD$Females / (DatabaseTSD$Females + DatabaseTSD$Males)

###
###
# Remove NA
ind.used <- !is.na(y)
###
###
sum(ind.used)

# Use data from a temperature-regulated chamber
ind.constant <- DatabaseTSD$Incubation.temperature.Constant == 'TRUE'


ind.all.required <- array(FALSE, length(y))
for (r in 1:length(y)) {
  if (ind.constant[r] && ind.used[r]) {
    ind.all.required[r] <- TRUE
  }
}

ind.constant <- ind.constant[ind.used]
X <- X[ind.used, ]
y <- y[ind.used]
X <- X[ind.constant, ]
y <- y[ind.constant]

dim(X) # 874, 168


for (item in list(species.matrix, subspecies.matrix, area.matrix, RMU.matrix, amp.ori, ipmean.ori, temperature.ori)) {
  # size_of_each_group
  # version "2022-03-29"
  print(dim(item)[2])
}


# Group infomation:
group_name <- c('species', 'subspecies', 'area', 'RMU', 'amplitude', 'incubation periods (days)', 'temperature')
size_of_each_group <- c(60, 10, 81, 14, 1, 1, 1)


file <- '/Users/xbb/Dropbox/collaborative_trees/embryogrowth/data_process_embryogrowth.rds'

# pyreadr only loads data.frame
# saveRDS(data.frame(X, y), file = file)

# readRDS(file)

obj <- list('DatabaseTSD' = DatabaseTSD, 'version' = DatabaseTSD.version())
file <- '/Users/xbb/Dropbox/collaborative_trees/embryogrowth/original_data_embryogrowth.rds'
sum(DatabaseTSD$Version == '2019-11-19')



###
###
# Data diagnosis
ind.all.required

'Caretta caretta'
'Lepidochelys olivacea'
'Chelonia mydas'
'Chelydra serpentina'
'Emys orbicularis'

which(species.ori == 'Emys orbicularis')

ratio.y <- DatabaseTSD$Females / (DatabaseTSD$Females + DatabaseTSD$Males)
temp.x <- DatabaseTSD$Incubation.temperature


ind_set <- which(species.ori == 'Chelydra serpentina')
plot(temp.x[ind_set], ratio.y[ind_set])

