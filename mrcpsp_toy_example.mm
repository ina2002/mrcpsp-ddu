************************************************************************
file with basedata            : mrcpsp_toyexample.bas
initial value random generator: 1424959589
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  8
horizon                       :  40
RESOURCES
  - renewable                 :  1   R
  - nonrenewable              :  0   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1      6      na      na       na       na
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          2           2   3
   2        2          2           4   5
   3        3          1           7
   4        2          1           6
   5        2          1           6
   6        2          1           7
   7        2          1           8
   8        1          0
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1
------------------------------------------------------------------------
   1     1      0       0
   2     1      3       6
         2      9       5
   3     1      3       9
         2      5       7
         3      7       6
   4     1      4       9
         2     10       4
   5     1      2       2
         2      5       2
   6     1      3       5
         2      7       5
   7     1      2       2
         2      6       1
   8     1      0       0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1
   10
************************************************************************
